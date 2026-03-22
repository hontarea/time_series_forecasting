from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

import optuna
import pandas as pd

from tsf.data.dataset import Dataset
from tsf.execution.backtester import Backtester
from tsf.execution.strategy import BaseStrategy, SignBasedStrategy
from tsf.execution.walk_forward import WalkForwardEngine
from tsf.utils.evaluation import Evaluation, Metric


class Optimizer:
    """
    Optuna hyperparameter optimiser.

    Args:
        engine : WalkForwardEngine
            The walk-forward engine whose model params will be tuned.
        dataset : Dataset
            Full market dataset (used by the backtester).
        metric : Metric
            The metric to optimise (e.g. Metric.SHARPE_RATIO).
        search_space : dict
            Mapping of param name and (type, *args[, kwargs_dict]).
            Example: {"alpha": ("float", 1e-2, 1e2, {"log": True})}.
        strategy : BaseStrategy, optional
            Strategy for return calculation (default SignBasedStrategy).
        n_trials : int
            Number of Optuna trials (default 20).
        direction : str
            "maximize" or "minimize" (default "maximize").
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler.
    """

    def __init__(
        self,
        engine: WalkForwardEngine,
        dataset: Dataset,
        metric: Metric,
        search_space: Dict[str, Tuple],
        strategy: Optional[BaseStrategy] = None,
        n_trials: int = 20,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
    ):
        self.engine = engine
        self.dataset = dataset
        self.metric = metric
        self.search_space = search_space
        self.strategy = strategy or SignBasedStrategy()
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study: Optional[optuna.Study] = None

    def run(self) -> None:
        """
        Create an Optuna study and run n_trials optimisation trials.

        Results are stored in self.study; access via best_params and best_value.
        """
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
        )
        self.study.optimize(self._objective, n_trials=self.n_trials)

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            params = self._suggest_params(trial)
            self.engine.set_model_params(params)

            predictions = self.engine.run()

            if predictions.empty or "prediction" not in predictions.columns:
                raise optuna.TrialPruned()

            bt = Backtester(
                dataset=self.dataset,
                strategy=self.strategy,
                horizon=self.engine.horizon,
            )
            result = bt.run(predictions["prediction"])
            score = result["metrics"].get(self.metric.value)

            if score is None:
                raise ValueError(f"Metric '{self.metric.value}' not computed.")

            print(f"Trial {trial.number}: {self.metric.value} = {score:.4f}")
            return score

        except Exception as e:
            print(f"Trial {trial.number} pruned/failed: {e}")
            raise optuna.TrialPruned()

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, (ptype, *args) in self.search_space.items():
            kwargs = {}
            if args and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]
            suggest_fn = getattr(trial, f"suggest_{ptype}")
            params[name] = suggest_fn(name, *args, **kwargs)
        return params

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.study.best_params if self.study else {}

    @property
    def best_value(self) -> float:
        return self.study.best_value if self.study else float("nan")
