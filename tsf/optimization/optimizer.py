from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna

from tsf.execution.walk_forward import WalkForwardEngine


class Optimizer:
    """
    Optuna hyperparameter optimiser.

    Scores each trial by the mean inner-validation loss across all
    walk-forward folds.  The test window is never used during
    optimisation, preventing information leakage.

    Args:
        engine : WalkForwardEngine
            The walk-forward engine whose model params will be tuned.
            Must be configured with val_ratio > 0 so that each fold
            produces a validation loss.
        search_space : dict
            Mapping of param name and (type, *args[, kwargs_dict]).
            Example: {"lr": ("float", 1e-4, 1e-1, {"log": True})}.
        n_trials : int
            Number of Optuna trials (default 20).
        direction : str
            "minimize" or "maximize" (default "minimize" — lower val
            loss is better).
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler.
    """

    def __init__(
        self,
        engine: WalkForwardEngine,
        search_space: Dict[str, Tuple],
        n_trials: int = 20,
        direction: str = "minimize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
    ):
        self.engine = engine
        self.search_space = search_space
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

            val_losses: list[float] = []
            for _, _, _, fold_idx, fold_val_loss in self.engine.run_fold_by_fold():
                if fold_val_loss is None:
                    raise optuna.TrialPruned()
                val_losses.append(fold_val_loss)
                running_avg = float(np.mean(val_losses))
                trial.report(running_avg, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not val_losses:
                raise optuna.TrialPruned()

            score = float(np.mean(val_losses))
            print(f"Trial {trial.number}: mean_val_loss = {score:.6f}")
            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
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
