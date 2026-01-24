import optuna
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Type
from src.utils.Evaluation import Metric, Evaluation
from src.learner.Learner import Learner
from src.datawrapper.DataWrapper import DataWrapper
from src.simulation.Simulation import Simulation, SignBasedStrategy

class Optimizer:
    def __init__(
        self,
        wrapper: DataWrapper,
        learner: Learner,
        metric: Metric,
        search_space: Dict[str, Tuple],
        simulation_class: Type[Simulation] = SignBasedStrategy, 
        n_trials: int = 50,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
    ):
        self.wrapper = wrapper
        self.learner = learner
        self.metric = metric
        self.search_space = search_space
        self.simulation_class = simulation_class
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study = None

    def run(self):
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

    def _suggest_params(self, trial):
        params = {}
        for name, (ptype, *args) in self.search_space.items():
            kwargs = {}
            if args and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]
            suggest_fn = getattr(trial, f"suggest_{ptype}")
            params[name] = suggest_fn(name, *args, **kwargs)
        return params

    def _objective(self, trial):
        try:
            params = self._suggest_params(trial)
            self.learner.reset_state()
            self.learner.set_model_params(params)
            
            result_wrapper = self.learner.compute()
            
            df = result_wrapper.get_dataframe()
            if 'prediction' not in df.columns:
                 raise optuna.TrialPruned()
            
            predictions = df['prediction']
            
            sim = self.simulation_class(self.wrapper)
            strategy_returns = sim.calculate_returns(predictions)
            all_metrics = Evaluation.compute_all(strategy_returns)
            
            score = all_metrics.get(self.metric.value)
            
            if score is None:
                raise ValueError(f"Metric {self.metric.value} could not be computed.")
                
            print(f"Trial {trial.number}: {self.metric.value} = {score:.4f}")
            return score

        except Exception as e:
            print(f"Trial {trial.number} pruned/failed: {e}")
            raise optuna.TrialPruned()

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.study.best_params if self.study else {}

    @property
    def best_value(self) -> float:
        return self.study.best_value if self.study else float("nan")