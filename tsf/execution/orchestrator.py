from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset
from tsf.agents.base import BaseAgent
from tsf.environments.trading import TradingEnv
from tsf.execution.walk_forward import WalkForwardEngine
from tsf.utils.evaluation import Evaluation


class Orchestrator:
    """
    Interleaves supervised walk-forward forecasting with RL execution.

    For each fold:
      - Call agent.learn() ``n_learn_steps`` times (uses transitions
         accumulated in the replay buffer from previous folds).
      - Reset the environment with the fold's forecast and actual prices.
      - Run a full episode.
      - Store all transitions for future learning.

    Args:
        engine        : WalkForwardEngine producing fold-by-fold forecasts
        agent         : RL agent (must implement BaseAgent)
        env           : TradingEnv instance
        dataset       : full market Dataset (used to look up close prices)
        n_learn_steps : gradient update calls between folds (default 4)
        verbose       : print fold-level progress (default False)
    """

    def __init__(
        self,
        engine: WalkForwardEngine,
        agent: BaseAgent,
        env: TradingEnv,
        dataset: Dataset,
        n_learn_steps: int = 4,
        verbose: bool = False,
    ) -> None:
        self.engine = engine
        self.agent = agent
        self.env = env
        self.dataset = dataset
        self.n_learn_steps = n_learn_steps
        self.verbose = verbose

    def run(self, annual_factor: int = 8760) -> dict:
        """
        Execute the full RL pipeline across all walk-forward folds.

        Returns a dict compatible with Evaluation.compute_all():
            strategy_returns : pd.Series  - per-step simple returns
            equity_curve     : pd.Series  - compounded capital (starts at 1)
            step_pnls        : pd.Series  - per-step log P&L
            positions        : pd.Series  - per-step position {-1, 0, 1}
            metrics          : dict       - scalar performance metrics
        """
        carry_position: float = 0.0
        all_step_pnls: List[float] = []
        all_positions: List[float] = []
        all_timestamps: List[pd.Timestamp] = []
        pred_history: List[Tuple[np.ndarray, np.ndarray]] = []

        close = self.dataset.close

        for fold_preds, train_ds, test_ds, fold_idx, _ in self.engine.run_fold_by_fold(
            verbose=self.verbose
        ):
            # Offline agent learning
            for _ in range(self.n_learn_steps):
                self.agent.learn()

            forecast = fold_preds["prediction"].to_numpy().astype(np.float32)

            test_start = test_ds.df.index[0]
            test_timestamps = test_ds.df.index[: self.env.horizon]

            start_pos = close.index.get_loc(test_start)
            actual_prices = close.iloc[start_pos : start_pos + self.env.horizon + 1].to_numpy()

            forecast_accuracy = _compute_forecast_accuracy(pred_history)
            volatility = _compute_volatility(train_ds, self.dataset)

            obs, _ = self.env.reset(
                options={
                    "forecast": forecast,
                    "actual_prices": actual_prices,
                    "carry_position": carry_position,
                    "forecast_accuracy": forecast_accuracy,
                    "volatility": volatility,
                }
            )

            # Online execution loop
            for h in range(self.env.horizon):
                action = self.agent.select_action(obs, explore=False)
                old_obs = obs
                obs, reward, terminated, _, info = self.env.step(action)
                self.agent.store_transition(old_obs, action, reward, obs, terminated)

                all_step_pnls.append(info["step_pnl"])
                all_positions.append(info["position"])
                all_timestamps.append(test_timestamps[h])

                if terminated:
                    break

            carry_position = info["position"]

            actual_returns_this_fold = np.log(
                actual_prices[1:] / actual_prices[:-1]
            ).astype(np.float32)
            pred_history.append((forecast, actual_returns_this_fold))

            if self.verbose:
                cum_ret = sum(all_step_pnls)
                eps = getattr(self.agent, "epsilon", float("nan"))
                print(
                    f"  Fold {fold_idx + 1} done | "
                    f"cum_log_ret={cum_ret:.4f} | epsilon={eps:.4f}"
                )

        pnl_series = pd.Series(all_step_pnls, index=all_timestamps, name="step_pnl")
        capital = np.exp(pnl_series.cumsum())
        strategy_returns = np.exp(pnl_series) - 1
        strategy_returns.name = "strategy_return"

        metrics = Evaluation.compute_all(strategy_returns, annual_factor=annual_factor)

        return {
            "strategy_returns": strategy_returns,
            "equity_curve": capital,
            "step_pnls": pnl_series,
            "positions": pd.Series(all_positions, index=all_timestamps, name="position"),
            "metrics": metrics,
        }


def _compute_forecast_accuracy(
    pred_history: List[Tuple[np.ndarray, np.ndarray]],
    window: int = 10,
) -> float:
    """Rolling MAE over the last ``window`` folds. Returns 0.0 if no history."""
    if not pred_history:
        return 0.0
    recent = pred_history[-window:]
    mae_values = [
        float(np.mean(np.abs(forecast - actual)))
        for forecast, actual in recent
    ]
    return float(np.mean(mae_values))


def _compute_volatility(train_ds: Dataset, dataset: Dataset) -> float:
    """Std of log-returns of close prices over the training window."""
    close = dataset.close
    train_index = train_ds.df.index
    if len(train_index) < 2:
        return 0.0
    start_ts = train_index[0]
    end_ts = train_index[-1]
    mask = (close.index >= start_ts) & (close.index <= end_ts)
    train_close = close.loc[mask]
    if len(train_close) < 2:
        return 0.0
    log_rets = np.log(train_close.values[1:] / train_close.values[:-1])
    return float(np.std(log_rets))
