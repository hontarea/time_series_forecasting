from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset
from tsf.execution.strategy import BaseStrategy, SignBasedStrategy
from tsf.utils.evaluation import Evaluation


class Backtester:
    """
    Per-window backtesting engine.

    The backtester operates at per-window granularity.  Each window
    corresponds to one prediction horizon (default 24 hours).  The model
    produces one forecast per window and the strategy holds one position
    for the entire window.  Transaction costs are incurred only when the
    position changes between consecutive windows.

    Capital is compounded via exp(cumulative log return) to remain
    consistent with the log-return prediction target.

    Args:
    
    dataset : Dataset
        The full market dataset (must contain a close price column).
    strategy : BaseStrategy, optional
        Signal-generation strategy (default SignBasedStrategy).
    horizon : int
        Number of time steps per prediction window (default 24).
    txn_cost : float
        Round-trip transaction cost deducted from a window's log return
        whenever the position changes (default 0.002 = 0.2 %).

    Example
    
        bt = Backtester(dataset, strategy=SignBasedStrategy(), horizon=24)
        result = bt.run(predictions["prediction"])
        print(result["metrics"])
    """

    def __init__(
        self,
        dataset: Dataset,
        strategy: Optional[BaseStrategy] = None,
        horizon: int = 24,
        txn_cost: float = 0.002,
    ):
        self.dataset = dataset
        self.strategy = strategy or SignBasedStrategy()
        self.horizon = horizon
        self.txn_cost = txn_cost

    # Helpers
    def _get_close(self) -> pd.Series:
        """Return the close-price series from the dataset."""
        ohlcv = getattr(self.dataset, "_ohlcv", {}) or {}
        close_col = ohlcv.get("close", "close")
        if close_col not in self.dataset.df.columns:
            raise ValueError(
                f"Close price column '{close_col}' not found in the dataset."
            )
        return self.dataset.df[close_col]

    #  Per-window construction
    def _build_window_df(self, predictions: pd.Series) -> pd.DataFrame:
        """
        Pair each per-window prediction with the realised cumulative sum of
        log return over that window.

        The walk-forward engine emits exactly one prediction per
        fold (the window-start observation).  This method computes
        the corresponding actual return:

            actual = ln(P_{t+H} / P_t)

        directly from close prices so that predicted and actual are on
        the same scale.

        Returns
 -
        pd.DataFrame
            Index : window-start timestamps (one per fold).
            Columns : predicted, actual.
        """
        close = self._get_close()
        actuals: list[float] = []

        for t in predictions.index:
            try:
                t_pos = close.index.get_loc(t)
                end_pos = t_pos + self.horizon
                if end_pos < len(close):
                    actual = float(
                        np.log(close.iloc[end_pos] / close.iloc[t_pos])
                    )
                else:
                    # Partial last window - use last available close
                    actual = float(
                        np.log(close.iloc[-1] / close.iloc[t_pos])
                    )
            except KeyError:
                actual = np.nan
            actuals.append(actual)

        return pd.DataFrame(
            {"predicted": predictions.values, "actual": actuals},
            index=predictions.index,
        )


    def run(self, predictions: pd.Series, annual_factor: int = 365) -> Dict:
        """
        Execute the per-window backtest.

        Args:

        predictions : pd.Series
            Model predictions indexed to align with the dataset.
        annual_factor : int
            Number of non-overlapping windows per year for
            annualisation (default 365 for daily windows).

        Returns:
        
        dict
            strategy_returns : pd.Series — per-window simple returns.
            equity_curve     : pd.Series — cumulative capital (starts at 1).
            market_returns   : pd.Series — per-window simple market returns.
            window_df        : pd.DataFrame — full per-window breakdown.
            metrics          : dict — scalar performance metrics.
        """
        window_df = self._build_window_df(predictions)

        if window_df.empty:
            empty = pd.Series(dtype=float)
            return {
                "strategy_returns": empty,
                "equity_curve": empty,
                "market_returns": empty,
                "window_df": window_df,
                "metrics": Evaluation.compute_all(empty, annual_factor),
            }

        # Signals (one per window) 
        signals = self.strategy.get_signal(window_df["predicted"])
        window_df["signal"] = signals

        # Transaction costs (on signal changes only)
        prev_signal = signals.shift(1, fill_value=0.0)
        signal_changed = (signals != prev_signal).astype(float)

        # Returns 
        gross_log_ret = signals * window_df["actual"]
        cost = np.log(1 - self.txn_cost) * signal_changed
        net_log_ret = gross_log_ret - cost

        window_df["gross_log_return"] = gross_log_ret
        window_df["txn_cost"] = cost
        window_df["net_log_return"] = net_log_ret

        # Capital (compound via exp) 
        capital = np.exp(net_log_ret.cumsum())
        window_df["capital"] = capital

        # Per-window simple returns (for metric computation)
        strategy_returns = np.exp(net_log_ret) - 1
        strategy_returns.name = "strategy_return"

        # Market returns (per-window, for buy-and-hold comparison)
        market_returns = np.exp(window_df["actual"]) - 1
        market_returns.name = "market_return"

        # Metrics 
        metrics = Evaluation.compute_all(
            strategy_returns, annual_factor=annual_factor,
        )

        return {
            "strategy_returns": strategy_returns,
            "equity_curve": capital,
            "market_returns": market_returns,
            "window_df": window_df,
            "metrics": metrics,
        }

    def __repr__(self) -> str:
        return (
            f"Backtester(strategy={self.strategy!r}, "
            f"horizon={self.horizon}, "
            f"txn_cost={self.txn_cost})"
        )