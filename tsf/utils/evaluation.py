from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Metric(Enum):
    """Available performance metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"


class Evaluation:
    """
    Static methods for computing financial performance metrics.
    """

    @staticmethod
    def compute_all(
        strategy_returns: pd.Series,
        annual_factor: int = 365,
    ) -> dict:
        """
        Compute all metrics at once.

        Args:
        
        strategy_returns : pd.Series
            Per-period simple returns.
        annual_factor : int
            Periods per year (365 for crypto-hourly, 252 for daily equities).
        """
        return {
            Metric.SHARPE_RATIO.value: Evaluation.sharpe_ratio(strategy_returns, annual_factor),
            Metric.TOTAL_RETURN.value: Evaluation.total_return(strategy_returns),
            Metric.MAX_DRAWDOWN.value: Evaluation.max_drawdown(strategy_returns),
            Metric.VOLATILITY.value: Evaluation.volatility(strategy_returns, annual_factor),
            Metric.WIN_RATE.value: Evaluation.win_rate(strategy_returns),
        }

    @staticmethod
    def sharpe_ratio(returns: pd.Series, annual_factor: int = 365) -> float:
        if returns.empty or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(annual_factor)

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        return float((1 + returns).prod() - 1)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def volatility(returns: pd.Series, annual_factor: int = 365) -> float:
        return float(returns.std() * np.sqrt(annual_factor))

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        active = returns[returns != 0]
        if active.empty:
            return 0.0
        return float(len(active[active > 0]) / len(active))
