import numpy as np
import pandas as pd
from enum import Enum

class Metric(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"

class Evaluation:
    """
    Calculates performance metrics from Simple Returns (Percentage Change).
    """
    
    @staticmethod
    def compute_all(strategy_returns: pd.Series, annual_factor: int = 365) -> dict:
        """
        Args:
            strategy_returns: Series of simple returns (e.g., 0.01, -0.02)
        """
        return {
            Metric.SHARPE_RATIO.value: Evaluation.sharpe_ratio(strategy_returns, annual_factor),
            Metric.TOTAL_RETURN.value: Evaluation.total_return(strategy_returns),
            Metric.MAX_DRAWDOWN.value: Evaluation.max_drawdown(strategy_returns),
            Metric.VOLATILITY.value: Evaluation.volatility(strategy_returns, annual_factor),
            Metric.WIN_RATE.value: Evaluation.win_rate(strategy_returns)
        }

    @staticmethod
    def sharpe_ratio(returns: pd.Series, annual_factor: int = 365) -> float:
        if returns.empty or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(annual_factor)

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        return (1 + returns).prod() - 1

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.cummax()
        
        # Drawdown percentage
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    @staticmethod
    def volatility(returns: pd.Series, annual_factor: int = 365) -> float:
        return returns.std() * np.sqrt(annual_factor)

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        active_returns = returns[returns != 0]
        if active_returns.empty:
            return 0.0
        return len(active_returns[active_returns > 0]) / len(active_returns)