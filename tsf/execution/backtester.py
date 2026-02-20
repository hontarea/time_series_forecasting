from __future__ import annotations

from typing import Dict, Optional, Type

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset
from tsf.execution.strategy import BaseStrategy, SignBasedStrategy
from tsf.utils.evaluation import Evaluation


class Backtester:
    """
    End-to-end backtesting engine.

    Args:
    
    dataset : Dataset
        The full market dataset.
    strategy : BaseStrategy
        The signal-generation strategy.
    returns_col : str
        Name of the column in dataset holding market simple returns.
        If the column does not exist but "log_return" does, simple
        returns are computed automatically.

    Example:

        bt = Backtester(dataset, strategy=SignBasedStrategy())
        result = bt.run(predictions_df["prediction"])
        print(result["metrics"])
    """

    def __init__(
        self,
        dataset: Dataset,
        strategy: Optional[BaseStrategy] = None,
        returns_col: str = "simple_return",
    ):
        self.dataset = dataset
        self.strategy = strategy or SignBasedStrategy()
        self.returns_col = returns_col

        self.market_returns = self._resolve_market_returns()

    def run(self, predictions: pd.Series, annual_factor: int = 365) -> Dict:
        """
        Execute the backtest.

        Args:
        
        predictions : pd.Series
            Model predictions, indexed to align with the dataset.
        annual_factor : int
            Annualisation factor for Sharpe / volatility (365 for crypto,
            252 for equities).

        Returns:
        
        dict with keys:
            "strategy_returns" - pd.Series of per-step returns.
            "equity_curve"     - pd.Series of cumulative wealth.
            "metrics"          - dict of scalar performance metrics.
        """
        strategy_returns = self.strategy.calculate_returns(
            predictions, self.market_returns
        )

        equity_curve = (1 + strategy_returns).cumprod()
        metrics = Evaluation.compute_all(strategy_returns, annual_factor=annual_factor)

        return {
            "strategy_returns": strategy_returns,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }

    def _resolve_market_returns(self) -> pd.Series:
        """
        Find/Compute a market simple-returns series from the dataset.
        """
        df = self.dataset.df
        if self.returns_col in df.columns:
            return df[self.returns_col]

        # Compute from log_return or close
        if "log_return" in df.columns:
            return np.exp(df["log_return"]) - 1

        if self.dataset.ohlcv.get("close") in df.columns:
            return df[self.dataset.ohlcv["close"]].pct_change().shift(-1)

        raise ValueError(
            f"Cannot resolve market returns. "
            f"Expected column '{self.returns_col}', 'log_return', or a 'close' column."
        )