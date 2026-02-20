from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset


class BaseStrategy(ABC):
    """
    Abstract strategy interface.

    Subclasses implement get_signal() to translate a prediction value
    into a position signal.
    """

    @abstractmethod
    def get_signal(self, predictions: pd.Series) -> pd.Series:
        """
        Map predictions to position signals.

        Args:

        predictions : pd.Series
            Raw model output, indexed by time.

        Returns:

        pd.Series
            Position signals, same index as input.
        """

    def calculate_returns(
        self,
        predictions: pd.Series,
        market_returns: pd.Series,
    ) -> pd.Series:
        """
        Compute strategy simple returns = position * market return.

        Args:

        predictions : pd.Series
            Model predictions.
        market_returns : pd.Series
            Market simple returns.

        Returns:
        
        pd.Series
            Strategy returns.
        """
        signals = self.get_signal(predictions)
        aligned_signals = signals.reindex(market_returns.index).fillna(0)
        return (aligned_signals * market_returns).dropna()


class SignBasedStrategy(BaseStrategy):
    """
    Directional strategy: long (+1) or short (-1) based on the sign
    of the prediction.
    """

    def get_signal(self, predictions: pd.Series) -> pd.Series:
        return np.sign(predictions)


class ThresholdStrategy(BaseStrategy):
    """
    Threshold strategy: go long when prediction exceeds "upper",
    short when below "lower", flat otherwise.

    Args:
    
    upper : float
        Threshold above which to go long (default 0.0).
    lower : float
        Threshold below which to go short (default 0.0).
    """

    def __init__(self, upper: float = 0.0, lower: float = 0.0):
        self.upper = upper
        self.lower = lower

    def get_signal(self, predictions: pd.Series) -> pd.Series:
        signals = pd.Series(0, index=predictions.index, dtype=float)
        signals[predictions > self.upper] = 1.0
        signals[predictions < self.lower] = -1.0
        return signals
