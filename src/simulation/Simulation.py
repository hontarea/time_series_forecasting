from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from src.datawrapper.DataWrapper import DataWrapper

class Simulation(ABC):
    def __init__(self, data_wrapper: DataWrapper):
        self.data_wrapper = data_wrapper
        self.market_data = data_wrapper.get_dataframe().copy()
        
        if 'log_return' in self.market_data.columns:
            self.market_simple_returns = np.exp(self.market_data['log_return']) - 1
        else:
            self.market_simple_returns = self.market_data['close'].pct_change()

    @abstractmethod
    def calculate_returns(self, predictions: pd.Series) -> pd.Series:
        """
        Converts predictions into Strategy Simple Returns (Percentage Change).
        """
        pass

class SignBasedStrategy(Simulation):
    """
    Directional Strategy:
    - Long (+1) / Short (-1) based on sign of prediction.
    - Returns are Simple Returns (e.g., 0.05 for 5% gain).
    """
    def calculate_returns(self, predictions: pd.Series) -> pd.Series:
        aligned_preds = predictions.reindex(self.market_simple_returns.index).fillna(0)
        signals = np.sign(aligned_preds)
        strategy_simple_returns = signals * self.market_simple_returns
        
        return strategy_simple_returns.dropna()

class ProportionalStrategy(Simulation):
    """
    Position size is proportional to the predicted return.
    Sensitive to magnitude changes, making it ideal for Optimization.
    """
    def __init__(self, data_wrapper: DataWrapper, leverage_scale: float = 100.0):
        super().__init__(data_wrapper)
        self.leverage_scale = leverage_scale

    def calculate_returns(self, predictions: pd.Series) -> pd.Series:
        leverage = predictions * self.leverage_scale
        
        leverage = leverage.clip(-2.0, 2.0)
        
        aligned_leverage = leverage.reindex(self.market_simple_returns.index).fillna(0)
        
        strategy_simple_returns = aligned_leverage * self.market_simple_returns
        return strategy_simple_returns.dropna()