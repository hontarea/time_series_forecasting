from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Data format declaration                                            
class DataFormat(Enum):
    """Declares the data format a model adapter expects."""
    TABULAR = "tabular"            # (X, y) as numpy arrays / DataFrames
    TORCH_LOADER = "torch_loader"  # PyTorch DataLoader


class BaseModel(ABC):
    """
    Universal model interface.

    Args:

    input_columns : list[str]
        Feature column names the model expects.  When set, the execution
        layer filters features to this subset before calling predict().
    """

    def __init__(self, input_columns: Optional[List[str]] = None, **kwargs):
        self.input_columns: List[str] = input_columns or []

    # Data format declaration                                            
    @property
    @abstractmethod
    def data_format(self) -> DataFormat:
        """
        Declare the data format this adapter expects.

        Subclasses must return a class DataFormat member so the
        execution layer (i.e. WalkForwardEngine) differs scikit-learn 
        model and torch model. 
        """

    # Core methods                                                      
    @abstractmethod
    def fit(self, X, y=None) -> None:
        """Train the model.  Signature depends on data_format."""

    @abstractmethod
    def predict(self, X) -> pd.DataFrame:
        """Return predictions.  Signature depends on data_format."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the model to an untrained state (preserving hyper-params)."""

    # Persistence (optional override, sensible defaults) 
    def save(self, path: str | Path) -> None:
        """Serialize the trained model to disk."""
        raise NotImplementedError(f"{type(self).__name__} does not implement save().")

    def load(self, path: str | Path) -> None:
        """Load a previously saved model from disk."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load().")

    # Hyperparameter management                                          
    def set_params(self, params: Dict) -> None:
        """
        Update hyper-parameters from a dict.  Subclasses may override
        to delegate to their wrapped estimator (e.g. sklearn's set_params).
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_params(self) -> Dict:
        """
        Return current hyper-parameters. Subclasses should override
        for framework-specific behaviour.
        """
        return {}