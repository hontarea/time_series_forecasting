from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import sklearn.base
from sklearn.base import BaseEstimator, is_classifier

from tsf.models.base import BaseModel, DataFormat

class SklearnAdapter(BaseModel):
    """
    Adapter for scikit-learn estimators.

    Args: 

    estimator : BaseEstimator
        A scikit-learn estimator instance (e.g. Ridge(alpha=1.0)).
    input_columns : list[str], optional
        Feature columns the model expects (filters applied by execution layer).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        input_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(input_columns=input_columns, **kwargs)
        self.estimator = estimator
        self.is_classifier_: bool = is_classifier(estimator)

    # Data format                                                        
    @property
    def data_format(self) -> DataFormat:
        return DataFormat.TABULAR

    #  Core methods                                                 
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        # in case y of the form (a,) or (,a) - transform to (a, 1) or (1, a) respectively
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        if hasattr(y, "ndim") and y.ndim > 1 and y.shape[1] == 1:
            y = y.squeeze()

        self.estimator.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.is_classifier_:
            if hasattr(self.estimator, "predict_proba"):
                probs = self.estimator.predict_proba(X)
                return probs[:, 1] if probs.shape[1] == 2 else probs
            else:

                return self.estimator.predict(X)
        else:
            return self.estimator.predict(X)

    def reset(self) -> None:
        """Clone the estimator to get an unfitted copy with the same params."""
        self.estimator = sklearn.base.clone(self.estimator)

    # Hyperparameters                                                    
    def set_params(self, params: Dict) -> None:
        """Delegate to the sklearn estimator's set_params."""
        self.estimator.set_params(**params)

    def get_params(self) -> Dict:
        return self.estimator.get_params()

    # Persistence                                                        
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.estimator, f)

    def load(self, path: str | Path) -> None:
        with open(Path(path), "rb") as f:
            self.estimator = pickle.load(f)
        self.is_classifier_ = is_classifier(self.estimator)

    # Repr                                                               
    def __repr__(self) -> str:
        return f"SklearnAdapter({self.estimator!r})"
