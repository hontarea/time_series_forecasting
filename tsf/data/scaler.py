from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from tsf.data.dataset import Dataset


class FeatureScaler:
    """
    Wraps an sklearn scaler to operate on class Dataset feature columns.

    Args:

    method : str
        "standard" (Z-score), "minmax" (0-1), or "robust"
    columns : list[str], optional
        Subset of feature columns to scale.  If None, all feature
        columns are scaled.
    """

    _SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(self, method: str = "standard", columns: Optional[List[str]] = None):
        if method not in self._SCALERS:
            raise ValueError(
                f"Unknown scaling method '{method}'. "
                f"Choose from: {list(self._SCALERS.keys())}"
            )
        self.method = method
        self.columns = columns
        self.scaler = self._SCALERS[method]() # Access a scaler mathod and initialize it
        self.fitted = False

    # Fit / Transform 
    def fit(self, dataset: Dataset) -> FeatureScaler:
        """
        Learn scaling parameters from the feature columns of dataset.

        Returns self for chaining.
        """
        cols = self._resolve_columns(dataset)
        self.scaler.fit(dataset.df[cols])
        self.fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply the learned scaling to dataset in place and return it.
        """
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform.")

        cols = self._resolve_columns(dataset)
        dataset.df[cols] = self.scaler.transform(dataset.df[cols])
        return dataset

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Combined method: fit then transform in one call."""
        return self.fit(dataset).transform(dataset)

    # Internals helpers                                                   
    def _resolve_columns(self, dataset: Dataset) -> List[str]:
        """Determine which columns to scale."""
        if self.columns:
            return self.columns
        return dataset.feature_cols

    def __repr__(self) -> str:
        return f"FeatureScaler(method='{self.method}', columns={self.columns}, fitted={self.fitted})"
