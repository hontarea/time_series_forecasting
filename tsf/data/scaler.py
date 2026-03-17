from __future__ import annotations

from typing import List, Optional

import numpy as np
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

    def __init__(self, method: str = "standard", columns: Optional[List[str]] = None, scale_labels: bool = True):
        if method not in self._SCALERS:
            raise ValueError(
                f"Unknown scaling method '{method}'. "
                f"Choose from: {list(self._SCALERS.keys())}"
            )
        self.method = method
        self.columns = columns
        self.scale_labels = scale_labels
        self.scaler = self._SCALERS[method]()
        self._label_scaler = self._SCALERS[method]() if scale_labels else None
        self.fitted = False

    def fit(self, dataset: Dataset) -> FeatureScaler:
        cols = self._resolve_columns(dataset)
        self.scaler.fit(dataset.df[cols])
        if self.scale_labels and self._label_scaler is not None:
            label_cols = dataset.label_cols
            if label_cols:
                self._label_scaler.fit(dataset.df[label_cols])
        self.fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform.")

        cols = self._resolve_columns(dataset)
        dataset.df[cols] = self.scaler.transform(dataset.df[cols])
        if self.scale_labels and self._label_scaler is not None:
            label_cols = dataset.label_cols
            if label_cols:
                dataset.df[label_cols] = self._label_scaler.transform(dataset.df[label_cols])
        return dataset

    def inverse_transform_labels(self, arr: np.ndarray) -> np.ndarray:
        """
        Inverse-transform label predictions back to the original scale.

        Args:
            arr : np.ndarray, shape (n, n_labels)
        Returns:
            np.ndarray of the same shape in original label units.
        """
        if not self.scale_labels or self._label_scaler is None:
            return arr
        print("Executed inverse scaler on labels")
        return self._label_scaler.inverse_transform(arr)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        return self.fit(dataset).transform(dataset)

    # Internals helpers                                                   
    def _resolve_columns(self, dataset: Dataset) -> List[str]:
        """Determine which columns to scale."""
        if self.columns:
            return self.columns
        return dataset.feature_cols

    def __repr__(self) -> str:
        return f"FeatureScaler(method='{self.method}', columns={self.columns}, scale_labels={self.scale_labels}, fitted={self.fitted})"
