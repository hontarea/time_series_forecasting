from __future__ import annotations

import copy
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class Dataset:
    """
    Unified data container for time-series modelling.

    Args: 

    dataframe : pd.DataFrame
        The underlying tabular data.
    feature_cols : list[str], optional
        Column names of the features
    label_cols : list[str], optional
        Column names of the prediction targets (labels).
    time_col : str, default "open_time_iso"
        Name of the datetime column used for temporal ordering.
    ohlcv_cols : dict, optional
        Mapping of canonical names with actual column names.
    """

    # Default OHLCV mapping (overridable per-instance)                   
    DEFAULT_OHLCV = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
                                                
    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        time_col: str = "open_time_iso",
        ohlcv_cols: Optional[dict] = None,
    ):
        self._df: pd.DataFrame = dataframe
        self._feature_cols: Set[str] = set(feature_cols) if feature_cols else set()
        self._label_cols: Set[str] = set(label_cols) if label_cols else set()
        self._time_col: str = time_col
        self._ohlcv: dict = ohlcv_cols or dict(self.DEFAULT_OHLCV)

        # Validate & sort by time on initialization
        self._ensure_datetime()

    # Properties                
    @property
    def df(self) -> pd.DataFrame:
        """The underlying DataFrame."""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        self._df = value

    @property
    def time_col(self) -> str:
        return self._time_col

    @property
    def feature_cols(self) -> List[str]:
        """Ordered list of current feature column names."""
        # Return in a stable order that respects the DataFrame column order
        return [c for c in self._df.columns if c in self._feature_cols]

    @property
    def label_cols(self) -> List[str]:
        """Ordered list of current label column names."""
        return [c for c in self._df.columns if c in self._label_cols]

    @property
    def ohlcv(self) -> dict:
        """OHLCV column name mapping."""
        return self._ohlcv

    @property
    def empty(self) -> bool:
        return self._df.empty

    @property
    def shape(self) -> tuple:
        return self._df.shape


    # OHLCV getters                                       
    @property
    def open(self) -> pd.Series:
        return self._df[self._ohlcv["open"]]

    @property
    def high(self) -> pd.Series:
        return self._df[self._ohlcv["high"]]

    @property
    def low(self) -> pd.Series:
        return self._df[self._ohlcv["low"]]

    @property
    def close(self) -> pd.Series:
        return self._df[self._ohlcv["close"]]

    @property
    def volume(self) -> pd.Series:
        return self._df[self._ohlcv["volume"]]


    # Feature management                                                 
    def get_features(self) -> pd.DataFrame:
        """Return a DataFrame view of the feature columns only."""
        return self._df[self.feature_cols]

    def add_features(self, data: pd.DataFrame | pd.Series, *, on: str | None = None) -> None:
        """
        Join new columns to the DataFrame and register them as features.

        Args:

        data : DataFrame | Series
            The new feature data.  Must be index-aligned or joinable via "on".
        on : str, optional
            Column name to join on (if not using index alignment).
        """
        if on:
            self._df = self._df.join(data, on=on)
        else:
            self._df = self._df.join(data)
        new_names = data.columns.tolist() if isinstance(data, pd.DataFrame) else [data.name]
        self._feature_cols.update(new_names) # Merge names of the new feature columns

    def remove_features(self, cols: List[str]) -> None:
        """Drop columns from both the DataFrame and the feature set."""
        self._df = self._df.drop(columns=cols, errors="ignore")
        self._feature_cols.difference_update(cols)


    # Label management                                                   
    def get_labels(self) -> pd.DataFrame:
        """Return a DataFrame view of the label columns."""
        return self._df[self.label_cols]

    def add_labels(self, data: pd.DataFrame | pd.Series) -> None:
        """Join new columns and register them as labels."""
        self._df = self._df.join(data)
        new_names = data.columns.tolist() if isinstance(data, pd.DataFrame) else [data.name]
        self._label_cols.update(new_names)

    def remove_labels(self, cols: List[str]) -> None:
        """Drop columns from both the DataFrame and the label set."""
        self._df = self._df.drop(columns=cols, errors="ignore")
        self._label_cols.difference_update(cols)

    
    #  Generic column access                     
    def get_columns(self, cols: List[str]) -> pd.DataFrame:
        """Return arbitrary columns without role annotation."""
        return self._df[cols]

    #  DataFrame mutation helpers                                         
    def dropna(self, *, reset_index: bool = True) -> None:
        """Drop rows with NaN values in-place, optionally resetting the index."""
        self._df.dropna(inplace=True)
        if reset_index:
            self._df.reset_index(drop=True, inplace=True)

    #  Model-ready data extraction                                        
    def get_tabular(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) as 2-D NumPy arrays for scikit-learn models.
        Each row is an independent entry. We specify the size of the 
        lookback window for each entry and the forecast horizon. 

        Returns:

        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_labels)
        """        
        X = self.get_features().to_numpy()
        y = self.get_labels().to_numpy().squeeze() # squeeze() collapses the shape (a, 1) or (1, a) to (a,) 
        return X, y

    def get_torch_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
    ):
        """
        Return a PyTorch DataLoader yielding (X_batch, y_batch)
        float32 tensors.

        Args:

        batch_size : int
            Mini-batch size (default 32).
        shuffle : bool

        Returns:

        torch.utils.data.DataLoader
        """
        try:
            import torch
            from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch is required for get_torch_loader(). "
                "Install it with: pip install torch"
            )

        X = self.get_features().to_numpy()
        y = self.get_labels().to_numpy()
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return TorchDataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_sequence_loader(
        self,
        lookback: int,
        horizon: int,
        batch_size: int = 32,
        shuffle: bool = False,
    ):
        """
        Return a PyTorch DataLoader of sliding-window sequences for
        training deep learning time-series models.

        Each sample is created by sliding a window of size
        lookback + horizon over the dataset with stride 1:

        - X[i] = features[i : i + lookback]          shape (lookback, F)
        - Y[i] = labels[i + lookback : i + lookback + horizon]  shape (horizon, L)

        Args:

        lookback : int
            Number of past time steps the model sees as input.
        horizon : int
            Number of future time steps the model must predict.
        batch_size : int
            Mini-batch size (default 32).
        shuffle : bool
            Whether to shuffle samples (True during training,
            False during evaluation).

        Returns:

        torch.utils.data.DataLoader
            Yields (X_batch, Y_batch) where
            X_batch has shape (B, lookback, n_features) and
            Y_batch has shape (B, horizon, n_labels).
        """
        try:
            import torch
            from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch is required for get_sequence_loader()."
                "Install it with: pip install torch"
            )

        features = self.get_features().to_numpy()  # (N, F)
        labels = self.get_labels().to_numpy()       # (N, L) or (N,)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        n = len(features)
        window = lookback + horizon
        if n < window:
            raise ValueError(
                f"Dataset has {n} rows but lookback ({lookback}) + "
                f"horizon ({horizon}) = {window} requires at least {window} rows."
            )

        num_samples = n - window + 1
        X_list = []
        Y_list = []
        for i in range(num_samples):
            X_list.append(features[i : i + lookback])
            Y_list.append(labels[i + lookback : i + lookback + horizon])

        X_arr = np.stack(X_list)  # (num_samples, lookback, F)
        Y_arr = np.stack(Y_list)  # (num_samples, horizon, L)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_arr, dtype=torch.float32)

        return TorchDataLoader(
            TensorDataset(X_tensor, Y_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
        )


    #  Slicing & Copying                                                  
    def slice_by_time(self, start: pd.Timestamp, end: pd.Timestamp) -> Dataset:
        """
        Return a new Dataset containing only rows from a specified period

        The returned Dataset shares column-role metadata but has its own
        DataFrame copy, so mutations are isolated.
        """
        time_series = self._df[self._time_col]
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)
        # Ensure timezone-naive comparison
        if time_series.dt.tz is not None:
            time_series = time_series.dt.tz_localize(None)

        mask = (time_series >= start) & (time_series < end)
        sliced_df = self._df.loc[mask].copy()

        return Dataset(
            dataframe=sliced_df,
            feature_cols=list(self._feature_cols),
            label_cols=list(self._label_cols),
            time_col=self._time_col,
            ohlcv_cols=dict(self._ohlcv),
        )

    def slice_by_index(self, start: int, end: int) -> Dataset:
        """
        Return a new Dataset for rows in the half-open range [start, end).
        """
        sliced_df = self._df.iloc[start:end].copy()
        return Dataset(
            dataframe=sliced_df,
            feature_cols=list(self._feature_cols),
            label_cols=list(self._label_cols),
            time_col=self._time_col,
            ohlcv_cols=dict(self._ohlcv),
        )

    def deepcopy(self) -> Dataset:
        """Return a fully independent clone."""
        return Dataset(
            dataframe=self._df.copy(),
            feature_cols=list(self._feature_cols),
            label_cols=list(self._label_cols),
            time_col=self._time_col,
            ohlcv_cols=copy.deepcopy(self._ohlcv),
        )


    # Time-column validation                      
    def _ensure_datetime(self) -> None:
        """Parse the time column to datetime and sort the DataFrame by it."""
        if self._time_col not in self._df.columns:
            raise ValueError(f"Time column '{self._time_col}' not found in dataset.")

        if not pd.api.types.is_datetime64_any_dtype(self._df[self._time_col]):
            try:
                self._df = self._df.copy()
                self._df[self._time_col] = pd.to_datetime(self._df[self._time_col])
            except Exception as e:
                raise ValueError(
                    f"Could not parse time column '{self._time_col}': {e}"
                ) from e

        self._df.sort_values(by=self._time_col, ascending=True, inplace=True, ignore_index=False)

    
    # Representation                                                    
    def __repr__(self) -> str:
        return (
            f"Dataset(shape={self.shape}, "
            f"features={len(self._feature_cols)}, "
            f"labels={len(self._label_cols)}, "
            f"time_col='{self._time_col}')"
        )

    def __len__(self) -> int:
        return len(self._df)
