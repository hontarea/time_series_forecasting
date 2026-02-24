from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

import numpy as np

from tsf.data.dataset import Dataset
from tsf.data.scaler import FeatureScaler
from tsf.data.window import WindowGenerator
from tsf.models.base import BaseModel, DataFormat

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class WalkForwardEngine:
    """
    Walk-forward validation runner.

    Args:

    model : BaseModel
        The model to train and evaluate.
    window : WindowGenerator
        The window configuration producing train/test splits.
    scaler : FeatureScaler, optional
        If provided, the scaler is fit() on each training fold and
        transform() on both train and test folds.
    reset_model : bool
        Whether to call model.reset() before each fold (default True).
        Set to False for incremental learning.
    lookback : int
        Number of past time steps used as model input.  Required
        (> 0) for torch-based models (DataFormat.TORCH_LOADER).
        Also controls the number of auto-injected lagged features
        for tabular models.  Default is 336 (14 days of 1-hour data).
    horizon : int
        Number of future time steps to predict.  Required (> 0) for
        torch-based models.  Should match the test_window of the
        WindowGenerator (in number of data points).  Default is 24
        (1 day of 1-hour data).

    Example:

        engine = WalkForwardEngine(
            model=my_model, window=wg, scaler=scaler,
            lookback=336, horizon=24,
        )
        predictions = engine.run()
        # predictions is a pd.DataFrame with columns ["prediction"]
    """

    def __init__(
        self,
        model: BaseModel,
        window: WindowGenerator,
        scaler: Optional[FeatureScaler] = None,
        reset_model: bool = True,
        lookback: int = 336,
        horizon: int = 24,
    ):
        self.model = model
        self.window = window
        self.scaler = scaler
        self.reset_model = reset_model
        self.lookback = lookback
        self.horizon = horizon
        self._lagged_features_added = False

        # Test_window should match horizon
        self._validate_test_window_horizon()

    # Validation helpers
    def _validate_test_window_horizon(self) -> None:
        """Warn if test_window duration does not match horizon in hours."""
        try:
            expected_td = pd.to_timedelta(f"{self.horizon}h")
            if self.window.test_window != expected_td:
                warnings.warn(
                    f"test_window ({self.window.test_window}) does not match "
                    f"horizon={self.horizon} hours ({expected_td}).  "
                    f"These should typically be equal so that each fold "
                    f"predicts exactly one horizon ahead.",
                    UserWarning,
                    stacklevel=3,
                )
        except Exception:
            pass  

    # Auto-inject lagged features for tabular models
    def _ensure_lagged_features(self) -> None:
        """
        If the model expects TABULAR data and the dataset does not
        already contain log_return_lag_* columns, auto-inject
        lagged log-return features using self.lookback as the lag count.

        This gives sklearn models the lookback context that torch models
        get from sequence windows.
        """
        if self.model.data_format != DataFormat.TABULAR:
            return
        if self._lagged_features_added:
            return

        ds = self.window.dataset
        existing = [c for c in ds.feature_cols if c.startswith("log_return_lag_")]
        if existing:
            self._lagged_features_added = True
            return

        from tsf.data.feature_engineer import _REGISTRY
        fn = _REGISTRY.get("LAGGED_RETURNS")

        fn(ds, lags=self.lookback)
        ds.dropna(reset_index=True)
        self._lagged_features_added = True

    # Embargo trimming
    def _trim_train(self, train_ds: Dataset) -> Dataset:
        """Remove the last ``horizon`` rows from the training dataset."""
        n = len(train_ds)
        if n <= self.horizon:
            warnings.warn(
                f"Training set ({n} rows) is not larger than horizon "
                f"({self.horizon}).  Skipping embargo trim.",
                UserWarning,
                stacklevel=2,
            )
            return train_ds
        return train_ds.slice_by_index(0, n - self.horizon)

    # Run loop
    def run(self, verbose: bool = False) -> pd.DataFrame:
        """
        Execute the walk-forward loop and return all predictions.

        Every fold follows the same uniform pipeline:

        Trim - remove the last horizon rows from the
           training set (embargo period).
        Reset - optionally reset the model weights.
        Scale - fit scaler on the trimmed training set, then
           transform both the trimmed set and the test set.
        Train - fit the model on the trimmed training set.
        Predict - generate predictions for the test set.
           For torch models the prediction context comes from the
           full (untrimmed) training tail so the model sees the
           freshest available features at prediction time.

        Returns:
        
        pd.DataFrame
            Predictions from every test fold, concatenated.
            Columns: ["prediction"].
            Index: aligned with the original Dataset.
        """
        # Auto-inject lagged features for tabular models
        self._ensure_lagged_features()
        all_predictions: list[pd.DataFrame] = []
        splits = list(self.window.get_splits())

        if not splits:
            return pd.DataFrame()

        for fold_idx, (train_ds, test_ds) in enumerate(splits):
            # Trim
            trimmed_ds = self._trim_train(train_ds)

            if verbose:
                print(
                    f"Fold {fold_idx + 1}/{len(splits)}: "
                    f"train={len(train_ds)} rows "
                    f"(trimmed={len(trimmed_ds)}), "
                    f"test={len(test_ds)} rows"
                )

            # Reset
            if self.reset_model:
                self.model.reset()

            # Scale (fit on trimmed, transform trimmed + test)
            if self.scaler is not None:
                self.scaler.fit(trimmed_ds)
                self.scaler.transform(trimmed_ds)
                self.scaler.transform(test_ds)

            # Train on trimmed, predict on test
            fold_preds = self._dispatch_fold(
                trimmed_ds, test_ds, fold_idx,
                train=True,
                full_train_ds=train_ds,
            )
            all_predictions.append(fold_preds)

        return self._finalize(all_predictions)

    # Fold dispatch helpers
    def _dispatch_fold(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        fold_idx: int,
        *,
        train: bool,
        full_train_ds: Optional[Dataset] = None,
    ) -> pd.DataFrame:
        """Route to the appropriate fold handler based on data format."""
        if self.model.data_format == DataFormat.TABULAR:
            return self._run_tabular_fold(
                train_ds, test_ds, fold_idx, train=train,
            )
        elif self.model.data_format == DataFormat.TORCH_LOADER:
            return self._run_torch_fold(
                train_ds, test_ds, fold_idx,
                train=train,
                full_train_ds=full_train_ds,
            )
        else:
            raise ValueError(
                f"Unsupported data format: {self.model.data_format}"
            )

    def _train_model(self, train_ds: Dataset, fold_idx: int) -> None:
        """Train the model on a training dataset (format-aware)."""
        if self.model.data_format == DataFormat.TABULAR:
            X_train = train_ds.get_features()
            y_train = train_ds.get_labels()
            if self.model.input_columns:
                X_train = X_train[self.model.input_columns]
            if X_train.isnull().values.any():
                raise ValueError(
                    f"Fold {fold_idx + 1}: training features contain NaN values."
                )
            self.model.fit(X_train, y_train)
        elif self.model.data_format == DataFormat.TORCH_LOADER:
            if not _TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for torch-based models.")
            if self.lookback <= 0 or self.horizon <= 0:
                raise ValueError(
                    "WalkForwardEngine requires lookback > 0 and horizon > 0 "
                    "for torch-based models.  Set them in the constructor."
                )
            batch_size = getattr(self.model, "batch_size", 32)
            train_loader = train_ds.get_sequence_loader(
                lookback=self.lookback,
                horizon=self.horizon,
                batch_size=batch_size,
                shuffle=True,
            )
            self.model.fit(train_loader)

    @staticmethod
    def _finalize(all_predictions: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate fold predictions and de-duplicate overlapping indices."""
        if not all_predictions:
            return pd.DataFrame()
        result = pd.concat(all_predictions)
        result = result[~result.index.duplicated(keep="last")]
        return result

    # Tabular fold                                
    def _run_tabular_fold(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        fold_idx: int,
        *,
        train: bool = True,
    ) -> pd.DataFrame:
        """Optionally train via _train_model(), then predict.

        Only the **first row** of ``test_ds`` is predicted — this is
        the window-start observation whose features encode the state at
        time *t*, and the model forecasts ln(P_{t+H}/P_t).  One
        prediction per fold matches the per-window backtester design.
        """
        if train:
            self._train_model(train_ds, fold_idx)

        X_test = test_ds.get_features().iloc[[0]]  # single row
        if self.model.input_columns:
            X_test = X_test[self.model.input_columns]
        if X_test.isnull().values.any():
            raise ValueError(
                f"Fold {fold_idx + 1}: test features contain NaN values."
            )

        preds = self.model.predict(X_test)
        return pd.DataFrame(
            preds, index=[test_ds.df.index[0]], columns=["prediction"],
        )

    # Torch fold                                    
    def _run_torch_fold(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        fold_idx: int,
        *,
        train: bool = True,
        full_train_ds: Optional[Dataset] = None,
    ) -> pd.DataFrame:
        """
        Sequence-based training and single-shot prediction for deep
        learning time-series models.

        Training phase (when train is True):
            Delegated to _train_model() using the (trimmed)
            train_ds.

        Prediction phase:
            The last lookback feature rows from full_train_ds
            (the untrimmed training window) are used as model input
            to produce a single-shot forecast of horizon steps.
            This ensures the prediction context includes the freshest
            data available at prediction time, even though that data
            was excluded from training (embargo period).
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for torch-based models.")

        if self.lookback <= 0 or self.horizon <= 0:
            raise ValueError(
                "WalkForwardEngine requires lookback > 0 and horizon > 0 "
                "for torch-based models.  Set them in the constructor."
            )

        if train:
            self._train_model(train_ds, fold_idx)

        # Prediction phase - use full (untrimmed) training tail
        pred_source = full_train_ds if full_train_ds is not None else train_ds
        features = pred_source.get_features().to_numpy()  # (N_full, F)
        X_pred = features[-self.lookback:]                 # (lookback, F)
        X_pred_tensor = torch.tensor(
            X_pred, dtype=torch.float32
        ).unsqueeze(0)  # (1, lookback, F)

        raw_preds = self.model.predict(X_pred_tensor)  # (1, horizon, L)

        # Reshape: keep only the first prediction step (window start)
        if raw_preds.ndim == 3:
            raw_preds = raw_preds.squeeze(0)  # (horizon, L)
        if raw_preds.ndim == 1:
            raw_preds = raw_preds.reshape(-1, 1)  # (horizon, 1)

        raw_preds = raw_preds[[0]]  # (1, L) — single window prediction

        label_cols = test_ds.label_cols
        if raw_preds.shape[1] == 1:
            col_names = ["prediction"]
        else:
            col_names = [f"prediction_{c}" for c in label_cols] if label_cols else [
                f"prediction_{i}" for i in range(raw_preds.shape[1])
            ]

        return pd.DataFrame(
            raw_preds,
            index=[test_ds.df.index[0]],
            columns=col_names,
        )

    # Hyperparameter pass-through for Optimizer integration
    def set_model_params(self, params: dict) -> None:
        """Forward hyperparameter updates to the model."""
        self.model.set_params(params)

    # Model persistence
    def save_model(self, path: str | Path) -> None:
        """Save the model from the last completed fold.

        After run() completes, self.model holds the weights
        trained on the most recent training window.  This is the model
        you would deploy for live prediction.
        """
        from pathlib import Path as _Path
        self.model.save(_Path(path))

    def __repr__(self) -> str:
        return (
            f"WalkForwardEngine(model={self.model!r}, "
            f"window={self.window!r}, "
            f"scaler={self.scaler!r})"
        )