from __future__ import annotations

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
        Ignored for tabular models.  Default is 336 (14 days of
        1-hour data).
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
                                                   
    def run(self, verbose: bool = False) -> pd.DataFrame:
        """
        Execute the walk-forward loop and return all predictions.

        The engine inspects self.model.data_format to decide how to
        extract data from each fold:

        - if DataFormat.TABULAR then Dataset.get_features() /
          get_labels().
        - if DataFormat.TORCH_LOADER then Dataset.get_torch_loader().

        Returns:
            pd.DataFrame
            Predictions from every test fold, concatenated.
            Columns: ["prediction"] (or multi-column for classifiers).
            Index: aligned with the original Dataset.
        """
        all_predictions: list[pd.DataFrame] = []

        for fold_idx, (train_ds, test_ds) in enumerate(self.window.get_splits()):
            if verbose:
                print(
                    f"Fold {fold_idx + 1}: "
                    f"train={len(train_ds)} rows, test={len(test_ds)} rows"
                )

            # Reset model(optional)
            if self.reset_model:
                self.model.reset()

            # Fit scaler on train, transform both(optional)
            if self.scaler is not None:
                self.scaler.fit(train_ds)
                self.scaler.transform(train_ds)
                self.scaler.transform(test_ds)

            # Run based on model data format
            if self.model.data_format == DataFormat.TABULAR:
                fold_preds = self._run_tabular_fold(train_ds, test_ds, fold_idx)
            elif self.model.data_format == DataFormat.TORCH_LOADER:
                fold_preds = self._run_torch_fold(train_ds, test_ds, fold_idx)
            else:
                raise ValueError(
                    f"Unsupported data format: {self.model.data_format}"
                )

            all_predictions.append(fold_preds)

        if not all_predictions:
            return pd.DataFrame()

        # Concatenate & remove duplicate indices (overlapping windows)
        result = pd.concat(all_predictions)
        result = result[~result.index.duplicated(keep="last")] # Keep the last prediction
        return result

    # Tabular fold                                
    def _run_tabular_fold(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        fold_idx: int,
    ) -> pd.DataFrame:
        """Extract DataFrames, validate, train, and predict."""
        X_train = train_ds.get_features()
        y_train = train_ds.get_labels()
        X_test = test_ds.get_features()

        # Filter to input_columns if specified
        if self.model.input_columns:
            X_train = X_train[self.model.input_columns]
            X_test = X_test[self.model.input_columns]

        # Validate
        if X_train.isnull().values.any():
            raise ValueError(
                f"Fold {fold_idx + 1}: training features contain NaN values."
            )
        if X_test.isnull().values.any():
            raise ValueError(
                f"Fold {fold_idx + 1}: test features contain NaN values."
            )

        self.model.fit(X_train, y_train)
        # return self.model.predict(X_test)
        preds = self.model.predict(X_test)
        return pd.DataFrame(preds, index=test_ds.df.index, columns=["prediction"])

    # Torch fold                                    
    def _run_torch_fold(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        fold_idx: int,
    ) -> pd.DataFrame:
        """
        Sequence-based training and single-shot prediction for deep
        learning time-series models.

        Training phase:
            The training dataset is sliced into overlapping sub-windows
            of (lookback, horizon) with stride 1.  Each sub-window
            produces one training sample:
            - X: features[i : i+lookback]              shape (lookback, F)
            - Y: labels[i+lookback : i+lookback+horizon]  shape (horizon, L)
            These samples are batched and shuffled for stochastic training.

        Prediction phase:
            The last "lookback" feature rows from the training period
            are used as model input to produce a single-shot forecast of
            ""horizon"" steps.  The predictions are aligned with the
            test_ds index.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for torch-based models.")

        if self.lookback <= 0 or self.horizon <= 0:
            raise ValueError(
                "WalkForwardEngine requires lookback > 0 and horizon > 0 "
                "for torch-based models.  Set them in the constructor."
            )

        batch_size = getattr(self.model, "batch_size", 32)

        # Training phase 
        train_loader = train_ds.get_sequence_loader(
            lookback=self.lookback,
            horizon=self.horizon,
            batch_size=batch_size,
            shuffle=True,  
        )
        self.model.fit(train_loader)

        # Prediction phase 
        # Take the last "lookback" feature rows from training data
        features = train_ds.get_features().to_numpy()  # (N_train, F)
        X_pred = features[-self.lookback:]               # (lookback, F)
        X_pred_tensor = torch.tensor(
            X_pred, dtype=torch.float32
        ).unsqueeze(0)  # (1, lookback, F)

        raw_preds = self.model.predict(X_pred_tensor)  # (1, horizon, L)

        # Reshape: remove batch dim, clip to actual test length
        if raw_preds.ndim == 3:
            raw_preds = raw_preds.squeeze(0)  # (horizon, L)
        if raw_preds.ndim == 1:
            raw_preds = raw_preds.reshape(-1, 1)  # (horizon, 1)

        n_test = len(test_ds)
        raw_preds = raw_preds[:n_test]  # safety clip

        # Determine column names from label cols
        label_cols = test_ds.label_cols
        if raw_preds.shape[1] == 1:
            col_names = ["prediction"]
        else:
            col_names = [f"prediction_{c}" for c in label_cols] if label_cols else [
                f"prediction_{i}" for i in range(raw_preds.shape[1])
            ]

        return pd.DataFrame(
            raw_preds,
            index=test_ds.df.index[:n_test],
            columns=col_names,
        )

    #  Hyperparameter pass-through for Optimizer integration            
    def set_model_params(self, params: dict) -> None:
        """Forward hyperparameter updates to the model."""
        self.model.set_params(params)

    def __repr__(self) -> str:
        return (
            f"WalkForwardEngine(model={self.model!r}, "
            f"window={self.window!r}, "
            f"scaler={self.scaler!r})"
        )