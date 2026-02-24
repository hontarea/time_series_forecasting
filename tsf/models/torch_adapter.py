from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsf.models.base import BaseModel, DataFormat

# Torch is an optional dependency - import only if DL models used
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TorchAdapter(BaseModel):
    """
    Adapter for PyTorch nn.Module models.

    Args:
    
    module : torch.nn.Module
        A PyTorch model instance (e.g. an LSTM network).
    lr : float
        Learning rate (default 1e-3).
    epochs : int
        Number of training epochs (default 50).
    batch_size : int
        Mini-batch size (default 32).  Used by the execution layer when
        building a DataLoader via Dataset.get_torch_loader().
    loss_fn : torch.nn.Module | None
        Loss function (default nn.MSELoss()).
    optimizer_cls : type | None
        Optimizer class (default torch.optim.Adam).
    device : str
        "cpu" or "cuda" (auto-detected if not specified).
    early_stopping_patience : int
        Stop after this many epochs with no improvement (0 = disabled).
    input_columns : list[str], optional
        Feature columns the model expects.
    """

    def __init__(
        self,
        module: "nn.Module | None" = None,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        loss_fn: "nn.Module | None" = None,
        optimizer_cls: "type | None" = None,
        device: Optional[str] = None,
        early_stopping_patience: int = 0,
        input_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(input_columns=input_columns, **kwargs)
        self._check_torch()

        self.module = module
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer_cls = optimizer_cls or torch.optim.Adam
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping_patience = early_stopping_patience

        if self.module is not None:
            self.module = self.module.to(self.device)

    @property
    def data_format(self) -> DataFormat:
        return DataFormat.TORCH_LOADER

    # Core fuctions
    def fit(self, loader: "TorchDataLoader") -> None:
        """
        Train the module for self.epochs over the given DataLoader.

        Args: 
        
        loader : torch.utils.data.DataLoader
            Yields (X_batch, y_batch) tensors.  Built by the
            execution layer via Dataset.get_torch_loader().
        """
        self._check_torch()
        if self.module is None:
            raise RuntimeError("No nn.Module has been set on this TorchAdapter.")

        self.module.train()
        optimizer = self.optimizer_cls(self.module.parameters(), lr=self.lr)

        best_loss = float("inf")
        patience_counter = 0
        n_samples = len(loader.dataset)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                # Pass y_batch to forward() to support teacher forcing
                preds = self.module(X_batch, y_batch)
                loss = self.loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= n_samples

            # Early stopping
            if self.early_stopping_patience > 0:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

    def predict(self, X) -> np.ndarray:
        """
        Run inference and return predictions as a NumPy array.

        Args:

        X : torch.Tensor or torch.utils.data.DataLoader
            Either a single tensor of shape ``(1, lookback, F)`` for
            single-shot forecasting, or a DataLoader yielding
            (X_batch, y_batch) pairs for batch evaluation.

        Returns:

        np.ndarray
            Predictions.  Shape depends on the model:
            - For sequence models: (batch, horizon, n_labels)
            - For flat models: (n_samples, n_labels)
        """
        self._check_torch()
        if self.module is None:
            raise RuntimeError("No nn.Module has been set on this TorchAdapter.")

        self.module.eval()

        # Single tensor input (e.g. from WalkForwardEngine prediction phase)
        if isinstance(X, torch.Tensor):
            with torch.no_grad():
                X = X.to(self.device)
                preds = self.module(X).cpu().numpy()
            return preds

        # DataLoader input (batch evaluation)
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in X:
                X_batch = X_batch.to(self.device)
                preds = self.module(X_batch).cpu().numpy() # Calculate the predicitons and move to the CPU 
                all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)

    def reset(self) -> None:
        """Reinitialise module weights."""
        if self.module is not None:
            for layer in self.module.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    #  Persistence
    def save(self, path: str | Path) -> None:
        """Save module weights, input_columns, and training params."""
        self._check_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.module.state_dict(),
            "input_columns": self.input_columns,
            "params": self.get_params(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str | Path) -> None:
        """Restore module weights and metadata from a checkpoint."""
        self._check_torch()
        checkpoint = torch.load(Path(path), map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.module.load_state_dict(checkpoint["state_dict"])
            self.input_columns = checkpoint.get("input_columns")
            for k, v in checkpoint.get("params", {}).items():
                if hasattr(self, k):
                    setattr(self, k, v)
        else:
            # Backwards-compatible: bare state_dict saved by old code
            self.module.load_state_dict(checkpoint)

    # Hyperparameters
    def get_params(self) -> Dict:
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "early_stopping_patience": self.early_stopping_patience,
        }

    @staticmethod
    def _check_torch() -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TorchAdapter. "
                "Install it with: pip install torch"
            )

    def __repr__(self) -> str:
        mod = type(self.module).__name__ if self.module else "None"
        return f"TorchAdapter(module={mod}, lr={self.lr}, epochs={self.epochs})"
