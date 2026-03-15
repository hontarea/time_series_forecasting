from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM encoder with an MLP forecasting head.

    Args:
        seq_len : int
            Input sequence length (lookback).  Accepted for interface
            consistency with other models but not used internally -
            the LSTM processes any-length sequences via shared weights.
        pred_len : int
            Forecast horizon.
        enc_in : int
            Number of input channels (features per time step).
        c_out : int
            Number of output channels (default 1).
        hidden_size : int
            Dimensionality of the LSTM hidden / cell states.
        num_layers : int
            Number of stacked LSTM layers (default 1).
        dropout : float
            Dropout applied between LSTM layers when num_layers > 1
            (default 0.0). Ignored for single-layer models.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Reshape in forward:  (B, pred_len * c_out) -> (B, pred_len, c_out)
        self.head = nn.Linear(hidden_size, pred_len * c_out)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, enc_in)
            y : ignored (accepted for TorchAdapter teacher-forcing compat).

        Returns:
            (batch, pred_len, c_out)
        """
        # Walk through the sequence step by step.
        _, (h_n, _) = self.encoder(x)

        # Take the final hidden state of the last layer.
        h_last = h_n[-1]

        # Project to horizon. 
        out = self.head(h_last)

        # Reshape to (B, pred_len, c_out) for interface consistency.
        out = out.view(x.size(0), self.pred_len, self.c_out)

        return out