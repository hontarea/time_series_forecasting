"""
Linear time-series forecasting model.

Architecture inspired by "Are Transformers Effective for Time Series Forecasting?"
(Zeng et al., 2023).  A single Linear layer maps the temporal dimension
(seq_len -> pred_len) independently per channel, then a projection layer
maps from enc_in channels down to c_out output channels.

Input  : (batch, seq_len, enc_in)
Output : (batch, pred_len, c_out)

When ``individual=True``, each input channel gets its own Linear layer
(no weight sharing across channels).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    Multivariate-in, univariate-out Linear forecaster.

    Args:
        seq_len : int
            Input sequence length (lookback).
        pred_len : int
            Forecast horizon.
        enc_in : int
            Number of input channels (features).
        c_out : int
            Number of output channels (labels).  Default 1.
        individual : bool
            If True, use a separate Linear per input channel.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int = 1,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.individual = individual

        # Temporal mapping: seq_len -> pred_len
        if self.individual:
            self.temporal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
        else:
            self.temporal = nn.Linear(seq_len, pred_len)

        # Channel projection: enc_in -> c_out
        self.projection = nn.Linear(enc_in, c_out)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, enc_in)
            y : ignored (accepted for TorchAdapter teacher-forcing compat).

        Returns:
            (batch, pred_len, c_out)
        """
        # x: (B, S, C)
        if self.individual:
            out = torch.zeros(
                x.size(0), self.pred_len, self.enc_in,
                dtype=x.dtype, device=x.device,
            )
            for i in range(self.enc_in):
                out[:, :, i] = self.temporal[i](x[:, :, i])
        else:
            # (B, S, C) -> (B, C, S) -> Linear -> (B, C, P) -> (B, P, C)
            out = self.temporal(x.permute(0, 2, 1)).permute(0, 2, 1)

        # (B, P, C) -> (B, P, c_out)
        out = self.projection(out)
        return out
