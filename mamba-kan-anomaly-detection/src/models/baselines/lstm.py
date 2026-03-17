"""LSTM-based anomaly detector baseline.

Provides a standard BiLSTM reconstruction model as a comparison baseline
for the Mamba-KAN architecture. Shares the same interface (forward returns
reconstructed + scores) for fair benchmarking.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LSTMDetector(nn.Module):
    """Bidirectional LSTM autoencoder for anomaly detection.

    A well-tuned LSTM baseline that follows the same encode-decode-score
    pattern as MambaKANDetector for apples-to-apples comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> LSTMDetector:
        return cls(
            input_dim=input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass matching MambaKANDetector interface.

        Args:
            x: Sensor window (batch, seq_len, input_dim).

        Returns:
            Tuple of (reconstructed, anomaly_scores).
        """
        projected = self.norm(self.input_proj(x))
        encoded, _ = self.encoder(projected)
        reconstructed = self.decoder(encoded)

        anomaly_scores = (reconstructed - x).pow(2).mean(dim=-1)
        return reconstructed, anomaly_scores

    def compute_loss(
        self,
        x: Tensor,
        reconstructed: Tensor,
        scores: Tensor,
    ) -> dict[str, Tensor]:
        recon_loss = F.mse_loss(reconstructed, x)
        return {"total": recon_loss, "reconstruction": recon_loss, "sparsity": torch.tensor(0.0)}

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}
