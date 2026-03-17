"""Transformer-based anomaly detector baseline.

Standard Transformer encoder with positional encoding, serving as the
attention-based comparison point for the Mamba-KAN architecture.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1]])


class TransformerDetector(nn.Module):
    """Transformer encoder autoencoder for anomaly detection.

    Uses multi-head self-attention to model temporal dependencies.
    Note the O(n²) complexity in sequence length — this is the key
    disadvantage that Mamba addresses with O(n) complexity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> TransformerDetector:
        return cls(
            input_dim=input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass matching MambaKANDetector interface."""
        projected = self.input_proj(x)
        encoded = self.encoder(self.pos_encoder(projected))
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
