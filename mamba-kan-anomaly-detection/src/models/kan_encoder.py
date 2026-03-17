"""Kolmogorov-Arnold Network (KAN) encoder for temporal feature extraction.

Implements KAN layers with learnable B-spline activation functions, providing
interpretable feature transformations that can be visualized to understand
what mathematical relationships the network discovers in sensor data.

Reference:
    Liu et al. (2024). "KAN: Kolmogorov-Arnold Networks." ICLR 2025 Oral.
    arXiv:2404.19756
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BSplineBasis(nn.Module):
    """Compute B-spline basis functions on a learnable grid.

    Generates a set of B-spline basis values for input activations,
    enabling smooth, differentiable, learnable activation functions
    parameterized as weighted sums of B-spline bases.
    """

    def __init__(
        self,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        grid_epsilon: float = 0.02,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform grid with padding for boundary splines
        n_knots = grid_size + 2 * spline_order + 1
        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * step,
            grid_range[1] + spline_order * step,
            n_knots,
        )
        self.register_buffer("grid", grid)
        self.grid_epsilon = grid_epsilon

    @property
    def n_bases(self) -> int:
        """Number of B-spline basis functions."""
        return self.grid_size + self.spline_order

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate B-spline bases at given positions.

        Args:
            x: Input tensor of any shape. Values should be roughly in grid_range.

        Returns:
            Basis values of shape (*x.shape, n_bases).
        """
        x_flat = x.unsqueeze(-1)  # (..., 1)
        grid = self.grid  # (n_knots,)

        # Cox-de Boor recursion: order 0
        bases = ((x_flat >= grid[:-1]) & (x_flat < grid[1:])).float()

        # Recursive higher-order basis computation
        for k in range(1, self.spline_order + 1):
            left_num = x_flat - grid[: -(k + 1)]
            left_den = grid[k:-1] - grid[: -(k + 1)]
            right_num = grid[k + 1 :] - x_flat
            right_den = grid[k + 1 :] - grid[1:(-k if -k != 0 else None)]

            left = left_num / (left_den + self.grid_epsilon) * bases[:, :, :-1]
            right = right_num / (right_den + self.grid_epsilon) * bases[:, :, 1:]
            bases = left + right

        return bases


class KANLayer(nn.Module):
    """Single KAN layer: replaces fixed activations with learnable splines.

    Each edge (i, j) in the network has its own B-spline activation function,
    parameterized by a set of learnable coefficients. A residual SiLU connection
    stabilizes training.

    Architecture per edge:
        φ_ij(x) = w_residual * SiLU(x) + w_spline * Σ(c_k * B_k(x))

    This design means the network can discover arbitrary smooth univariate
    functions on each connection — making it inherently more interpretable
    than standard MLPs with fixed activations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        residual_weight: float = 1.0,
        grid_epsilon: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual_weight = residual_weight

        self.basis = BSplineBasis(grid_size, spline_order, grid_range, grid_epsilon)

        # Spline coefficients: one set per (in, out) edge
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.basis.n_bases)
            * (1.0 / math.sqrt(in_features * self.basis.n_bases))
        )

        # Residual path weight
        self.residual_linear = nn.Linear(in_features, out_features)

        # Layernorm for stable training
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Apply KAN transformation.

        Args:
            x: Input of shape (batch, ..., in_features).

        Returns:
            Output of shape (batch, ..., out_features).
        """
        # B-spline path: compute basis, then weighted sum
        bases = self.basis(x)  # (batch, ..., in_features, n_bases)
        spline_out = torch.einsum("...ib,oib->...o", bases, self.spline_weight)

        # Residual path: standard linear + SiLU
        residual_out = self.residual_linear(F.silu(x))

        # Combine
        output = spline_out + self.residual_weight * residual_out
        return self.norm(output)

    def get_spline_activations(
        self, x_range: tuple[float, float] = (-2.0, 2.0), n_points: int = 200
    ) -> tuple[Tensor, Tensor]:
        """Extract learned activation curves for visualization.

        Args:
            x_range: Input range to evaluate over.
            n_points: Number of evaluation points.

        Returns:
            Tuple of (x_values, activation_curves) where activation_curves
            has shape (out_features, in_features, n_points).
        """
        x = torch.linspace(x_range[0], x_range[1], n_points, device=self.spline_weight.device)
        bases = self.basis(x)  # (n_points, n_bases)

        # Compute activation for each (in, out) pair
        # bases shape: (n_points, n_bases) — need to broadcast over in_features
        bases_expanded = bases.unsqueeze(0).expand(self.in_features, -1, -1)
        curves = torch.einsum("oib,ibn->oin", self.spline_weight, bases_expanded)

        return x, curves


class TemporalKANEncoder(nn.Module):
    """Multi-layer KAN encoder for temporal feature extraction.

    Progressively transforms raw sensor channels through stacked KAN layers,
    learning interpretable per-channel transformations before feeding into
    the Mamba sequence model.

    The encoder operates channel-wise at each timestep, discovering optimal
    feature representations through learned spline functions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        residual_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * n_layers

        for i in range(len(dims) - 1):
            layers.append(
                KANLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    residual_weight=residual_weight,
                )
            )
            if i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Encode sensor windows through stacked KAN layers.

        Args:
            x: Sensor window of shape (batch, seq_len, input_dim).

        Returns:
            Encoded features of shape (batch, seq_len, hidden_dim).
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def get_all_spline_activations(
        self,
    ) -> list[tuple[Tensor, Tensor]]:
        """Extract spline curves from all KAN layers for visualization."""
        activations = []
        for layer in self.layers:
            if isinstance(layer, KANLayer):
                activations.append(layer.get_spline_activations())
        return activations
