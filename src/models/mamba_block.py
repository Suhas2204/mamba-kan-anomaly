"""Mamba selective state-space model block.

Pure PyTorch implementation of the Mamba architecture that captures long-range
temporal dependencies with linear complexity O(n) instead of the O(n²) cost
of attention-based transformers.

The key innovation is the *selective* state-space mechanism: the model's
recurrence parameters (Δ, B, C) are input-dependent, allowing it to focus
on relevant parts of the sequence and ignore irrelevant context.

Reference:
    Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces." arXiv:2312.00752
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelectiveSSM(nn.Module):
    """Core selective state-space mechanism.

    Implements the discretized state-space recurrence with input-dependent
    parameters, enabling content-aware sequence processing.

    State equation (discretized):
        h_t = Ā * h_{t-1} + B̄ * x_t
        y_t = C * h_t

    Where Ā, B̄ are derived from continuous parameters (A, B) via the
    zero-order hold discretization with input-dependent step size Δ.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # State matrix A: initialized as negative diagonal (stable dynamics)
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.register_buffer("A_log", torch.log(A))

        # Input-dependent projections: x → (Δ, B, C)
        self.x_to_delta = nn.Linear(d_model, self.dt_rank, bias=False)
        self.x_to_B = nn.Linear(d_model, state_dim, bias=False)
        self.x_to_C = nn.Linear(d_model, state_dim, bias=False)

        # Δ projection back to d_model
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize Δ bias for proper time-scale range
        dt_init_std = self.dt_rank**-0.5
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.bias, math.log(dt_min), math.log(dt_max))
        else:
            nn.init.constant_(self.dt_proj.bias, math.log(0.01))

        # Skip connection scaling
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Run selective SSM scan over the input sequence.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Compute input-dependent parameters
        delta = F.softplus(self.dt_proj(self.x_to_delta(x)))  # (B, L, D)
        B = self.x_to_B(x)  # (B, L, N)
        C = self.x_to_C(x)  # (B, L, N)
        A = -torch.exp(self.A_log)  # (D, N), negative for stability

        return self._sequential_scan(x, delta, A, B, C)

    def _sequential_scan(
        self,
        x: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        """Recurrent scan — straightforward implementation for clarity.

        For production with long sequences, this would use a parallel scan.
        The sequential version is used here for portability and debuggability.
        """
        batch, seq_len, d_model = x.shape
        state_dim = self.state_dim

        # Discretize: A_bar = exp(Δ * A), B_bar = Δ * B
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)

        # Initialize hidden state
        h = torch.zeros(batch, d_model, state_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)

            # Output: y = C * h + D * x (skip connection)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x[:, t]
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """Complete Mamba block with gated architecture.

    Follows the original Mamba design:
        1. Linear projection to expanded dimension
        2. Depthwise convolution for local context
        3. Selective SSM for long-range dynamics
        4. Gated output with SiLU activation
        5. Residual connection + LayerNorm

    The expansion factor (typically 2) controls the internal width,
    trading compute for representational capacity.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expansion_factor: int = 2,
        dt_rank: int | str = "auto",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expansion_factor

        # Input projection: split into SSM path and gate path
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Local context via depthwise conv (before SSM)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=conv_kernel,
            groups=d_inner,
            padding=conv_kernel - 1,
            bias=True,
        )

        # Core selective SSM
        self.ssm = SelectiveSSM(
            d_model=d_inner,
            state_dim=state_dim,
            dt_rank=dt_rank,
        )

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Process sequence through Mamba block.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        residual = x
        x = self.norm(x)

        # Project and split into SSM path + gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Local convolution (causal: trim future)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :x_ssm.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM for long-range context
        x_ssm_out = self.ssm(x_conv)

        # Gated output
        z_gate = F.silu(z)
        x_out = x_ssm_out * z_gate

        # Project back and residual
        output = self.out_proj(x_out)
        output = self.dropout(output)

        return output + residual


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba for non-causal anomaly detection.

    Unlike language modeling (which must be causal), anomaly detection benefits
    from seeing both past and future context. This module runs two Mamba
    blocks — forward and backward — and merges their representations.

    This is architecturally similar to BiLSTM but with O(n) complexity
    instead of O(n²) for equivalent transformer-based approaches.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        merge: str = "concat",  # concat | add | gate
    ) -> None:
        super().__init__()
        self.merge = merge

        self.forward_layers = nn.ModuleList(
            [
                MambaBlock(d_model, state_dim, conv_kernel, expansion_factor, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.backward_layers = nn.ModuleList(
            [
                MambaBlock(d_model, state_dim, conv_kernel, expansion_factor, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        if merge == "concat":
            self.merge_proj = nn.Linear(d_model * 2, d_model, bias=False)
        elif merge == "gate":
            self.gate_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Bidirectional Mamba encoding.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Merged representation of shape (batch, seq_len, d_model).
        """
        # Forward pass
        fwd = x
        for layer in self.forward_layers:
            fwd = layer(fwd)

        # Backward pass (reverse → process → reverse back)
        bwd = x.flip(dims=[1])
        for layer in self.backward_layers:
            bwd = layer(bwd)
        bwd = bwd.flip(dims=[1])

        # Merge directions
        if self.merge == "concat":
            merged = self.merge_proj(torch.cat([fwd, bwd], dim=-1))
        elif self.merge == "add":
            merged = fwd + bwd
        elif self.merge == "gate":
            gate = torch.sigmoid(self.gate_proj(torch.cat([fwd, bwd], dim=-1)))
            merged = gate * fwd + (1 - gate) * bwd
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge}")

        return merged
