"""Unit tests for Mamba-KAN model components."""

import pytest
import torch

from src.models.kan_encoder import BSplineBasis, KANLayer, TemporalKANEncoder
from src.models.mamba_block import BidirectionalMamba, MambaBlock, SelectiveSSM
from src.models.mamba_kan import MambaKANDetector
from src.models.baselines.lstm import LSTMDetector
from src.models.baselines.transformer import TransformerDetector


BATCH, SEQ_LEN, INPUT_DIM, HIDDEN_DIM = 4, 32, 8, 16


# ── KAN Tests ────────────────────────────────────────────────

class TestBSplineBasis:
    def test_output_shape(self):
        basis = BSplineBasis(grid_size=8, spline_order=3)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = basis(x)
        assert out.shape == (BATCH, SEQ_LEN, INPUT_DIM, basis.n_bases)

    def test_non_negative(self):
        basis = BSplineBasis(grid_size=8, spline_order=3)
        x = torch.linspace(-1, 1, 100)
        out = basis(x)
        assert (out >= -0.01).all(), "B-spline bases should be approximately non-negative"


class TestKANLayer:
    def test_forward_shape(self):
        layer = KANLayer(in_features=INPUT_DIM, out_features=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = layer(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_spline_extraction(self):
        layer = KANLayer(in_features=INPUT_DIM, out_features=HIDDEN_DIM)
        x_vals, curves = layer.get_spline_activations()
        assert x_vals.shape[0] == 200
        assert curves.shape == (HIDDEN_DIM, INPUT_DIM, 200)


class TestTemporalKANEncoder:
    def test_forward_shape(self):
        encoder = TemporalKANEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, n_layers=2)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = encoder(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_spline_collection(self):
        encoder = TemporalKANEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, n_layers=2)
        activations = encoder.get_all_spline_activations()
        assert len(activations) == 2


# ── Mamba Tests ──────────────────────────────────────────────

class TestSelectiveSSM:
    def test_forward_shape(self):
        ssm = SelectiveSSM(d_model=HIDDEN_DIM, state_dim=8)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ssm(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)


class TestMambaBlock:
    def test_forward_shape(self):
        block = MambaBlock(d_model=HIDDEN_DIM, state_dim=8, expansion_factor=2)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_residual_connection(self):
        block = MambaBlock(d_model=HIDDEN_DIM, state_dim=8)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        # Output should not be all zeros (residual ensures signal flow)
        assert out.abs().sum() > 0


class TestBidirectionalMamba:
    @pytest.mark.parametrize("merge", ["concat", "add", "gate"])
    def test_merge_strategies(self, merge):
        model = BidirectionalMamba(d_model=HIDDEN_DIM, n_layers=1, merge=merge)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)


# ── Full Model Tests ─────────────────────────────────────────

class TestMambaKANDetector:
    @pytest.fixture
    def model(self):
        return MambaKANDetector(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
            n_kan_layers=1, n_mamba_layers=1,
        )

    def test_forward_returns_tuple(self, model):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        reconstructed, scores = model(x)
        assert reconstructed.shape == (BATCH, SEQ_LEN, INPUT_DIM)
        assert scores.shape == (BATCH, SEQ_LEN)

    def test_loss_computation(self, model):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        reconstructed, scores = model(x)
        losses = model.compute_loss(x, reconstructed, scores)
        assert "total" in losses
        assert "reconstruction" in losses
        assert losses["total"].requires_grad

    def test_parameter_counting(self, model):
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert "kan_encoder" in counts
        assert "mamba_encoder" in counts

    def test_spline_extraction(self, model):
        splines = model.get_learned_functions()
        assert len(splines) > 0


# ── Baseline Tests ───────────────────────────────────────────

class TestBaselines:
    def test_lstm_interface(self):
        model = LSTMDetector(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        reconstructed, scores = model(x)
        assert reconstructed.shape == (BATCH, SEQ_LEN, INPUT_DIM)
        assert scores.shape == (BATCH, SEQ_LEN)

    def test_transformer_interface(self):
        model = TransformerDetector(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        reconstructed, scores = model(x)
        assert reconstructed.shape == (BATCH, SEQ_LEN, INPUT_DIM)
        assert scores.shape == (BATCH, SEQ_LEN)


# ── Gradient Flow ────────────────────────────────────────────

class TestGradientFlow:
    def test_gradients_flow_through_full_model(self):
        model = MambaKANDetector(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
            n_kan_layers=1, n_mamba_layers=1,
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM, requires_grad=True)
        reconstructed, scores = model(x)
        loss = scores.mean() + (reconstructed - x).pow(2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
