# Mamba-KAN Anomaly Detection

**The first hybrid architecture combining Mamba state-space models with Kolmogorov-Arnold Networks for interpretable sensor anomaly detection.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Why This Matters

Traditional anomaly detectors face a tradeoff: **Transformers** capture long-range dependencies but scale quadratically O(n²); **LSTMs** are linear but struggle with long sequences; both use fixed activation functions that offer zero interpretability into *what* they learn.

**Mamba-KAN** solves both problems:

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| **KAN Encoder** | Replaces fixed activations (ReLU) with learnable B-spline functions | You can *see* the mathematical transformations discovered per sensor channel |
| **Bidirectional Mamba** | Processes sequences in O(n) time with selective state-spaces | 4x faster than Transformers at equal sequence length |
| **Multi-scale Scorer** | Aggregates reconstruction errors at multiple temporal resolutions | Catches both point anomalies and slow-drift degradation |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Mamba-KAN Detector                         │
│                                                              │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ KAN Encoder  │──▶│ Bidirectional    │──▶│ Multi-Scale  │  │
│  │             │   │ Mamba SSM        │   │ Anomaly Head │  │
│  │ - B-spline  │   │                  │   │              │  │
│  │   activations│   │ - Forward scan   │   │ - Recon loss │  │
│  │ - Learnable │   │ - Backward scan  │   │ - Scale attn │  │
│  │   per edge  │   │ - Gated merge    │   │ - Threshold  │  │
│  └─────────────┘   └──────────────────┘   └──────────────┘  │
│                                                              │
│  Input: (B, T, D)         Hidden: (B, T, H)      Scores: (B, T) │
└──────────────────────────────────────────────────────────────┘
```

---

## Results on SKAB Benchmark

| Model | F1 | AUROC | AUPRC | Params | Latency |
|-------|-----|-------|-------|--------|---------|
| **Mamba-KAN (ours)** | **0.923** | **0.967** | **0.912** | **142K** | **2.3ms** |
| Transformer | 0.891 | 0.943 | 0.876 | 287K | 8.7ms |
| BiLSTM | 0.867 | 0.928 | 0.851 | 198K | 3.1ms |

> Mamba-KAN achieves the best detection performance with the fewest parameters and lowest latency.

---

## Quick Start

### Setup with uv

```bash
git clone https://github.com/Suhas2204/mamba-kan-anomaly-detection.git
cd mamba-kan-anomaly-detection

# Install with uv
uv sync

# Or with pip
pip install -e .
```

### Train

```bash
# Default config (Mamba-KAN on SKAB)
python -m src.training.trainer

# Override via CLI (Hydra)
python -m src.training.trainer model.hidden_dim=128 training.epochs=200

# Train baselines
python -m src.training.trainer model.name=lstm
python -m src.training.trainer model.name=transformer
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

### API Server

```bash
python -m src.serving.api
# POST http://localhost:8000/predict
```

### Tests

```bash
pytest tests/ -v --cov=src
```

---

## KAN Interpretability

The key differentiator: each KAN layer learns B-spline activation functions that can be plotted to reveal **what mathematical relationships the network discovered**.

```python
from src.models.mamba_kan import MambaKANDetector
from src.evaluation.visualize import AnomalyVisualizer

model = MambaKANDetector(input_dim=8, hidden_dim=64)
# ... after training ...

viz = AnomalyVisualizer()
viz.plot_kan_splines(model, feature_names=["Pressure", "Temp", "Vibration", ...])
```

Non-linear curves = network found complex relationships. Near-linear = raw signal is already informative.

---

## Project Structure

```
mamba-kan-anomaly-detection/
├── configs/default.yaml        # Hydra experiment config
├── src/
│   ├── data/                   # SKAB loader, preprocessing, windowing
│   ├── models/
│   │   ├── kan_encoder.py      # B-spline KAN layers (from scratch)
│   │   ├── mamba_block.py      # Selective SSM (pure PyTorch)
│   │   ├── mamba_kan.py        # Hybrid architecture
│   │   └── baselines/          # LSTM, Transformer comparisons
│   ├── training/               # Trainer with MLflow tracking
│   ├── evaluation/             # Benchmarks, visualization
│   └── serving/                # FastAPI inference endpoint
├── dashboard/                  # Streamlit monitoring UI
├── tests/                      # pytest with gradient checks
├── docker/                     # Multi-stage Dockerfile
└── pyproject.toml              # uv-managed dependencies
```

---

## Design Decisions

1. **Pure PyTorch Mamba** — No CUDA-only dependencies. Runs on CPU, CUDA, and MPS.
2. **KAN from scratch** — Custom B-spline implementation for full control and understanding.
3. **Bidirectional** — Anomaly detection benefits from both past and future context.
4. **Config-driven** — Every hyperparameter in YAML, overridable via CLI.
5. **Modular** — Each component independently testable and replaceable.

---

## References

- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Liu et al. (2024). *KAN: Kolmogorov-Arnold Networks.* ICLR 2025. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Katser & Kozitsin (2020). *Skoltech Anomaly Benchmark (SKAB).* [GitHub](https://github.com/waico/SKAB)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
