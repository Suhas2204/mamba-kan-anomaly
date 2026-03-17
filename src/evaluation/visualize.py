"""Visualization suite for Mamba-KAN anomaly detection.

Provides publication-quality plots for:
    - Learned KAN spline activation functions (the interpretability showcase)
    - Anomaly score timelines with ground-truth overlay
    - Reconstruction quality comparisons
    - Benchmark comparison charts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from numpy.typing import NDArray
from loguru import logger

# Publication-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

PALETTE = {
    "normal": "#2ecc71",
    "anomaly": "#e74c3c",
    "score": "#3498db",
    "threshold": "#f39c12",
    "reconstruction": "#9b59b6",
}


class AnomalyVisualizer:
    """Generates all visualization artifacts for the Mamba-KAN project.

    Example:
        >>> viz = AnomalyVisualizer(save_dir="results/")
        >>> viz.plot_kan_splines(model, feature_names=["Pressure", "Temp", ...])
        >>> viz.plot_anomaly_timeline(scores, labels, threshold)
    """

    def __init__(self, save_dir: str | Path = "results/") -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def plot_kan_splines(
        self,
        model: torch.nn.Module,
        feature_names: list[str] | None = None,
        layer_idx: int = 0,
        max_edges: int = 16,
    ) -> plt.Figure:
        """Visualize learned KAN spline activation functions.

        This is the key interpretability feature: each subplot shows the
        mathematical function the network discovered for a specific
        (input_feature → hidden_unit) connection.

        Args:
            model: Trained MambaKANDetector.
            feature_names: Sensor channel names for axis labels.
            layer_idx: Which KAN layer to visualize.
            max_edges: Maximum number of edge functions to display.

        Returns:
            Matplotlib figure.
        """
        activations = model.get_learned_functions()
        if not activations:
            logger.warning("No KAN layers found in model.")
            return plt.figure()

        x_vals, curves = activations[layer_idx]
        x_np = x_vals.detach().cpu().numpy()
        curves_np = curves.detach().cpu().numpy()

        n_out, n_in, n_points = curves_np.shape
        n_plots = min(n_out * n_in, max_edges)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        plot_idx = 0
        for i in range(n_out):
            for j in range(n_in):
                if plot_idx >= n_plots:
                    break

                ax = axes[plot_idx]
                curve = curves_np[i, j]

                ax.plot(x_np, curve, color="#2c3e50", linewidth=2.0)
                ax.fill_between(x_np, curve, alpha=0.15, color="#3498db")
                ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
                ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                in_name = feature_names[j] if feature_names and j < len(feature_names) else f"in_{j}"
                ax.set_title(f"φ({in_name}) → h_{i}", fontsize=10)
                ax.set_xlabel("Input activation")
                ax.set_ylabel("Output")
                ax.grid(True, alpha=0.2)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            "Learned KAN Spline Activations — Layer 1",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        path = self._save_dir / "kan_spline_activations.png"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        logger.info(f"Saved KAN spline plot → {path}")
        return fig

    def plot_anomaly_timeline(
        self,
        scores: NDArray[np.float32],
        labels: NDArray[np.int32],
        threshold: float,
        title: str = "Anomaly Detection Timeline",
    ) -> plt.Figure:
        """Plot anomaly scores against ground-truth labels.

        Args:
            scores: Per-timestep anomaly scores.
            labels: Binary ground-truth labels.
            threshold: Decision threshold.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, height_ratios=[3, 1])

        timesteps = np.arange(len(scores))

        # Top: anomaly scores
        ax1.plot(timesteps, scores, color=PALETTE["score"], linewidth=0.8, alpha=0.9, label="Score")
        ax1.axhline(y=threshold, color=PALETTE["threshold"], linestyle="--", linewidth=1.5,
                     label=f"Threshold ({threshold:.4f})")

        # Shade anomaly regions
        anomaly_mask = labels.astype(bool)
        ax1.fill_between(timesteps, 0, scores.max() * 1.1, where=anomaly_mask,
                         color=PALETTE["anomaly"], alpha=0.15, label="Ground Truth Anomaly")

        ax1.set_ylabel("Anomaly Score")
        ax1.set_title(title, fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.grid(True, alpha=0.2)

        # Bottom: binary labels
        ax2.fill_between(timesteps, 0, labels, step="mid",
                         color=PALETTE["anomaly"], alpha=0.7, label="Label")
        predictions = (scores >= threshold).astype(int)
        ax2.step(timesteps, predictions * 0.5, color=PALETTE["score"],
                 linewidth=1.5, where="mid", label="Predicted")
        ax2.set_ylabel("Label")
        ax2.set_xlabel("Timestep")
        ax2.set_yticks([0, 1])
        ax2.legend(loc="upper right")

        fig.tight_layout()
        path = self._save_dir / "anomaly_timeline.png"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        logger.info(f"Saved timeline → {path}")
        return fig

    def plot_reconstruction(
        self,
        original: NDArray[np.float32],
        reconstructed: NDArray[np.float32],
        feature_names: list[str] | None = None,
        n_features: int = 4,
    ) -> plt.Figure:
        """Compare original vs. reconstructed sensor signals."""
        n_features = min(n_features, original.shape[-1])
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
        if n_features == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"Sensor {i}"
            ax.plot(original[:, i], color="#2c3e50", linewidth=1.0, label="Original", alpha=0.8)
            ax.plot(reconstructed[:, i], color=PALETTE["reconstruction"],
                    linewidth=1.0, label="Reconstructed", alpha=0.8)
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.2)

        axes[0].set_title("Signal Reconstruction Quality", fontsize=13, fontweight="bold")
        axes[-1].set_xlabel("Timestep")

        fig.tight_layout()
        path = self._save_dir / "reconstruction_comparison.png"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        logger.info(f"Saved reconstruction plot → {path}")
        return fig

    def plot_benchmark_comparison(
        self,
        results_df,
    ) -> plt.Figure:
        """Bar chart comparing model performance metrics."""
        metrics = ["f1", "auroc", "auprc"]
        available = [m for m in metrics if m in results_df.columns]

        fig, axes = plt.subplots(1, len(available) + 1, figsize=(5 * (len(available) + 1), 5))

        colors = sns.color_palette("Set2", n_colors=len(results_df))

        for idx, metric in enumerate(available):
            ax = axes[idx]
            bars = ax.bar(results_df["model"], results_df[metric], color=colors)
            ax.set_title(metric.upper(), fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, axis="y")

            for bar, val in zip(bars, results_df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", fontsize=9)

        # Latency comparison
        ax = axes[-1]
        bars = ax.bar(results_df["model"], results_df["latency_ms"], color=colors)
        ax.set_title("Latency (ms)", fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        for bar, val in zip(bars, results_df["latency_ms"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=9)

        fig.suptitle("Model Benchmark Comparison", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        path = self._save_dir / "benchmark_comparison.png"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        logger.info(f"Saved benchmark chart → {path}")
        return fig
