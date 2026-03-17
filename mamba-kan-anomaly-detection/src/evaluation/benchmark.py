"""Cross-model benchmarking framework.

Runs all registered models (Mamba-KAN, LSTM, Transformer) on the same data
split with identical hyperparameters, producing a comparison table of metrics,
parameter counts, and inference latency.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from src.data.loader import SensorDataModule
from src.training.trainer import Trainer, build_model


class BenchmarkRunner:
    """Run comparative benchmarks across model architectures.

    Trains each model with identical data splits and hyperparameters,
    then collects metrics, parameter counts, and latency measurements
    into a structured comparison table.

    Example:
        >>> runner = BenchmarkRunner(cfg, data_module)
        >>> results_df = runner.run(["mamba_kan", "lstm", "transformer"])
        >>> print(results_df.to_markdown())
    """

    def __init__(self, cfg: DictConfig, data_module: SensorDataModule) -> None:
        self._cfg = cfg
        self._dm = data_module

    def run(
        self,
        model_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Execute benchmarks for specified models.

        Args:
            model_names: List of model identifiers. Defaults to all registered.

        Returns:
            DataFrame with one row per model and columns for each metric.
        """
        if model_names is None:
            model_names = ["mamba_kan", "lstm", "transformer"]

        results = []

        for name in model_names:
            logger.info(f"{'='*60}")
            logger.info(f"Benchmarking: {name}")
            logger.info(f"{'='*60}")

            self._cfg.model.name = name
            model = build_model(self._cfg, input_dim=self._dm.input_dim)

            trainer = Trainer(self._cfg, model, self._dm)
            train_results = trainer.fit()

            latency = self._measure_latency(model, trainer.device)
            param_count = model.count_parameters()

            row = {
                "model": name,
                **train_results["test_metrics"],
                "params": param_count.get("total", sum(param_count.values())),
                "latency_ms": latency,
            }
            results.append(row)

        df = pd.DataFrame(results)
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)

        logger.info("\n" + df.to_string(index=False))
        return df

    def _measure_latency(
        self,
        model: nn.Module,
        device: torch.device,
        n_runs: int = 100,
    ) -> float:
        """Measure mean inference latency in milliseconds."""
        model.eval()
        dummy = torch.randn(1, self._cfg.data.window_size, self._dm.input_dim).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        return float(np.mean(times))
