"""Training orchestrator with MLflow experiment tracking.

Handles the complete training lifecycle: model instantiation, optimization,
learning rate scheduling, early stopping, checkpointing, and experiment
logging — all driven by Hydra configuration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from src.data.loader import SensorDataModule
from src.models.mamba_kan import MambaKANDetector
from src.models.baselines.lstm import LSTMDetector
from src.models.baselines.transformer import TransformerDetector
from src.training.metrics import AnomalyMetrics


# ── Model Registry ───────────────────────────────────────────

_MODEL_REGISTRY = {
    "mamba_kan": MambaKANDetector,
    "lstm": LSTMDetector,
    "transformer": TransformerDetector,
}


def build_model(cfg: DictConfig, input_dim: int) -> nn.Module:
    """Instantiate a model from the registry based on config."""
    model_name = cfg.model.name
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[model_name].from_config(cfg, input_dim)


# ── Trainer ──────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss plateaus."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """End-to-end training orchestrator.

    Manages the training loop, validation, checkpointing, and MLflow logging
    for any model that implements the (reconstructed, scores) forward interface.

    Example:
        >>> trainer = Trainer(cfg, model, data_module)
        >>> trainer.fit()
        >>> results = trainer.evaluate()
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        data_module: SensorDataModule,
    ) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg.experiment.device)
        self.model = model.to(self.device)
        self.dm = data_module

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
        )
        self.metrics = AnomalyMetrics(point_adjust=cfg.evaluation.point_adjust)

        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_val_loss = float("inf")

    def fit(self) -> dict[str, Any]:
        """Execute the full training loop.

        Returns:
            Dict with training history and best metrics.
        """
        cfg = self.cfg
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "learning_rate": [],
        }

        if cfg.mlflow.enabled:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)

        with mlflow.start_run(run_name=cfg.experiment.name) if cfg.mlflow.enabled else _noop_ctx():
            if cfg.mlflow.enabled:
                mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
                param_counts = self.model.count_parameters()
                mlflow.log_params({f"params_{k}": v for k, v in param_counts.items()})

            logger.info(f"Training {cfg.model.name} | {param_counts.get('total', '?')} parameters")
            logger.info(f"Device: {self.device}")

            for epoch in range(1, cfg.training.epochs + 1):
                t0 = time.time()

                train_loss = self._train_epoch()
                val_loss = self._validate_epoch()
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["learning_rate"].append(lr)

                # Checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch)

                # Logging
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch:03d}/{cfg.training.epochs} | "
                        f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                        f"LR: {lr:.2e} | {elapsed:.1f}s"
                    )

                if cfg.mlflow.enabled and epoch % cfg.mlflow.log_every_n_steps == 0:
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss, "lr": lr},
                        step=epoch,
                    )

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Early stopping
                if self.early_stopping.step(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Final evaluation
            self._load_best_checkpoint()
            eval_results = self.evaluate()

            if cfg.mlflow.enabled:
                mlflow.log_metrics({f"test_{k}": v for k, v in eval_results.to_dict().items()})

            logger.info(f"Final test metrics: {eval_results}")

        return {"history": history, "test_metrics": eval_results.to_dict()}

    def evaluate(self):
        """Evaluate model on test set."""
        self.model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for windows, targets, labels in self.dm.test_loader:
                windows = windows.to(self.device)
                _, scores = self.model(windows)

                # Aggregate per-window scores (mean over timesteps)
                window_scores = scores.mean(dim=-1).cpu().numpy()
                all_scores.append(window_scores)
                all_labels.append(labels.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        return self.metrics.compute(scores, labels)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for windows, targets, labels in self.dm.train_loader:
            windows = windows.to(self.device)

            self.optimizer.zero_grad()
            reconstructed, scores = self.model(windows)
            losses = self.model.compute_loss(windows, reconstructed, scores)
            losses["total"].backward()

            if self.cfg.training.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip_val
                )

            self.optimizer.step()
            total_loss += losses["total"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for windows, targets, labels in self.dm.val_loader:
            windows = windows.to(self.device)
            reconstructed, scores = self.model(windows)
            losses = self.model.compute_loss(windows, reconstructed, scores)
            total_loss += losses["total"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": self.best_val_loss,
            },
            path,
        )

    def _load_best_checkpoint(self) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg.training
        optimizers = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        opt_cls = optimizers.get(cfg.optimizer, AdamW)
        return opt_cls(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def _build_scheduler(self):
        cfg = self.cfg.training
        if cfg.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.scheduler_params.T_max,
                eta_min=cfg.scheduler_params.eta_min,
            )
        elif cfg.scheduler == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.5)
        elif cfg.scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        return None

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)


class _noop_ctx:
    """No-op context manager when MLflow is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ── CLI Entry Point ──────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Train a model from CLI with Hydra config overrides.

    Usage:
        python -m src.training.trainer
        python -m src.training.trainer model.name=lstm training.epochs=50
    """
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    dm = SensorDataModule(cfg)
    dm.setup()

    if cfg.model.input_dim is None:
        cfg.model.input_dim = dm.input_dim

    model = build_model(cfg, input_dim=dm.input_dim)
    trainer = Trainer(cfg, model, dm)
    results = trainer.fit()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
