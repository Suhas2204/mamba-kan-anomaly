"""Sensor dataset loaders and PyTorch DataModule.

Provides a unified interface for loading SKAB and other sensor benchmark
datasets into windowed PyTorch DataLoaders suitable for anomaly detection.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from src.data.preprocessing import SensorPreprocessor, SplitConfig, WindowConfig


# ── Dataset Registry ─────────────────────────────────────────

_SKAB_URL = (
    "https://github.com/waico/SKAB/archive/refs/heads/master.zip"
)

_DATASET_REGISTRY: dict[str, str] = {
    "skab": _SKAB_URL,
}


# ── PyTorch Dataset ──────────────────────────────────────────

class SensorWindowDataset(Dataset):
    """PyTorch dataset wrapping pre-extracted sensor windows.

    Each sample is a tuple of (window_tensor, target_tensor, label).
    """

    def __init__(
        self,
        windows: NDArray[np.float32],
        targets: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> None:
        assert len(windows) == len(targets) == len(labels)
        self.windows = torch.from_numpy(windows).float()
        self.targets = torch.from_numpy(targets).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx], self.labels[idx]


# ── SKAB Dataset Loader ──────────────────────────────────────

class SKABDataset:
    """Skoltech Anomaly Benchmark dataset loader.

    Downloads and parses the SKAB benchmark, which contains multivariate
    sensor readings from a water circulation testbed with labeled anomalies
    across multiple fault scenarios.

    Reference:
        Katser & Kozitsin (2020). "Skoltech Anomaly Benchmark (SKAB)."
        https://github.com/waico/SKAB

    Example:
        >>> skab = SKABDataset(root_dir="data/")
        >>> features, labels, metadata = skab.load()
    """

    SENSOR_COLUMNS = [
        "Accelerometer1RMS",
        "Accelerometer2RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS",
    ]

    def __init__(self, root_dir: str | Path = "data/") -> None:
        self._root = Path(root_dir)
        self._skab_dir = self._root / "SKAB-master"

    def load(self) -> tuple[NDArray[np.float32], NDArray[np.int32], dict[str, Any]]:
        """Load and concatenate all SKAB experiment CSVs.

        Returns:
            Tuple of (features, labels, metadata) where:
                - features: (n_timesteps, 8) sensor readings                     # eg 2D array it looks like [[0.1, 0.2, ..., 0.3], [0.2, 0.1, ..., 0.4], ...]
                - labels: (n_timesteps,) binary anomaly labels                   # eg it looks like [0, 0, 1, 0, ...] where 1 indicates an anomaly
                - metadata: dict with dataset statistics                         # eg metdata = { "n_timesteps": 100000, "n_features": 8, "n_experiments": 15, "anomaly_ratio": 0.05, "feature_names": [...], "experiments": [...] }
        """ 
        self._ensure_downloaded()

        all_features: list[NDArray] = []
        all_labels: list[NDArray] = []
        experiment_names: list[str] = []

        data_dir = self._skab_dir / "data"
        csv_files = sorted(data_dir.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        for csv_path in csv_files:
            df = self._parse_single_csv(csv_path)
            if df is None:
                continue

            features = df[self.SENSOR_COLUMNS].values.astype(np.float32)
            labels = df["anomaly"].values.astype(np.int32) if "anomaly" in df.columns else (
                np.zeros(len(df), dtype=np.int32)
            )

            all_features.append(features)
            all_labels.append(labels)
            experiment_names.append(csv_path.stem)

        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        metadata = {
            "n_timesteps": len(combined_features),
            "n_features": combined_features.shape[1],
            "n_experiments": len(experiment_names),
            "anomaly_ratio": float(combined_labels.mean()),
            "feature_names": self.SENSOR_COLUMNS,
            "experiments": experiment_names,
        }

        logger.info(
            f"SKAB loaded: {metadata['n_timesteps']} timesteps, "
            f"{metadata['n_features']} features, "
            f"{metadata['anomaly_ratio']:.2%} anomalous"
        )

        return combined_features, combined_labels, metadata

    def _parse_single_csv(self, path: Path) -> pd.DataFrame | None:
        """Parse a single SKAB CSV with error handling."""
        try:
            df = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True)
            required = set(self.SENSOR_COLUMNS)
            if not required.issubset(set(df.columns)):
                logger.debug(f"Skipping {path.name}: missing required columns")
                return None
            return df.dropna(subset=self.SENSOR_COLUMNS)
        except Exception as e:
            logger.warning(f"Failed to parse {path.name}: {e}")
            return None

    def _ensure_downloaded(self) -> None:
        """Download SKAB if not already cached."""
        if self._skab_dir.exists():
            logger.debug("SKAB dataset already cached.")
            return

        self._root.mkdir(parents=True, exist_ok=True)
        zip_path = self._root / "skab.zip"

        logger.info("Downloading SKAB dataset...")
        response = requests.get(_SKAB_URL, stream=True, timeout=120)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Extracting SKAB archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._root)

        zip_path.unlink()
        logger.info(f"SKAB ready at {self._skab_dir}")


# ── DataModule (orchestrates everything) ─────────────────────

class SensorDataModule:
    """Top-level data orchestrator for sensor anomaly detection.

    Chains dataset loading → preprocessing → splitting → DataLoader creation
    into a single coherent interface. Inspired by PyTorch Lightning's
    LightningDataModule pattern.

    Example:
        >>> dm = SensorDataModule(cfg)
        >>> dm.setup()
        >>> for batch in dm.train_loader:
        ...     windows, targets, labels = batch
    """

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._preprocessor: SensorPreprocessor | None = None
        self._train_ds: SensorWindowDataset | None = None
        self._val_ds: SensorWindowDataset | None = None
        self._test_ds: SensorWindowDataset | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def input_dim(self) -> int:
        """Number of sensor channels."""
        return self._metadata.get("n_features", 0)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def setup(self) -> None:
        """Execute the full data pipeline: load → preprocess → split → window."""
        dataset_name = self._cfg.data.dataset

        if dataset_name == "skab":
            features, labels, self._metadata = SKABDataset(self._cfg.data.root_dir).load()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self._preprocessor = SensorPreprocessor(
            scaler_type=self._cfg.data.scaler,
            window_cfg=WindowConfig(
                size=self._cfg.data.window_size,
                stride=self._cfg.data.stride,
            ),
            split_cfg=SplitConfig(
                train_ratio=self._cfg.data.train_ratio,
                val_ratio=self._cfg.data.val_ratio,
                test_ratio=self._cfg.data.test_ratio,
            ),
        )

        splits = self._preprocessor.temporal_split(features, labels)

        self._preprocessor.fit(splits.train_features)

        train_w, train_t, train_l = self._preprocessor.transform(
            splits.train_features, splits.train_labels
        )
        val_w, val_t, val_l = self._preprocessor.transform(
            splits.val_features, splits.val_labels
        )
        test_w, test_t, test_l = self._preprocessor.transform(
            splits.test_features, splits.test_labels
        )

        self._train_ds = SensorWindowDataset(train_w, train_t, train_l)
        self._val_ds = SensorWindowDataset(val_w, val_t, val_l)
        self._test_ds = SensorWindowDataset(test_w, test_t, test_l)

        logger.info(
            f"DataModule ready — "
            f"train: {len(self._train_ds)}, val: {len(self._val_ds)}, test: {len(self._test_ds)}"
        )

    @property
    def preprocessor(self) -> SensorPreprocessor:
        if self._preprocessor is None:
            raise RuntimeError("Call .setup() before accessing the preprocessor.")
        return self._preprocessor

    @property
    def train_loader(self) -> DataLoader:
        return self._build_loader(self._train_ds, shuffle=True)

    @property
    def val_loader(self) -> DataLoader:
        return self._build_loader(self._val_ds, shuffle=False)

    @property
    def test_loader(self) -> DataLoader:
        return self._build_loader(self._test_ds, shuffle=False)

    def _build_loader(self, dataset: SensorWindowDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call .setup() before accessing data loaders.")
        return DataLoader(
            dataset,
            batch_size=self._cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=self._cfg.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
