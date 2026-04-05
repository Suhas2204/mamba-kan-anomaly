"""Sensor data preprocessing pipeline.

Handles normalization, sliding-window segmentation, and train/val/test
partitioning for multivariate time-series sensor data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from loguru import logger


class ScalerType(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


@dataclass
class WindowConfig:
    """Configuration for sliding-window segmentation."""

    size: int = 64
    stride: int = 1
    horizon: int = 1  # steps ahead for forecasting objective


@dataclass
class SplitConfig:
    """Configuration for temporal train/val/test splits."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


@dataclass
class WindowedSample:
    """A single windowed segment with its label."""

    features: NDArray[np.float32]  # (window_size, n_features)
    target: NDArray[np.float32]    # (horizon, n_features) or reconstruction target
    label: int                      # 0 = normal, 1 = anomalous


@dataclass
class SplitArrays:
    """Container for partitioned arrays after temporal split."""

    train_features: NDArray[np.float32]
    train_labels: NDArray[np.float32]
    val_features: NDArray[np.float32]
    val_labels: NDArray[np.float32]
    test_features: NDArray[np.float32]
    test_labels: NDArray[np.float32]


class SensorPreprocessor:
    """End-to-end preprocessing for multivariate sensor streams.

    Applies normalization, sliding-window extraction, and temporal splitting
    while preserving temporal ordering (no shuffling across time).

    Example:
        >>> preprocessor = SensorPreprocessor(scaler_type="standard", window_size=64)
        >>> preprocessor.fit(train_data)
        >>> windows, targets, labels = preprocessor.transform(data, anomaly_labels)
    """

    _SCALER_MAP = {
        ScalerType.STANDARD: StandardScaler,
        ScalerType.MINMAX: MinMaxScaler,
        ScalerType.ROBUST: RobustScaler,
    }

    def __init__(
        self,
        scaler_type: Literal["standard", "minmax", "robust"] = "standard",
        window_cfg: WindowConfig | None = None,
        split_cfg: SplitConfig | None = None,
        clip_outliers: bool = True,
        clip_sigma: float = 5.0,
    ) -> None:
        self._scaler_type = ScalerType(scaler_type)
        self._scaler = self._SCALER_MAP[self._scaler_type]()
        self._window_cfg = window_cfg or WindowConfig()
        self._split_cfg = split_cfg or SplitConfig()
        self._clip_outliers = clip_outliers
        self._clip_sigma = clip_sigma
        self._is_fitted = False

    @property
    def n_features(self) -> int | None:
        """Number of sensor channels seen during fitting."""
        if not self._is_fitted:
            return None
        return self._scaler.n_features_in_

    def fit(self, data: NDArray[np.float32]) -> SensorPreprocessor:
        """Fit scaler on training data only.

        Args:
            data: Raw sensor readings of shape (n_timesteps, n_features).

        Returns:
            Self for method chaining.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array (timesteps, features), got shape {data.shape}")

        logger.info(
            f"Fitting {self._scaler_type.value} scaler on {data.shape[0]} timesteps, "
            f"{data.shape[1]} features"
        )
        self._scaler.fit(data)
        self._is_fitted = True
        return self

    def transform(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int32] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
        """Normalize and segment raw sensor data into overlapping windows.

        Args:
            data: Raw sensor readings (n_timesteps, n_features).
            labels: Per-timestep anomaly labels (n_timesteps,). Defaults to all zeros.

        Returns:
            Tuple of (windows, targets, window_labels) where:
                - windows: (n_windows, window_size, n_features)
                - targets: (n_windows, horizon, n_features)
                - window_labels: (n_windows,) — 1 if any timestep in window is anomalous
        """
        self._check_fitted()

        normalized = self._scaler.transform(data)

        if self._clip_outliers:
            normalized = np.clip(normalized, -self._clip_sigma, self._clip_sigma)

        if labels is None:
            labels = np.zeros(data.shape[0], dtype=np.int32)

        windows, targets, window_labels = self._extract_windows(normalized, labels)

        logger.info(
            f"Extracted {len(windows)} windows | "
            f"Anomalous: {window_labels.sum()} ({window_labels.mean() * 100:.1f}%)"
        )

        return windows, targets, window_labels

    def fit_transform(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int32] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
        """Convenience method: fit on data then transform it."""
        return self.fit(data).transform(data, labels)

    def temporal_split(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> SplitArrays:
        """Split data temporally into train/val/test without shuffling.

        Args:
            data: Full dataset (n_timesteps, n_features).
            labels: Per-timestep labels (n_timesteps,).

        Returns:
            SplitArrays with partitioned feature and label arrays.
        """
        n = len(data)
        cfg = self._split_cfg

        train_end = int(n * cfg.train_ratio)
        val_end = train_end + int(n * cfg.val_ratio)

        logger.info(
            f"Temporal split — train: [0:{train_end}], "
            f"val: [{train_end}:{val_end}], test: [{val_end}:{n}]"
        )

        return SplitArrays(
            train_features=data[:train_end],
            train_labels=labels[:train_end],
            val_features=data[train_end:val_end],
            val_labels=labels[train_end:val_end],
            test_features=data[val_end:],
            test_labels=labels[val_end:],
        )

    def inverse_transform(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reverse normalization for interpretability."""
        self._check_fitted()
        return self._scaler.inverse_transform(data)

    def _extract_windows(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
        """Segment time series into overlapping windows using stride tricks."""
        cfg = self._window_cfg
        n_timesteps, n_features = data.shape

        effective_len = cfg.size + cfg.horizon
        if n_timesteps < effective_len:
            raise ValueError(
                f"Data length ({n_timesteps}) < window_size + horizon ({effective_len})"
            )

        indices = range(0, n_timesteps - effective_len + 1, cfg.stride)

        windows = np.stack([data[i : i + cfg.size] for i in indices], dtype=np.float32)
        targets = np.stack(
            [data[i + cfg.size : i + effective_len] for i in indices], dtype=np.float32
        )
        window_labels = np.array(
            [int(labels[i : i + cfg.size].any()) for i in indices], dtype=np.int32
        )

        return windows, targets, window_labels

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call .fit() first.")
