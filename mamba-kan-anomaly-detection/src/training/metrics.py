"""Anomaly detection evaluation metrics.

Implements standard and time-series-specific metrics including point-adjusted
F1 scoring, dynamic thresholding, and composite evaluation for fair comparison
of anomaly detectors on continuous sensor data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricResult:
    """Container for evaluation metrics."""

    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    best_threshold: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "best_threshold": self.best_threshold,
        }

    def __repr__(self) -> str:
        return (
            f"F1={self.f1:.4f} | P={self.precision:.4f} | R={self.recall:.4f} | "
            f"AUROC={self.auroc:.4f} | AUPRC={self.auprc:.4f}"
        )


class AnomalyMetrics:
    """Compute anomaly detection metrics with optional point-adjust protocol.

    Point-adjust (PA%k) is the standard evaluation protocol for time-series
    anomaly detection: if any point within an anomaly segment is detected,
    the entire segment is considered correctly identified. This accounts for
    the practical reality that detecting an anomaly slightly early or late
    is still operationally useful.

    Example:
        >>> metrics = AnomalyMetrics(point_adjust=True)
        >>> result = metrics.compute(scores, labels)
        >>> print(result)  # F1=0.9234 | P=0.8912 | R=0.9578 | ...
    """

    def __init__(self, point_adjust: bool = True) -> None:
        self._point_adjust = point_adjust

    def compute(
        self,
        scores: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> MetricResult:
        """Compute all metrics with optimal threshold selection.

        Args:
            scores: Per-timestep anomaly scores (higher = more anomalous).
            labels: Ground-truth binary labels (1 = anomaly).

        Returns:
            MetricResult with precision, recall, F1, AUROC, AUPRC.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.int32).ravel()

        assert len(scores) == len(labels), "Score and label lengths must match"

        # Threshold-free metrics
        auroc = self._safe_auroc(scores, labels)
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall_curve, precision_curve)

        # Find optimal threshold via F1 maximization
        best_threshold = self._find_optimal_threshold(scores, labels)
        predictions = (scores >= best_threshold).astype(np.int32)

        # Apply point-adjust if enabled
        if self._point_adjust:
            predictions = self._apply_point_adjust(predictions, labels)

        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        return MetricResult(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            auroc=float(auroc),
            auprc=float(auprc),
            best_threshold=float(best_threshold),
        )

    def _find_optimal_threshold(
        self,
        scores: NDArray,
        labels: NDArray,
        n_candidates: int = 200,
    ) -> float:
        """Search for the threshold that maximizes F1 score."""
        thresholds = np.percentile(scores, np.linspace(80, 99.9, n_candidates))
        best_f1, best_thresh = 0.0, float(np.percentile(scores, 95))

        for thresh in thresholds:
            preds = (scores >= thresh).astype(np.int32)
            if self._point_adjust:
                preds = self._apply_point_adjust(preds, labels)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        return best_thresh

    @staticmethod
    def _apply_point_adjust(
        predictions: NDArray[np.int32],
        labels: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Apply point-adjust: if any point in an anomaly segment is detected,
        credit the entire segment as correctly detected."""
        adjusted = predictions.copy()
        segments = _find_anomaly_segments(labels)

        for start, end in segments:
            if predictions[start:end].any():
                adjusted[start:end] = 1

        return adjusted

    @staticmethod
    def _safe_auroc(scores: NDArray, labels: NDArray) -> float:
        """AUROC with fallback for edge cases (all-normal or all-anomaly)."""
        if len(np.unique(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5


def _find_anomaly_segments(labels: NDArray[np.int32]) -> list[tuple[int, int]]:
    """Extract contiguous anomaly segments as (start, end) index pairs."""
    segments = []
    in_segment = False
    start = 0

    for i, label in enumerate(labels):
        if label == 1 and not in_segment:
            start = i
            in_segment = True
        elif label == 0 and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(labels)))

    return segments
