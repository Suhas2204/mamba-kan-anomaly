"""Unit tests for data preprocessing pipeline."""

import numpy as np
import pytest

from src.data.preprocessing import SensorPreprocessor, SplitConfig, WindowConfig


class TestSensorPreprocessor:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(1000, 4).astype(np.float32)

    @pytest.fixture
    def sample_labels(self):
        labels = np.zeros(1000, dtype=np.int32)
        labels[200:220] = 1
        labels[500:510] = 1
        return labels

    def test_fit_transform_shapes(self, sample_data, sample_labels):
        prep = SensorPreprocessor(window_cfg=WindowConfig(size=32, stride=4))
        windows, targets, labels = prep.fit_transform(sample_data, sample_labels)

        assert windows.ndim == 3
        assert windows.shape[1] == 32
        assert windows.shape[2] == 4
        assert targets.ndim == 3
        assert len(windows) == len(labels)

    def test_temporal_split_preserves_order(self, sample_data, sample_labels):
        prep = SensorPreprocessor()
        splits = prep.temporal_split(sample_data, sample_labels)

        assert len(splits.train_features) == 700
        assert len(splits.val_features) == 150
        assert len(splits.test_features) == 150

    def test_split_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_transform_before_fit_raises(self, sample_data):
        prep = SensorPreprocessor()
        with pytest.raises(RuntimeError, match="fitted"):
            prep.transform(sample_data)

    def test_scaler_types(self, sample_data, sample_labels):
        for scaler in ["standard", "minmax", "robust"]:
            prep = SensorPreprocessor(scaler_type=scaler, window_cfg=WindowConfig(size=16))
            windows, _, _ = prep.fit_transform(sample_data, sample_labels)
            assert not np.isnan(windows).any()

    def test_window_labels_capture_anomalies(self, sample_data, sample_labels):
        prep = SensorPreprocessor(window_cfg=WindowConfig(size=16, stride=8))
        _, _, window_labels = prep.fit_transform(sample_data, sample_labels)
        assert window_labels.sum() > 0

    def test_inverse_transform(self, sample_data, sample_labels):
        prep = SensorPreprocessor()
        prep.fit(sample_data)
        normalized = prep._scaler.transform(sample_data)
        recovered = prep.inverse_transform(normalized)
        np.testing.assert_allclose(recovered, sample_data, atol=1e-5)
