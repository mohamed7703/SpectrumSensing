"""Unit Tests for RadioML 2018.01A Dataset Ingestion and Preprocessing Pipeline.

Tests synthetic frame generation, I/Q tensor shape (1024, 2), SNR filtering,
binary hypothesis labeling, train/val/test splits, and batch generation.
"""

import pytest
import numpy as np

from src.training.dataset_loader import (
    RadioMLDataset,
    SyntheticRadioMLGenerator,
    DatasetLoader,
    MODULATION_CLASSES,
    ALL_SNR_LEVELS,
)


class TestSyntheticRadioMLGenerator:
    """Test synthetic RadioML signal synthesis."""

    def test_single_frame_shape_and_dtype(self) -> None:
        """Every generated frame must have shape (1024, 2) and float32 dtype."""
        gen = SyntheticRadioMLGenerator(random_seed=42)
        frame = gen.generate_single_frame(modulation="QPSK", snr_db=10.0, num_samples=1024)

        assert frame.shape == (1024, 2)
        assert frame.dtype == np.float32
        assert np.all(np.isfinite(frame))

    def test_h0_noise_frame_properties(self) -> None:
        """H0 noise frame must be complex zero-mean circular Gaussian noise."""
        gen = SyntheticRadioMLGenerator(random_seed=42)
        frame = gen.generate_single_frame(modulation="NOISE", snr_db=0.0, num_samples=4096)

        i_rail = frame[:, 0]
        q_rail = frame[:, 1]

        # Mean should be close to 0
        assert np.isclose(np.mean(i_rail), 0.0, atol=0.08)
        assert np.isclose(np.mean(q_rail), 0.0, atol=0.08)
        # Power should be close to 1
        total_power = np.mean(i_rail**2 + q_rail**2)
        assert np.isclose(total_power, 1.0, atol=0.1)

    def test_dataset_generation_binary_classes(self) -> None:
        """Dataset must contain both H0 (0) and H1 (1) binary labels."""
        gen = SyntheticRadioMLGenerator(random_seed=42)
        ds = gen.generate_dataset(frames_per_class=5, modulations=["BPSK", "QPSK"], snrs=[0, 10], h0_ratio=0.5)

        assert len(ds) > 0
        assert set(np.unique(ds.y_binary)) == {0, 1}
        assert ds.x.ndim == 3
        assert ds.x.shape[1:] == (1024, 2)


class TestRadioMLDataset:
    """Test dataset filtering, partitioning, and batching."""

    def test_snr_filtering(self) -> None:
        """Filter dataset to SNR in [-8, 20] dB."""
        gen = SyntheticRadioMLGenerator(random_seed=42)
        ds = gen.generate_dataset(frames_per_class=4, modulations=["BPSK"], snrs=[-14, -8, 0, 12, 24])

        filtered = ds.filter_by_snr(min_snr_db=-8.0, max_snr_db=20.0)

        # H1 signals outside [-8, 20] must be excluded
        h1_snrs = filtered.snr[filtered.y_binary == 1]
        assert np.all(h1_snrs >= -8.0)
        assert np.all(h1_snrs <= 20.0)
        # H0 noise must remain present
        assert np.any(filtered.y_binary == 0)

    def test_train_val_test_split(self) -> None:
        """Split must divide dataset accurately without overlapping indices."""
        gen = SyntheticRadioMLGenerator(random_seed=42)
        ds = gen.generate_dataset(frames_per_class=10, modulations=["BPSK", "QPSK"], snrs=[0, 10])

        total_len = len(ds)
        train_ds, val_ds, test_ds = ds.train_val_test_split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

        assert len(train_ds) + len(val_ds) + len(test_ds) == total_len
        assert np.isclose(len(train_ds) / total_len, 0.70, atol=0.05)
        assert np.isclose(len(val_ds) / total_len, 0.15, atol=0.05)
        assert np.isclose(len(test_ds) / total_len, 0.15, atol=0.05)

    def test_batch_generator_shapes(self) -> None:
        """Batch generator must yield batches of size (B, 1024, 2) and (B,)."""
        gen = SyntheticRadioMLGenerator(random_seed=42)
        ds = gen.generate_dataset(frames_per_class=8, modulations=["QPSK"], snrs=[0, 10])

        batches = list(ds.batch_generator(batch_size=16, shuffle=False))
        assert len(batches) > 0

        for batch_x, batch_y in batches:
            assert batch_x.shape[1:] == (1024, 2)
            assert len(batch_x) == len(batch_y)
            assert batch_x.dtype == np.float32


class TestDatasetLoader:
    """Test unified loader and fallback behavior."""

    def test_load_or_generate_synthetic_fallback(self) -> None:
        """When HDF5 file does not exist, loader must transparently generate synthetic data."""
        ds = DatasetLoader.load_or_generate(
            hdf5_path="non_existent_file.hdf5",
            min_snr_db=-8.0,
            max_snr_db=30.0,
            num_synthetic_frames=128,
            random_seed=42,
        )

        assert isinstance(ds, RadioMLDataset)
        assert len(ds) > 0
        assert ds.x.shape[1:] == (1024, 2)
        assert set(np.unique(ds.y_binary)).issubset({0, 1})
