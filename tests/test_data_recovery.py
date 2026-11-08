"""Check data provenance safeguards and hypothesis-independent synthesis."""

from pathlib import Path
import h5py
import numpy as np
import pytest

from src.training.dataset_loader import DatasetLoader, SyntheticRadioMLGenerator


def test_explicit_missing_dataset_does_not_silently_become_synthetic(tmp_path: Path) -> None:
    """A misspelled real-data path must fail visibly instead of changing the study."""
    with pytest.raises(FileNotFoundError):
        DatasetLoader.load_or_generate(tmp_path / "missing.hdf5")


def test_hdf5_limit_applies_after_snr_filter(tmp_path: Path) -> None:
    """A low-SNR sorted prefix must not consume the requested selected-frame cap."""
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as stream:
        stream["X"] = np.arange(6 * 16 * 2, dtype=np.float32).reshape(6, 16, 2)
        stream["Y"] = np.eye(24, dtype=np.float32)[np.arange(6)]
        stream["Z"] = np.array([-20, -18, -8, 0, 10, 20]).reshape(-1, 1)
    data = DatasetLoader.load_from_hdf5(path, min_snr_db=0, max_samples=2)
    assert len(data) == 2
    np.testing.assert_array_equal(data.snr, [0, 10])
    assert np.all(data.y_binary == 1)  # HDF5 alone never establishes binary Pfa.


def test_synthetic_h1_retains_energy_with_fixed_h0_noise_power() -> None:
    """At +10dB the received mean power is about 11, without class-specific normalization."""
    generator = SyntheticRadioMLGenerator(random_seed=1)
    h0 = generator.generate_single_frame("NOISE", 10, num_samples=100000)
    h1 = generator.generate_single_frame("QPSK", 10, num_samples=100000)
    assert np.mean(np.sum(h0 ** 2, axis=-1)) == pytest.approx(1, rel=.02)
    assert np.mean(np.sum(h1 ** 2, axis=-1)) == pytest.approx(11, rel=.02)


def test_unsupported_modulation_is_not_silently_qpsk() -> None:
    """Synthetic coverage must not be inflated with mislabeled fallback waveforms."""
    with pytest.raises(ValueError, match="supported"):
        SyntheticRadioMLGenerator().generate_single_frame("invented_modulation", 10)


@pytest.mark.parametrize("h0_ratio", [-.1, 1., float("nan")])
def test_invalid_hypothesis_ratio_rejected(h0_ratio: float) -> None:
    """Reject undefined class proportions rather than silently generating a study."""
    with pytest.raises(ValueError):
        SyntheticRadioMLGenerator().generate_dataset(frames_per_class=1, h0_ratio=h0_ratio)
