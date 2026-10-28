"""Unit tests for binary hypothesis testing module.

Verifies statistical correctness of H0/H1 signals, power allocations,
modulations, and AWGN generators against theoretical derivations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
    generate_awgn,
    generate_pu_signal,
    simulate_received_signal,
)


class TestAWGNGeneration:
    """Test suite for Additive White Gaussian Noise generator."""

    def test_complex_awgn_shape_and_dtype(self):
        """Verify complex AWGN produces correct length and complex128 type."""
        samples = generate_awgn(num_samples=1024, noise_variance=1.0, complex_noise=True)
        assert len(samples) == 1024
        assert np.iscomplexobj(samples)

    def test_real_awgn_shape_and_dtype(self):
        """Verify real AWGN produces float type."""
        samples = generate_awgn(num_samples=1024, noise_variance=1.0, complex_noise=False)
        assert len(samples) == 1024
        assert not np.iscomplexobj(samples)

    def test_awgn_statistical_moments(self):
        """Verify empirical mean and variance match theoretical zero mean and sigma_w^2."""
        rng = np.random.default_rng(42)
        n_samples = 200_000
        noise_var = 2.5
        samples = generate_awgn(
            num_samples=n_samples, noise_variance=noise_var, complex_noise=True, rng=rng
        )

        emp_mean = np.mean(samples)
        emp_var = np.var(samples)

        # Standard error of mean: sqrt(var / N) = sqrt(2.5 / 200_000) ~ 0.0035
        assert np.abs(emp_mean) < 0.03
        # Empirical variance within 3% of theoretical
        assert np.isclose(emp_var, noise_var, rtol=0.03)

    def test_invalid_noise_variance(self):
        """Negative or zero noise variance should raise ValueError."""
        with pytest.raises(ValueError):
            generate_awgn(num_samples=100, noise_variance=0.0)
        with pytest.raises(ValueError):
            generate_awgn(num_samples=100, noise_variance=-1.0)


class TestPUSignalGeneration:
    """Test suite for Primary User signal generation across modulation schemes."""

    @pytest.mark.parametrize("mod", ["bpsk", "qpsk", "sinusoid", "gaussian"])
    def test_signal_power_conservation(self, mod):
        """Verify average power E[|s(n)|^2] matches target power across all modulations."""
        rng = np.random.default_rng(123)
        n_samples = 100_000
        target_power = 2.0
        sig = generate_pu_signal(
            num_samples=n_samples,
            signal_power=target_power,
            modulation=mod,
            complex_signal=True,
            rng=rng,
        )

        emp_power = np.mean(np.abs(sig) ** 2)
        assert np.isclose(emp_power, target_power, rtol=0.03)

    def test_bpsk_constellation(self):
        """Verify BPSK symbols are strictly on the real axis with magnitude sqrt(P)."""
        power = 4.0
        sig = generate_pu_signal(
            num_samples=1000, signal_power=power, modulation="bpsk", complex_signal=True
        )
        assert np.all(np.isclose(np.imag(sig), 0.0))
        assert np.all(np.isclose(np.abs(np.real(sig)), 2.0))

    def test_qpsk_constellation(self):
        """Verify QPSK symbols have equal magnitude sqrt(P) with 4 discrete phases."""
        power = 2.0
        sig = generate_pu_signal(
            num_samples=1000, signal_power=power, modulation="qpsk", complex_signal=True
        )
        magnitudes = np.abs(sig)
        assert np.all(np.isclose(magnitudes, np.sqrt(power)))


class TestReceivedSignalSimulation:
    """Test suite for H0/H1 received signals and SNR scaling."""

    def test_h0_has_no_signal_component(self):
        """Under H0, empirical power equals noise variance."""
        rng = np.random.default_rng(777)
        config = HypothesisConfig(num_samples=100_000, noise_variance=1.5, snr_db=10.0)
        r = simulate_received_signal(hypothesis=0, config=config, rng=rng)

        emp_power = np.mean(np.abs(r) ** 2)
        assert np.isclose(emp_power, config.noise_variance, rtol=0.03)

    def test_h1_power_matches_snr_formula(self):
        """Under H1, power equals sigma_w^2 * (1 + 10^(SNR/10))."""
        rng = np.random.default_rng(888)
        snr_db = 6.0  # 10^(0.6) ~ 3.981
        noise_var = 1.0
        config = HypothesisConfig(
            num_samples=100_000,
            noise_variance=noise_var,
            snr_db=snr_db,
            modulation="bpsk",
        )
        r = simulate_received_signal(hypothesis=1, config=config, channel_gain=1.0, rng=rng)

        expected_power = noise_var * (1.0 + 10.0 ** (snr_db / 10.0))
        emp_power = np.mean(np.abs(r) ** 2)
        assert np.isclose(emp_power, expected_power, rtol=0.03)

    def test_invalid_hypothesis_raises_error(self):
        """Invalid hypothesis integer raises ValueError."""
        config = HypothesisConfig()
        with pytest.raises(ValueError):
            simulate_received_signal(hypothesis=2, config=config)  # type: ignore


class TestHypothesisTestingSimulator:
    """Test suite for the batch simulator class."""

    def test_batch_generation_dimensions(self):
        """Verify batch output tensor shapes matching RadioML (N, L, 2)."""
        sim = HypothesisTestingSimulator(HypothesisConfig(num_samples=512), seed=42)
        x_batch, y_batch = sim.generate_batch(num_frames=50, as_iq_channels=True)

        assert x_batch.shape == (50, 512, 2)
        assert y_batch.shape == (50,)
        assert set(np.unique(y_batch)).issubset({0, 1})

    def test_batch_complex_format(self):
        """Verify batch output tensor shape (N, L) when complex format requested."""
        sim = HypothesisTestingSimulator(HypothesisConfig(num_samples=256), seed=42)
        x_batch, y_batch = sim.generate_batch(num_frames=20, as_iq_channels=False)

        assert x_batch.shape == (20, 256)
        assert np.iscomplexobj(x_batch)

    def test_prior_probability_p_h1(self):
        """Verify proportion of H1 frames matches specified prior probability p_h1."""
        sim = HypothesisTestingSimulator(seed=999)
        _, y_batch = sim.generate_batch(num_frames=5000, p_h1=0.7)

        prop_h1 = np.mean(y_batch)
        assert np.isclose(prop_h1, 0.7, atol=0.03)
