"""Unit tests for noise models, calibrated AWGN, and the SNR wall phenomenon.

Verifies standard deviation matching the paper (sigma=0.708), theoretical SNR wall
calculations, infinite sample complexity barriers, and threshold degradation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.math_models.noise_models import (
    CalibratedAWGNGenerator,
    NoiseModelConfig,
    compute_sample_complexity,
    compute_snr_wall,
)
from src.math_models.energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    compute_energy_statistic,
)


class TestCalibratedAWGN:
    """Test suite for calibrated noise generation matching the paper."""

    def test_paper_calibrated_variance(self):
        """Paper uses sigma = 0.708, yielding variance approx 0.5013."""
        cfg = NoiseModelConfig(std_dev=0.708)
        assert np.isclose(cfg.variance, 0.501264, atol=1e-5)

        gen = CalibratedAWGNGenerator(cfg, seed=42)
        noise = gen.generate(200_000)

        assert np.isclose(np.mean(noise), 0.0, atol=0.01)
        assert np.isclose(np.var(noise), cfg.variance, rtol=0.02)

    def test_uncertain_noise_generation_bounds(self):
        """Uncertain noise variance is strictly within [sigma^2 / rho, rho * sigma^2]."""
        cfg = NoiseModelConfig(std_dev=1.0, uncertainty_db=1.0)
        gen = CalibratedAWGNGenerator(cfg, seed=99)

        min_var = cfg.variance / cfg.rho
        max_var = cfg.variance * cfg.rho

        for _ in range(20):
            _, perturbed_var = gen.generate_uncertain_noise(1000)
            assert min_var <= perturbed_var <= max_var


class TestSNRWallDerivations:
    """Test suite for SNR wall equations."""

    def test_snr_wall_calculation_values(self):
        """Verify known theoretical values: 0.2 dB uncertainty gives SNR wall ~ -10.4 dB."""
        _, snr_wall_db = compute_snr_wall(uncertainty_db=0.2)
        assert np.isclose(snr_wall_db, -10.36, atol=0.1)

    def test_snr_wall_monotonicity_with_uncertainty(self):
        """Greater noise uncertainty shifts SNR wall higher (worse detection ceiling)."""
        _, wall_02 = compute_snr_wall(0.2)
        _, wall_05 = compute_snr_wall(0.5)
        _, wall_10 = compute_snr_wall(1.0)

        assert wall_02 < wall_05 < wall_10

    def test_non_positive_uncertainty_raises_error(self):
        """Non-positive uncertainty bound raises ValueError."""
        with pytest.raises(ValueError):
            compute_snr_wall(0.0)
        with pytest.raises(ValueError):
            compute_snr_wall(-0.5)


class TestSampleComplexity:
    """Test suite for sample complexity under noise uncertainty."""

    def test_infinite_samples_below_snr_wall(self):
        """Below the SNR wall, required sample size L is strictly infinite."""
        # For 0.5 dB uncertainty, SNR wall is ~ -6.37 dB
        _, wall_db = compute_snr_wall(0.5)

        l_below = compute_sample_complexity(snr_db=wall_db - 1.0, uncertainty_db=0.5)
        assert np.isinf(l_below)

    def test_finite_samples_above_snr_wall(self):
        """Above the SNR wall, sample size is finite and decreases as SNR rises."""
        _, wall_db = compute_snr_wall(0.5)

        l_near = compute_sample_complexity(snr_db=wall_db + 2.0, uncertainty_db=0.5)
        l_far = compute_sample_complexity(snr_db=wall_db + 8.0, uncertainty_db=0.5)

        assert not np.isinf(l_near)
        assert not np.isinf(l_far)
        assert l_near > l_far


class TestDetectorFailureUnderUncertainty:
    """Verify performance breakdown when receiver operates under perturbed noise."""

    def test_false_alarm_collapse_with_noise_rise(self):
        """If actual noise variance rises by +0.5 dB above nominal, Pfa blows up from 5% to > 25%."""
        l_samples = 1024
        nominal_var = 1.0
        pfa_target = 0.05

        detector = EnergyDetector(EnergyDetectorConfig(num_samples=l_samples, noise_variance=nominal_var, pfa_target=pfa_target))

        # True noise is 0.5 dB higher: 10^(0.05) ~ 1.122
        perturbed_var = nominal_var * (10.0 ** (0.5 / 10.0))
        rng = np.random.default_rng(42)

        # Generate 5000 H0 frames with elevated noise floor
        std_comp = np.sqrt(perturbed_var / 2.0)
        noise_frames = rng.normal(0, std_comp, size=(5000, l_samples)) + 1j * rng.normal(0, std_comp, size=(5000, l_samples))

        emp_pfa = np.mean(detector.decide_batch(noise_frames) == 1)

        # Uncalibrated detector treats the noise rise as primary user signals!
        assert emp_pfa > 0.25
