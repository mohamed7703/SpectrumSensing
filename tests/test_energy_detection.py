"""Unit tests for the energy detection module.

Verifies test statistics, CFAR threshold calculation (Exact Chi-Square & CLT),
analytical detection probability, and empirical Monte Carlo convergence.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from scipy import stats

from src.math_models.energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    compute_energy_statistic,
    compute_theoretical_threshold,
    analytical_pd,
)
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


class TestEnergyStatisticCalculation:
    """Test suite for energy statistic extraction."""

    def test_single_complex_frame_statistic(self):
        """Average power equals mean(|r|^2)."""
        r = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])  # each |r|^2 = 2
        stat = compute_energy_statistic(r, normalize=False)
        assert np.isclose(stat, 2.0)

    def test_iq_tensor_shape_statistic(self):
        """IQ shape (N, L, 2) properly calculates I^2 + Q^2."""
        batch = np.ones((5, 1024, 2))  # I=1, Q=1 -> |r|^2 = 2
        stats_batch = compute_energy_statistic(batch, normalize=False)
        assert stats_batch.shape == (5,)
        assert np.all(np.isclose(stats_batch, 2.0))

    def test_normalized_statistic_degrees_of_freedom(self):
        """Normalized complex noise statistic follows Chi-square with 2L DoF."""
        rng = np.random.default_rng(42)
        n_trials = 10_000
        l_samples = 50
        noise_var = 1.0

        # Circularly symmetric complex Gaussian noise
        std = np.sqrt(noise_var / 2.0)
        noise = rng.normal(0, std, size=(n_trials, l_samples)) + 1j * rng.normal(0, std, size=(n_trials, l_samples))

        t_norm = compute_energy_statistic(noise, normalize=True, noise_variance=noise_var)

        # Theoretical Chi-square(2L): Mean = 2L, Variance = 4L
        expected_mean = 2 * l_samples
        expected_var = 4 * l_samples

        assert np.isclose(np.mean(t_norm), expected_mean, rtol=0.03)
        assert np.isclose(np.var(t_norm), expected_var, rtol=0.05)


class TestThresholdDerivation:
    """Test suite for threshold calculation."""

    def test_exact_vs_clt_approximation_agreement(self):
        """For L=1024, exact Chi-Square and CLT threshold agree within 0.2%."""
        pfa = 0.05
        l_samples = 1024
        var_w = 1.0

        thresh_exact = compute_theoretical_threshold(pfa, l_samples, var_w, method="exact")
        thresh_clt = compute_theoretical_threshold(pfa, l_samples, var_w, method="clt")

        assert np.isclose(thresh_exact, thresh_clt, rtol=0.005)

    def test_threshold_monotonicity_with_pfa(self):
        """Stricter (smaller) PFA requires higher threshold."""
        t_01 = compute_theoretical_threshold(pfa_target=0.01, num_samples=1024)
        t_05 = compute_theoretical_threshold(pfa_target=0.05, num_samples=1024)
        t_10 = compute_theoretical_threshold(pfa_target=0.10, num_samples=1024)

        assert t_01 > t_05 > t_10

    def test_invalid_pfa_raises_error(self):
        """pfa outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            compute_theoretical_threshold(pfa_target=0.0, num_samples=100)
        with pytest.raises(ValueError):
            compute_theoretical_threshold(pfa_target=1.0, num_samples=100)


class TestAnalyticalDetectionProbability:
    """Test suite for analytical Pd derivations."""

    def test_pd_monotonicity_with_snr(self):
        """Pd increases monotonically as SNR improves within the dynamic transition range."""
        pfa = 0.05
        pd_low = analytical_pd(pfa=pfa, snr_db=-16.0, num_samples=1024)
        pd_mid = analytical_pd(pfa=pfa, snr_db=-13.0, num_samples=1024)
        pd_high = analytical_pd(pfa=pfa, snr_db=-10.0, num_samples=1024)

        assert 0.0 < pd_low < pd_mid < pd_high < 1.0

    def test_high_snr_approaches_unity(self):
        """At SNR >= 4 dB with L=1024, Pd is virtually 1.0."""
        pd = analytical_pd(pfa=0.01, snr_db=5.0, num_samples=1024)
        assert np.isclose(pd, 1.0, atol=1e-3)

    def test_very_low_snr_approaches_pfa(self):
        """At SNR -> -infinity, detector cannot distinguish signal; Pd -> Pfa."""
        pfa = 0.1
        pd = analytical_pd(pfa=pfa, snr_db=-30.0, num_samples=1024)
        assert np.isclose(pd, pfa, atol=0.02)


class TestEnergyDetectorEmpiricalEvaluation:
    """Test suite for Monte Carlo convergence of the EnergyDetector class."""

    def test_empirical_pfa_matches_target(self):
        """Under H0, empirical false alarm rate aligns with target Pfa within confidence interval."""
        pfa_target = 0.05
        n_trials = 10_000
        l_samples = 1024

        cfg = EnergyDetectorConfig(num_samples=l_samples, pfa_target=pfa_target, method="exact")
        detector = EnergyDetector(cfg)

        sim = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples), seed=42)
        h0_batch, _ = sim.generate_batch(num_frames=n_trials, p_h1=0.0, as_iq_channels=True)

        ground_truth = np.zeros(n_trials, dtype=int)
        res = detector.evaluate_performance(h0_batch, ground_truth)

        # Standard error of proportion: sqrt(p*(1-p)/N) = sqrt(0.05*0.95/10000) ~ 0.0022
        assert np.isclose(res["pfa"], pfa_target, atol=0.008)

    def test_empirical_pd_matches_analytical_theory(self):
        """Under H1, empirical Pd aligns with non-central Chi-square theoretical Pd."""
        pfa_target = 0.05
        snr_db = -5.0  # Moderately low SNR to test intermediate detection probabilities
        n_trials = 10_000
        l_samples = 1024

        cfg = EnergyDetectorConfig(num_samples=l_samples, pfa_target=pfa_target, method="exact")
        detector = EnergyDetector(cfg)

        theory_pd = analytical_pd(pfa=pfa_target, snr_db=snr_db, num_samples=l_samples, method="exact")

        sim = HypothesisTestingSimulator(
            HypothesisConfig(num_samples=l_samples, snr_db=snr_db, modulation="bpsk"),
            seed=101,
        )
        h1_batch, _ = sim.generate_batch(num_frames=n_trials, p_h1=1.0, as_iq_channels=True)

        ground_truth = np.ones(n_trials, dtype=int)
        res = detector.evaluate_performance(h1_batch, ground_truth)

        # Empirical Pd must match analytical Pd within 3 standard errors
        assert np.isclose(res["pd"], theory_pd, atol=0.015)
