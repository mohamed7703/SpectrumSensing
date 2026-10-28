"""Unit tests for wireless channel fading models.

Verifies analytical PDFs (Rayleigh, Rician, Nakagami-m), integration to unity,
energy conservation of stochastic channel coefficients, and detection degradation
under fading channels as documented in the research plan.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from scipy import integrate

from src.math_models.channel_fading import (
    FadingChannel,
    FadingChannelConfig,
    generate_fading_coefficients,
    nakagami_pdf,
    rayleigh_pdf,
    rician_pdf,
)
from src.math_models.energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    compute_energy_statistic,
)
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


class TestAnalyticalFadingPDFs:
    """Test suite for analytical probability density functions."""

    def test_rayleigh_pdf_properties(self):
        """Rayleigh PDF integrates to 1.0 and achieves mode at a = sigma."""
        sigma = 1.0 / np.sqrt(2.0)
        a_vals = np.linspace(0.0, 5.0, 1000)
        pdf_vals = rayleigh_pdf(a_vals, sigma=sigma)

        area = integrate.trapezoid(pdf_vals, a_vals)
        assert np.isclose(area, 1.0, atol=0.01)

        # Mode occurs at a = sigma
        max_idx = np.argmax(pdf_vals)
        assert np.isclose(a_vals[max_idx], sigma, atol=0.02)

    def test_rician_pdf_reduces_to_rayleigh_at_k0(self):
        """At K=0, Rician PDF strictly converges to Rayleigh PDF with sigma = sqrt(omega/2)."""
        omega = 1.0
        sigma = np.sqrt(omega / 2.0)
        a_vals = np.linspace(0.01, 3.5, 200)

        r_pdf = rayleigh_pdf(a_vals, sigma=sigma)
        ric_pdf = rician_pdf(a_vals, k_factor=0.0, average_power=omega)

        assert np.allclose(r_pdf, ric_pdf, atol=1e-3)

    def test_nakagami_pdf_reduces_to_rayleigh_at_m1(self):
        """At m=1, Nakagami-m PDF strictly reduces to Rayleigh PDF."""
        omega = 1.0
        sigma = np.sqrt(omega / 2.0)
        a_vals = np.linspace(0.01, 3.5, 200)

        r_pdf = rayleigh_pdf(a_vals, sigma=sigma)
        nak_pdf = nakagami_pdf(a_vals, m_param=1.0, average_power=omega)

        assert np.allclose(r_pdf, nak_pdf, atol=1e-3)

    def test_pdf_integrals_to_unity(self):
        """Verify Rician and Nakagami PDFs integrate to 1.0."""
        a_vals = np.linspace(0.0, 6.0, 1000)

        ric_area = integrate.trapezoid(rician_pdf(a_vals, k_factor=4.0, average_power=1.0), a_vals)
        nak_area = integrate.trapezoid(nakagami_pdf(a_vals, m_param=2.0, average_power=1.0), a_vals)

        assert np.isclose(ric_area, 1.0, atol=0.01)
        assert np.isclose(nak_area, 1.0, atol=0.01)


class TestFadingCoefficients:
    """Test suite for random channel coefficient generation."""

    @pytest.mark.parametrize("fading", ["rayleigh", "rician", "nakagami", "awgn"])
    def test_expected_power_conservation(self, fading):
        """E[|h|^2] equals configured average power Omega across all fading distributions."""
        rng = np.random.default_rng(42)
        n_samples = 150_000
        target_omega = 2.0

        cfg = FadingChannelConfig(
            fading_type=fading,
            average_power=target_omega,
            rician_k=3.0,
            nakagami_m=1.5,
        )
        h = generate_fading_coefficients(n_samples, cfg, rng=rng)

        emp_power = np.mean(np.abs(h) ** 2)
        assert np.isclose(emp_power, target_omega, rtol=0.03)

    def test_rayleigh_phase_uniformity(self):
        """Rayleigh complex phase is uniformly distributed in [-pi, pi]."""
        rng = np.random.default_rng(999)
        cfg = FadingChannelConfig(fading_type="rayleigh")
        h = generate_fading_coefficients(50_000, cfg, rng=rng)
        phases = np.angle(h)

        mean_phase = np.mean(phases)
        var_phase = np.var(phases)

        # Uniform [-pi, pi] has mean 0, variance (2pi)^2 / 12 = pi^2 / 3 ~ 3.2898
        assert np.isclose(mean_phase, 0.0, atol=0.05)
        assert np.isclose(var_phase, (np.pi**2) / 3.0, rtol=0.03)


class TestFadingChannelApplication:
    """Test suite for FadingChannel class."""

    def test_block_vs_fast_fading(self):
        """Block fading keeps h constant; fast fading varies h across samples."""
        sig = np.ones(512, dtype=complex)

        chan_block = FadingChannel(FadingChannelConfig(fading_type="rayleigh", block_fading=True), seed=1)
        _, h_block = chan_block.apply_channel(sig)
        assert np.all(h_block == h_block[0])

        chan_fast = FadingChannel(FadingChannelConfig(fading_type="rayleigh", block_fading=False), seed=2)
        _, h_fast = chan_fast.apply_channel(sig)
        assert not np.all(h_fast == h_fast[0])


class TestDetectionUnderFading:
    """Verify performance characteristics under fading stated in the deployment plan."""

    def test_fading_degradation_in_transition_regime(self):
        """Under Rayleigh fading, deep fades degrade detection: at -8 dB, Pd drops from >99% to ~74%."""
        l_samples = 1024
        n_frames = 2000
        snr_db = -8.0

        detector = EnergyDetector(EnergyDetectorConfig(num_samples=l_samples, pfa_target=0.05))

        # Under Rayleigh block fading
        chan = FadingChannel(FadingChannelConfig(fading_type="rayleigh", block_fading=True), seed=42)
        sim = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples, snr_db=snr_db), seed=42)

        # Generate H1 frames with fading applied
        frames = []
        for _ in range(n_frames):
            h_val = generate_fading_coefficients(1, chan.config, rng=sim.rng)[0]
            r = sim.generate_batch(num_frames=1, p_h1=1.0, channel_gain=h_val, as_iq_channels=True)[0][0]
            frames.append(r)

        batch_faded = np.array(frames)
        labels = np.ones(n_frames, dtype=int)
        res = detector.evaluate_performance(batch_faded, labels)

        # Without fading at -8 dB, Pd > 0.99.
        # Under Rayleigh fading, deep fades drop Pd to ~0.74 (< 0.80)
        assert res["pd"] < 0.80

    def test_fading_recovery_at_eight_db(self):
        """At SNR >= 8 dB, signal energy overcomes fading margin, restoring Pd > 95%."""
        l_samples = 1024
        n_frames = 2000
        snr_db = 8.0

        detector = EnergyDetector(EnergyDetectorConfig(num_samples=l_samples, pfa_target=0.05))
        chan = FadingChannel(FadingChannelConfig(fading_type="rayleigh", block_fading=True), seed=99)
        sim = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples, snr_db=snr_db), seed=99)

        frames = []
        for _ in range(n_frames):
            h_val = generate_fading_coefficients(1, chan.config, rng=sim.rng)[0]
            r = sim.generate_batch(num_frames=1, p_h1=1.0, channel_gain=h_val, as_iq_channels=True)[0][0]
            frames.append(r)

        batch_faded = np.array(frames)
        labels = np.ones(n_frames, dtype=int)
        res = detector.evaluate_performance(batch_faded, labels)

        # Plan specification: At SNR >= 8 dB, detection reliably exceeds 95%
        assert res["pd"] >= 0.95
