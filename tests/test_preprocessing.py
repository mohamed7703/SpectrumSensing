"""Unit tests for the physical-layer I/Q preprocessing and augmentation pipeline.

Tests DC offset removal, Gram-Schmidt I/Q imbalance correction, power normalization,
frame chunking, calibrated noise injection, and CFO/phase augmentations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.preprocessing.iq_normalizer import (
    correct_iq_imbalance,
    normalize_power,
    remove_dc_offset,
)
from src.preprocessing.frame_builder import (
    chunk_stream_to_frames,
    complex_to_iq,
    iq_to_complex,
)
from src.preprocessing.noise_injector import (
    inject_batch_snr_sweep,
    inject_noise_at_snr,
)
from src.preprocessing.augmentation import (
    apply_cfo,
    apply_random_phase_rotation,
    apply_time_shift,
    augment_frame,
)
from src.dsp.fft_utils import compute_fft


class TestIQNormalizer:
    """Test suite for I/Q normalization and impairment correction."""

    def test_dc_offset_elimination(self):
        """DC bias added to both rails is strictly reduced to zero."""
        rng = np.random.default_rng(42)
        ac_sig = rng.normal(0, 1, 1024) + 1j * rng.normal(0, 1, 1024)
        dc_bias = 0.75 - 0.40j
        distorted = ac_sig + dc_bias

        cleaned = remove_dc_offset(distorted)
        assert np.abs(np.mean(cleaned)) < 1e-12

    def test_dc_offset_iq_2d_format(self):
        """Array of shape (L, 2) has mean removed per rail."""
        distorted_2d = np.ones((512, 2)) * 3.5
        cleaned = remove_dc_offset(distorted_2d)
        assert np.allclose(np.mean(cleaned, axis=0), [0.0, 0.0], atol=1e-12)

    def test_iq_imbalance_correction(self):
        """Gram-Schmidt orthogonalization restores orthogonality between skewed rails."""
        rng = np.random.default_rng(123)
        n = 100_000
        # Generate independent standard normals
        i_true = rng.normal(0, 1, n)
        q_true = rng.normal(0, 1, n)

        # Inject 20% amplitude mismatch and 15 degree phase skew
        gain_err = 1.25
        phase_skew_rad = np.radians(15.0)

        i_distorted = i_true
        q_distorted = gain_err * (np.sin(phase_skew_rad) * i_true + np.cos(phase_skew_rad) * q_true)
        sig_distorted = i_distorted + 1j * q_distorted

        # Prior to correction, I and Q are strongly correlated
        cov_prior = np.mean(i_distorted * q_distorted)
        assert abs(cov_prior) > 0.2

        # Correct
        sig_clean = correct_iq_imbalance(sig_distorted)
        i_clean = np.real(sig_clean)
        q_clean = np.imag(sig_clean)

        # Post-correction: correlation is nullified and variances match
        cov_post = float(np.mean(i_clean * q_clean))
        var_ratio = float(np.var(i_clean) / np.var(q_clean))

        assert abs(cov_post) < 0.015
        assert np.isclose(var_ratio, 1.0, atol=0.03)

    def test_power_normalization_exact_target(self):
        """Signal with arbitrary energy is scaled to target power exactly."""
        rng = np.random.default_rng(999)
        sig = 15.0 * (rng.normal(0, 1, 1024) + 1j * rng.normal(0, 1, 1024))
        target_p = 2.5

        norm_sig = normalize_power(sig, target_power=target_p)
        measured_p = float(np.mean(np.abs(norm_sig) ** 2))
        assert np.isclose(measured_p, target_p, atol=1e-10)


class TestFrameBuilder:
    """Test suite for continuous stream framing."""

    def test_drop_remainder_chunking(self):
        """Chunking 3000 samples into 1024-length frames yields 2 frames when remainder dropped."""
        stream = np.arange(3000, dtype=complex)
        frames = chunk_stream_to_frames(stream, frame_length=1024, remainder_mode="drop", as_iq_channels=True)

        assert frames.shape == (2, 1024, 2)
        # Check first sample of second frame
        assert frames[1, 0, 0] == 1024.0

    def test_pad_remainder_chunking(self):
        """Chunking 3000 samples with padding yields 3 frames."""
        stream = np.ones(3000, dtype=complex)
        frames = chunk_stream_to_frames(stream, frame_length=1024, remainder_mode="pad", as_iq_channels=True)

        assert frames.shape == (3, 1024, 2)
        # Padded elements at end of 3rd frame are zero
        assert np.all(frames[2, (3000 - 2048) :, :] == 0.0)

    def test_complex_iq_conversions(self):
        """Roundtrip conversion between (L,) complex and (L, 2) real is lossless."""
        orig = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j])
        iq = complex_to_iq(orig)
        assert iq.shape == (3, 2)

        rec = iq_to_complex(iq)
        assert np.allclose(orig, rec)


class TestCalibratedNoiseInjection:
    """Test suite for calibrated noise addition."""

    def test_measured_snr_matches_target_db(self):
        """Signal + injected noise achieves requested SNR within 0.15 dB."""
        rng = np.random.default_rng(42)
        n = 100_000
        # Pure signal with known unit power
        t = np.arange(n)
        clean_signal = np.exp(1j * 0.1 * t)
        target_snr_db = 6.0

        noisy, noise_var = inject_noise_at_snr(clean_signal, snr_db=target_snr_db, rng=rng)

        # Measure noise component directly
        injected_noise = noisy - clean_signal
        measured_noise_var = np.var(injected_noise)
        measured_snr_linear = np.mean(np.abs(clean_signal) ** 2) / measured_noise_var
        measured_snr_db = 10.0 * np.log10(measured_snr_linear)

        assert np.isclose(measured_snr_db, target_snr_db, atol=0.15)

    def test_batch_snr_sweep_generation(self):
        """Batch sweep across 4 SNR levels quadruples frame count."""
        clean = np.ones((5, 1024, 2))
        snrs = [-10.0, -5.0, 0.0, 5.0]

        all_noisy, snr_labels = inject_batch_snr_sweep(clean, snrs)
        assert all_noisy.shape == (20, 1024, 2)
        assert len(snr_labels) == 20


class TestDataAugmentations:
    """Test suite for physical-layer augmentations."""

    def test_cfo_spectral_shift(self):
        """Applying CFO shifts the signal's spectral peak by exactly delta_f."""
        fs = 1e6
        cfo_target = 1.25e5  # +125 kHz shift
        n = 1024
        tone = np.exp(1j * 2 * np.pi * 5e4 * (np.arange(n) / fs))  # 50 kHz tone

        shifted = apply_cfo(tone, freq_offset_hz=cfo_target, sample_rate=fs)

        freqs, spec = compute_fft(shifted, sample_rate=fs, window=None)
        peak_freq = freqs[np.argmax(np.abs(spec))]

        # Expected new peak at 50 kHz + 125 kHz = 175 kHz
        assert abs(peak_freq - 1.75e5) < (fs / n)

    def test_phase_rotation_energy_conservation(self):
        """Phase rotation changes angle but strictly preserves amplitude and energy."""
        sig = np.array([1.0 + 2.0j, 3.0 - 1.0j])
        orig_energy = np.sum(np.abs(sig) ** 2)

        rotated = apply_random_phase_rotation(sig)
        new_energy = np.sum(np.abs(rotated) ** 2)

        assert np.isclose(orig_energy, new_energy, atol=1e-12)

    def test_time_shift_cyclic(self):
        """Cyclic roll shifts values by exact index count."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        rolled = apply_time_shift(x, 1)
        assert np.all(rolled == [4.0, 1.0, 2.0, 3.0])
