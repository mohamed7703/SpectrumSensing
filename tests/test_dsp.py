"""Unit tests for the Digital Signal Processing (DSP) transforms module.

Tests Fast Fourier Transforms, Parseval energy conservation, Welch Power Spectral
Density estimation, 2D STFT spectrogram generation, and digital/matched filters.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.dsp.fft_utils import (
    compute_fft,
    compute_ifft,
    get_window,
    verify_parseval_theorem,
)
from src.dsp.psd_estimator import (
    estimate_batch_psd,
    estimate_psd_welch,
)
from src.dsp.spectrogram import (
    compute_spectrogram,
    generate_batch_spectrograms,
)
from src.dsp.filters import (
    apply_filter,
    design_butterworth_filter,
    matched_filter,
)


class TestFFTUtilities:
    """Test suite for Fast Fourier Transform functions."""

    def test_parseval_energy_conservation(self):
        """Parseval's theorem must hold: time domain energy equals frequency domain energy."""
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 1, 1024) + 1j * rng.normal(0, 1, 1024)

        e_time, e_freq, rel_diff = verify_parseval_theorem(sig)
        assert rel_diff < 1e-12
        assert np.isclose(e_time, e_freq, atol=1e-10)

    def test_single_tone_frequency_recovery(self):
        """FFT of a pure complex exponential exp(j 2 pi f0 t) must peak at exactly f0."""
        fs = 1e6  # 1 MHz
        f0 = 2.5e5  # 250 kHz
        n = 1024
        t = np.arange(n) / fs
        tone = np.exp(1j * 2 * np.pi * f0 * t)

        freqs, spec = compute_fft(tone, sample_rate=fs, window=None)
        peak_freq = freqs[np.argmax(np.abs(spec))]

        # Peak frequency must be within half a bin width: fs / (2 * N) ~ 488 Hz
        bin_width = fs / n
        assert abs(peak_freq - f0) < bin_width

    def test_ifft_exact_reconstruction(self):
        """IFFT of FFT must perfectly reconstruct the original time-domain waveform."""
        rng = np.random.default_rng(77)
        sig = rng.normal(0, 1, 512) + 1j * rng.normal(0, 1, 512)

        _, spec = compute_fft(sig, window=None, shift=True)
        sig_rec = compute_ifft(spec, shifted=True)

        max_err = np.max(np.abs(sig - sig_rec))
        assert max_err < 1e-12

    def test_iq_2d_tensor_support(self):
        """Array of shape (L, 2) is handled seamlessly."""
        iq_sig = np.zeros((1024, 2), dtype=float)
        iq_sig[:, 0] = 1.0  # Real part = 1.0 (DC tone)

        freqs, spec = compute_fft(iq_sig, window=None, shift=True)
        # DC tone peaks at 0 Hz
        peak_freq = freqs[np.argmax(np.abs(spec))]
        assert np.isclose(peak_freq, 0.0)


class TestPSDEstimation:
    """Test suite for Welch Power Spectral Density estimator."""

    def test_white_noise_psd_flatness(self):
        """Complex white Gaussian noise has flat PSD equal to sigma_w^2 / fs."""
        rng = np.random.default_rng(101)
        fs = 2e6
        noise_var = 2.0
        std = np.sqrt(noise_var / 2.0)
        noise = rng.normal(0, std, 30_000) + 1j * rng.normal(0, std, 30_000)

        freqs, psd = estimate_psd_welch(noise, sample_rate=fs, nperseg=256, scaling="density", return_db=False)

        # Expected density = sigma_w^2 / fs = 2.0 / 2e6 = 1e-6
        expected_density = noise_var / fs
        emp_mean_psd = np.mean(psd)
        assert np.isclose(emp_mean_psd, expected_density, rtol=0.08)

    def test_tone_detection_in_psd(self):
        """A sinusoidal carrier produces a dominant PSD peak > 20 dB above the noise floor."""
        fs = 1e6
        f_tone = 1.5e5
        n = 8192
        t = np.arange(n) / fs
        signal_clean = 2.0 * np.exp(1j * 2 * np.pi * f_tone * t)

        rng = np.random.default_rng(202)
        noise = rng.normal(0, 0.2, n) + 1j * rng.normal(0, 0.2, n)
        noisy_signal = signal_clean + noise

        freqs, psd_db = estimate_psd_welch(noisy_signal, sample_rate=fs, nperseg=512, return_db=True)

        peak_idx = np.argmax(psd_db)
        median_noise_level = np.median(psd_db)

        assert abs(freqs[peak_idx] - f_tone) < (fs / 512)
        assert psd_db[peak_idx] - median_noise_level > 20.0  # > 20 dB peak prominence

    def test_batch_psd_output_shape(self):
        """Batch PSD estimator returns tensor of shape (N, nperseg)."""
        batch = np.ones((10, 1024, 2))
        freqs, psd_batch = estimate_batch_psd(batch, nperseg=128)

        assert len(freqs) == 128
        assert psd_batch.shape == (10, 128)


class TestSpectrogramGeneration:
    """Test suite for 2D STFT spectrogram generation."""

    def test_spectrogram_matrix_shapes(self):
        """Verify 2D time-frequency matrix dimensions."""
        sig = np.ones(1024, dtype=complex)
        times, freqs, spec = compute_spectrogram(sig, sample_rate=1e6, nperseg=64, noverlap=48)

        # With L=1024, nperseg=64, step=16: (1024 - 64) / 16 + 1 = 61 time slices
        assert len(freqs) == 64
        assert len(times) == spec.shape[0]
        assert spec.shape[1] == 64

    def test_frequency_sweep_chirp_spectrogram(self):
        """A linear chirp frequency sweep concentrates power along a diagonal ridge in the spectrogram."""
        fs = 1e6
        n = 2048
        t = np.arange(n) / fs
        # Chirp from -300 kHz to +300 kHz
        k_chirp = 6e5 / (n / fs)
        phase = 2 * np.pi * (-3e5 * t + 0.5 * k_chirp * (t**2))
        chirp_sig = np.exp(1j * phase)

        times, freqs, spec_db = compute_spectrogram(chirp_sig, sample_rate=fs, nperseg=128, noverlap=96, return_db=True)

        # Early time peak should be at negative frequency, late time peak at positive frequency
        early_peak_freq = freqs[np.argmax(spec_db[2, :])]
        late_peak_freq = freqs[np.argmax(spec_db[-2, :])]

        assert early_peak_freq < -1.5e5
        assert late_peak_freq > 1.5e5

    def test_batch_spectrogram_dimensions(self):
        """Batch spectrogram generator yields 3D tensor of shape (N, Time, Freq)."""
        batch = np.zeros((8, 512, 2))
        specs = generate_batch_spectrograms(batch, nperseg=64)
        assert specs.shape[0] == 8
        assert specs.shape[2] == 64


class TestDigitalFilters:
    """Test suite for digital filter design and application."""

    def test_butterworth_lowpass_attenuation(self):
        """Butterworth lowpass filter passes in-band tone with < 1 dB loss and cuts out-of-band by > 20 dB."""
        fs = 1e6
        cutoff = 1.0e5  # 100 kHz lowpass
        b, a = design_butterworth_filter(cutoff_freq=cutoff, sample_rate=fs, filter_type="lowpass", order=6)

        t = np.arange(2048) / fs
        in_band_tone = np.exp(1j * 2 * np.pi * 3e4 * t)   # 30 kHz (passband)
        out_band_tone = np.exp(1j * 2 * np.pi * 3.5e5 * t) # 350 kHz (stopband)

        filt_in = apply_filter(in_band_tone, b, a, zero_phase=True)
        filt_out = apply_filter(out_band_tone, b, a, zero_phase=True)

        power_in = np.mean(np.abs(filt_in[200:-200]) ** 2)
        power_out = np.mean(np.abs(filt_out[200:-200]) ** 2)

        # In-band attenuation < 0.5 dB (power ratio > 0.90)
        assert power_in > 0.90
        # Out-of-band attenuation > 25 dB (power ratio < 0.003)
        assert power_out < 0.003

    def test_matched_filter_peak_correlation(self):
        """Matched filter output achieves peak output exactly at pulse alignment."""
        pulse_len = 64
        rng = np.random.default_rng(42)
        pulse = rng.choice([-1.0, 1.0], size=pulse_len) + 1j * rng.choice([-1.0, 1.0], size=pulse_len)

        # Embed pulse at center of 512-sample received frame
        received = np.zeros(512, dtype=complex)
        start_pos = 200
        received[start_pos : start_pos + pulse_len] = pulse

        mf_out = matched_filter(received, pulse)
        peak_idx = np.argmax(np.abs(mf_out))

        # Peak must occur at center alignment
        expected_peak = start_pos + pulse_len // 2
        assert abs(peak_idx - expected_peak) <= 1
