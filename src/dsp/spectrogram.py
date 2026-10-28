"""Short-Time Fourier Transform (STFT) Spectrogram Generation.

Transforms 1D complex time-series signals into 2D time-frequency power representations:
.. math::
    S(t, f) = \\left| \\sum_{n=-\\infty}^{\\infty} x(n) \\cdot w(n - t) \\cdot e^{-j 2\\pi f n} \\right|^2

Spectrogram arrays are used as direct 2D image inputs for deep learning models,
spectral occupancy watermarking, and jamming classification (Sensors 2024 / Plan).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import signal

from .fft_utils import WindowType


def compute_spectrogram(
    x: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    noverlap: int | None = None,
    window: WindowType = "hann",
    return_db: bool = True,
    dynamic_range_db: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute 2D Short-Time Fourier Transform power spectrogram.

    Args:
        x: Complex baseband signal of length :math:`L`, or array of shape :math:`(L, 2)`.
        sample_rate: Physical sampling frequency :math:`f_s` in Hz.
        nperseg: Time window length for each localized FFT slice.
        noverlap: Number of overlapping samples between adjacent time slices. Defaults to 75%.
        window: Windowing function name.
        return_db: If True, returns power spectral density in decibels :math:`10 \log_{10}(S)`.
        dynamic_range_db: Dynamic range clipping threshold in dB relative to peak power.

    Returns:
        Tuple of:
            - `times`: 1D array of time bin centers in seconds.
            - `frequencies`: 1D array of centered frequency bins in Hz :math:`[-f_s/2, +f_s/2)`.
            - `spectrogram`: 2D real matrix of shape :math:`(N_{\text{time}}, N_{\text{freq}})`.
    """
    if x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x):
        sig = x[:, 0] + 1j * x[:, 1]
    else:
        sig = np.asarray(x, dtype=complex)

    l_samples = len(sig)
    if nperseg > l_samples:
        nperseg = l_samples

    if noverlap is None:
        noverlap = int(nperseg * 0.75)

    freqs, times, sxx = signal.spectrogram(
        sig,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )

    # Shift frequencies so zero frequency is centered
    freqs_shifted = np.fft.fftshift(freqs)
    # sxx has shape (freq_bins, time_bins) -> shift along frequency axis (axis 0)
    sxx_shifted = np.fft.fftshift(sxx, axes=0)

    # Transpose to (time_bins, freq_bins) for conventional ML feature matrices
    spec_matrix = sxx_shifted.T

    if return_db:
        peak_val = np.max(spec_matrix)
        floor_val = peak_val * (10.0 ** (-dynamic_range_db / 10.0))
        spec_safe = np.maximum(spec_matrix, max(floor_val, 1e-18))
        spec_db = 10.0 * np.log10(spec_safe)
        return times, freqs_shifted, spec_db

    return times, freqs_shifted, spec_matrix


def generate_batch_spectrograms(
    batch_x: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    return_db: bool = True,
) -> np.ndarray:
    """Generate 2D spectrogram images for an entire batch of frames.

    Args:
        batch_x: Array of shape :math:`(N, L)` complex or :math:`(N, L, 2)` real.
        sample_rate: Sampling frequency.
        nperseg: STFT segment size.
        return_db: Whether to compute log-power.

    Returns:
        3D tensor of shape :math:`(N, N_{\text{time}}, N_{\text{freq}})`.
    """
    n_frames = batch_x.shape[0]
    specs = []

    for i in range(n_frames):
        _, _, s = compute_spectrogram(
            batch_x[i],
            sample_rate=sample_rate,
            nperseg=nperseg,
            return_db=return_db,
        )
        specs.append(s)

    return np.array(specs)
