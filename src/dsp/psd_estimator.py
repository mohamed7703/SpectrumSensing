"""Power Spectral Density (PSD) Estimation via Welch's Method.

Computes smoothed Power Spectral Density for complex RF baseband signals:
.. math::
    \\hat{P}_{xx}(f) = \\frac{1}{K} \\sum_{i=0}^{K-1} \\frac{1}{f_s S} \\left| \\sum_{n=0}^{M-1} x_i(n) w(n) e^{-j 2\\pi f n / f_s} \\right|^2
Where:
    - :math:`M` is the segment length (nperseg).
    - :math:`S = \\sum_{n=0}^{M-1} |w(n)|^2` is the window energy normalization factor.
    - :math:`K` is the total number of overlapping sub-segments.
"""

from __future__ import annotations

from typing import Literal, Tuple
import numpy as np
from scipy import signal

from .fft_utils import WindowType


def estimate_psd_welch(
    x: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: WindowType = "hann",
    scaling: Literal["density", "spectrum"] = "density",
    return_db: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Estimate Power Spectral Density using Welch's averaged modified periodogram.

    Args:
        x: Complex baseband signal of length :math:`L`, or array of shape :math:`(L, 2)`.
        sample_rate: Physical sampling frequency :math:`f_s` in Hz.
        nperseg: Length of each segment for FFT averaging.
        noverlap: Number of points of overlap between segments. Defaults to 50% (`nperseg // 2`).
        window: Windowing function applied to segments.
        scaling: 'density' computes V^2/Hz, 'spectrum' computes V^2.
        return_db: If True, returns PSD in decibels :math:`10 \log_{10}(\text{PSD})`.

    Returns:
        Tuple of:
            - `frequencies`: Centered frequencies spanning :math:`[-f_s/2, +f_s/2)`.
            - `psd`: Power spectral density array (linear or dB).
    """
    if x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x):
        sig = x[:, 0] + 1j * x[:, 1]
    else:
        sig = np.asarray(x, dtype=complex)

    l_samples = len(sig)
    if nperseg > l_samples:
        nperseg = l_samples

    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = signal.welch(
        sig,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=False,
        scaling=scaling,
    )

    # Center frequencies and spectrum around zero baseband
    freqs_shifted = np.fft.fftshift(freqs)
    psd_shifted = np.fft.fftshift(psd)

    if return_db:
        # Avoid log(0) with machine epsilon
        psd_safe = np.maximum(psd_shifted, 1e-18)
        return freqs_shifted, 10.0 * np.log10(psd_safe)

    return freqs_shifted, psd_shifted


def estimate_batch_psd(
    batch_x: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 256,
    return_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate PSD across a batch of observation frames.

    Args:
        batch_x: Tensor of shape (N, L) complex or (N, L, 2) real.
        sample_rate: Physical sampling rate.
        nperseg: Segment length.
        return_db: Whether to output in dB.

    Returns:
        Tuple of (frequencies, psd_tensor) where `psd_tensor` has shape `(N, nperseg)`.
    """
    n_frames = batch_x.shape[0]
    psd_list = []
    freqs = None

    for i in range(n_frames):
        f, p = estimate_psd_welch(
            batch_x[i],
            sample_rate=sample_rate,
            nperseg=nperseg,
            return_db=return_db,
        )
        if freqs is None:
            freqs = f
        psd_list.append(p)

    return freqs, np.array(psd_list)
