"""Fast Fourier Transform (FFT) Utilities for Complex Baseband Signals.

Provides windowed FFT, centered spectra, and Parseval's energy conservation:
.. math::
    X(k) = \\sum_{n=0}^{N-1} x(n) \\cdot w(n) \\cdot e^{-j 2\\pi k n / N}
.. math::
    \\sum_{n=0}^{N-1} |x(n)|^2 = \\frac{1}{N} \\sum_{k=0}^{N-1} |X(k)|^2 \\quad \\text{(Parseval's Theorem)}
"""

from __future__ import annotations

from typing import Literal, Tuple
import numpy as np
from scipy import signal


WindowType = Literal["hann", "hamming", "blackman", "boxcar", "flattop"]


def get_window(window_type: WindowType, num_samples: int) -> np.ndarray:
    """Retrieve an energy-normalized or standard window function.

    Args:
        window_type: Name of window ('hann', 'hamming', 'blackman', 'boxcar', 'flattop').
        num_samples: Length of window vector.

    Returns:
        1D real NumPy array of shape (num_samples,).
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    if window_type == "boxcar":
        return np.ones(num_samples, dtype=float)
    return signal.get_window(window_type, num_samples, fftbins=True)


def compute_fft(
    x: np.ndarray,
    n_fft: int | None = None,
    window: WindowType | None = "hann",
    shift: bool = True,
    sample_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute discrete Fast Fourier Transform with windowing and frequency centering.

    Args:
        x: Input signal array. Can be 1D complex of length :math:`L`, or 2D real of shape :math:`(L, 2)`.
        n_fft: Size of FFT. If None, uses `len(x)`. If larger, zero-padding is applied.
        window: Window name to apply prior to FFT to reduce spectral leakage, or None for rectangular.
        shift: If True, centers zero-frequency bin to :math:`[-f_s/2, +f_s/2)`.
        sample_rate: Physical sampling frequency :math:`f_s` in Hz.

    Returns:
        Tuple of:
            - `frequencies`: 1D array of frequency bins in Hz.
            - `spectrum`: Complex FFT values.
    """
    # Convert from (L, 2) I/Q format if needed
    if x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x):
        sig = x[:, 0] + 1j * x[:, 1]
    else:
        sig = np.asarray(x, dtype=complex)

    l_samples = len(sig)
    fft_size = n_fft or l_samples

    if fft_size < l_samples:
        raise ValueError(f"n_fft ({fft_size}) cannot be smaller than signal length ({l_samples})")

    if window is not None:
        win = get_window(window, l_samples)
        # Scale window to preserve signal amplitude
        win_norm = win / np.mean(win)
        windowed_sig = sig * win_norm
    else:
        windowed_sig = sig

    spectrum = np.fft.fft(windowed_sig, n=fft_size)
    freqs = np.fft.fftfreq(fft_size, d=1.0 / sample_rate)

    if shift:
        spectrum = np.fft.fftshift(spectrum)
        freqs = np.fft.fftshift(freqs)

    return freqs, spectrum


def compute_ifft(
    spectrum: np.ndarray,
    shifted: bool = True,
) -> np.ndarray:
    """Compute Inverse Fast Fourier Transform (IFFT).

    Args:
        spectrum: Complex spectrum array.
        shifted: If True, spectrum is centered and requires `ifftshift` before IFFT.

    Returns:
        1D reconstructed complex time-domain signal.
    """
    spec = np.fft.ifftshift(spectrum) if shifted else spectrum
    return np.fft.ifft(spec)


def verify_parseval_theorem(x: np.ndarray) -> Tuple[float, float, float]:
    r"""Verify Parseval's energy theorem between time and frequency domain.

    .. math::
        E_{\text{time}} = \sum_{n=0}^{N-1} |x(n)|^2, \quad
        E_{\text{freq}} = \frac{1}{N} \sum_{k=0}^{N-1} |X(k)|^2

    Args:
        x: Input signal array.

    Returns:
        Tuple of (energy_time, energy_freq, relative_difference).
    """
    sig = x[:, 0] + 1j * x[:, 1] if (x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)) else np.asarray(x, dtype=complex)
    n = len(sig)

    e_time = float(np.sum(np.abs(sig) ** 2))
    # Unwindowed, unshifted FFT
    _, x_f = compute_fft(sig, window=None, shift=False)
    e_freq = float(np.sum(np.abs(x_f) ** 2) / n)

    rel_diff = abs(e_time - e_freq) / e_time if e_time > 0 else 0.0
    return e_time, e_freq, rel_diff
