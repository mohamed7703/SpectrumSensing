"""Digital Filter Design and Filtering Utilities for Cognitive Radios.

Provides FIR/IIR digital filter design (Butterworth, Chebyshev, Matched filter)
with zero-phase forward-backward filtering:
.. math::
    y(n) = \\sum_{k=0}^{M} b_k x(n-k) - \\sum_{k=1}^{N} a_k y(n-k)
.. math::
    h_{\\text{matched}}(n) = s^*(L - 1 - n) \\quad \\text{(Matched Filter Template)}
"""

from __future__ import annotations

from typing import Literal, Tuple
import numpy as np
from scipy import signal


FilterType = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def design_butterworth_filter(
    cutoff_freq: float | Tuple[float, float],
    sample_rate: float,
    filter_type: FilterType = "lowpass",
    order: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a digital Butterworth IIR filter.

    Args:
        cutoff_freq: Cutoff frequency in Hz (single float for low/highpass, tuple for bandpass/stop).
        sample_rate: Physical sampling frequency :math:`f_s` in Hz.
        filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
        order: Filter order.

    Returns:
        Tuple of filter transfer function coefficients `(b, a)`.
    """
    nyquist = 0.5 * sample_rate

    if isinstance(cutoff_freq, (tuple, list)):
        norm_cutoff = [f / nyquist for f in cutoff_freq]
    else:
        norm_cutoff = cutoff_freq / nyquist

    b, a = signal.butter(N=order, Wn=norm_cutoff, btype=filter_type, analog=False, output="ba")
    return b, a


def apply_filter(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply IIR/FIR filter to input baseband signal.

    Args:
        x: Input signal array (1D complex or 2D real).
        b: Numerator coefficients.
        a: Denominator coefficients.
        zero_phase: If True, uses forward-backward filtering (`filtfilt`) to prevent phase distortion.

    Returns:
        Filtered signal array with matching shape and dtype.
    """
    is_iq = x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)
    sig = x[:, 0] + 1j * x[:, 1] if is_iq else np.asarray(x, dtype=complex)

    if zero_phase:
        # Filter real and imaginary parts independently to avoid complex filtfilt issues
        filtered_real = signal.filtfilt(b, a, np.real(sig))
        filtered_imag = signal.filtfilt(b, a, np.imag(sig))
        out_sig = filtered_real + 1j * filtered_imag
    else:
        out_sig = signal.lfilter(b, a, sig)

    if is_iq:
        return np.stack([np.real(out_sig), np.imag(out_sig)], axis=-1)
    return out_sig


def matched_filter(
    received_signal: np.ndarray,
    template_pulse: np.ndarray,
) -> np.ndarray:
    r"""Apply optimal matched filter to maximize Signal-to-Noise Ratio (SNR) in AWGN.

    .. math::
        h_{\text{matched}}(n) = s^*(L - 1 - n)
        y(n) = (r * h_{\text{matched}})(n)

    Args:
        received_signal: Received complex baseband signal :math:`r(n)`.
        template_pulse: Known transmitted primary user pulse waveform :math:`s(n)`.

    Returns:
        Matched filter output sequence :math:`y(n)`.
    """
    rec = np.asarray(received_signal, dtype=complex)
    tmpl = np.asarray(template_pulse, dtype=complex)

    # Time-reversed complex conjugate
    matched_ir = np.conj(tmpl[::-1])
    # Energy normalize template
    matched_ir = matched_ir / np.sqrt(np.sum(np.abs(tmpl) ** 2))

    return signal.convolve(rec, matched_ir, mode="same")
