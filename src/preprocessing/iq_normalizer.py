"""I/Q Signal Normalization and Frontend Impairment Correction.

Provides algorithms to rectify physical RF frontend direct-conversion impairments:
1. DC Offset Removal (Carrier Leakage):
   .. math::
       x_{\\text{clean}}(n) = x(n) - \\mathbb{E}[x(n)]
2. I/Q Imbalance Correction (Amplitude & Phase Mismatch):
   .. math::
       I'(n) = \\frac{I(n)}{\\sigma_I}, \\quad
       Q'(n) = \\frac{Q(n) - \\rho_{IQ} I'(n)}{\\sqrt{1 - \\rho_{IQ}^2}}
3. Power Normalization:
   .. math::
       x_{\\text{norm}}(n) = \\frac{x(n)}{\\sqrt{\\frac{1}{L} \\sum_{k=0}^{L-1} |x(k)|^2}}
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    r"""Remove DC offset (carrier leakage / local oscillator feedthrough).

    .. math::
        x_{\text{clean}}(n) = x(n) - \frac{1}{L} \sum_{k=0}^{L-1} x(k)

    Args:
        x: Complex 1D array of length :math:`L`, or 2D real array of shape :math:`(L, 2)`.

    Returns:
        Zero-mean array with identical shape and dtype.
    """
    is_iq = x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)
    if is_iq:
        dc_i = np.mean(x[:, 0])
        dc_q = np.mean(x[:, 1])
        out = x.copy()
        out[:, 0] -= dc_i
        out[:, 1] -= dc_q
        return out

    sig = np.asarray(x, dtype=complex)
    return sig - np.mean(sig)


def correct_iq_imbalance(x: np.ndarray) -> np.ndarray:
    r"""Estimate and correct I/Q amplitude and phase imbalance using Gram-Schmidt orthogonalization.

    In zero-IF direct-conversion SDR frontends, analog mixer phase and gain mismatch
    distorts the orthogonality of In-Phase and Quadrature rails. This function restores
    orthogonality:
    .. math::
        \\mathbb{E}[I'(n) Q'(n)] = 0, \\quad \\mathbb{E}[I'^2(n)] = \\mathbb{E}[Q'^2(n)]

    Args:
        x: Complex 1D array of length :math:`L`, or 2D real array of shape :math:`(L, 2)`.

    Returns:
        Imbalance-corrected complex array.
    """
    is_iq = x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)
    i_rail = x[:, 0] if is_iq else np.real(x)
    q_rail = x[:, 1] if is_iq else np.imag(x)

    # 1. Remove DC bias first
    i_zero = i_rail - np.mean(i_rail)
    q_zero = q_rail - np.mean(q_rail)

    # 2. Normalize I rail to unit variance
    std_i = float(np.std(i_zero))
    if std_i > 1e-12:
        i_norm = i_zero / std_i
    else:
        i_norm = i_zero

    # 3. Gram-Schmidt orthogonalization: subtract projection of Q onto I_norm
    proj_q_on_i = float(np.mean(q_zero * i_norm))
    q_orth = q_zero - proj_q_on_i * i_norm

    # 4. Normalize Q_orth to unit variance
    std_q_orth = float(np.std(q_orth))
    if std_q_orth > 1e-12:
        q_norm = q_orth / std_q_orth
    else:
        q_norm = q_orth

    # Equalize average energy
    corrected_complex = (i_norm + 1j * q_norm) / np.sqrt(2.0)

    if is_iq:
        return np.stack([np.real(corrected_complex), np.imag(corrected_complex)], axis=-1)
    return corrected_complex


def normalize_power(
    x: np.ndarray,
    target_power: float = 1.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    r"""Scale input signal to achieve exact target average power: :math:`\mathbb{E}[|x(n)|^2] = P_{\text{target}}`.

    Args:
        x: Signal array (1D complex or 2D real).
        target_power: Desired average frame power (default: 1.0).
        epsilon: Safety offset to prevent divide-by-zero on silent frames.

    Returns:
        Power-normalized array.
    """
    is_iq = x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)
    if is_iq:
        inst_power = x[:, 0] ** 2 + x[:, 1] ** 2
    else:
        inst_power = np.abs(x) ** 2

    current_power = float(np.mean(inst_power))
    if current_power < epsilon:
        return x.copy()

    scale_factor = np.sqrt(target_power / current_power)
    return x * scale_factor
