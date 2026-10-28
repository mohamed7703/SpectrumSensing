"""Physical Layer Data Augmentation for RF Signals.

Provides realistic RF physical-layer augmentations to improve generalization
and robustness against over-the-air channel imperfections:
1. Carrier Frequency Offset (CFO):
   .. math::
       x_{\\text{cfo}}(n) = x(n) \\cdot e^{j 2\\pi \\Delta f \\cdot n / f_s}
2. Random Phase Rotation (Phase Jitter):
   .. math::
       x_{\\text{phase}}(n) = x(n) \\cdot e^{j \\theta}, \\quad \\theta \\sim \\mathcal{U}[0, 2\\pi)
3. Time Delay / Cyclic Shift:
   .. math::
       x_{\\text{shift}}(n) = x((n - \\Delta n) \\bmod L)
4. Random Amplitude Gain Fluctuation (AGC drift):
   .. math::
       x_{\\text{gain}}(n) = g \\cdot x(n), \\quad g \\sim \\mathcal{U}[g_{\\min}, g_{\\max}]
"""

from __future__ import annotations

import numpy as np


def apply_cfo(
    x: np.ndarray,
    freq_offset_hz: float,
    sample_rate: float = 1.0,
) -> np.ndarray:
    r"""Apply Carrier Frequency Offset (CFO) to baseband signal.

    Args:
        x: Complex 1D array of length :math:`L`, or 2D real array :math:`(L, 2)`.
        freq_offset_hz: Frequency offset :math:`\Delta f` in Hz.
        sample_rate: Physical sampling frequency :math:`f_s` in Hz.

    Returns:
        Frequency-shifted array with matching shape.
    """
    is_iq = x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)
    sig = x[:, 0] + 1j * x[:, 1] if is_iq else np.asarray(x, dtype=complex)

    n = np.arange(len(sig))
    cfo_phasor = np.exp(1j * 2.0 * np.pi * freq_offset_hz * n / sample_rate)
    shifted = sig * cfo_phasor

    if is_iq:
        return np.stack([np.real(shifted), np.imag(shifted)], axis=-1)
    return shifted


def apply_random_phase_rotation(
    x: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Apply uniform random phase rotation :math:`e^{j \theta}` where :math:`\theta \sim \mathcal{U}[0, 2\pi)`.

    Args:
        x: Input signal.
        rng: Optional NumPy random generator.

    Returns:
        Rotated signal array.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_iq = x.ndim == 2 and x.shape[-1] == 2 and not np.iscomplexobj(x)
    sig = x[:, 0] + 1j * x[:, 1] if is_iq else np.asarray(x, dtype=complex)

    theta = rng.uniform(0.0, 2.0 * np.pi)
    rotated = sig * np.exp(1j * theta)

    if is_iq:
        return np.stack([np.real(rotated), np.imag(rotated)], axis=-1)
    return rotated


def apply_time_shift(
    x: np.ndarray,
    shift_samples: int,
) -> np.ndarray:
    """Apply cyclic time shift to simulate timing offsets.

    Args:
        x: Input signal.
        shift_samples: Number of samples to roll.

    Returns:
        Time-shifted array.
    """
    return np.roll(x, shift=shift_samples, axis=0)


def augment_frame(
    x: np.ndarray,
    max_cfo_fraction: float = 0.05,
    sample_rate: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply stochastic chain of augmentations (CFO, phase, cyclic shift, gain).

    Args:
        x: Input frame.
        max_cfo_fraction: Maximum CFO as a fraction of sample rate.
        sample_rate: Sampling rate.
        rng: Optional random generator.

    Returns:
        Augmented frame.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Random CFO
    cfo_hz = rng.uniform(-max_cfo_fraction * sample_rate, max_cfo_fraction * sample_rate)
    out = apply_cfo(x, cfo_hz, sample_rate)

    # 2. Random phase rotation
    out = apply_random_phase_rotation(out, rng)

    # 3. Random cyclic shift
    max_shift = len(x) // 8
    shift = int(rng.integers(-max_shift, max_shift + 1))
    out = apply_time_shift(out, shift)

    # 4. Random gain fluctuation [0.8, 1.2]
    gain = rng.uniform(0.8, 1.2)
    return out * gain
