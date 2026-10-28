"""Calibrated Noise Injection for Synthetic Signal Generation.

Injects complex Additive White Gaussian Noise (AWGN) to achieve exact target
Signal-to-Noise Ratios (SNR) in decibels:
.. math::
    \\text{SNR}_{\\text{dB}} = 10 \\log_{10}\\left(\\frac{P_s}{\\sigma_w^2}\\right)
    \\implies \\sigma_w^2 = \\frac{P_s}{10^{\\text{SNR}_{\\text{dB}} / 10}}
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def inject_noise_at_snr(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    r"""Add calibrated AWGN to a clean signal to achieve the specified SNR in dB.

    Args:
        signal: Clean input signal (1D complex or 2D real I/Q).
        snr_db: Desired Signal-to-Noise Ratio in dB.
        rng: Optional NumPy random generator instance.

    Returns:
        Tuple of:
            - `noisy_signal`: Signal with additive white noise applied.
            - `noise_variance`: Variance of the injected noise :math:`\sigma_w^2`.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_iq = signal.ndim == 2 and signal.shape[-1] == 2 and not np.iscomplexobj(signal)
    if is_iq:
        sig = signal[:, 0] + 1j * signal[:, 1]
    else:
        sig = np.asarray(signal, dtype=complex)

    sig_power = float(np.mean(np.abs(sig) ** 2))
    if sig_power <= 0:
        raise ValueError("Signal average power must be > 0 to inject calibrated noise")

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = sig_power / snr_linear

    # Circularly symmetric complex Gaussian noise
    std_comp = np.sqrt(noise_var / 2.0)
    noise = rng.normal(0.0, std_comp, size=len(sig)) + 1j * rng.normal(0.0, std_comp, size=len(sig))

    noisy_complex = sig + noise

    if is_iq:
        return np.stack([np.real(noisy_complex), np.imag(noisy_complex)], axis=-1), noise_var
    return noisy_complex, noise_var


def inject_batch_snr_sweep(
    clean_frames: np.ndarray,
    snr_levels_db: list[float] | np.ndarray,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate noisy frames across an array of SNR levels (e.g. RadioML -20 to +30 dB).

    Args:
        clean_frames: Clean input batch of shape (N, L, 2) or (N, L).
        snr_levels_db: List of target SNR values in dB.
        rng: Optional NumPy random generator instance.

    Returns:
        Tuple of:
            - `all_noisy_frames`: Array of shape (N * num_snrs, L, 2) or (N * num_snrs, L).
            - `snr_labels`: 1D array of assigned SNR levels of length N * num_snrs.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy_list = []
    labels_list = []

    for snr in snr_levels_db:
        for frame in clean_frames:
            noisy, _ = inject_noise_at_snr(frame, snr_db=float(snr), rng=rng)
            noisy_list.append(noisy)
            labels_list.append(snr)

    return np.array(noisy_list), np.array(labels_list)
