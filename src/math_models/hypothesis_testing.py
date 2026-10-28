"""Binary Hypothesis Testing Module for Cognitive Radio Spectrum Sensing.

This module models the foundational binary hypothesis formulation:
.. math::
    \\mathcal{H}_0: r(n) = w(n)
.. math::
    \\mathcal{H}_1: r(n) = h \\cdot s(n) + w(n)

Where:
    - :math:`r(n)` is the complex received baseband sample at discrete time index :math:`n`.
    - :math:`s(n)` is the transmitted primary user (PU) signal with average power :math:`\\sigma_s^2`.
    - :math:`w(n) \\sim \\mathcal{CN}(0, \\sigma_w^2)` is additive white Gaussian noise (AWGN)
      with zero mean and total variance :math:`\\sigma_w^2`.
    - :math:`h` is the complex channel gain (:math:`|h|=1` under AWGN, or fading random variable).
    - Signal-to-Noise Ratio (SNR) is defined as:
      .. math::
          \\text{SNR} = \\frac{|h|^2 \\sigma_s^2}{\\sigma_w^2}, \\quad
          \\text{SNR}_{\\text{dB}} = 10 \\log_{10}(\\text{SNR})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np


ModulationType = Literal["bpsk", "qpsk", "sinusoid", "gaussian"]


@dataclass
class HypothesisConfig:
    """Configuration parameters for binary hypothesis testing.

    Attributes:
        num_samples: Number of samples per sensing observation window (L).
        noise_variance: Total noise power variance :math:`\\sigma_w^2`.
        snr_db: Signal-to-Noise Ratio in decibels :math:`\\text{SNR}_{\\text{dB}}`.
        modulation: Modulation scheme for the Primary User signal.
        complex_signal: Whether samples are complex baseband :math:`I + jQ` (True) or real (False).
    """

    num_samples: int = 1024
    noise_variance: float = 1.0
    snr_db: float = 0.0
    modulation: ModulationType = "bpsk"
    complex_signal: bool = True


def generate_awgn(
    num_samples: int,
    noise_variance: float = 1.0,
    complex_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Generate Additive White Gaussian Noise (AWGN) samples.

    For complex noise :math:`w(n) = w_I(n) + j w_Q(n)`:
    .. math::
        w_I(n), w_Q(n) \\sim \\mathcal{N}\\left(0, \\frac{\\sigma_w^2}{2}\\right)
        \\mathbb{E}[|w(n)|^2] = \\sigma_w^2

    For real noise:
    .. math::
        w(n) \\sim \\mathcal{N}(0, \\sigma_w^2)

    Args:
        num_samples: Number of noise samples to draw.
        noise_variance: Total noise variance :math:`\\sigma_w^2`. Must be strictly positive.
        complex_noise: If True, generate circularly symmetric complex Gaussian noise.
        rng: Optional NumPy random generator instance for reproducibility.

    Returns:
        1D NumPy array of length `num_samples` containing noise samples.
    """
    if noise_variance <= 0:
        raise ValueError(f"noise_variance must be > 0, got {noise_variance}")

    if rng is None:
        rng = np.random.default_rng()

    if complex_noise:
        std_per_component = np.sqrt(noise_variance / 2.0)
        noise_real = rng.normal(loc=0.0, scale=std_per_component, size=num_samples)
        noise_imag = rng.normal(loc=0.0, scale=std_per_component, size=num_samples)
        return noise_real + 1j * noise_imag
    else:
        std = np.sqrt(noise_variance)
        return rng.normal(loc=0.0, scale=std, size=num_samples)


def generate_pu_signal(
    num_samples: int,
    signal_power: float = 1.0,
    modulation: ModulationType = "bpsk",
    complex_signal: bool = True,
    carrier_normalized_freq: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Generate primary user (PU) transmitted signal :math:`s(n)`.

    The signal is generated such that its expected power satisfies:
    .. math::
        \\mathbb{E}[|s(n)|^2] = P_s = \\sigma_s^2

    Args:
        num_samples: Number of discrete samples.
        signal_power: Expected average signal power :math:`P_s`.
        modulation: Modulation type ('bpsk', 'qpsk', 'sinusoid', 'gaussian').
        complex_signal: Whether output signal is complex baseband.
        carrier_normalized_freq: Normalized frequency :math:`f/f_s` for sinusoidal modulation.
        rng: Optional NumPy random generator instance.

    Returns:
        1D NumPy array of length `num_samples` containing the PU signal.
    """
    if signal_power <= 0:
        raise ValueError(f"signal_power must be > 0, got {signal_power}")

    if rng is None:
        rng = np.random.default_rng()

    amplitude = np.sqrt(signal_power)

    if modulation == "bpsk":
        bits = rng.integers(0, 2, size=num_samples)
        symbols = 2 * bits - 1  # in {-1, +1}
        if complex_signal:
            return amplitude * symbols.astype(complex)
        return amplitude * symbols.astype(float)

    elif modulation == "qpsk":
        bits_i = rng.integers(0, 2, size=num_samples)
        bits_q = rng.integers(0, 2, size=num_samples)
        symbols_i = 2 * bits_i - 1
        symbols_q = 2 * bits_q - 1
        scale = amplitude / np.sqrt(2.0)
        if complex_signal:
            return scale * (symbols_i + 1j * symbols_q)
        # Real projection
        return scale * symbols_i.astype(float)

    elif modulation == "sinusoid":
        n = np.arange(num_samples)
        phase_offset = rng.uniform(0, 2 * np.pi)
        phase = 2 * np.pi * carrier_normalized_freq * n + phase_offset
        if complex_signal:
            return amplitude * np.exp(1j * phase)
        return np.sqrt(2.0) * amplitude * np.cos(phase)

    elif modulation == "gaussian":
        if complex_signal:
            std = np.sqrt(signal_power / 2.0)
            return rng.normal(0, std, num_samples) + 1j * rng.normal(0, std, num_samples)
        return rng.normal(0, np.sqrt(signal_power), num_samples)

    else:
        raise ValueError(f"Unsupported modulation scheme: {modulation}")


def simulate_received_signal(
    hypothesis: Literal[0, 1],
    config: HypothesisConfig,
    channel_gain: complex | float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Simulate a single observation frame :math:`r(n)` under :math:`\mathcal{H}_0` or :math:`\mathcal{H}_1`.

    Under :math:`\mathcal{H}_0`:
    .. math::
        r(n) = w(n), \\quad w(n) \\sim \\mathcal{CN}(0, \\sigma_w^2)

    Under :math:`\mathcal{H}_1`:
    .. math::
        r(n) = h \\cdot s(n) + w(n)

    Where :math:`s(n)` power is scaled so that:
    .. math::
        \\text{SNR} = 10^{\\text{snr\_db} / 10} = \\frac{|h|^2 P_s}{\\sigma_w^2}
        \\implies P_s = \\frac{\\sigma_w^2 \\cdot 10^{\\text{snr\_db} / 10}}{|h|^2}

    Args:
        hypothesis: 0 for idle spectrum (:math:`\mathcal{H}_0`), 1 for active PU (:math:`\mathcal{H}_1`).
        config: HypothesisConfig instance.
        channel_gain: Complex or real channel multiplier :math:`h`.
        rng: Optional NumPy random generator instance.

    Returns:
        1D NumPy array representing received signal :math:`r(n)`.
    """
    if hypothesis not in (0, 1):
        raise ValueError(f"hypothesis must be 0 or 1, got {hypothesis}")

    if rng is None:
        rng = np.random.default_rng()

    # Always generate noise
    noise = generate_awgn(
        num_samples=config.num_samples,
        noise_variance=config.noise_variance,
        complex_noise=config.complex_signal,
        rng=rng,
    )

    if hypothesis == 0:
        return noise

    # Calculate required PU signal power to achieve requested SNR after channel gain
    snr_linear = 10.0 ** (config.snr_db / 10.0)
    target_signal_power = config.noise_variance * snr_linear

    signal = generate_pu_signal(
        num_samples=config.num_samples,
        signal_power=target_signal_power,
        modulation=config.modulation,
        complex_signal=config.complex_signal,
        rng=rng,
    )

    return channel_gain * signal + noise


class HypothesisTestingSimulator:
    """Simulator generating batches of labeled spectrum sensing observation frames.

    Provides datasets formatted as complex 1D arrays or 2-channel real arrays :math:`[L, 2]`
    representing In-Phase (I) and Quadrature (Q) components matching the RadioML format.
    """

    def __init__(self, config: HypothesisConfig | None = None, seed: int | None = None):
        """Initialize the simulator.

        Args:
            config: HypothesisConfig settings. Uses default if None.
            seed: Optional integer seed for reproducible random sequences.
        """
        self.config = config or HypothesisConfig()
        self.rng = np.random.default_rng(seed)

    def generate_batch(
        self,
        num_frames: int,
        p_h1: float = 0.5,
        channel_gain: complex | float = 1.0,
        as_iq_channels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Generate a batch of observation frames with corresponding ground-truth labels.

        Args:
            num_frames: Number of observation frames to generate.
            p_h1: Prior probability of primary user activity :math:`P(\\mathcal{H}_1)`.
            channel_gain: Channel multiplier :math:`h`.
            as_iq_channels: If True, returns shape :math:`(N, L, 2)` where index 0 is I and
                            index 1 is Q. If False, returns complex array :math:`(N, L)`.

        Returns:
            Tuple of:
                - `X`: Signal array of shape :math:`(N, L, 2)` or :math:`(N, L)`.
                - `y`: Binary labels array of length :math:`N` containing values in {0, 1}.
        """
        if not (0.0 <= p_h1 <= 1.0):
            raise ValueError(f"p_h1 must be within [0, 1], got {p_h1}")

        # Sample hypotheses for each frame
        labels = (self.rng.uniform(size=num_frames) < p_h1).astype(int)

        frames = []
        for label in labels:
            r = simulate_received_signal(
                hypothesis=int(label),  # type: ignore
                config=self.config,
                channel_gain=channel_gain,
                rng=self.rng,
            )
            frames.append(r)

        raw_complex = np.array(frames)  # shape (N, L)

        if as_iq_channels:
            # Format into (N, L, 2) where axis -1 is [I, Q]
            i_component = np.real(raw_complex)
            q_component = np.imag(raw_complex)
            return np.stack([i_component, q_component], axis=-1), labels

        return raw_complex, labels

    def theoretical_statistics(self, channel_gain: complex | float = 1.0) -> dict[str, float]:
        r"""Compute theoretical statistical parameters under :math:`\mathcal{H}_0` and :math:`\mathcal{H}_1`.

        Returns dictionary with:
            - `var_h0`: Expected variance under :math:`\mathcal{H}_0` (:math:`\sigma_w^2`)
            - `var_h1`: Expected variance under :math:`\mathcal{H}_1` (:math:`|h|^2 P_s + \sigma_w^2`)
            - `snr_linear`: Linear SNR ratio
        """
        snr_linear = 10.0 ** (self.config.snr_db / 10.0)
        var_h0 = self.config.noise_variance
        var_h1 = var_h0 * (1.0 + snr_linear)

        return {
            "var_h0": float(var_h0),
            "var_h1": float(var_h1),
            "snr_linear": float(snr_linear),
            "snr_db": float(self.config.snr_db),
        }
