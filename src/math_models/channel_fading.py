"""Wireless Channel Fading Models for Cognitive Radio Spectrum Sensing.

Implements envelope probability density functions (PDFs), random complex channel
coefficient generators, and block/fast fading channels:
- Rayleigh Fading (No direct Line-of-Sight, rich scattering):
  .. math::
      f_{\\text{Rayleigh}}(a; \\sigma) = \\frac{a}{\\sigma^2} \\exp\\left(-\\frac{a^2}{2\\sigma^2}\\right), \\quad a \\ge 0
- Rician Fading (Dominant Line-of-Sight component + multipath):
  .. math::
      f_{\\text{Rician}}(a; K, \\Omega) = \\frac{2(K+1)a}{\\Omega} \\exp\\left(-K - \\frac{(K+1)a^2}{\\Omega}\\right) I_0\\left(2a\\sqrt{\\frac{K(K+1)}{\\Omega}}\\right)
  where :math:`K` is the specular-to-diffuse power ratio and :math:`I_0(\\cdot)` is the modified Bessel function.
- Nakagami-:math:`m` Fading (Generalized multipath fading model):
  .. math::
      f_{\\text{Nakagami}}(a; m, \\Omega) = \\frac{2 m^m}{\\Gamma(m) \\Omega^m} a^{2m-1} \\exp\\left(-\\frac{m}{\\Omega} a^2\\right)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
from scipy import special, stats


FadingType = Literal["awgn", "rayleigh", "rician", "nakagami"]


@dataclass
class FadingChannelConfig:
    """Configuration parameters for wireless fading channel.

    Attributes:
        fading_type: Type of fading distribution ('awgn', 'rayleigh', 'rician', 'nakagami').
        average_power: Expected channel power gain :math:`\\Omega = \\mathbb{E}[|h|^2]`.
        rician_k: Linear Rician K-factor (ratio of LoS to diffuse power). Ignored if not rician.
        nakagami_m: Nakagami fading parameter :math:`m \\ge 0.5`. Ignored if not nakagami.
        block_fading: If True, channel remains constant over each frame (quasi-static).
                      If False, channel varies sample-by-sample (fast fading).
    """

    fading_type: FadingType = "rayleigh"
    average_power: float = 1.0
    rician_k: float = 4.0
    nakagami_m: float = 1.5
    block_fading: bool = True


def rayleigh_pdf(a: np.ndarray | float, sigma: float = 1.0 / np.sqrt(2.0)) -> np.ndarray | float:
    r"""Compute Rayleigh probability density function.

    .. math::
        f(a; \\sigma) = \\frac{a}{\\sigma^2} \\exp\\left(-\\frac{a^2}{2\\sigma^2}\\right), \\quad a \\ge 0

    Args:
        a: Envelope value(s) :math:`a \\ge 0`.
        sigma: Scale parameter :math:`\\sigma > 0`, where :math:`\\mathbb{E}[a^2] = 2\\sigma^2`.

    Returns:
        Probability density value(s).
    """
    a_arr = np.asarray(a)
    pdf = np.zeros_like(a_arr, dtype=float)
    pos_mask = a_arr >= 0.0

    a_pos = a_arr[pos_mask]
    pdf[pos_mask] = (a_pos / (sigma**2)) * np.exp(-(a_pos**2) / (2.0 * sigma**2))

    return float(pdf) if np.isscalar(a) else pdf


def rician_pdf(
    a: np.ndarray | float,
    k_factor: float = 4.0,
    average_power: float = 1.0,
) -> np.ndarray | float:
    r"""Compute Rician probability density function.

    .. math::
        f(a; K, \\Omega) = \\frac{2(K+1)a}{\\Omega} \\exp\\left(-K - \\frac{(K+1)a^2}{\\Omega}\\right)
                           I_0\\left(2a\\sqrt{\\frac{K(K+1)}{\\Omega}}\\right)

    Args:
        a: Envelope value(s) :math:`a \\ge 0`.
        k_factor: Linear Rician factor :math:`K \\ge 0`.
        average_power: Total average power :math:`\\Omega > 0`.

    Returns:
        Probability density value(s).
    """
    if k_factor < 0:
        raise ValueError(f"k_factor must be >= 0, got {k_factor}")
    if average_power <= 0:
        raise ValueError(f"average_power must be > 0, got {average_power}")

    a_arr = np.asarray(a)
    pdf = np.zeros_like(a_arr, dtype=float)
    pos_mask = a_arr >= 0.0

    a_pos = a_arr[pos_mask]
    coeff = 2.0 * (k_factor + 1.0) / average_power
    exponent = -k_factor - ((k_factor + 1.0) * (a_pos**2) / average_power)
    bessel_arg = 2.0 * a_pos * np.sqrt(k_factor * (k_factor + 1.0) / average_power)

    pdf[pos_mask] = coeff * a_pos * np.exp(exponent) * special.i0(bessel_arg)

    return float(pdf) if np.isscalar(a) else pdf


def nakagami_pdf(
    a: np.ndarray | float,
    m_param: float = 1.5,
    average_power: float = 1.0,
) -> np.ndarray | float:
    r"""Compute Nakagami-:math:`m` probability density function.

    .. math::
        f(a; m, \\Omega) = \\frac{2 m^m}{\\Gamma(m) \\Omega^m} a^{2m-1} \\exp\\left(-\\frac{m}{\\Omega} a^2\\right)

    Args:
        a: Envelope value(s) :math:`a \\ge 0`.
        m_param: Shape parameter :math:`m \\ge 0.5`.
        average_power: Total average power :math:`\\Omega > 0`.

    Returns:
        Probability density value(s).
    """
    if m_param < 0.5:
        raise ValueError(f"m_param must be >= 0.5, got {m_param}")
    if average_power <= 0:
        raise ValueError(f"average_power must be > 0, got {average_power}")

    a_arr = np.asarray(a)
    pdf = np.zeros_like(a_arr, dtype=float)
    pos_mask = a_arr >= 0.0

    a_pos = a_arr[pos_mask]
    coeff = (2.0 * (m_param**m_param)) / (special.gamma(m_param) * (average_power**m_param))
    term = (a_pos ** (2.0 * m_param - 1.0)) * np.exp(-(m_param / average_power) * (a_pos**2))

    pdf[pos_mask] = coeff * term
    return float(pdf) if np.isscalar(a) else pdf


def generate_fading_coefficients(
    num_samples: int,
    config: FadingChannelConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Generate random complex channel coefficients :math:`h = h_I + j h_Q`.

    Normalized such that :math:`\mathbb{E}[|h|^2] = \Omega = \text{average\_power}`.

    Args:
        num_samples: Number of channel coefficients to generate.
        config: FadingChannelConfig settings.
        rng: Optional NumPy random generator instance.

    Returns:
        1D complex array of length `num_samples`.
    """
    if rng is None:
        rng = np.random.default_rng()

    omega = config.average_power

    if config.fading_type == "awgn":
        # Non-fading line-of-sight: constant magnitude sqrt(omega)
        return np.full(num_samples, np.sqrt(omega), dtype=complex)

    elif config.fading_type == "rayleigh":
        # Two independent zero-mean Gaussians with variance omega/2
        sigma_comp = np.sqrt(omega / 2.0)
        h_i = rng.normal(0.0, sigma_comp, size=num_samples)
        h_q = rng.normal(0.0, sigma_comp, size=num_samples)
        return h_i + 1j * h_q

    elif config.fading_type == "rician":
        k = config.rician_k
        # Specular component power: s^2 = K*omega / (K + 1)
        # Diffuse component power: 2*sigma^2 = omega / (K + 1)
        specular_amp = np.sqrt(k * omega / (k + 1.0))
        sigma_comp = np.sqrt(omega / (2.0 * (k + 1.0)))

        h_i = rng.normal(specular_amp, sigma_comp, size=num_samples)
        h_q = rng.normal(0.0, sigma_comp, size=num_samples)
        return h_i + 1j * h_q

    elif config.fading_type == "nakagami":
        m = config.nakagami_m
        # Envelope squared follows Gamma distribution: shape=m, scale=omega/m
        power_samples = rng.gamma(shape=m, scale=omega / m, size=num_samples)
        envelopes = np.sqrt(power_samples)
        # Random uniform phase in [0, 2pi)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)
        return envelopes * np.exp(1j * phases)

    else:
        raise ValueError(f"Unsupported fading type: {config.fading_type}")


class FadingChannel:
    """Wireless fading channel simulator applying stochastic multipath fading."""

    def __init__(
        self,
        config: FadingChannelConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize the fading channel simulator.

        Args:
            config: FadingChannelConfig options. Uses defaults if None.
            seed: Optional integer seed for reproducibility.
        """
        self.config = config or FadingChannelConfig()
        self.rng = np.random.default_rng(seed)

    def apply_channel(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Apply fading channel multiplication to an input signal: :math:`r(n) = h(n) \cdot s(n)`.

        Args:
            signal: 1D complex array of length :math:`L`.

        Returns:
            Tuple of:
                - `faded_signal`: Complex array of length :math:`L`.
                - `channel_h`: Array of channel coefficient(s) applied.
        """
        n_samples = len(signal)

        if self.config.block_fading:
            # Single complex scalar per frame
            h = generate_fading_coefficients(1, self.config, rng=self.rng)[0]
            faded = h * signal
            return faded, np.full(n_samples, h, dtype=complex)
        else:
            # Sample-by-sample fast fading
            h = generate_fading_coefficients(n_samples, self.config, rng=self.rng)
            faded = h * signal
            return faded, h
