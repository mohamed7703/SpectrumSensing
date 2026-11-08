"""Noise Models and SNR Wall Analysis for Cognitive Radio Spectrum Sensing.

Implements calibrated AWGN generation (matching the research paper's parameters)
and theoretical/empirical modeling of the **SNR Wall** phenomenon:
.. math::
    \\sigma_w^2 \\in \\left[\\frac{\\sigma_n^2}{\\rho}, \\rho \\cdot \\sigma_n^2\\right], \\quad \\rho > 1
    \\implies B_{\\text{dB}} = 10 \\log_{10}(\\rho)

The fundamental SNR wall below which an energy detector cannot reliably detect
regardless of infinite sample accumulation (:math:`L \\to \\infty`):
.. math::
    \\text{SNR}_{\\text{wall}} = \\rho - \\frac{1}{\\rho}
.. math::
    \\text{SNR}_{\\text{wall, dB}} = 10 \\log_{10}\\left(\\rho - \\frac{1}{\\rho}\\right)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import special


@dataclass
class NoiseModelConfig:
    """Configuration for noise generation and uncertainty modeling.

    Attributes:
        std_dev: Standard deviation :math:`\\sigma_w` of noise (paper uses :math:`\\sigma=0.708`).
        uncertainty_db: Noise variance uncertainty bound :math:`B_{\\text{dB}} = 10 \\log_{10}(\\rho)`.
        complex_noise: Whether to generate complex baseband noise :math:`w_I + j w_Q`.
    """

    std_dev: float = 0.708
    uncertainty_db: float = 0.5
    complex_noise: bool = True

    @property
    def variance(self) -> float:
        """Total noise power variance :math:`\\sigma_w^2`."""
        return float(self.std_dev**2)

    @property
    def rho(self) -> float:
        """Linear uncertainty factor :math:`\\rho = 10^{B_{\\text{dB}} / 10}`."""
        return float(10.0 ** (self.uncertainty_db / 10.0))


def compute_snr_wall(uncertainty_db: float) -> Tuple[float, float]:
    r"""Compute the theoretical SNR wall for an energy detector.

    .. math::
        \\rho = 10^{B_{\\text{dB}} / 10}, \\quad
        \\text{SNR}_{\\text{wall}} = \\rho - \\frac{1}{\\rho}

    Args:
        uncertainty_db: Noise uncertainty bound in dB (:math:`B_{\\text{dB}} > 0`).

    Returns:
        Tuple of (snr_wall_linear, snr_wall_db).
    """
    if uncertainty_db <= 0:
        raise ValueError(f"uncertainty_db must be > 0, got {uncertainty_db}")

    rho = 10.0 ** (uncertainty_db / 10.0)
    snr_wall_lin = rho - (1.0 / rho)
    snr_wall_db = 10.0 * np.log10(snr_wall_lin)

    return float(snr_wall_lin), float(snr_wall_db)


def compute_sample_complexity(
    snr_db: float,
    uncertainty_db: float,
    pfa_target: float = 0.05,
    pd_target: float = 0.90,
) -> float:
    r"""Return a conservative sufficient sample count for robust fixed-energy sensing.

    Nominal noise power is one, actual v is in [1/rho, rho], and fixed
    transmit signal power S = 10**(snr_db/10), independent of v. Complex AWGN
    gives Var(T|H0) <= rho**2/N and Var(T|H1) <= (rho**2+2*rho*S)/N.
    Cantelli's one-sided inequality supplies a uniform sufficient threshold
    rho + rho*sqrt((1-alpha)/(alpha*N)). To attain Pd >= beta it suffices that
    sqrt(N) >= [rho*sqrt((1-alpha)/alpha) +
    sqrt((rho**2+2*rho*S)*beta/(1-beta))] / [S-(rho-1/rho)].

    This bound is conservative and is NOT the minimum required sample size.
    The former approximation omitted variance factors and missed its targets.
    An exact H0-CFAR threshold at v=rho is no larger than this sufficient
    threshold, so it also attains the stated lower Pd bound. No guarantee is
    made for correlated noise, random signal energy, or unbounded uncertainty.

    Args:
        snr_db: Fixed signal power relative to nominal noise, before uncertainty.
        uncertainty_db: Noise uncertainty in dB.
        pfa_target: Target False Alarm rate :math:`\\alpha`.
        pd_target: Target Detection rate :math:`\\beta`.

    Returns:
        Conservative sufficient sample count, or infinity at/below the robust
        mean-separation wall. Infinity describes this uniform guarantee, not
        impossibility of all average-case or finite-sample discrimination.
    """
    if (not np.isfinite(snr_db) or not np.isfinite(uncertainty_db)
            or not 0 < pfa_target < pd_target < 1):
        raise ValueError("finite SNR/uncertainty and 0 < pfa_target < pd_target < 1 are required")
    snr_lin = 10.0 ** (snr_db / 10.0)
    snr_wall_lin, _ = compute_snr_wall(uncertainty_db)

    if snr_lin <= snr_wall_lin:
        return float("inf")

    rho = 10.0 ** (uncertainty_db / 10.0)
    numerator = (rho * np.sqrt((1.0 - pfa_target) / pfa_target)
                 + np.sqrt((rho**2 + 2.0 * rho * snr_lin) * pd_target / (1.0 - pd_target)))
    denominator = snr_lin - snr_wall_lin

    l_required = (numerator / denominator) ** 2
    return float(np.ceil(l_required))


class CalibratedAWGNGenerator:
    """Calibrated Gaussian noise generator matching the Sensors 2024 paper specification.

    By default, generates AWGN with standard deviation :math:`\\sigma_w = 0.708`
    (corresponding to total variance :math:`\\sigma_w^2 \\approx 0.5013`).
    """

    def __init__(self, config: NoiseModelConfig | None = None, seed: int | None = None):
        """Initialize calibrated noise generator.

        Args:
            config: NoiseModelConfig instance. Uses defaults if None.
            seed: Optional integer seed for reproducibility.
        """
        self.config = config or NoiseModelConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, num_samples: int) -> np.ndarray:
        r"""Generate a 1D vector of calibrated AWGN samples.

        For complex baseband noise:
        .. math::
            w(n) = w_I(n) + j w_Q(n), \\quad
            w_I, w_Q \\sim \\mathcal{N}\\left(0, \\frac{\\sigma_w^2}{2}\\right)
        """
        if self.config.complex_noise:
            std_comp = self.config.std_dev / np.sqrt(2.0)
            return self.rng.normal(0.0, std_comp, size=num_samples) + 1j * self.rng.normal(
                0.0, std_comp, size=num_samples
            )
        else:
            return self.rng.normal(0.0, self.config.std_dev, size=num_samples)

    def generate_uncertain_noise(
        self,
        num_samples: int,
        uncertainty_factor: float | None = None,
    ) -> Tuple[np.ndarray, float]:
        r"""Generate noise with perturbed variance within the uncertainty interval :math:`[\sigma_n^2/\rho, \rho \sigma_n^2]`.

        Args:
            num_samples: Number of samples.
            uncertainty_factor: Multiplier in :math:`[1/\\rho, \\rho]`. If None, sampled uniformly.

        Returns:
            Tuple of (noise_array, actual_perturbed_variance).
        """
        rho = self.config.rho
        if uncertainty_factor is None:
            # Uniform in log-scale [1/rho, rho]
            log_rho = np.log(rho)
            u = self.rng.uniform(-log_rho, log_rho)
            uncertainty_factor = np.exp(u)

        perturbed_var = self.config.variance * uncertainty_factor
        perturbed_std = np.sqrt(perturbed_var)

        if self.config.complex_noise:
            std_comp = perturbed_std / np.sqrt(2.0)
            noise = self.rng.normal(0.0, std_comp, size=num_samples) + 1j * self.rng.normal(
                0.0, std_comp, size=num_samples
            )
        else:
            noise = self.rng.normal(0.0, perturbed_std, size=num_samples)

        return noise, float(perturbed_var)
