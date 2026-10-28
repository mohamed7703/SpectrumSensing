"""Energy Detection Decision Statistic Module for Spectrum Sensing.

Implements the classical energy detection test statistic, threshold derivations
under the Constant False Alarm Rate (CFAR) principle, and exact/asymptotic
statistical distributions:
.. math::
    D(Y) = \\frac{1}{L} \\sum_{n=0}^{L-1} |r(n)|^2

Under :math:`\\mathcal{H}_0`:
.. math::
    T_{\\text{norm}} = \\frac{2}{\\sigma_w^2} \\sum_{n=0}^{L-1} |r(n)|^2 \\sim \\chi^2_{2L}
    \\quad \\text{(Central } \\chi^2 \\text{ with } 2L \\text{ degrees of freedom)}

Under :math:`\\mathcal{H}_1`:
.. math::
    T_{\\text{norm}} \\sim \\chi'^2_{2L}(\\lambda)
    \\quad \\text{(Non-central } \\chi^2 \\text{ with non-centrality } \\lambda = 2L \\cdot \\text{SNR)}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
from scipy import stats, special


@dataclass
class EnergyDetectorConfig:
    """Configuration for the Energy Detector.

    Attributes:
        num_samples: Number of observation samples :math:`L`.
        noise_variance: Known or estimated noise variance :math:`\\sigma_w^2`.
        pfa_target: Desired Probability of False Alarm for CFAR threshold setting.
        method: Calculation method ('exact' using Chi-Square, or 'clt' using Gaussian approximation).
    """

    num_samples: int = 1024
    noise_variance: float = 1.0
    pfa_target: float = 0.05
    method: Literal["exact", "clt"] = "exact"


def compute_energy_statistic(
    samples: np.ndarray,
    normalize: bool = False,
    noise_variance: float = 1.0,
) -> float | np.ndarray:
    r"""Compute the energy test statistic :math:`D(Y)` over observation samples.

    For unnormalized statistic:
    .. math::
        D(Y) = \\frac{1}{L} \\sum_{n=0}^{L-1} |r(n)|^2

    For normalized statistic:
    .. math::
        T_{\\text{norm}} = \\frac{2}{\\sigma_w^2} \\sum_{n=0}^{L-1} |r(n)|^2

    Args:
        samples: 1D array of shape :math:`(L,)` or 2D array of shape :math:`(N, L)` (complex),
                 or 3D array of shape :math:`(N, L, 2)` (real I/Q).
        normalize: If True, scale by :math:`2 / \\sigma_w^2` to match standard Chi-Square distribution.
        noise_variance: Known noise power :math:`\\sigma_w^2` used when `normalize=True`.

    Returns:
        Scalar energy (for single frame) or 1D array of length :math:`N` (for batch).
    """
    if noise_variance <= 0:
        raise ValueError(f"noise_variance must be > 0, got {noise_variance}")

    # Handle I/Q array format (..., L, 2)
    if samples.ndim >= 2 and samples.shape[-1] == 2 and not np.iscomplexobj(samples):
        # |r(n)|^2 = I(n)^2 + Q(n)^2
        inst_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
    else:
        inst_power = np.abs(samples) ** 2

    # Average over sample window (axis -1)
    unnormalized_energy = np.mean(inst_power, axis=-1)
    sample_length = samples.shape[-2] if (samples.ndim >= 2 and samples.shape[-1] == 2 and not np.iscomplexobj(samples)) else samples.shape[-1]

    if normalize:
        # T_norm = unnormalized_energy * (2 * L / sigma_w^2)
        return unnormalized_energy * (2.0 * sample_length / noise_variance)

    return unnormalized_energy


def compute_theoretical_threshold(
    pfa_target: float,
    num_samples: int,
    noise_variance: float = 1.0,
    method: Literal["exact", "clt"] = "exact",
) -> float:
    r"""Calculate the decision threshold :math:`\gamma` for a target false alarm rate.

    Under CFAR (Constant False Alarm Rate), :math:`\gamma` is chosen such that:
    .. math::
        P_{\\text{FA}} = P(D(Y) \\ge \\gamma \\mid \\mathcal{H}_0) = \\alpha_{\\text{target}}

    Exact Chi-Square Method:
    .. math::
        \\gamma_{\\text{norm}} = F_{\\chi^2_{2L}}^{-1}(1 - P_{\\text{FA}}), \\quad
        \\gamma = \\gamma_{\\text{norm}} \\cdot \\frac{\\sigma_w^2}{2 L}

    Central Limit Theorem (CLT) Gaussian Approximation:
    .. math::
        \\gamma = \\sigma_w^2 \\left(1 + \\frac{Q^{-1}(P_{\\text{FA}})}{\\sqrt{L}}\\right)

    Args:
        pfa_target: Target Probability of False Alarm :math:`P_{\\text{FA}} \\in (0, 1)`.
        num_samples: Number of observation samples :math:`L`.
        noise_variance: Noise variance :math:`\\sigma_w^2`.
        method: 'exact' or 'clt'.

    Returns:
        Decision threshold :math:`\\gamma` on the scale of sample average power.
    """
    if not (0.0 < pfa_target < 1.0):
        raise ValueError(f"pfa_target must be strictly between 0 and 1, got {pfa_target}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    if method == "exact":
        # Degrees of freedom for complex baseband is 2L
        dof = 2 * num_samples
        gamma_norm = stats.chi2.ppf(1.0 - pfa_target, df=dof)
        return float(gamma_norm * (noise_variance / (2.0 * num_samples)))

    elif method == "clt":
        # Q^-1(pfa) = sqrt(2) * erfinv(1 - 2*pfa)
        q_inv = np.sqrt(2.0) * special.erfinv(1.0 - 2.0 * pfa_target)
        return float(noise_variance * (1.0 + q_inv / np.sqrt(num_samples)))

    else:
        raise ValueError(f"Unsupported method: {method}")


def analytical_pd(
    pfa: float,
    snr_db: float,
    num_samples: int,
    noise_variance: float = 1.0,
    method: Literal["exact", "clt"] = "exact",
) -> float:
    r"""Compute the theoretical Probability of Detection :math:`P_d` for a given :math:`P_{\\text{FA}}` and SNR.

    Exact Chi-Square formulation:
    .. math::
        P_d = 1 - F_{\\chi'^2_{2L}(\\lambda)}\\left(F_{\\chi^2_{2L}}^{-1}(1 - P_{\\text{FA}})\\right)
    where :math:`\\lambda = 2L \\cdot \\text{SNR}`.

    CLT Gaussian approximation:
    .. math::
        P_d = Q\\left( \\frac{\\gamma - \\sigma_w^2(1 + \\text{SNR})}{\\sigma_w^2 (1 + \\text{SNR}) / \\sqrt{L}} \\right)

    Args:
        pfa: Probability of False Alarm.
        snr_db: Signal-to-Noise Ratio in dB.
        num_samples: Sample length :math:`L`.
        noise_variance: Noise variance :math:`\\sigma_w^2`.
        method: 'exact' or 'clt'.

    Returns:
        Theoretical Probability of Detection :math:`P_d \\in [0, 1]`.
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    dof = 2 * num_samples

    if method == "exact":
        # Critical value on chi2(2L)
        crit_val = stats.chi2.ppf(1.0 - pfa, df=dof)
        # Non-centrality parameter
        nc_param = 2.0 * num_samples * snr_linear
        # Survival function of non-central chi-square (1 - CDF)
        return float(stats.ncx2.sf(crit_val, df=dof, nc=nc_param))

    elif method == "clt":
        gamma = compute_theoretical_threshold(pfa, num_samples, noise_variance, method="clt")
        mu_1 = noise_variance * (1.0 + snr_linear)
        sigma_1 = mu_1 / np.sqrt(num_samples)
        z = (gamma - mu_1) / sigma_1
        # Q(z) = 0.5 * erfc(z / sqrt(2))
        return float(0.5 * special.erfc(z / np.sqrt(2.0)))

    else:
        raise ValueError(f"Unsupported method: {method}")


class EnergyDetector:
    """Configurable Energy Detector for Cognitive Radio Spectrum Sensing.

    Computes decision statistics and evaluates empirical detection rates.
    """

    def __init__(self, config: EnergyDetectorConfig | None = None):
        """Initialize the energy detector.

        Args:
            config: EnergyDetectorConfig instance. Uses defaults if None.
        """
        self.config = config or EnergyDetectorConfig()
        self.threshold = compute_theoretical_threshold(
            pfa_target=self.config.pfa_target,
            num_samples=self.config.num_samples,
            noise_variance=self.config.noise_variance,
            method=self.config.method,
        )

    def set_threshold(self, threshold: float) -> None:
        """Manually override the decision threshold."""
        if threshold <= 0:
            raise ValueError(f"Threshold must be > 0, got {threshold}")
        self.threshold = threshold

    def decide(self, samples: np.ndarray) -> int:
        r"""Perform binary detection decision on a single observation frame.

        .. math::
            \\hat{\\mathcal{H}} = \\begin{cases}
                1 & \\text{if } D(Y) \\ge \\gamma \\quad (\\text{PU Present}) \\\\
                0 & \\text{if } D(Y) < \\gamma \\quad (\\text{Spectrum Idle})
            \\end{cases}

        Args:
            samples: Received frame array of length :math:`L`.

        Returns:
            Decision integer (1 for :math:`\\mathcal{H}_1`, 0 for :math:`\\mathcal{H}_0`).
        """
        stat = float(compute_energy_statistic(samples, normalize=False, noise_variance=self.config.noise_variance))
        return 1 if stat >= self.threshold else 0

    def decide_batch(self, batch_samples: np.ndarray) -> np.ndarray:
        """Perform binary detection decisions across a batch of frames.

        Args:
            batch_samples: Array of shape :math:`(N, L)` or :math:`(N, L, 2)`.

        Returns:
            1D NumPy array of length :math:`N` containing binary decisions in {0, 1}.
        """
        stats_arr = compute_energy_statistic(batch_samples, normalize=False, noise_variance=self.config.noise_variance)
        return (stats_arr >= self.threshold).astype(int)

    def evaluate_performance(
        self,
        batch_samples: np.ndarray,
        ground_truth: np.ndarray,
    ) -> dict[str, float]:
        r"""Compute empirical detection metrics from a test batch.

        .. math::
            P_d = \\frac{\\text{Correct detections}}{\\text{Total actual } \\mathcal{H}_1}, \\quad
            P_{\\text{FA}} = \\frac{\\text{False alarms}}{\\text{Total actual } \\mathcal{H}_0}

        Args:
            batch_samples: Received frames tensor.
            ground_truth: True binary labels (0 for H0, 1 for H1).

        Returns:
            Dictionary with empirical `pd`, `pfa`, `accuracy`, and sample counts.
        """
        predictions = self.decide_batch(batch_samples)

        h0_mask = ground_truth == 0
        h1_mask = ground_truth == 1

        n_h0 = int(np.sum(h0_mask))
        n_h1 = int(np.sum(h1_mask))

        false_alarms = int(np.sum((predictions == 1) & h0_mask))
        correct_detections = int(np.sum((predictions == 1) & h1_mask))

        emp_pfa = false_alarms / n_h0 if n_h0 > 0 else 0.0
        emp_pd = correct_detections / n_h1 if n_h1 > 0 else 0.0
        accuracy = float(np.mean(predictions == ground_truth))

        return {
            "pfa": emp_pfa,
            "pd": emp_pd,
            "accuracy": accuracy,
            "threshold": self.threshold,
            "n_h0": n_h0,
            "n_h1": n_h1,
            "false_alarms": false_alarms,
            "correct_detections": correct_detections,
        }
