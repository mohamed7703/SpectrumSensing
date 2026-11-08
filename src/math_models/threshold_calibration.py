"""Distribution-free H0 tolerance thresholds, conditional on iid stationary scores."""

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ThresholdCalibration:
    """Threshold and order-statistic calibration provenance."""
    threshold: float
    frames: int
    order_rank: int
    target_pfa: float
    confidence_level: float


def calibrate_score_threshold(scores: np.ndarray, target_pfa: float = .05,
                              confidence_level: float = .95) -> ThresholdCalibration:
    """Choose a strict upper order statistic from independent H0 scores.

    Args:
        scores: Finite real vector from H0 calibration frames only.
        target_pfa: Upper population false alarm rate in (0,1).
        confidence_level: Marginal calibration confidence in (0,1).

    For continuous F, P(F(X_(k)) >= q) = P(Bin(n,q) <= k-1).
    Strict rejection of ties is conservative for discrete scores. No confidence
    guarantee applies under drift, dependence, or tuning on evaluation frames.
    Each calibration is marginal; multiple comparisons need joint-error control.
    """
    values = np.asarray(scores)
    if (values.ndim != 1 or len(values) == 0 or values.dtype.kind not in "fiu"
            or not np.all(np.isfinite(values))):
        raise ValueError("calibration requires a nonempty finite real score vector")
    if not 0 < target_pfa < 1 or not 0 < confidence_level < 1:
        raise ValueError("target_pfa and confidence_level must be in (0,1)")
    values = values.astype(np.float64)
    n = len(values)
    rank = int(stats.binom.ppf(confidence_level, n, 1-target_pfa)) + 1
    if rank > n:
        raise ValueError("insufficient calibration frames for tolerance/confidence")
    threshold = float(np.nextafter(np.partition(values, rank-1)[rank-1], np.inf))
    return ThresholdCalibration(threshold, n, rank, target_pfa, confidence_level)
