"""Independent exact calibration probability and tie-boundary checks."""

import numpy as np
import pytest
from scipy import stats
from src.math_models.threshold_calibration import calibrate_score_threshold


def test_selected_rank_meets_confidence_and_preceding_rank_does_not() -> None:
    """Check calibration probability directly using a binomial CDF."""
    result = calibrate_score_threshold(np.arange(400), .05, .95)
    assert stats.binom.cdf(result.order_rank-1, 400, .95) >= .95
    assert stats.binom.cdf(result.order_rank-2, 400, .95) < .95


def test_tied_scores_are_rejected_strictly_in_float64() -> None:
    """A constant float32 score must not produce universal false alarms."""
    values = np.ones(100, dtype=np.float32)
    result = calibrate_score_threshold(values)
    assert not np.any(values.astype(np.float64) >= result.threshold)


@pytest.mark.parametrize("scores", [np.array([]), np.ones((3, 2)), np.array([np.nan]), np.array([1j])])
def test_invalid_calibration_scores_rejected(scores: np.ndarray) -> None:
    """Reject empty, matrix, nonfinite or complex inputs before choosing a threshold."""
    with pytest.raises(ValueError):
        calibrate_score_threshold(scores)
