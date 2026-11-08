"""Exact binomial endpoints checked against elementary independent formulas."""

import numpy as np
import pytest

from src.math_models.detection_metrics import binomial_rate_interval


def test_zero_events_one_sided_upper_bound():
    """For zero events, P(X=0)=(1-p)^n determines the upper confidence bound."""
    low, high = binomial_rate_interval(0, 100, alternative="less")
    assert low == 0
    assert high == pytest.approx(1 - .05 ** (1 / 100))


def test_all_successes_one_sided_lower_bound():
    """For n successes, P(X=n)=p^n determines the lower confidence bound."""
    low, high = binomial_rate_interval(299, 299, alternative="greater")
    assert low == pytest.approx(.05 ** (1 / 299))
    assert high == 1


def test_two_sided_single_trial_and_audit_detection_interval():
    """Single-trial endpoint is elementary; 22/25 remains imprecise."""
    assert binomial_rate_interval(0, 1) == pytest.approx((0, .975))
    assert binomial_rate_interval(1, 1) == pytest.approx((.025, 1))
    assert binomial_rate_interval(22, 25) == pytest.approx((.6878096927, .9745346034))


@pytest.mark.parametrize("successes,trials", [(-1, 5), (6, 5), (0, 0), (1.5, 5), (True, 5)])
def test_invalid_counts_rejected(successes, trials):
    """Confidence intervals require real, nonempty integer trial counts."""
    with pytest.raises(ValueError):
        binomial_rate_interval(successes, trials)


@pytest.mark.parametrize("confidence", [0, 1, np.nan, np.inf])
def test_invalid_confidence_rejected(confidence):
    """A confidence level must be finite and strictly interior to (0,1)."""
    with pytest.raises(ValueError):
        binomial_rate_interval(1, 2, confidence_level=confidence)
