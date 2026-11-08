"""Independent audit counterexamples for empirical binary metrics."""

import numpy as np
import pytest

from src.evaluation.evaluator import BenchmarkEvaluator
from src.math_models.detection_metrics import compute_detection_metrics, compute_empirical_roc
from src.training.trainer import compute_binary_metrics


def rank_auc(labels, scores):
    """Independent pairwise oracle for small fixtures, with half credit for ties."""
    positives = np.asarray(scores)[np.asarray(labels) == 1]
    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    return float(np.mean((positives[:, None] > negatives) + .5 * (positives[:, None] == negatives)))


@pytest.mark.parametrize("scores,expected", [
    ([1.1, 1.2, 2., 3.], 1.),
    ([.5001, .5002, .5003, .5004], 1.),
    ([.7, .7, .7, .7], .5),
    ([4., 3., 2., 1.], 0.),
    ([.2, .8, .2, .8], .5),
])
@pytest.mark.parametrize("backend", ["math", "benchmark"])
def test_exact_roc_audit_fixtures(scores, expected, backend):
    """All score transitions, including tied FPRs and tiny separations, matter."""
    labels = np.array([0, 0, 1, 1])
    scores = np.array(scores)
    result = (compute_empirical_roc(scores, labels, num_thresholds=2) if backend == "math"
              else BenchmarkEvaluator().compute_roc(labels, scores, num_thresholds=2))
    assert result.auc == pytest.approx(expected, abs=1e-12)
    assert result.auc == pytest.approx(rank_auc(labels, scores), abs=1e-12)
    assert np.all(np.diff(result.pfa) >= 0)
    assert np.all(np.diff(result.pd) >= 0)
    assert (result.pfa[0], result.pd[0]) == (0, 0)
    assert (result.pfa[-1], result.pd[-1]) == (1, 1)
    assert np.isposinf(result.thresholds[0])
    for threshold, pfa, pd in zip(result.thresholds, result.pfa, result.pd):
        assert pfa == pytest.approx(np.mean(scores[labels == 0] >= threshold))
        assert pd == pytest.approx(np.mean(scores[labels == 1] >= threshold))


@pytest.mark.parametrize("seed", [1, 17, 42])
def test_all_metric_entrypoints_match_independent_rank_oracle(seed):
    """Tied and interleaved scores must agree across math, benchmark and training."""
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 50)
    scores = rng.integers(0, 8, 100) / 7.
    expected = rank_auc(labels, scores)
    assert compute_empirical_roc(scores, labels).auc == pytest.approx(expected)
    assert BenchmarkEvaluator().compute_roc(labels, scores).auc == pytest.approx(expected)
    assert compute_binary_metrics(labels, scores).auc == pytest.approx(expected)


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([0, 1], [.5]), ([0, 2], [.1, .9]),
    ([0, .9], [.1, .9]), ([0, np.nan], [.1, .9]),
    ([0, 1], [.1, np.inf]), ([0, 1], [.1, np.nan]),
    ([[0, 1], [0, 1]], [[.1, .9], [.2, .8]]),
])
@pytest.mark.parametrize("backend", ["math", "benchmark", "training"])
def test_invalid_metric_data_are_rejected(labels, scores, backend):
    """Malformed or nonfinite observations must not be cast into credible rates."""
    with pytest.raises(ValueError):
        if backend == "math":
            compute_empirical_roc(np.asarray(scores), np.asarray(labels))
        elif backend == "benchmark":
            BenchmarkEvaluator().compute_roc(np.asarray(labels), np.asarray(scores))
        else:
            compute_binary_metrics(np.asarray(labels), np.asarray(scores))


@pytest.mark.parametrize("label", [0, 1])
def test_roc_rejects_absent_class_and_rates_remain_undefined(label):
    """No observations of a class cannot establish its conditional error rate."""
    labels = np.full(4, label)
    scores = np.array([.1, .2, .8, .9])
    with pytest.raises(ValueError):
        BenchmarkEvaluator().compute_roc(labels, scores)
    with pytest.raises(ValueError):
        compute_empirical_roc(scores, labels)
    training = compute_binary_metrics(labels, scores)
    detection = compute_detection_metrics((scores >= .5).astype(int), labels)
    assert np.isnan(training.auc)
    assert np.isnan(training.detection_probability if label == 0 else training.false_alarm_rate)
    assert np.isnan(detection.pd if label == 0 else detection.pfa)


def test_training_probabilities_and_thresholds_are_validated():
    """AUC permits raw scores, but probability-based BCE does not."""
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 1]), np.array([-.1, 1.1]))
    with pytest.raises(ValueError):
        BenchmarkEvaluator().compute_confusion_matrix(np.array([0, 1]), np.array([.1, .9]), threshold=np.nan)


def test_benchmark_eer_interpolates_a_roc_segment():
    """A tied block crosses FPR=FNR at 2/3 between (.5,0) and (1,1)."""
    result = BenchmarkEvaluator().compute_roc(np.array([0, 0, 1]), np.array([3., 2., 2.]))
    assert result.auc == pytest.approx(.25)
    assert result.eer == pytest.approx(2 / 3)
