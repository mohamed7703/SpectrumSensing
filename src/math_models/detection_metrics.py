"""Detection Metrics and ROC Analysis for Cognitive Radio Spectrum Sensing.

This module provides metrics and Receiver Operating Characteristic (ROC) curve
evaluations according to IEEE 802.22 cognitive radio sensing standards:
- Probability of Detection:
  .. math::
      P_d = P(\\hat{\\mathcal{H}}_1 = \\mathcal{H}_1 \\mid \\mathcal{H}_1) = \\frac{\\text{Correct Detections}}{\\text{Total Actual } \\mathcal{H}_1}
- Probability of False Alarm:
  .. math::
      P_{\\text{FA}} = P(\\hat{\\mathcal{H}}_1 = \\mathcal{H}_1 \\mid \\mathcal{H}_0) = \\frac{\\text{False Alarms}}{\\text{Total Actual } \\mathcal{H}_0}
- Probability of Missed Detection:
  .. math::
      P_m = 1 - P_d = \\frac{\\text{Missed Detections}}{\\text{Total Actual } \\mathcal{H}_1}
- Receiver Operating Characteristic (ROC) curve:
  .. math::
      P_d(P_{\\text{FA}}) \\quad \\forall \\; \\gamma \\in (-\\infty, +\\infty)
- Area Under the Curve (AUC):
  .. math::
      \\text{AUC} = \\int_0^1 P_d(P_{\\text{FA}}) \\, dP_{\\text{FA}}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import integrate
from scipy import stats

from .energy_detection import analytical_pd


def validate_binary_inputs(
    values: np.ndarray,
    ground_truth: np.ndarray,
    *,
    binary_values: bool = False,
    probabilities: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate paired real vectors before casting labels or calculating rates.

    Args:
        values: Finite scores, probabilities, or binary decisions, as a vector
            or a single-column array. Matrix inputs are rejected.
        ground_truth: Corresponding nonempty labels containing only 0 and 1.
        binary_values: Require values to be binary decisions.
        probabilities: Require values to lie in [0, 1].

    Returns:
        Flattened float64 values and int64 labels, retaining original labels
        until their validity has been checked. A single class is allowed for
        confusion counts; ROC callers must require both classes separately.
    """
    vectors = []
    for name, raw in (("values", values), ("ground_truth", ground_truth)):
        array = np.asarray(raw)
        if array.ndim != 1 and not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(f"{name} must be a vector or a single-column array")
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite real numbers")
        vectors.append(array.reshape(-1))
    scores, labels = vectors
    if len(labels) == 0 or len(scores) != len(labels):
        raise ValueError("values and ground_truth must be nonempty and have equal length")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("ground_truth must contain only labels 0 and 1")
    if binary_values and not np.all((scores == 0) | (scores == 1)):
        raise ValueError("predictions must contain only binary decisions 0 and 1")
    if probabilities and not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return scores.astype(np.float64), labels.astype(np.int64)


def binomial_rate_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Return an exact Clopper-Pearson interval for independent Bernoulli trials.

    Args:
        successes: Number of observed events, between zero and trials.
        trials: Positive integer number of independent opportunities.
        confidence_level: Confidence level strictly between zero and one.
        alternative: 'two-sided', 'greater' (lower bound), or 'less' (upper bound).

    Returns:
        Lower and upper probability bounds. This interval does not account for
        dependence between radio frames or variation between trained models.
    """
    if (isinstance(successes, (bool, np.bool_)) or isinstance(trials, (bool, np.bool_))
            or not isinstance(successes, (int, np.integer))
            or not isinstance(trials, (int, np.integer)) or trials <= 0
            or not 0 <= successes <= trials):
        raise ValueError("successes and trials must be integers with 0 <= successes <= trials and trials > 0")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    interval = stats.binomtest(int(successes), int(trials), alternative=alternative).proportion_ci(
        confidence_level=confidence_level, method="exact"
    )
    return float(interval.low), float(interval.high)


@dataclass
class ConfusionMatrix:
    """Confusion matrix representation for binary hypothesis testing.

    Attributes:
        true_negatives: Number of correct H0 decisions (idle spectrum detected as idle).
        false_positives: Number of false alarms (idle spectrum detected as occupied).
        false_negatives: Number of missed detections (occupied spectrum detected as idle).
        true_positives: Number of correct detections (occupied spectrum detected as occupied).
    """

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def total_h0(self) -> int:
        """Total actual H0 (idle) events."""
        return self.true_negatives + self.false_positives

    @property
    def total_h1(self) -> int:
        """Total actual H1 (PU active) events."""
        return self.false_negatives + self.true_positives

    @property
    def total_samples(self) -> int:
        """Total number of evaluated frames."""
        return self.total_h0 + self.total_h1


@dataclass
class DetectionMetrics:
    """Quantitative spectrum sensing evaluation metrics.

    Attributes:
        pd: Probability of Detection :math:`P_d`.
        pfa: Probability of False Alarm :math:`P_{\\text{FA}}`.
        pm: Probability of Missed Detection :math:`P_m = 1 - P_d`.
        accuracy: Overall classification accuracy :math:`(TP + TN) / Total`.
        precision: Positive Predictive Value :math:`TP / (TP + FP)`.
        f1_score: Harmonic mean of precision and recall.
        confusion_matrix: Detailed counts breakdown.
    """

    pd: float
    pfa: float
    pm: float
    accuracy: float
    precision: float
    f1_score: float
    confusion_matrix: ConfusionMatrix


@dataclass
class ROCCurveResult:
    """Container for Receiver Operating Characteristic (ROC) curve data.

    Attributes:
        pfa: 1D array of False Alarm probabilities, monotonically increasing in [0, 1].
        pd: 1D array of Detection probabilities corresponding to each Pfa entry.
        thresholds: 1D array of threshold values associated with each (Pfa, Pd) coordinate.
        auc: Area Under the ROC Curve in [0, 1]; reversed rankings may be below 0.5.
    """

    pfa: np.ndarray
    pd: np.ndarray
    thresholds: np.ndarray
    auc: float


def compute_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> ConfusionMatrix:
    r"""Compute confusion matrix counts from binary predictions and true labels.

    Args:
        predictions: 1D array of binary predictions in {0, 1}.
        ground_truth: 1D array of true labels in {0, 1}.

    Returns:
        ConfusionMatrix dataclass instance.
    """
    preds, truth = validate_binary_inputs(predictions, ground_truth, binary_values=True)

    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))
    tp = int(np.sum((preds == 1) & (truth == 1)))

    return ConfusionMatrix(
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
    )


def compute_detection_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> DetectionMetrics:
    r"""Calculate all spectrum sensing performance metrics.

    .. math::
        P_d = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}, \\quad
        P_{\\text{FA}} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}, \\quad
        P_m = 1 - P_d

    Args:
        predictions: Binary predicted decisions in {0, 1}.
        ground_truth: True hypothesis labels in {0, 1}.

    Returns:
        DetectionMetrics dataclass containing all derived probabilities.
    """
    cm = compute_confusion_matrix(predictions, ground_truth)

    pd = cm.true_positives / cm.total_h1 if cm.total_h1 > 0 else float("nan")
    pfa = cm.false_positives / cm.total_h0 if cm.total_h0 > 0 else float("nan")
    pm = 1.0 - pd

    accuracy = (cm.true_positives + cm.true_negatives) / cm.total_samples if cm.total_samples > 0 else 0.0
    denom_prec = cm.true_positives + cm.false_positives
    precision = cm.true_positives / denom_prec if denom_prec > 0 else 0.0

    denom_f1 = 2 * cm.true_positives + cm.false_positives + cm.false_negatives
    f1 = (2 * cm.true_positives) / denom_f1 if denom_f1 > 0 else 0.0

    return DetectionMetrics(
        pd=pd,
        pfa=pfa,
        pm=pm,
        accuracy=accuracy,
        precision=precision,
        f1_score=f1,
        confusion_matrix=cm,
    )


def compute_empirical_roc(
    decision_scores: np.ndarray,
    ground_truth: np.ndarray,
    num_thresholds: int = 500,
) -> ROCCurveResult:
    r"""Generate empirical Receiver Operating Characteristic (ROC) curve from scores.

    Traverses every distinct score in descending order, accepting each tied
    block together. The first threshold is positive infinity (all negative).
    Trapezoidal integration then agrees with pairwise ranking and half-credit
    for tied positive/negative scores, without resampling or FPR-only sorting.

    Args:
        decision_scores: 1D array of continuous decision statistics (e.g. Energy or Soft outputs).
        ground_truth: 1D array of ground truth labels (0 for H0, 1 for H1).
        num_thresholds: Retained compatibility argument (integer >= 2). All
            distinct thresholds are returned regardless of this value so a
            plotting resolution cannot change the scientific metric.

    Returns:
        ROCCurveResult containing monotonic Pfa, Pd, thresholds, and trapezoidal AUC.
    """
    scores, labels = validate_binary_inputs(decision_scores, ground_truth)
    if (isinstance(num_thresholds, (bool, np.bool_))
            or not isinstance(num_thresholds, (int, np.integer)) or num_thresholds < 2):
        raise ValueError("num_thresholds must be an integer >= 2")
    n_h1 = int(np.sum(labels == 1))
    n_h0 = len(labels) - n_h1
    if n_h0 == 0 or n_h1 == 0:
        raise ValueError("ground_truth must contain at least one sample of both class 0 and class 1")
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    positive = labels[order] == 1
    block_ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(scores) - 1]
    tp = np.cumsum(positive, dtype=np.int64)[block_ends]
    fp = (block_ends + 1) - tp
    pfa = np.r_[0.0, fp / n_h0]
    pd = np.r_[0.0, tp / n_h1]
    return ROCCurveResult(
        pfa=pfa,
        pd=pd,
        thresholds=np.r_[np.inf, ordered_scores[block_ends]],
        auc=float(integrate.trapezoid(pd, pfa)),
    )


def compute_theoretical_roc(
    snr_db: float,
    num_samples: int = 1024,
    noise_variance: float = 1.0,
    pfa_points: int = 200,
    method: str = "exact",
) -> ROCCurveResult:
    r"""Compute theoretical energy detector ROC curve across specified SNR.

    Evaluates:
    .. math::
        P_d(P_{\text{FA}}) = 1 - F_{\chi'^2_{2L}(\lambda)}\left(F_{\chi^2_{2L}}^{-1}(1 - P_{\text{FA}})\right)

    Args:
        snr_db: Signal-to-Noise Ratio in dB.
        num_samples: Number of observation samples :math:`L`.
        noise_variance: Known noise power :math:`\sigma_w^2`.
        pfa_points: Number of discrete evaluation points in :math:`P_{\text{FA}} \in [10^{-4}, 1.0]`.
        method: 'exact' or 'clt'.

    Returns:
        ROCCurveResult containing theoretical curve and integrated theoretical AUC.
    """
    # Use logarithmic spacing at low PFA combined with linear spacing
    pfa_log = np.logspace(-4, -0.001, pfa_points)
    pfa_vals = np.sort(np.unique(np.concatenate([[0.0], pfa_log, [1.0]])))

    pd_vals = []
    for p in pfa_vals:
        if p <= 0.0:
            pd_vals.append(0.0)
        elif p >= 1.0:
            pd_vals.append(1.0)
        else:
            val = analytical_pd(
                pfa=float(p),
                snr_db=snr_db,
                num_samples=num_samples,
                noise_variance=noise_variance,
                method=method,  # type: ignore
            )
            pd_vals.append(val)

    pd_arr = np.clip(np.array(pd_vals), 0.0, 1.0)
    auc = float(integrate.trapezoid(pd_arr, pfa_vals))

    return ROCCurveResult(
        pfa=pfa_vals,
        pd=pd_arr,
        thresholds=np.array([]),  # Implicit thresholds
        auc=auc,
    )
