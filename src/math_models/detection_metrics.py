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

from .energy_detection import analytical_pd


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
        auc: Area Under the ROC Curve :math:`\\in [0.5, 1.0]`.
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
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != ground_truth ({len(ground_truth)})"
        )

    preds = predictions.astype(int)
    truth = ground_truth.astype(int)

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

    pd = cm.true_positives / cm.total_h1 if cm.total_h1 > 0 else 0.0
    pfa = cm.false_positives / cm.total_h0 if cm.total_h0 > 0 else 0.0
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

    Sweeps decision threshold :math:`\gamma` from below minimum score to above maximum score.
    Computes :math:`(P_{\text{FA}}(\gamma), P_d(\gamma))` at each point.

    Args:
        decision_scores: 1D array of continuous decision statistics (e.g. Energy or Soft outputs).
        ground_truth: 1D array of ground truth labels (0 for H0, 1 for H1).
        num_thresholds: Number of discrete evaluation thresholds across score dynamic range.

    Returns:
        ROCCurveResult containing monotonic Pfa, Pd, thresholds, and trapezoidal AUC.
    """
    if len(decision_scores) != len(ground_truth):
        raise ValueError("decision_scores and ground_truth must have equal length")

    scores = np.asarray(decision_scores, dtype=float)
    labels = np.asarray(ground_truth, dtype=int)

    h0_scores = scores[labels == 0]
    h1_scores = scores[labels == 1]

    if len(h0_scores) == 0 or len(h1_scores) == 0:
        raise ValueError("ground_truth must contain at least one sample of both class 0 and class 1")

    # Generate thresholds spanning dynamic range
    min_score = np.min(scores)
    max_score = np.max(scores)
    margin = (max_score - min_score) * 0.02
    thresholds = np.linspace(min_score - margin, max_score + margin, num_thresholds)

    # Sort thresholds in descending order so that Pfa increases from 0 to 1
    sorted_thresh = np.sort(thresholds)[::-1]

    pfa_list = []
    pd_list = []

    n_h0 = float(len(h0_scores))
    n_h1 = float(len(h1_scores))

    for th in sorted_thresh:
        pfa_val = np.sum(h0_scores >= th) / n_h0
        pd_val = np.sum(h1_scores >= th) / n_h1
        pfa_list.append(pfa_val)
        pd_list.append(pd_val)

    pfa_arr = np.array(pfa_list)
    pd_arr = np.array(pd_list)

    # Ensure strictly bounded in [0, 1] and monotonic
    pfa_clean = np.clip(pfa_arr, 0.0, 1.0)
    pd_clean = np.clip(pd_arr, 0.0, 1.0)

    # Prepend (0,0) and append (1,1) if not already present
    if pfa_clean[0] > 0.0 or pd_clean[0] > 0.0:
        pfa_clean = np.insert(pfa_clean, 0, 0.0)
        pd_clean = np.insert(pd_clean, 0, 0.0)
        sorted_thresh = np.insert(sorted_thresh, 0, sorted_thresh[0] + margin)

    if pfa_clean[-1] < 1.0 or pd_clean[-1] < 1.0:
        pfa_clean = np.append(pfa_clean, 1.0)
        pd_clean = np.append(pd_clean, 1.0)
        sorted_thresh = np.append(sorted_thresh, sorted_thresh[-1] - margin)

    # Sort by Pfa ascending for integration
    sort_idx = np.argsort(pfa_clean)
    pfa_sorted = pfa_clean[sort_idx]
    pd_sorted = pd_clean[sort_idx]
    thresh_sorted = sorted_thresh[sort_idx]

    # Compute AUC using trapezoidal rule
    auc = float(integrate.trapezoid(pd_sorted, pfa_sorted))

    return ROCCurveResult(
        pfa=pfa_sorted,
        pd=pd_sorted,
        thresholds=thresh_sorted,
        auc=auc,
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
