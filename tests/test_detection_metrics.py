"""Unit tests for the detection metrics and ROC analysis module.

Tests confusion matrix accounting, empirical/analytical ROC curves,
monotonicity constraints, and Area Under the Curve (AUC) integration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.math_models.detection_metrics import (
    ConfusionMatrix,
    DetectionMetrics,
    compute_confusion_matrix,
    compute_detection_metrics,
    compute_empirical_roc,
    compute_theoretical_roc,
)
from src.math_models.energy_detection import (
    EnergyDetectorConfig,
    EnergyDetector,
    compute_energy_statistic,
)
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


class TestConfusionMatrixAndMetrics:
    """Test suite for basic detection classification metrics."""

    def test_confusion_matrix_counts(self):
        """Verify exact cell counts in confusion matrix."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0])

        cm = compute_confusion_matrix(y_pred, y_true)
        assert cm.true_negatives == 2
        assert cm.false_positives == 2
        assert cm.false_negatives == 1
        assert cm.true_positives == 3
        assert cm.total_h0 == 4
        assert cm.total_h1 == 4
        assert cm.total_samples == 8

    def test_perfect_classifier_metrics(self):
        """Perfect classifier achieves Pd=1, Pfa=0, Accuracy=1, F1=1."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = compute_detection_metrics(y_pred, y_true)
        assert metrics.pd == 1.0
        assert metrics.pfa == 0.0
        assert metrics.pm == 0.0
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.f1_score == 1.0

    def test_length_mismatch_raises_error(self):
        """Arrays of differing sizes raise ValueError."""
        with pytest.raises(ValueError):
            compute_confusion_matrix(np.zeros(5), np.zeros(6))


class TestEmpiricalROC:
    """Test suite for empirical ROC curve generation."""

    def test_perfectly_separable_scores_auc(self):
        """Separable distributions yield AUC = 1.0."""
        scores = np.array([0.1, 0.2, 0.3, 0.9, 1.1, 1.2])
        labels = np.array([0, 0, 0, 1, 1, 1])

        roc = compute_empirical_roc(scores, labels, num_thresholds=50)
        assert np.isclose(roc.auc, 1.0)
        assert roc.pfa[0] == 0.0 and roc.pd[0] == 0.0
        assert roc.pfa[-1] == 1.0 and roc.pd[-1] == 1.0

    def test_random_scores_auc_near_half(self):
        """Identically distributed scores yield AUC near 0.5."""
        rng = np.random.default_rng(42)
        scores = rng.normal(0, 1, size=10_000)
        labels = rng.integers(0, 2, size=10_000)

        roc = compute_empirical_roc(scores, labels, num_thresholds=200)
        assert np.isclose(roc.auc, 0.5, atol=0.03)

    def test_single_class_raises_error(self):
        """Labels containing only one class raise ValueError."""
        with pytest.raises(ValueError):
            compute_empirical_roc(np.array([1, 2, 3]), np.array([0, 0, 0]))


class TestTheoreticalROC:
    """Test suite for theoretical energy detection ROC curves."""

    def test_high_snr_theoretical_auc_near_unity(self):
        """At SNR = 5 dB with L=1024, theoretical AUC is virtually 1.0."""
        roc = compute_theoretical_roc(snr_db=5.0, num_samples=1024)
        assert np.isclose(roc.auc, 1.0, atol=1e-3)
        assert np.all(np.diff(roc.pfa) >= 0)  # Monotonic Pfa

    def test_low_snr_theoretical_auc(self):
        """At SNR = -20 dB, detector struggles; AUC approaches 0.55."""
        roc = compute_theoretical_roc(snr_db=-20.0, num_samples=1024)
        assert 0.50 <= roc.auc <= 0.65

    def test_empirical_vs_theoretical_roc_auc_convergence(self):
        """Empirical ROC AUC on simulated frames aligns with theoretical AUC within 0.02."""
        snr_db = -10.0
        l_samples = 1024
        n_frames = 6000

        # Theoretical ROC
        theory_roc = compute_theoretical_roc(snr_db=snr_db, num_samples=l_samples)

        # Empirical simulation
        sim = HypothesisTestingSimulator(
            HypothesisConfig(num_samples=l_samples, snr_db=snr_db),
            seed=555,
        )
        x_batch, y_batch = sim.generate_batch(num_frames=n_frames, p_h1=0.5, as_iq_channels=True)
        scores = compute_energy_statistic(x_batch, normalize=False)

        emp_roc = compute_empirical_roc(scores, y_batch, num_thresholds=300)

        assert np.isclose(emp_roc.auc, theory_roc.auc, atol=0.02)
