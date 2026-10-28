"""Unit tests for Benchmark Evaluation Suite (P4-005)."""

from __future__ import annotations

import pytest
import numpy as np

from src.evaluation.evaluator import BenchmarkEvaluator, ROCResult, SNRSweepResult
from src.models import BaselineCNN, BatchNormCNN, GAPCNN, BiGRUModel


class TestBenchmarkEvaluator:
    """Test suite for ROC curve generation, SNR sweeps, and confusion matrices."""

    @pytest.fixture
    def evaluator(self) -> BenchmarkEvaluator:
        return BenchmarkEvaluator(random_seed=42)

    def test_compute_roc_perfect_separation(self, evaluator: BenchmarkEvaluator) -> None:
        """Verify ROC computation on perfectly separable distributions."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.15, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95])
        roc = evaluator.compute_roc(y_true, scores)
        assert isinstance(roc, ROCResult)
        assert pytest.approx(roc.auc, rel=1e-3) == 1.0
        assert pytest.approx(roc.eer, abs=1e-2) == 0.0
        assert np.all(roc.pfa >= 0.0) and np.all(roc.pfa <= 1.0)
        assert np.all(roc.pd >= 0.0) and np.all(roc.pd <= 1.0)

    def test_compute_confusion_matrix(self, evaluator: BenchmarkEvaluator) -> None:
        """Verify 2x2 confusion matrix counts."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.1, 0.9])  # TN=1, FP=1, FN=1, TP=1
        cm = evaluator.compute_confusion_matrix(y_true, scores, threshold=0.5)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP

    def test_model_snr_sweep(self, evaluator: BenchmarkEvaluator) -> None:
        """Verify model detection probability sweep over SNR range."""
        model = GAPCNN()
        snrs = [-10.0, 0.0, 10.0]
        res = evaluator.evaluate_model_snr_sweep(model, snrs=snrs, frames_per_snr=10, target_pfa=0.05)
        assert isinstance(res, SNRSweepResult)
        assert len(res.pd) == len(snrs)
        assert np.all(res.pd >= 0.0) and np.all(res.pd <= 1.0)
        assert np.all(res.pfa >= 0.0) and np.all(res.pfa <= 1.0)

    def test_energy_detector_snr_wall(self, evaluator: BenchmarkEvaluator) -> None:
        """Verify energy detector collapse below the SNR wall (-8.2 dB)."""
        snrs = [-16.0, -12.0, -4.0, 10.0]
        res = evaluator.evaluate_energy_detector_snr_sweep(
            snrs=snrs,
            num_samples=512,
            noise_uncertainty_db=1.0,
            target_pfa=0.05,
            trials_per_snr=40,
        )
        assert isinstance(res, SNRSweepResult)
        # Low SNR below SNR wall (-16 dB) should have very low Pd (close to Pfa ~0.05)
        assert res.pd[0] < 0.25
        # High SNR (+10 dB) should have high Pd (> 0.80)
        assert res.pd[-1] > 0.80
