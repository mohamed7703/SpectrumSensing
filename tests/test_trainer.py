"""Unit tests for Deep Learning Training Pipeline and Optimization (P4-004)."""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np

from src.training.loss_functions import BinaryCrossEntropyLoss
from src.training.trainer import (
    compute_binary_metrics,
    EvaluationMetrics,
    SGDOptimizer,
    EarlyStopping,
    ModelCheckpoint,
    TrainingHistory,
    SpectrumSensingTrainer,
)
from src.models import BaselineCNN, BatchNormCNN, GAPCNN, BiGRUModel


class TestLossFunctions:
    """Test suite for Binary Cross-Entropy loss and numerical stabilization."""

    def test_bce_basic_computation(self) -> None:
        """Test BCE against known analytical calculation."""
        loss_fn = BinaryCrossEntropyLoss(weight_decay=0.0)
        y_true = np.array([1, 0], dtype=np.int32)
        y_pred = np.array([0.8, 0.2], dtype=np.float64)
        # Loss = -0.5 * (ln(0.8) + ln(0.8)) = -ln(0.8) = 0.22314355
        expected = -np.log(0.8)
        assert pytest.approx(loss_fn(y_true, y_pred), rel=1e-4) == expected

    def test_bce_extreme_bounds_no_nan(self) -> None:
        """Test numerical stability with extreme probabilities 0.0 and 1.0."""
        loss_fn = BinaryCrossEntropyLoss(eps=1e-15)
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.0, 1.0, 1.0, 0.0])
        loss = loss_fn(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss > 0.0

    def test_bce_weight_decay_penalty(self) -> None:
        """Test that L2 weight decay regularizer adds (λ/2) ||W||^2."""
        wd = 0.01
        loss_fn = BinaryCrossEntropyLoss(weight_decay=wd)
        y_true = np.array([1, 0])
        y_pred = np.array([0.5, 0.5])
        weights = np.array([1.0, 2.0, -1.0])  # ||W||^2 = 1 + 4 + 1 = 6
        # Expected = ln(2) + 0.5 * 0.01 * 6 = ln(2) + 0.03
        expected = np.log(2.0) + 0.5 * wd * 6.0
        assert pytest.approx(loss_fn(y_true, y_pred, weights=weights), rel=1e-4) == expected

    def test_gradient_wrt_logits(self) -> None:
        """Verify analytical gradient calculation (p - y) / N."""
        loss_fn = BinaryCrossEntropyLoss()
        y_true = np.array([1, 0])
        y_pred = np.array([0.8, 0.1])
        grad = loss_fn.gradient_wrt_logits(y_true, y_pred)
        expected = np.array([0.8 - 1.0, 0.1 - 0.0]) / 2.0
        np.testing.assert_allclose(grad, expected)


class TestDetectionMetrics:
    """Test suite for cognitive detection metrics (Pd, Pfa, AUC, Accuracy)."""

    def test_perfect_predictions(self) -> None:
        """Verify metrics under perfect detection."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.05, 0.1, 0.9, 0.95])
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0
        assert metrics.detection_probability == 1.0
        assert metrics.false_alarm_rate == 0.0
        assert metrics.precision == 1.0
        assert metrics.auc == 1.0

    def test_imperfect_predictions(self) -> None:
        """Verify metrics with false alarms and missed detections."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.8, 0.1, 0.2, 0.9])  # 1 FP, 1 FN, 1 TN, 1 TP
        metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
        assert metrics.accuracy == 0.5
        assert metrics.detection_probability == 0.5
        assert metrics.false_alarm_rate == 0.5


class TestOptimizerAndCallbacks:
    """Test suite for SGD optimization and training callbacks."""

    def test_sgd_optimizer_step(self) -> None:
        """Verify SGD parameter update with learning rate and decay."""
        sgd = SGDOptimizer(learning_rate=0.1, momentum=0.0, weight_decay=0.01)
        params = np.array([1.0, -1.0])
        grads = np.array([0.5, -0.5])
        # g_eff = grads + wd * params = [0.5 + 0.01*1.0, -0.5 + 0.01*(-1.0)] = [0.51, -0.51]
        # p_new = params - 0.1 * g_eff = [1.0 - 0.051, -1.0 - (-0.051)] = [0.949, -0.949]
        updated = sgd.update("w1", params, grads)
        np.testing.assert_allclose(updated, np.array([0.949, -0.949]))

    def test_early_stopping_trigger(self) -> None:
        """Verify early stopping triggers after patience exhausted."""
        es = EarlyStopping(monitor="val_loss", patience=2, min_delta=1e-3)
        # Epoch 1: best = 0.5
        assert not es.on_epoch_end(1, {"val_loss": 0.5})
        # Epoch 2: 0.52 (no improvement, wait=1)
        assert not es.on_epoch_end(2, {"val_loss": 0.52})
        # Epoch 3: 0.53 (no improvement, wait=2 >= patience -> triggers stop)
        assert es.on_epoch_end(3, {"val_loss": 0.53})
        assert es.should_stop

    def test_model_checkpoint(self) -> None:
        """Verify model checkpoint saves metadata upon improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model_ckpt")
            ckpt = ModelCheckpoint(ckpt_path, monitor="val_loss")
            model = BaselineCNN()
            # Epoch 1: improves
            saved1 = ckpt.on_epoch_end(1, {"val_loss": 0.4}, model)
            assert saved1
            # Epoch 2: worsens, should not save
            saved2 = ckpt.on_epoch_end(2, {"val_loss": 0.5}, model)
            assert not saved2


class TestSpectrumSensingTrainer:
    """Test suite for high-level Trainer fit and evaluation pipeline."""

    @pytest.fixture
    def dummy_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Generate dummy synthetic I/Q dataset."""
        n_samples = 32
        x = np.random.randn(n_samples, 1024, 2).astype(np.float32)
        y = np.random.choice([0, 1], size=n_samples).astype(np.int32)
        return (x[:24], y[:24]), (x[24:], y[24:])

    def test_fit_and_history(self, dummy_data) -> None:
        """Verify complete training loop generates history logs."""
        train_data, val_data = dummy_data
        trainer = SpectrumSensingTrainer()
        model = GAPCNN()

        history = trainer.fit(
            model=model,
            train_data=train_data,
            val_data=val_data,
            epochs=3,
            batch_size=8,
            verbose=False,
        )

        assert len(history["loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_pd"]) == 3
        assert all(0.0 <= acc <= 1.0 for acc in history["val_accuracy"])

    def test_evaluate(self, dummy_data) -> None:
        """Verify trainer evaluation on held-out test data."""
        _, test_data = dummy_data
        trainer = SpectrumSensingTrainer()
        model = BiGRUModel()

        metrics = trainer.evaluate(model, test_data)
        assert isinstance(metrics, EvaluationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.auc <= 1.0
