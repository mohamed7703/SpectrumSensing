r"""Training Coordinator and Optimization Pipeline for Spectrum Sensing Models.

Implements SGD optimization (lr=0.001, momentum=0, weight_decay=1e-4), Binary Cross-Entropy
loss, early stopping, checkpointing, and comprehensive metric evaluation (Accuracy, Pd, Pfa, AUC)
matching Abdelbaset et al. (Sensors 2024).

Mathematical Formulation:
    1. Parameter Update (SGD with Weight Decay):
       $$\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t) + \lambda \boldsymbol{\theta}_t$$
       $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_{t+1}$$
       where $\eta = 10^{-3}$, $\mu = 0$, $\lambda = 10^{-4}$.

    2. Sensing Metrics:
       $$\text{Accuracy} = \frac{TP + TN}{N}$$
       $$P_d = \frac{TP}{TP + FN}, \quad P_{fa} = \frac{FP}{FP + TN}$$
       $$\text{AUC} = \int_{0}^1 P_d(P_{fa}) \, dP_{fa}$$
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import yaml
import numpy as np

from src.models.base_model import BaseSpectrumClassifier
from src.training.loss_functions import BinaryCrossEntropyLoss


@dataclass
class EvaluationMetrics:
    """Detection and classification metrics for cognitive spectrum sensing."""
    loss: float
    accuracy: float
    detection_probability: float  # Pd (Recall)
    false_alarm_rate: float       # Pfa
    precision: float
    f1_score: float
    auc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "pd": self.detection_probability,
            "pfa": self.false_alarm_rate,
            "precision": self.precision,
            "f1_score": self.f1_score,
            "auc": self.auc,
        }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    loss_fn: Optional[BinaryCrossEntropyLoss] = None,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute comprehensive cognitive radio detection metrics.

    Args:
        y_true: Ground truth binary labels (0 for H0, 1 for H1).
        y_pred_proba: Predicted detection probabilities in [0.0, 1.0].
        loss_fn: Loss function instance (defaults to BCE).
        threshold: Decision boundary probability (default 0.5).

    Returns:
        EvaluationMetrics dataclass instance.
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64).ravel()

    if loss_fn is None:
        loss_fn = BinaryCrossEntropyLoss()

    loss = loss_fn(y_true, y_pred_proba)
    decisions = (y_pred_proba >= threshold).astype(np.int32)

    tp = np.sum((decisions == 1) & (y_true == 1))
    tn = np.sum((decisions == 0) & (y_true == 0))
    fp = np.sum((decisions == 1) & (y_true == 0))
    fn = np.sum((decisions == 0) & (y_true == 1))

    total = len(y_true)
    accuracy = float((tp + tn) / max(total, 1))

    # Detection metrics
    pd = float(tp / max(tp + fn, 1))
    pfa = float(fp / max(fp + tn, 1))
    precision = float(tp / max(tp + fp, 1))
    f1 = float(2 * precision * pd / max(precision + pd, 1e-12))

    # Fast AUC calculation via trapezoidal rule over sorted thresholds
    # AUC is rank-statistic P(Y1 > Y0)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos > 0 and n_neg > 0:
        pos_scores = y_pred_proba[pos_mask]
        neg_scores = y_pred_proba[neg_mask]
        # Exact Wilcoxon-Mann-Whitney U statistic: U / (n_pos * n_neg)
        # Vectorized comparison:
        u_stat = np.sum(pos_scores[:, None] > neg_scores[None, :]) + 0.5 * np.sum(pos_scores[:, None] == neg_scores[None, :])
        auc = float(u_stat / (n_pos * n_neg))
    else:
        auc = 0.5

    return EvaluationMetrics(
        loss=loss,
        accuracy=accuracy,
        detection_probability=pd,
        false_alarm_rate=pfa,
        precision=precision,
        f1_score=f1,
        auc=auc,
    )


class SGDOptimizer:
    """Stochastic Gradient Descent optimizer with weight decay and momentum."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        momentum: float = 0.0,
        weight_decay: float = 0.0001,
    ) -> None:
        r"""Initialize SGD optimizer.

        Args:
            learning_rate: Step size $\eta$ (default 0.001 per paper).
            momentum: Momentum factor $\mu$ (default 0.0).
            weight_decay: L2 parameter regularizer $\lambda$ (default 1e-4).
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities: Dict[str, np.ndarray] = {}

    def update(self, param_name: str, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update a parameter array with SGD rule."""
        if param_name not in self.velocities:
            self.velocities[param_name] = np.zeros_like(params)

        # Gradient with weight decay
        g = grads + self.weight_decay * params
        v = self.momentum * self.velocities[param_name] + g
        self.velocities[param_name] = v

        return params - self.learning_rate * v


class EarlyStopping:
    """Early stopping callback to prevent overfitting and minimize compute."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 6,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ) -> None:
        """Initialize early stopping.

        Args:
            monitor: Metric name to monitor ('val_loss' or 'val_auc').
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change in monitored quantity to qualify as improvement.
            restore_best_weights: Whether to restore model weights from best epoch.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.wait_epochs = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights: Optional[Any] = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: Optional[Any] = None) -> bool:
        """Check whether training should stop after an epoch."""
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return False

        is_loss = "loss" in self.monitor
        # Improvement condition
        if self.best_score is None:
            self.best_score = current_score
            self.wait_epochs = 0
            return False

        if is_loss:
            improved = (self.best_score - current_score) > self.min_delta
        else:
            improved = (current_score - self.best_score) > self.min_delta

        if improved:
            self.best_score = current_score
            self.wait_epochs = 0
        else:
            self.wait_epochs += 1
            if self.wait_epochs >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                return True

        return False


class ModelCheckpoint:
    """Checkpoint callback saving best model states during training."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
    ) -> None:
        """Initialize checkpoint callback."""
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score: Optional[float] = None
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: BaseSpectrumClassifier) -> bool:
        """Save model checkpoint if performance improved."""
        score = metrics.get(self.monitor)
        if score is None:
            return False

        is_loss = "loss" in self.monitor
        improved = (
            self.best_score is None
            or (is_loss and score < self.best_score)
            or (not is_loss and score > self.best_score)
        )

        if improved:
            self.best_score = score
            # Save checkpoint metadata file
            meta_path = f"{self.filepath}_best_epoch_{epoch}.txt"
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"Model: {model.name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"{self.monitor}: {score:.6f}\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
            return True
        return False


class TrainingHistory:
    """Container recording training and validation metrics across epochs."""

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_pd": [],
            "val_pfa": [],
            "val_auc": [],
        }

    def record(self, epoch_metrics: Dict[str, float]) -> None:
        """Record dictionary of epoch metrics into history sequences."""
        for k, v in epoch_metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

    def __getitem__(self, key: str) -> List[float]:
        return self.history[key]


class SpectrumSensingTrainer:
    """High-level Training Coordinator for Spectrum Sensing Deep Learning Models."""

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> None:
        """Initialize training coordinator.

        Args:
            config_path: Path to YAML training configuration.
            learning_rate: Override learning rate (default 0.001 from config).
            weight_decay: Override weight decay (default 0.0001 from config).
        """
        self.config = self._load_config(config_path)
        t_cfg = self.config.get("training", {})
        opt_cfg = t_cfg.get("optimizer", {})

        lr = learning_rate or opt_cfg.get("learning_rate", 0.001)
        wd = weight_decay or opt_cfg.get("weight_decay", 0.0001)
        mom = opt_cfg.get("momentum", 0.0)

        self.loss_fn = BinaryCrossEntropyLoss(weight_decay=wd)
        self.optimizer = SGDOptimizer(learning_rate=lr, momentum=mom, weight_decay=wd)

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def fit(
        self,
        model: BaseSpectrumClassifier,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 10,
        batch_size: int = 128,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train spectrum classifier over given dataset.

        Args:
            model: Spectrum sensing classifier instance.
            train_data: Tuple of (X_train, y_train).
            val_data: Tuple of (X_val, y_val).
            epochs: Total epochs to train.
            batch_size: Batch size per gradient step.
            early_stopping: Optional early stopping callback.
            checkpoint: Optional model checkpoint callback.
            verbose: Print epoch-by-epoch training progress.

        Returns:
            TrainingHistory instance with recorded metrics.
        """
        x_train, y_train = train_data
        x_val, y_val = val_data
        n_train = len(x_train)

        history = TrainingHistory()

        if verbose:
            print(f"\nStarting training for '{model.name}': {epochs} epochs, batch size {batch_size}")
            print(f"Train samples: {n_train}, Validation samples: {len(x_val)}")

        for epoch in range(1, epochs + 1):
            t_start = time.perf_counter()

            # Shuffle training dataset each epoch
            indices = np.random.permutation(n_train)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch processing with SGD gradient updates
            train_preds = []
            for b_idx in range(0, n_train, batch_size):
                b_x = x_shuffled[b_idx: b_idx + batch_size]
                b_y = y_shuffled[b_idx: b_idx + batch_size]

                # Check if model has trainable decision head
                if hasattr(model, "extract_features") and hasattr(model, "weights") and hasattr(model, "bias"):
                    feats = model.extract_features(b_x)
                    logits = np.dot(feats, model.weights) + model.bias
                    b_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

                    # Analytical gradient of BCE with respect to logits: (p - y) / B
                    err = (b_pred - b_y) / max(len(b_y), 1)
                    grad_w = np.dot(feats.T, err)
                    grad_b = np.sum(err)

                    # SGD parameter update with weight decay and momentum
                    model.weights = self.optimizer.update(f"{model.name}_w", model.weights, grad_w)
                    b_arr = np.array([model.bias], dtype=np.float32)
                    model.bias = float(self.optimizer.update(f"{model.name}_b", b_arr, np.array([grad_b], dtype=np.float32))[0])
                else:
                    b_pred = model.predict_proba(b_x)

                train_preds.append(b_pred)

            train_preds_all = np.concatenate(train_preds)
            train_metrics = compute_binary_metrics(y_shuffled, train_preds_all, self.loss_fn)

            # Validation pass
            val_preds = model.predict_proba(x_val)
            val_metrics = compute_binary_metrics(y_val, val_preds, self.loss_fn)

            epoch_time = time.perf_counter() - t_start

            epoch_record = {
                "loss": train_metrics.loss,
                "accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_pd": val_metrics.detection_probability,
                "val_pfa": val_metrics.false_alarm_rate,
                "val_auc": val_metrics.auc,
            }
            history.record(epoch_record)

            if verbose:
                print(
                    f"Epoch {epoch:2d}/{epochs:2d} [{epoch_time:4.2f}s] - "
                    f"loss: {train_metrics.loss:.4f} - acc: {train_metrics.accuracy:.4f} | "
                    f"val_loss: {val_metrics.loss:.4f} - val_acc: {val_metrics.accuracy:.4f} - "
                    f"val_Pd: {val_metrics.detection_probability:.4f} - val_AUC: {val_metrics.auc:.4f}"
                )

            # Checkpoint callback
            if checkpoint:
                checkpoint.on_epoch_end(epoch, epoch_record, model)

            # Early stopping callback
            if early_stopping and early_stopping.on_epoch_end(epoch, epoch_record, model):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring best score: {early_stopping.best_score:.4f}")
                break

        return history

    def evaluate(
        self,
        model: BaseSpectrumClassifier,
        test_data: Tuple[np.ndarray, np.ndarray],
        threshold: float = 0.5,
    ) -> EvaluationMetrics:
        """Evaluate trained model on test dataset.

        Args:
            model: Spectrum sensing classifier.
            test_data: Tuple of (X_test, y_test).
            threshold: Decision boundary probability.

        Returns:
            EvaluationMetrics object.
        """
        x_test, y_test = test_data
        preds = model.predict_proba(x_test)
        return compute_binary_metrics(y_test, preds, self.loss_fn, threshold=threshold)
