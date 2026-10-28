r"""Loss Functions and Regularizers for Spectrum Sensing Deep Learning Models.

Implements Binary Cross-Entropy (BCE) loss with numerical stability safeguards and
L2 weight regularization per Abdelbaset et al. (Sensors 2024).

Mathematical Formulation:
    For ground truth binary hypothesis labels $y_i \in \{0, 1\}$ ($\mathcal{H}_0 = 0, \mathcal{H}_1 = 1$)
    and predicted probabilities $\hat{p}_i = P(\mathcal{H}_1 \mid \mathbf{X}_i) \in [0, 1]$:

    $$\mathcal{L}_{\text{BCE}}(\mathbf{y}, \hat{\mathbf{p}}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

    With L2 weight decay penalty $\lambda$:
    $$\mathcal{L}_{\text{total}}(\mathbf{y}, \hat{\mathbf{p}}, \mathbf{W}) = \mathcal{L}_{\text{BCE}}(\mathbf{y}, \hat{\mathbf{p}}) + \frac{\lambda}{2} \sum_{k} w_k^2$$

    Derivative with respect to pre-sigmoid logit $z_i$ (where $\hat{p}_i = \sigma(z_i)$):
    $$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z_i} = \hat{p}_i - y_i$$
"""

from __future__ import annotations

from typing import Optional, Union, Any
import numpy as np


class BinaryCrossEntropyLoss:
    """Binary Cross-Entropy Loss with L2 Regularization Support."""

    def __init__(
        self,
        eps: float = 1e-15,
        weight_decay: float = 0.0001,
        pos_weight: Optional[float] = None,
    ) -> None:
        r"""Initialize Binary Cross-Entropy loss.

        Args:
            eps: Epsilon value to prevent log(0) numerical instability.
            weight_decay: L2 regularization coefficient $\lambda$ (default 1e-4).
            pos_weight: Optional weighting factor for positive class ($\mathcal{H}_1$).
        """
        self.eps = eps
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute scalar BCE loss over batch.

        Args:
            y_true: Ground truth binary labels of shape (N,) or (N, 1).
            y_pred: Predicted probabilities in [0.0, 1.0] of shape (N,) or (N, 1).
            weights: Optional flat weight vector or parameter array for L2 decay.

        Returns:
            Scalar cross-entropy loss value.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Numerical stability clipping
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)

        if self.pos_weight is not None:
            # Weighted BCE
            term1 = self.pos_weight * y_true * np.log(p)
            term0 = (1.0 - y_true) * np.log(1.0 - p)
            bce = -np.mean(term1 + term0)
        else:
            bce = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

        # Add L2 penalty if weights are provided
        if weights is not None and self.weight_decay > 0.0:
            w_flat = np.asarray(weights, dtype=np.float64).ravel()
            l2_reg = 0.5 * self.weight_decay * np.sum(w_flat ** 2)
            return float(bce + l2_reg)

        return float(bce)

    def gradient_wrt_logits(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        r"""Compute analytical gradient of BCE loss with respect to pre-sigmoid logits.

        For $\hat{p}_i = \sigma(z_i)$, the gradient is $\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N}(\hat{p}_i - y_i)$.

        Args:
            y_true: Ground truth labels (N,).
            y_pred: Probabilities (N,).

        Returns:
            Array of gradients of shape (N,).
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        return (y_pred - y_true) / max(len(y_true), 1)

    def to_keras(self) -> Any:
        """Return equivalent tf.keras.losses.BinaryCrossentropy instance."""
        try:
            import tensorflow as tf
            return tf.keras.losses.BinaryCrossentropy(from_logits=False)
        except ImportError:
            raise ImportError("TensorFlow is required to instantiate Keras loss.")
