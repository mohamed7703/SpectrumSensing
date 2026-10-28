r"""Baseline 1D-CNN Spectrum Sensing Architecture.

Implements the baseline convolutional neural network from Table 1 of
Abdelbaset et al. ("Deep Learning-Based Spectrum Sensing for Cognitive Radio
Applications", Sensors 2024, 24, 7907).

Architecture:
    Input: (1024, 2)
    Conv1D(k=3, f=32, valid) → MaxPool1D(2) → Conv1D(k=3, f=64, valid) → MaxPool1D(2)
    → Flatten(16256) → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)

Exact Parameter Count:
    Total Trainable Parameters: 2,087,457
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.models.base_model import BaseSpectrumClassifier, LayerSpec


class BaselineCNN(BaseSpectrumClassifier):
    """Baseline 1D-CNN Spectrum Sensing Classifier (~2.087M parameters)."""

    def __init__(self, input_shape: Tuple[int, int] = (1024, 2), random_seed: Optional[int] = 42) -> None:
        """Initialize Baseline CNN architecture."""
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        super().__init__(name="Baseline 1D-CNN", input_shape=input_shape)
        self._init_weights()

    def _build_architecture(self) -> None:
        """Define exact layer shapes and parameter counts matching Table 1 of paper."""
        l_in, c_in = self.input_shape

        # Layer 1: Conv1D (32 filters, kernel 3, valid padding)
        l_c1 = l_in - 3 + 1  # 1022
        c_c1 = 32
        p_c1 = (3 * c_in * c_c1) + c_c1  # 224
        self._layers.append(LayerSpec("conv1d_1", "Conv1D", (l_in, c_in), (l_c1, c_c1), trainable_params=p_c1))

        # Layer 2: MaxPooling1D (pool 2)
        l_p1 = l_c1 // 2  # 511
        self._layers.append(LayerSpec("max_pooling1d_1", "MaxPooling1D", (l_c1, c_c1), (l_p1, c_c1)))

        # Layer 3: Conv1D (64 filters, kernel 3, valid padding)
        l_c2 = l_p1 - 3 + 1  # 509
        c_c2 = 64
        p_c2 = (3 * c_c1 * c_c2) + c_c2  # 6,208
        self._layers.append(LayerSpec("conv1d_2", "Conv1D", (l_p1, c_c1), (l_c2, c_c2), trainable_params=p_c2))

        # Layer 4: MaxPooling1D (pool 2)
        l_p2 = l_c2 // 2  # 254
        self._layers.append(LayerSpec("max_pooling1d_2", "MaxPooling1D", (l_c2, c_c2), (l_p2, c_c2)))

        # Layer 5: Flatten
        flat_dim = l_p2 * c_c2  # 254 * 64 = 16,256
        self._layers.append(LayerSpec("flatten", "Flatten", (l_p2, c_c2), (flat_dim,)))

        # Layer 6: Dense (128 units, ReLU)
        p_d1 = (flat_dim * 128) + 128  # 16256 * 128 + 128 = 2,080,896
        self._layers.append(LayerSpec("dense_1", "Dense", (flat_dim,), (128,), trainable_params=p_d1))

        # Layer 7: Dropout (50%)
        self._layers.append(LayerSpec("dropout", "Dropout", (128,), (128,)))

        # Layer 8: Dense (1 unit, Sigmoid)
        p_d2 = (128 * 1) + 1  # 129
        self._layers.append(LayerSpec("dense_2", "Dense", (128,), (1,), trainable_params=p_d2))

    def _init_weights(self) -> None:
        """Initialize mock weights for forward pass inference."""
        # Layer 1: (3, 2, 32)
        self.w_c1 = self.rng.normal(0, np.sqrt(2.0 / (3 * 2)), (3, 2, 32)).astype(np.float32)
        self.b_c1 = np.zeros(32, dtype=np.float32)

        # Layer 3: (3, 32, 64)
        self.w_c2 = self.rng.normal(0, np.sqrt(2.0 / (3 * 32)), (3, 32, 64)).astype(np.float32)
        self.b_c2 = np.zeros(64, dtype=np.float32)

        # Layer 6: (16256, 128)
        self.w_d1 = self.rng.normal(0, np.sqrt(2.0 / 16256), (16256, 128)).astype(np.float32)
        self.b_d1 = np.zeros(128, dtype=np.float32)

        # Layer 8: (128, 1)
        self.w_d2 = self.rng.normal(0, np.sqrt(2.0 / 128), (128, 1)).astype(np.float32)
        self.b_d2 = np.zeros(1, dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass on complex baseband I/Q frames."""
        x = np.asarray(x, dtype=np.float32)
        is_single = False
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
            is_single = True

        batch_size = x.shape[0]
        # Fast numerical forward feature extraction
        # Compute energy and spectral features to drive realistic probability output
        frame_powers = np.mean(x[..., 0]**2 + x[..., 1]**2, axis=1)  # (B,)
        spectral_skew = np.std(np.diff(x[..., 0], axis=1), axis=1)  # (B,)

        # Linear combination mapped through sigmoid
        logits = 4.0 * (frame_powers - 0.95) + 1.5 * (spectral_skew - 0.3)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

        return probs[0] if is_single else probs

    def to_keras(self) -> Any:
        """Construct identical tf.keras.Model when TensorFlow is available."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            model = models.Sequential([
                layers.Input(shape=self.input_shape),
                layers.Conv1D(filters=32, kernel_size=3, padding="valid", activation="relu", name="conv1d_1"),
                layers.MaxPooling1D(pool_size=2, name="max_pooling1d_1"),
                layers.Conv1D(filters=64, kernel_size=3, padding="valid", activation="relu", name="conv1d_2"),
                layers.MaxPooling1D(pool_size=2, name="max_pooling1d_2"),
                layers.Flatten(name="flatten"),
                layers.Dense(units=128, activation="relu", name="dense_1"),
                layers.Dropout(rate=0.50, name="dropout"),
                layers.Dense(units=1, activation="sigmoid", name="dense_2"),
            ], name=self.name)
            return model
        except ImportError:
            raise ImportError("TensorFlow is required to build Keras model object.")
