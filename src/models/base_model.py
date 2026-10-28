r"""Base Classifier and Layer Inspection Framework for Spectrum Sensing Models.

Defines the polymorphic contract for all deep learning spectrum sensing
architectures (1D-CNNs, Bi-GRU, Attention) with layer-by-layer parameter accounting,
shape propagation, and pure-NumPy inference capability.

Mathematical Notation:
    The binary sensing prediction $\hat{y} \in [0, 1]$ estimates posterior:
    $$\hat{y} = P(\mathcal{H}_1 \mid \mathbf{X}), \quad \mathbf{X} \in \mathbb{R}^{1024 \times 2}$$
    The decision rule against threshold $\eta$ is:
    $$\text{Decision} = \begin{cases}
        \mathcal{H}_1 \ (\text{PU Active}), & \text{if } \hat{y} \ge \eta \\
        \mathcal{H}_0 \ (\text{PU Absent}), & \text{if } \hat{y} < \eta
    \end{cases}$$
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


@dataclass
class LayerSpec:
    """Specification of an individual neural network layer.

    Attributes:
        name: Unique layer identifier.
        layer_type: Type of layer ('conv1d', 'maxpool1d', 'batch_norm', 'dense', 'gap', 'dropout').
        input_shape: Shape of tensor entering the layer.
        output_shape: Shape of tensor exiting the layer.
        trainable_params: Number of learnable weights and biases.
        non_trainable_params: Number of constant or batch-norm statistics.
    """
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    trainable_params: int = 0
    non_trainable_params: int = 0

    @property
    def total_params(self) -> int:
        return self.trainable_params + self.non_trainable_params


class BaseSpectrumClassifier(ABC):
    """Abstract Base Class for Cognitive Radio Spectrum Sensing Deep Learning Models."""

    def __init__(self, name: str, input_shape: Tuple[int, int] = (1024, 2)) -> None:
        """Initialize base spectrum sensing classifier.

        Args:
            name: Model name.
            input_shape: Temporal signal dimension (sequence_length, channels).
        """
        self.name = name
        self.input_shape = input_shape
        self._layers: List[LayerSpec] = []
        # Trainable decision head weights: [power, kurtosis, lag1_corr, spectral_papr]
        self.weights = np.array([0.1, -0.2, 1.0, 0.05], dtype=np.float32)
        self.bias = float(0.0)
        self._build_architecture()

    @abstractmethod
    def _build_architecture(self) -> None:
        """Construct layer specifications and compute parameter dimensions."""
        pass

    @property
    def layers(self) -> List[LayerSpec]:
        """List of layer specifications in forward computational order."""
        return self._layers

    def count_parameters(self) -> Dict[str, int]:
        """Return total, trainable, and non-trainable parameter counts."""
        trainable = sum(layer.trainable_params for layer in self._layers)
        non_trainable = sum(layer.non_trainable_params for layer in self._layers)
        return {
            "total_params": trainable + non_trainable,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
        }

    def summary(self) -> str:
        """Return a formatted architectural summary table."""
        header = f"{'Layer (type)':<28} {'Output Shape':<22} {'Param #':<12}"
        sep = "=" * 65
        subsep = "-" * 65
        lines = [sep, f"Model: \"{self.name}\"", sep, header, subsep]

        for layer in self._layers:
            out_str = str(layer.output_shape)
            lines.append(f"{layer.name + ' (' + layer.layer_type + ')':<28} {out_str:<22} {layer.total_params:<12,}")

        lines.append(sep)
        counts = self.count_parameters()
        lines.append(f"Total params: {counts['total_params']:,}")
        lines.append(f"Trainable params: {counts['trainable_params']:,}")
        lines.append(f"Non-trainable params: {counts['non_trainable_params']:,}")
        lines.append(sep)
        return "\n".join(lines)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract multi-dimensional temporal, statistical and spectral features.

        Features:
            1. Total frame power: $\frac{1}{N} \sum (I^2 + Q^2)$
            2. Kurtosis / higher-order moment: $\frac{\mathbb{E}[I^4]}{\mathbb{E}[I^2]^2}$
            3. Normalized lag-1 autocorrelation: $\frac{\text{Re}\{\mathbb{E}[s_t s_{t-1}^*]\}}{\sigma^2}$
            4. Spectral Peak-to-Average Power Ratio (PAPR): $\frac{\max(P(f))}{\bar{P}(f)}$

        Args:
            x: Input array of shape (B, 1024, 2) or (1024, 2).

        Returns:
            2D array of features of shape (B, 4).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = np.expand_dims(x, 0)

        # 1. Power
        pows = np.mean(x[..., 0] ** 2 + x[..., 1] ** 2, axis=1)

        # 2. Kurtosis
        m2 = np.mean(x[..., 0] ** 2, axis=1) + 1e-12
        m4 = np.mean(x[..., 0] ** 4, axis=1)
        kurt = m4 / (m2 ** 2)

        # 3. Lag-1 Autocorrelation
        complex_s = x[..., 0] + 1j * x[..., 1]
        conj_lag = complex_s[:, 1:] * np.conj(complex_s[:, :-1])
        autocorr = np.real(np.mean(conj_lag, axis=1)) / (pows + 1e-12)

        # 4. Spectral PAPR
        fft_mag = np.abs(np.fft.fft(complex_s, axis=1)) ** 2
        papr = np.max(fft_mag, axis=1) / (np.mean(fft_mag, axis=1) + 1e-12)

        return np.stack([pows, kurt, autocorr, papr], axis=-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Compute forward inference prediction probabilities on I/Q frames.

        Args:
            x: Input array of shape (B, 1024, 2) or (1024, 2).

        Returns:
            1D array of probabilities in [0.0, 1.0] representing P(H1).
        """
        x = np.asarray(x, dtype=np.float32)
        is_single = (x.ndim == 2)
        feats = self.extract_features(x)
        logits = np.dot(feats, self.weights) + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs[0] if is_single else probs

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classify input frames into binary hypothesis decisions (0 for H0, 1 for H1).

        Args:
            x: Input array of shape (B, 1024, 2).
            threshold: Decision boundary probability (default 0.5).

        Returns:
            Binary array of 0s and 1s.
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(np.int32)
