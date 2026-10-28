"""Deep Learning Spectrum Sensing Models Module.

Provides 1D-CNN architectures (Baseline, BatchNorm, GAP), Bi-GRU recurrent models,
and model factory functions.
"""

from src.models.base_model import BaseSpectrumClassifier, LayerSpec
from src.models.cnn_baseline import BaselineCNN
from src.models.cnn_batchnorm import BatchNormCNN
from src.models.cnn_gap import GAPCNN


def get_model(model_name: str, input_shape: tuple = (1024, 2), **kwargs) -> BaseSpectrumClassifier:
    """Instantiate a spectrum sensing model by name.

    Args:
        model_name: One of 'baseline', 'batchnorm', 'gap'.
        input_shape: Input frame shape (default 1024, 2).
        **kwargs: Additional parameters for model constructor.

    Returns:
        BaseSpectrumClassifier instance.
    """
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    if "baseline" in key:
        return BaselineCNN(input_shape=input_shape, **kwargs)
    elif "batch" in key or "bn" in key:
        return BatchNormCNN(input_shape=input_shape, **kwargs)
    elif "gap" in key or "pool" in key:
        return GAPCNN(input_shape=input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model architecture '{model_name}'. Choose from 'baseline', 'batchnorm', 'gap'.")


__all__ = [
    "BaseSpectrumClassifier",
    "LayerSpec",
    "BaselineCNN",
    "BatchNormCNN",
    "GAPCNN",
    "get_model",
]
