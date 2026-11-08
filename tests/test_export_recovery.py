"""Export must reject feature wrappers and require genuine integer calibration."""

import pytest
import numpy as np
from src.models import GAPCNN
from src.models.model_export import ModelExporter, quantize_affine_int8


@pytest.mark.parametrize("method", ["export_tflite", "export_onnx"])
def test_export_rejects_architecture_wrapper_without_trained_native_graph(tmp_path, method: str) -> None:
    """A wrapper cannot be rebuilt randomly or replaced with a fabricated file."""
    exporter = ModelExporter(str(tmp_path))
    with pytest.raises(TypeError, match="native"):
        getattr(exporter, method)(GAPCNN())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("values", [np.array([1., 2., 3.]), np.array([-3., -2., -1.]), np.ones(8)])
def test_affine_quantization_preserves_one_sided_and_constant_ranges(values: np.ndarray) -> None:
    """Clipped zero points must not destroy positive/negative or constant weights."""
    quantized, summary = quantize_affine_int8(values)
    reconstructed = summary.scale * (quantized.astype(np.float64)-summary.zero_point)
    assert np.max(np.abs(values-reconstructed)) <= summary.scale/2 + 1e-6
