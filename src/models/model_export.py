r"""Model Export and Edge Quantization Pipeline for Cognitive Radio SDR Deployment (P4-005).

Provides post-training quantization (FP16, INT8 affine scaling), weight serialization,
TFLite FlatBuffer export, ONNX conversion, and embedded edge latency profiling
for Raspberry Pi 4/5, NVIDIA Jetson, and SDR host controllers.

Mathematical Formulation:
    1. Uniform Affine Quantization (INT8):
       For real-valued tensor $\mathbf{W} \in [\alpha, \beta]$:
       $$S = \frac{\beta - \alpha}{2^b - 1} = \frac{\max(\mathbf{W}) - \min(\mathbf{W})}{255}$$
       $$Z = \text{round}\left(-\frac{\alpha}{S}\right)$$
       $$\mathbf{Q} = \text{clip}\left(\text{round}\left(\frac{\mathbf{W}}{S}\right) + Z, 0, 255\right)$$
       $$\hat{\mathbf{W}} = S \cdot (\mathbf{Q} - Z)$$

    2. Edge Memory Footprint:
       $$M_{\text{FP32}} = 4 \times N_{\text{params}} \text{ bytes}$$
       $$M_{\text{FP16}} = 2 \times N_{\text{params}} \text{ bytes} \quad (50\% \text{ reduction})$$
       $$M_{\text{INT8}} = 1 \times N_{\text{params}} \text{ bytes} \quad (75\% \text{ reduction})$$
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from src.models.base_model import BaseSpectrumClassifier


@dataclass
class QuantizationSummary:
    """Summary of quantization compression and reconstruction precision."""
    mode: str
    original_bytes: int
    quantized_bytes: int
    compression_ratio: float
    scale: float
    zero_point: int
    mean_squared_error: float
    max_absolute_error: float


@dataclass
class EdgeBenchmarkReport:
    """Edge profiling report containing latency, memory, and throughput."""
    model_name: str
    target_format: str
    mean_latency_us: float
    p95_latency_us: float
    min_latency_us: float
    throughput_fps: float
    model_size_kb: float
    realtime_compliant: bool  # Mean latency < 2300 us


def quantize_affine_int8(weights: np.ndarray) -> Tuple[np.ndarray, QuantizationSummary]:
    """Perform uniform affine INT8 quantization on floating point weights.

    Args:
        weights: Float32 weights array.

    Returns:
        Tuple of (quantized_uint8_array, QuantizationSummary).
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be nonempty and finite")
    # The representable affine range must include zero before rounding/clipping
    # the zero point; otherwise one-sided/constant arrays saturate incorrectly.
    w_min = min(0.0, float(np.min(w)))
    w_max = max(0.0, float(np.max(w)))

    if w_min == w_max:
        scale = 1.0
        zero_point = 0
        q = np.zeros_like(w, dtype=np.uint8)
    else:
        scale = (w_max - w_min) / 255.0
        zero_point = int(np.clip(np.round(-w_min / scale), 0, 255))
        q = np.clip(np.round(w / scale) + zero_point, 0, 255).astype(np.uint8)

    # De-quantize to measure reconstruction distortion
    w_recon = scale * (q.astype(np.float32) - zero_point)
    mse = float(np.mean((w - w_recon) ** 2))
    mae = float(np.max(np.abs(w - w_recon)))

    orig_bytes = w.nbytes
    quant_bytes = q.nbytes
    comp_ratio = orig_bytes / max(quant_bytes, 1)

    summary = QuantizationSummary(
        mode="INT8",
        original_bytes=orig_bytes,
        quantized_bytes=quant_bytes,
        compression_ratio=comp_ratio,
        scale=scale,
        zero_point=zero_point,
        mean_squared_error=mse,
        max_absolute_error=mae,
    )
    return q, summary


def quantize_fp16(weights: np.ndarray) -> Tuple[np.ndarray, QuantizationSummary]:
    """Perform half-precision float16 quantization.

    Args:
        weights: Float32 weights array.

    Returns:
        Tuple of (float16_array, QuantizationSummary).
    """
    w = np.asarray(weights, dtype=np.float32)
    w_f16 = w.astype(np.float16)
    w_recon = w_f16.astype(np.float32)

    mse = float(np.mean((w - w_recon) ** 2))
    mae = float(np.max(np.abs(w - w_recon)))

    orig_bytes = w.nbytes
    quant_bytes = w_f16.nbytes
    comp_ratio = orig_bytes / max(quant_bytes, 1)

    summary = QuantizationSummary(
        mode="FP16",
        original_bytes=orig_bytes,
        quantized_bytes=quant_bytes,
        compression_ratio=comp_ratio,
        scale=1.0,
        zero_point=0,
        mean_squared_error=mse,
        max_absolute_error=mae,
    )
    return w_f16, summary


class ModelExporter:
    """Model export coordinator for Edge SDR deployment."""

    def __init__(self, output_dir: str = "saved_models/exported") -> None:
        """Initialize model exporter."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_weights_npz(
        self,
        model: BaseSpectrumClassifier,
        filename: Optional[str] = None,
    ) -> str:
        """Export model parameters, layer specs, and decision weights to NumPy NPZ format.

        Args:
            model: Spectrum sensing classifier.
            filename: Target filename (.npz).

        Returns:
            Path to exported file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = filename or f"{model.name.lower().replace(' ', '_')}_weights.npz"
        out_path = os.path.join(self.output_dir, base_name)

        # Collect metadata
        metadata = {
            "name": model.name,
            "inference_backend": model.inference_backend,
            "exported_state": "four-feature logistic head only; neural architecture is metadata",
            "input_shape": list(model.input_shape),
            "param_counts": model.count_parameters(),
            "layers": [
                {
                    "name": l.name,
                    "type": l.layer_type,
                    "input_shape": list(l.input_shape),
                    "output_shape": list(l.output_shape),
                    "trainable_params": l.trainable_params,
                }
                for l in model.layers
            ],
        }

        weights_dict = {
            "metadata_json": np.array(json.dumps(metadata)),
            "decision_weights": model.weights,
            "decision_bias": np.array([model.bias], dtype=np.float32),
        }

        np.savez_compressed(out_path, **weights_dict)
        return out_path

    def export_tflite(
        self,
        model: Any,
        quantization: str = "none",
        filename: Optional[str] = None,
        representative_data: Optional[np.ndarray] = None,
    ) -> str:
        """Export model to TensorFlow Lite FlatBuffer format (.tflite).

        Args:
            model: Existing native Keras graph whose current weights are exported.
            quantization: Quantization mode ('none', 'fp16', 'int8').
            filename: Target filename.
            representative_data: Training-only float32 I/Q calibration frames,
                required for full integer input/operations/output quantization.

        Returns:
            Path to exported .tflite file.
        """
        if isinstance(model, BaseSpectrumClassifier):
            raise TypeError("a native trained Keras graph is required; feature wrappers cannot be exported as neural models")
        import tensorflow as tf
        if not isinstance(model, tf.keras.Model):
            raise TypeError("native tf.keras.Model required")
        quantization = quantization.lower()
        if quantization not in ("none", "fp16", "int8"):
            raise ValueError("quantization must be none, fp16 or int8")
        os.makedirs(self.output_dir, exist_ok=True)
        suffix = f"_{quantization.lower()}" if quantization != "none" else ""
        base_name = filename or f"{model.name.lower().replace(' ', '_')}{suffix}.tflite"
        out_path = os.path.join(self.output_dir, base_name)

        if os.path.exists(out_path):
            raise FileExistsError(out_path)
        # Trace the existing trained state through the public ConcreteFunction
        # API. This avoids the legacy _get_save_spec path in some Keras/TF pairs.
        module = tf.Module()
        module.model = model
        @tf.function(input_signature=[tf.TensorSpec([1, *model.input_shape[1:]], tf.float32, name="iq_input")])
        def inference(x: Any) -> Any:
            """Execute this trained graph in inference mode for TFLite conversion."""
            return module.model(x, training=False)
        module.inference = inference
        converter = tf.lite.TFLiteConverter.from_concrete_functions([inference.get_concrete_function()], module)
        if quantization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            if representative_data is None:
                raise ValueError("full INT8 export requires representative training data")
            representative_data = np.asarray(representative_data, dtype=np.float32)
            if (len(representative_data) == 0 or representative_data.shape[1:] != model.input_shape[1:]
                    or not np.all(np.isfinite(representative_data))):
                raise ValueError("invalid representative I/Q frames")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: ([frame[None]] for frame in representative_data)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        if quantization == "int8" and (interpreter.get_input_details()[0]["dtype"] != np.int8
                                       or interpreter.get_output_details()[0]["dtype"] != np.int8):
            raise ValueError("converter did not produce integer input/output")
        with open(out_path, "xb") as stream:
            stream.write(tflite_model)
        return out_path

    def export_onnx(
        self,
        model: Any,
        filename: Optional[str] = None,
    ) -> str:
        """Export model to Open Neural Network Exchange (ONNX) format (.onnx).

        Args:
            model: Spectrum sensing classifier.
            filename: Target filename.

        Returns:
            Path to exported .onnx file.
        """
        if isinstance(model, BaseSpectrumClassifier):
            raise TypeError("a native trained Keras graph is required; feature wrappers cannot be exported as neural models")
        import tensorflow as tf
        if not isinstance(model, tf.keras.Model):
            raise TypeError("native tf.keras.Model required")
        import tf2onnx
        import onnx
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = filename or f"{model.name.lower().replace(' ', '_')}.onnx"
        out_path = os.path.join(self.output_dir, base_name)

        if os.path.exists(out_path):
            raise FileExistsError(out_path)
        signature = [tf.TensorSpec([None, *model.input_shape[1:]], tf.float32, name="iq_input")]
        @tf.function(input_signature=signature)
        def inference(x: Any) -> Any:
            """Execute the supplied graph in inference mode for ONNX tracing."""
            return model(x, training=False)
        model_proto, _ = tf2onnx.convert.from_function(inference, input_signature=signature)
        onnx.checker.check_model(model_proto)
        with open(out_path, "xb") as stream:
            stream.write(model_proto.SerializeToString())
        return out_path

    def profile_edge_inference(
        self,
        model: BaseSpectrumClassifier,
        num_iterations: int = 150,
    ) -> EdgeBenchmarkReport:
        """Profile the host four-feature NumPy path, not an SDR or neural edge runtime.

        Args:
            model: Spectrum sensing classifier.
            num_iterations: Benchmark inference iterations.

        Returns:
            EdgeBenchmarkReport dataclass.
        """
        sample_frame = np.random.randn(1024, 2).astype(np.float32)

        # Warmup
        for _ in range(15):
            _ = model.predict_proba(sample_frame)

        latencies_us = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model.predict_proba(sample_frame)
            elapsed = (time.perf_counter() - t0) * 1e6  # microseconds
            latencies_us.append(elapsed)

        lat_arr = np.array(latencies_us)
        mean_lat = float(np.mean(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))
        min_lat = float(np.min(lat_arr))
        fps = float(1e6 / max(mean_lat, 1e-6))

        size_kb = (model.weights.nbytes + np.dtype(np.float32).itemsize) / 1024.0

        return EdgeBenchmarkReport(
            model_name=model.name,
            target_format="Host NumPy four-feature logistic classifier",
            mean_latency_us=mean_lat,
            p95_latency_us=p95_lat,
            min_latency_us=min_lat,
            throughput_fps=fps,
            model_size_kb=size_kb,
            realtime_compliant=(mean_lat < 2300.0),
        )
