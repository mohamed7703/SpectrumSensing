"""Unit tests for Model Export and Edge Quantization (P4-006)."""

from __future__ import annotations

import os
import json
import tempfile
import pytest
import numpy as np

from src.models import (
    BaselineCNN,
    BatchNormCNN,
    GAPCNN,
    BiGRUModel,
    ModelExporter,
    QuantizationSummary,
    EdgeBenchmarkReport,
    quantize_affine_int8,
    quantize_fp16,
)


class TestEdgeQuantization:
    """Test suite for post-training quantization algorithms."""

    def test_quantize_affine_int8(self) -> None:
        """Verify INT8 affine quantization mapping, compression, and reconstruction error."""
        w = np.linspace(-2.0, 3.0, 1000, dtype=np.float32)
        q, summary = quantize_affine_int8(w)

        assert isinstance(summary, QuantizationSummary)
        assert summary.mode == "INT8"
        assert q.dtype == np.uint8
        assert q.shape == w.shape
        assert np.all(q >= 0) and np.all(q <= 255)
        # 4 bytes -> 1 byte = exactly 4.0x compression
        assert summary.compression_ratio == 4.0
        # Reconstruction MSE should be small (< 0.001)
        assert summary.mean_squared_error < 1e-3

    def test_quantize_fp16(self) -> None:
        """Verify FP16 quantization mapping and precision."""
        w = np.random.randn(500).astype(np.float32)
        q, summary = quantize_fp16(w)

        assert isinstance(summary, QuantizationSummary)
        assert summary.mode == "FP16"
        assert q.dtype == np.float16
        # 4 bytes -> 2 bytes = exactly 2.0x compression
        assert summary.compression_ratio == 2.0
        assert summary.mean_squared_error < 1e-6


class TestModelExporter:
    """Test suite for format serialization (NPZ, TFLite, ONNX) and edge profiling."""

    @pytest.fixture
    def exporter(self, tmp_path) -> ModelExporter:
        return ModelExporter(output_dir=str(tmp_path))

    def test_export_weights_npz(self, exporter: ModelExporter) -> None:
        """Verify NPZ weight export and metadata deserialization."""
        model = GAPCNN()
        out_path = exporter.export_weights_npz(model)
        assert os.path.exists(out_path)
        assert out_path.endswith(".npz")

        # Reload and inspect
        data = np.load(out_path)
        assert "metadata_json" in data
        assert "decision_weights" in data
        meta = json.loads(str(data["metadata_json"]))
        assert meta["name"] == "GAP 1D-CNN"
        assert meta["param_counts"]["total_params"] == 14881

    def test_export_tflite_formats(self, exporter: ModelExporter) -> None:
        """Verify TFLite export across FP32, FP16, and INT8 quantization modes."""
        model = BatchNormCNN()
        for mode in ["none", "fp16", "int8"]:
            out_path = exporter.export_tflite(model, quantization=mode)
            assert os.path.exists(out_path)
            assert out_path.endswith(".tflite")
            assert os.path.getsize(out_path) > 0

    def test_export_onnx(self, exporter: ModelExporter) -> None:
        """Verify ONNX export and graph signature metadata."""
        model = BiGRUModel()
        out_path = exporter.export_onnx(model)
        assert os.path.exists(out_path)
        assert out_path.endswith(".onnx")
        assert os.path.getsize(out_path) > 0

    def test_profile_edge_inference(self, exporter: ModelExporter) -> None:
        """Verify edge inference profiling and real-time compliance (< 2.3 ms)."""
        model = GAPCNN()
        report = exporter.profile_edge_inference(model, num_iterations=50)
        assert isinstance(report, EdgeBenchmarkReport)
        assert report.mean_latency_us > 0.0
        assert report.p95_latency_us >= report.mean_latency_us
        assert report.throughput_fps > 0.0
        # Latency must satisfy 2.3 ms SDR real-time limit
        assert report.realtime_compliant is True
        assert report.mean_latency_us < 2300.0
