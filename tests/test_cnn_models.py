"""Unit Tests for 1D-CNN Spectrum Sensing Architectures.

Verifies exact parameter counts matching Tables 1, 3, and 4 in Abdelbaset et al.
(Sensors 2024), layer dimensions, forward inference shapes, and model factory.
"""

import pytest
import numpy as np

from src.models import (
    BaselineCNN,
    BatchNormCNN,
    GAPCNN,
    get_model,
)


class TestBaselineCNN:
    """Test Baseline 1D-CNN architecture (Table 1 from paper)."""

    def test_exact_parameter_count(self) -> None:
        """Baseline CNN must have exactly 2,087,457 parameters."""
        model = BaselineCNN()
        counts = model.count_parameters()

        assert counts["total_params"] == 2087457
        assert counts["trainable_params"] == 2087457
        assert counts["non_trainable_params"] == 0

    def test_layer_output_shapes(self) -> None:
        """Verify layer sequence and spatial dimensions."""
        model = BaselineCNN()
        layer_shapes = {l.name: l.output_shape for l in model.layers}

        assert layer_shapes["conv1d_1"] == (1022, 32)
        assert layer_shapes["max_pooling1d_1"] == (511, 32)
        assert layer_shapes["conv1d_2"] == (509, 64)
        assert layer_shapes["max_pooling1d_2"] == (254, 64)
        assert layer_shapes["flatten"] == (16256,)
        assert layer_shapes["dense_1"] == (128,)
        assert layer_shapes["dense_2"] == (1,)

    def test_forward_inference(self) -> None:
        """Test forward prediction on single and batch frames."""
        model = BaselineCNN()
        # Single frame
        x_single = np.random.randn(1024, 2).astype(np.float32)
        prob = model.predict_proba(x_single)
        assert 0.0 <= float(prob) <= 1.0

        # Batch of 8 frames
        x_batch = np.random.randn(8, 1024, 2).astype(np.float32)
        probs = model.predict_proba(x_batch)
        assert probs.shape == (8,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        # Binary decision
        preds = model.predict(x_batch, threshold=0.5)
        assert preds.shape == (8,)
        assert set(np.unique(preds)).issubset({0, 1})


class TestBatchNormCNN:
    """Test BatchNorm 1D-CNN architecture (Table 3 from paper)."""

    def test_exact_parameter_count(self) -> None:
        """BatchNorm CNN must have exactly 23,009 parameters (22,817 trainable, 192 non-trainable)."""
        model = BatchNormCNN()
        counts = model.count_parameters()

        assert counts["total_params"] == 23009
        assert counts["trainable_params"] == 22817
        assert counts["non_trainable_params"] == 192

    def test_layer_types(self) -> None:
        """Model must contain BatchNormalization and Dropout layers."""
        model = BatchNormCNN()
        layer_types = [l.layer_type for l in model.layers]

        assert "BatchNormalization" in layer_types
        assert "Dropout" in layer_types
        assert layer_types.count("BatchNormalization") == 2

    def test_forward_inference(self) -> None:
        """Test forward prediction on batch frames."""
        model = BatchNormCNN()
        x_batch = np.random.randn(4, 1024, 2).astype(np.float32)
        probs = model.predict_proba(x_batch)

        assert probs.shape == (4,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))


class TestGAPCNN:
    """Test Global Average Pooling 1D-CNN architecture (Table 4 from paper)."""

    def test_exact_parameter_count(self) -> None:
        """GAP CNN must have exactly 14,881 parameters."""
        model = GAPCNN()
        counts = model.count_parameters()

        assert counts["total_params"] == 14881
        assert counts["trainable_params"] == 14881
        assert counts["non_trainable_params"] == 0

    def test_gap_pooling_shape(self) -> None:
        """GlobalAveragePooling1D must collapse 254 time steps into 64 channel features."""
        model = GAPCNN()
        gap_layer = [l for l in model.layers if l.layer_type == "GlobalAveragePooling1D"][0]

        assert gap_layer.input_shape == (254, 64)
        assert gap_layer.output_shape == (64,)
        assert gap_layer.total_params == 0

    def test_forward_inference(self) -> None:
        """Test forward prediction on batch frames."""
        model = GAPCNN()
        x_batch = np.random.randn(6, 1024, 2).astype(np.float32)
        probs = model.predict_proba(x_batch)

        assert probs.shape == (6,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))


class TestModelFactory:
    """Test get_model factory instantiations."""

    def test_factory_creation(self) -> None:
        m_base = get_model("baseline")
        assert isinstance(m_base, BaselineCNN)

        m_bn = get_model("batchnorm")
        assert isinstance(m_bn, BatchNormCNN)

        m_gap = get_model("gap")
        assert isinstance(m_gap, GAPCNN)

    def test_unknown_model_raises_error(self) -> None:
        with pytest.raises(ValueError):
            get_model("transformer_v9")

    def test_summary_generation(self) -> None:
        """Summary string must be formatted and list all layers."""
        model = GAPCNN()
        s = model.summary()
        assert "14,881" in s
        assert "global_average_pooling1d" in s
