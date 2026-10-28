"""Unit tests for Bi-GRU Recurrent Temporal Architecture (P4-003)."""

from __future__ import annotations

import os
import yaml
import pytest
import numpy as np

from src.models.bigru_model import BiGRUModel
from src.models import get_model


class TestBiGRUArchitecture:
    """Test suite for BiGRUModel architecture, parameter counts, and inference."""

    @pytest.fixture
    def model(self) -> BiGRUModel:
        return BiGRUModel(input_shape=(1024, 2), gru_units=128, dense_units=64, dropout_rate=0.5)

    def test_parameter_counts(self, model: BiGRUModel) -> None:
        """Verify exact parameter counts of Bi-GRU network."""
        counts = model.count_parameters()
        # Bi-GRU: 2 * (2*3*128 + 128*3*128 + 2*3*128) = 2 * (768 + 49152 + 768) = 101,376
        # Dense1: 256 * 64 + 64 = 16,448
        # Output: 64 * 1 + 1 = 65
        # Total = 117,889
        assert counts["total_params"] == 117889
        assert counts["trainable_params"] == 117889
        assert counts["non_trainable_params"] == 0

    def test_layer_specifications(self, model: BiGRUModel) -> None:
        """Verify individual layer specs, dimensions and names."""
        layers = model.layers
        assert len(layers) == 4

        # Recurrent layer
        assert layers[0].name == "bidirectional_gru"
        assert layers[0].layer_type == "Bidirectional(GRU)"
        assert layers[0].input_shape == (1024, 2)
        assert layers[0].output_shape == (256,)
        assert layers[0].trainable_params == 101376

        # Projection Dense
        assert layers[1].name == "dense_projection"
        assert layers[1].input_shape == (256,)
        assert layers[1].output_shape == (64,)
        assert layers[1].trainable_params == 16448

        # Dropout
        assert layers[2].name == "dropout"
        assert layers[2].input_shape == (64,)
        assert layers[2].output_shape == (64,)
        assert layers[2].trainable_params == 0

        # Output Dense
        assert layers[3].name == "dense_output"
        assert layers[3].input_shape == (64,)
        assert layers[3].output_shape == (1,)
        assert layers[3].trainable_params == 65

    def test_factory_instantiation(self) -> None:
        """Test instantiating model through get_model() factory."""
        for name in ["bigru", "Bi-GRU", "gru", "recurrent"]:
            m = get_model(name)
            assert isinstance(m, BiGRUModel)
            assert m.count_parameters()["total_params"] == 117889

    def test_predict_proba_shapes_and_bounds(self, model: BiGRUModel) -> None:
        """Verify inference output shapes and valid probability range [0, 1]."""
        batch = np.random.randn(8, 1024, 2).astype(np.float32)
        probs = model.predict_proba(batch)
        assert probs.shape == (8,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        # Single frame
        single = np.random.randn(1024, 2).astype(np.float32)
        prob_single = model.predict_proba(single)
        assert isinstance(prob_single, (float, np.floating))
        assert 0.0 <= prob_single <= 1.0

    def test_predict_binary_decisions(self, model: BiGRUModel) -> None:
        """Verify binary hypothesis decisions (0 or 1)."""
        batch = np.random.randn(5, 1024, 2).astype(np.float32)
        preds = model.predict(batch, threshold=0.5)
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1})

    def test_recurrent_step_mechanics(self, model: BiGRUModel) -> None:
        """Verify mathematical GRU step execution."""
        x_t = np.array([0.5, -0.5], dtype=np.float32)
        h_prev = np.zeros(128, dtype=np.float32)
        weights = {
            "W_x": np.ones((2, 384), dtype=np.float32) * 0.01,
            "W_h": np.ones((128, 384), dtype=np.float32) * 0.01,
            "b_in": np.zeros(384, dtype=np.float32),
            "b_rec": np.zeros(384, dtype=np.float32),
        }
        h_next = model.recurrent_step(x_t, h_prev, weights)
        assert h_next.shape == (128,)
        assert np.all(np.isfinite(h_next))
        # Hidden state values should be within [-1, 1] due to tanh candidate activation
        assert np.all(h_next >= -1.0) and np.all(h_next <= 1.0)

    def test_forward_recurrent_sequence(self, model: BiGRUModel) -> None:
        """Verify sequence propagation through forward and backward GRU paths."""
        seq = np.random.randn(1024, 2).astype(np.float32)
        merged_state = model.forward_recurrent_sequence(seq, subsample_step=8)
        assert merged_state.shape == (256,)
        assert np.all(np.isfinite(merged_state))

    def test_summary_formatting(self, model: BiGRUModel) -> None:
        """Verify model summary formatting string."""
        summary = model.summary()
        assert "Bi-GRU Temporal" in summary
        assert "117,889" in summary
        assert "bidirectional_gru" in summary

    def test_config_consistency(self) -> None:
        """Verify model matches configurations in model_config.yaml."""
        config_path = os.path.join("configs", "model_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        bigru_cfg = cfg["models"]["bigru"]
        assert bigru_cfg["target_params"] == 117889
        assert bigru_cfg["gru_units"] == 128
        assert bigru_cfg["dense_units"] == 64
