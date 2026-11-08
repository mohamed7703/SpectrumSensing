"""Native neural graph checks, distinct from historical feature-head tests."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from src.models import BaselineCNN, BatchNormCNN, GAPCNN
from src.training.keras_trainer import KerasSensingTrainer
from src.training.dataset_loader import SyntheticRadioMLGenerator
from src.models.model_export import ModelExporter


@pytest.mark.parametrize("factory", [BaselineCNN, BatchNormCNN, GAPCNN])
def test_declared_architecture_matches_executed_graph(factory) -> None:
    """Compare declared trainable/state counts and layer shapes with the real graph."""
    declared = factory()
    native = declared.to_keras()
    counts = declared.count_parameters()
    assert native.count_params() == counts["total_params"]
    assert sum(int(np.prod(w.shape)) for w in native.trainable_weights) == counts["trainable_params"]
    assert sum(int(np.prod(w.shape)) for w in native.non_trainable_weights) == counts["non_trainable_params"]
    for layer in native.layers:
        specification = next(s for s in declared.layers if s.name == layer.name)
        assert tuple(layer.output.shape[1:]) == specification.output_shape


def test_native_gradients_checkpoint_and_optimizer_restore(tmp_path) -> None:
    """Training changes convolutional weights; checkpoint predictions and optimizer resume agree."""
    tf.keras.utils.set_random_seed(193)
    native = GAPCNN().to_keras()
    initial = native.get_layer("conv1d_1").kernel.numpy().copy()
    data = SyntheticRadioMLGenerator(3).generate_dataset(4, ["QPSK"], [0, 10])
    validation = SyntheticRadioMLGenerator(4).generate_dataset(2, ["QPSK"], [0, 10])
    trainer = KerasSensingTrainer(native, {"name": "SGD", "learning_rate": .01, "momentum": .5, "weight_decay": .0001}, 193)
    trainer.fit(data.x, data.y_binary, validation.x, validation.y_binary,
                run_directory=tmp_path/"run", epochs=2, batch_size=4, patience=1, min_delta=0)
    assert np.linalg.norm(trainer.model.get_layer("conv1d_1").kernel.numpy() - initial) > 0
    restored = tf.keras.models.load_model(tmp_path/"run"/"best.keras", safe_mode=True)
    np.testing.assert_allclose(restored(validation.x, training=False), trainer.model(validation.x, training=False), atol=1e-7)
    assert int(restored.optimizer.iterations.numpy()) > 0
    for expected, actual in zip(restored.optimizer.variables, trainer.model.optimizer.variables):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    # Use the same deterministic gradient to exercise saved SGD state. Two
    # training=True calls would draw different dropout masks and are not equal.
    for model in [restored, trainer.model]:
        with tf.GradientTape() as tape:
            predictions = model(data.x[:4], training=False)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data.y_binary[:4, None].astype(np.float32), predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    for expected, actual in zip(restored.get_weights(), trainer.model.get_weights()):
        np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_native_trainer_rejects_historical_feature_wrapper() -> None:
    """A declared CNN with feature inference must never silently enter neural training."""
    with pytest.raises(TypeError, match="native"):
        KerasSensingTrainer(GAPCNN(), {"name": "SGD"}, 1)


def test_int8_requires_representative_training_frames(tmp_path) -> None:
    """An INT8 label without integer calibration is rejected without an artifact."""
    with pytest.raises(ValueError, match="representative"):
        ModelExporter(str(tmp_path)).export_tflite(GAPCNN().to_keras(), "int8")
    assert not list(tmp_path.iterdir())
