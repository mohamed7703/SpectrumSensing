"""Native neural training with full restorable Keras checkpoints.

No synthetic fallback, feature extraction, or export stand-in is used here.
This establishes executed computation; generalization requires separate data.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.math_models.detection_metrics import validate_binary_inputs
from src.training.trainer import compute_binary_metrics


def frame_hashes(frames: np.ndarray) -> list[str]:
    """Hash each supplied float32 frame to detect exact duplicate partition entries."""
    return [hashlib.sha256(np.ascontiguousarray(frame, dtype=np.float32).tobytes()).hexdigest() for frame in frames]


class KerasSensingTrainer:
    """Train an actual Keras graph and restore the best model and optimizer state."""

    def __init__(self, model: Any, optimizer_config: dict, random_seed: int) -> None:
        """Compile the supplied native graph using explicitly configured SGD settings."""
        import tensorflow as tf
        if not isinstance(model, tf.keras.Model):
            raise TypeError("native tf.keras.Model required; architecture/feature wrappers are not neural graphs")
        if optimizer_config.get("name") != "SGD":
            raise ValueError("this recovery protocol requires explicitly configured SGD")
        self.model = model
        self.seed = random_seed
        self.optimizer_config = dict(optimizer_config)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(
            learning_rate=optimizer_config["learning_rate"], momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"]), loss="binary_crossentropy")
        self.history: list[dict] = []
        self.best_epoch: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, *, run_directory: Path, epochs: int, batch_size: int,
            patience: int, min_delta: float) -> list[dict]:
        """Train on given arrays; select by validation BCE and preserve an exclusive run.

        Args:
            x_train, y_train: Explicit training I/Q frames and binary labels.
            x_val, y_val: Independent model-selection partition, never threshold calibration.
            run_directory: New output directory; existing paths are rejected.
            epochs, batch_size: Positive training schedule values.
            patience: Positive number of validation epochs without improvement.
            min_delta: Nonnegative required validation BCE improvement.

        Exact duplicate checks do not establish independence of adjacent frames
        or common captures. Group/session provenance remains a caller requirement.
        Keras weight_decay is decoupled SGD decay, not a BCE penalty term.
        """
        import tensorflow as tf
        if any(isinstance(v, bool) or int(v) != v or v < 1 for v in [epochs, batch_size, patience]):
            raise ValueError("epochs, batch_size and patience must be positive integers")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        x_train = np.asarray(x_train, dtype=np.float32)
        x_val = np.asarray(x_val, dtype=np.float32)
        partitions = []
        for name, x, y in [("train", x_train, y_train), ("validation", x_val, y_val)]:
            _, labels = validate_binary_inputs(np.zeros(len(x)), y)
            if x.shape[1:] != self.model.input_shape[1:] or not np.all(np.isfinite(x)):
                raise ValueError(f"{name}: invalid I/Q shape or values")
            if len(np.unique(labels)) != 2:
                raise ValueError(f"{name}: both hypotheses are required")
            partitions.append(labels.astype(np.float32))
        y_train, y_val = partitions
        train_ids, val_ids = frame_hashes(x_train), frame_hashes(x_val)
        if set(train_ids) & set(val_ids):
            raise ValueError("exact duplicate frames cross training/validation partitions")
        run_directory = Path(run_directory)
        run_directory.mkdir(parents=True, exist_ok=False)
        checkpoint = run_directory / "best.keras"
        manifest = {"scope": "native computation, not paper reproduction", "backend": "tensorflow_keras",
                    "tensorflow": tf.__version__, "seed": self.seed,
                    "optimizer": self.optimizer_config, "epochs_limit": epochs, "batch_size": batch_size,
                    "patience": patience, "min_delta": min_delta, "input_shape": self.model.input_shape,
                    "train_frame_sha256": train_ids, "validation_frame_sha256": val_ids,
                    "train_labels_sha256": hashlib.sha256(y_train.tobytes()).hexdigest(),
                    "validation_labels_sha256": hashlib.sha256(y_val.tobytes()).hexdigest()}
        (run_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rng = np.random.default_rng(self.seed)
        best_loss = float("inf")
        waiting = 0
        for epoch in range(epochs):
            indices = rng.permutation(len(x_train))
            batch_losses = []
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset:offset+batch_size]
                self.model.reset_metrics()
                loss = float(self.model.train_on_batch(x_train[batch], y_train[batch]))
                if not np.isfinite(loss):
                    raise FloatingPointError("nonfinite native training loss")
                batch_losses.append((loss, len(batch)))
            probabilities = np.concatenate([np.asarray(self.model(x_val[offset:offset+batch_size], training=False)).reshape(-1)
                                            for offset in range(0, len(x_val), batch_size)])
            metrics = compute_binary_metrics(y_val, probabilities)
            row = {"epoch": epoch + 1, "loss": sum(v*n for v, n in batch_losses)/len(x_train),
                   "val_loss": metrics.loss, "val_accuracy": metrics.accuracy, "val_auc": metrics.auc}
            self.history.append(row)
            if row["val_loss"] < best_loss - min_delta:
                best_loss = row["val_loss"]
                self.best_epoch = epoch + 1
                waiting = 0
                self.model.save(checkpoint)
            else:
                waiting += 1
            if waiting >= patience:
                break
        self.model = tf.keras.models.load_model(checkpoint, compile=True, safe_mode=True)
        record = {"best_epoch": self.best_epoch, "history": self.history,
                  "restored_optimizer_iterations": int(self.model.optimizer.iterations.numpy()),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (run_directory / "training.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        return self.history
