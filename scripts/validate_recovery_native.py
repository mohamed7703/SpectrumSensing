"""Exercise real CNN gradients/checkpoints/FlatBuffers on explicitly synthetic fixtures.

No dataset is downloaded. Scores from these small runs do not reproduce the
paper, validate RadioML generalization, or establish device deadlines.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models import get_model, ModelExporter
from src.training.dataset_loader import SyntheticRadioMLGenerator
from src.training.keras_trainer import KerasSensingTrainer, frame_hashes
from src.math_models.energy_detection import compute_energy_statistic
from src.math_models.threshold_calibration import calibrate_score_threshold
from src.math_models.detection_metrics import binomial_rate_interval, compute_empirical_roc


def predict(model, frames: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the existing native graph on supplied frames, using bounded batches."""
    return np.concatenate([np.asarray(model(frames[offset:offset+batch_size], training=False)).reshape(-1)
                           for offset in range(0, len(frames), batch_size)]).astype(np.float64)


def describe_scores(scores: np.ndarray, labels: np.ndarray, snrs: np.ndarray,
                    calibration: dict, config: dict) -> dict:
    """Report empirical counts, exact intervals and AUC without target-based filtering."""
    output = {"calibration": calibration, "auc": compute_empirical_roc(scores, labels).auc}
    decisions = np.asarray(scores, dtype=np.float64) >= calibration["threshold"]
    h0 = labels == 0
    fa = int(np.count_nonzero(decisions[h0]))
    output["h0"] = {"false_alarms": fa, "frames": int(h0.sum()), "pfa": fa/h0.sum(),
                    "interval": binomial_rate_interval(fa, int(h0.sum()), config["interval_confidence"])}
    output["h1_by_snr"] = []
    for snr in np.unique(snrs[labels == 1]):
        mask = (labels == 1) & (snrs == snr)
        detections = int(np.count_nonzero(decisions[mask]))
        output["h1_by_snr"].append({"snr_db": float(snr), "detections": detections,
                                    "frames": int(mask.sum()), "pd": detections/mask.sum(),
                                    "interval": binomial_rate_interval(detections, int(mask.sum()), config["interval_confidence"])})
    return output


def tflite_scores(path: str, frames: np.ndarray) -> tuple[np.ndarray, dict]:
    """Load and execute the actual FlatBuffer, respecting its quantization metadata."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=path, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scores = []
    clipped = 0
    for frame in frames:
        value = frame[None].astype(np.float32)
        if np.issubdtype(inp["dtype"], np.integer):
            scale, zero = inp["quantization"]
            raw = np.rint(value/scale + zero)
            limits = np.iinfo(inp["dtype"])
            clipped += int(np.count_nonzero((raw < limits.min) | (raw > limits.max)))
            value = np.clip(raw, limits.min, limits.max).astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], value)
        interpreter.invoke()
        score = interpreter.get_tensor(out["index"]).astype(np.float64)
        if np.issubdtype(out["dtype"], np.integer):
            scale, zero = out["quantization"]
            score = (score-zero)*scale
        scores.append(float(score.reshape(-1)[0]))
    return np.asarray(scores), {"input_dtype": np.dtype(inp["dtype"]).name,
                                "output_dtype": np.dtype(out["dtype"]).name,
                                "input_quantization": inp["quantization"], "output_quantization": out["quantization"],
                                "clipped_input_values": clipped, "total_input_values": int(frames.size)}


def main() -> None:
    """Execute the configured native smoke experiment and preserve exclusive artifacts."""
    import tensorflow as tf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_directory.exists() or args.model_directory.exists():
        parser.error("use new output and model directories; historical artifacts will not be overwritten")
    tf.config.threading.set_intra_op_parallelism_threads(config["runtime"]["intra_op_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(config["runtime"]["inter_op_threads"])
    data = config["data"]
    partitions = {}
    for name, count_key in [("train", "training_frames_per_condition"), ("validation", "validation_frames_per_condition"),
                            ("test", "test_frames_per_condition")]:
        partitions[name] = SyntheticRadioMLGenerator(data[f"{name}_seed"]).generate_dataset(
            data[count_key], data["modulations"], data["snrs_db"], data["h0_fraction"])
    generator = SyntheticRadioMLGenerator(data["calibration_seed"])
    calibration_x = np.asarray([generator.generate_single_frame("NOISE", 0.) for _ in range(data["calibration_h0_frames"])])
    hashes = {name: frame_hashes(ds.x) for name, ds in partitions.items()}
    hashes["calibration_h0"] = frame_hashes(calibration_x)
    seen = set()
    for values in hashes.values():
        if len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError("exact duplicate frames within or between partitions")
        seen.update(values)
    args.output_directory.mkdir(parents=True)
    args.model_directory.mkdir(parents=True)
    train, validation, test = [partitions[name] for name in ["train", "validation", "test"]]
    evaluation = config["evaluation"]
    def calibrate(scores: np.ndarray) -> dict:
        """Calibrate the supplied H0 scores with the fixed experiment policy."""
        return asdict(calibrate_score_threshold(scores, evaluation["target_pfa"], evaluation["calibration_confidence"]))
    report = {"scope": config["scope"], "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "python": platform.python_version(), "tensorflow": tf.__version__, "keras": tf.keras.__version__,
              "devices": [str(device) for device in tf.config.list_physical_devices()],
              "partition_sizes": {name: len(values) for name, values in hashes.items()}, "partition_frame_sha256": hashes,
              "limits": ["Small two-epoch synthetic fixture; no estimate of RadioML or paper performance",
                         "Intervals and calibration confidence are marginal, not simultaneous across models/conditions",
                         "Exact duplicate checks do not establish independence of real capture groups",
                         "No named edge hardware, end-to-end timing, or power measurement",
                         "Full checkpoints restore weights/state/optimizer; bitwise dropout RNG continuation is not asserted"],
              "models": {}}
    energy_scores = compute_energy_statistic(test.x)
    report["energy_detector"] = describe_scores(energy_scores, test.y_binary, test.snr,
                                                calibrate(compute_energy_statistic(calibration_x)), evaluation)
    raw_scores = {"labels": test.y_binary, "snr": test.snr, "energy": energy_scores}
    for name in config["models"]:
        tf.keras.utils.set_random_seed(config["model_seed"])
        native = get_model(name).to_keras()
        initial = native.get_layer("conv1d_1").kernel.numpy().copy()
        trainer = KerasSensingTrainer(native, config["training"]["optimizer"], config["model_seed"])
        schedule = config["training"]
        trainer.fit(train.x, train.y_binary, validation.x, validation.y_binary,
                    run_directory=args.model_directory/name, epochs=schedule["epochs"], batch_size=schedule["batch_size"],
                    patience=schedule["patience"], min_delta=schedule["min_delta"])
        native = trainer.model
        scores = predict(native, test.x, schedule["batch_size"])
        reference = tf.keras.models.load_model(args.model_directory/name/"best.keras", safe_mode=True)
        reload_error = float(np.max(np.abs(scores-predict(reference, test.x, schedule["batch_size"])) ))
        convolution_change = float(np.linalg.norm(native.get_layer("conv1d_1").kernel.numpy()-initial))
        if convolution_change == 0 or reload_error > 1e-6:
            raise AssertionError("native training or checkpoint parity failed")
        result = describe_scores(scores, test.y_binary, test.snr,
                                 calibrate(predict(native, calibration_x, schedule["batch_size"])), evaluation)
        result.update({"count_params": native.count_params(), "best_epoch": trainer.best_epoch,
                       "convolution_kernel_change_l2": convolution_change, "checkpoint_reload_max_error": reload_error,
                       "optimizer_iterations": int(native.optimizer.iterations.numpy()), "history": trainer.history,
                       "checkpoint": str((args.model_directory/name/"best.keras").resolve()), "exports": {}})
        exporter = ModelExporter(str(args.model_directory/name/"exports"))
        parity_count = config["export"]["parity_frames"]
        indices = np.random.default_rng(data["test_seed"]).permutation(len(test))[:parity_count]
        for mode in config["export"]["modes"]:
            path = exporter.export_tflite(native, mode, representative_data=train.x[:config["export"]["representative_frames"]])
            exported, details = tflite_scores(path, test.x[indices])
            errors = np.abs(exported-scores[indices])
            result["exports"][mode] = {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
                                      "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "runtime": details,
                                      "parity_frames": len(indices), "max_absolute_score_error": float(errors.max()),
                                      "mean_absolute_score_error": float(errors.mean()),
                                      "disagreement_at_native_threshold": int(np.count_nonzero((exported >= result["calibration"]["threshold"]) != (scores[indices] >= result["calibration"]["threshold"])))}
        report["models"][name] = result
        raw_scores[name] = scores
        print(json.dumps({"model": name, "kernel_change_l2": convolution_change, "checkpoint_error": reload_error,
                          "auc_smoke_only": result["auc"], "exports": {mode: values["bytes"] for mode, values in result["exports"].items()}}), flush=True)
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for directory in ["models", "training", "math_models"]
                              for path in sorted((ROOT/"src"/directory).glob("*.py"))}
    np.savez_compressed(args.output_directory/"scores.npz", **raw_scores)
    (args.output_directory/"record.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
