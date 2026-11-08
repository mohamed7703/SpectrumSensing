"""Investigate INT8 threshold transfer and fixed-threshold gain drift locally."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.validate_recovery_native import predict, tflite_scores, describe_scores
from src.training.dataset_loader import SyntheticRadioMLGenerator
from src.training.keras_trainer import frame_hashes
from src.math_models.energy_detection import compute_energy_statistic
from src.math_models.threshold_calibration import calibrate_score_threshold


def main() -> None:
    """Evaluate fixed existing checkpoints under an explicit, new stress configuration."""
    import tensorflow as tf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-record", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error("output directory exists; choose a new run")
    source = json.loads(args.native_record.read_text(encoding="utf-8"))
    stress = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tf.config.threading.set_intra_op_parallelism_threads(stress["intra_op_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(stress["inter_op_threads"])
    config = source["config"]
    data = config["data"]
    evaluation = config["evaluation"]
    test = SyntheticRadioMLGenerator(data["test_seed"]).generate_dataset(
        data["test_frames_per_condition"], data["modulations"], data["snrs_db"], data["h0_fraction"])
    generator = SyntheticRadioMLGenerator(data["calibration_seed"])
    calibration_x = np.asarray([generator.generate_single_frame("NOISE", 0.) for _ in range(data["calibration_h0_frames"])])
    if (frame_hashes(test.x) != source["partition_frame_sha256"]["test"] or
            frame_hashes(calibration_x) != source["partition_frame_sha256"]["calibration_h0"]):
        raise ValueError("regenerated fixture identity differs from the frozen source run")
    args.output_directory.mkdir(parents=True)
    result = {"scope": stress["scope"], "config": stress, "source_record_sha256": hashlib.sha256(args.native_record.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "models": {},
              "limits": ["No retraining or threshold selection on test frames", "All calibration uses nominal gain only",
                         "Gain scales both signal and noise, retaining the input SNR", "Synthetic fixture results do not reproduce the paper",
                         "Counts/intervals are marginal; no significance claim across comparisons"]}
    raw = {"labels": test.y_binary, "snr": test.snr}
    def calibrate(scores: np.ndarray) -> dict:
        """Calibrate only the supplied nominal-gain H0 score vector."""
        return asdict(calibrate_score_threshold(scores, evaluation["target_pfa"], evaluation["calibration_confidence"]))
    energy_calibration = calibrate(compute_energy_statistic(calibration_x))
    result["energy_detector"] = []
    for drift in stress["receiver_gain_drift_db"]:
        scores = compute_energy_statistic(test.x * 10**(drift/20))
        result["energy_detector"].append({"gain_drift_db": drift, "metrics": describe_scores(scores, test.y_binary, test.snr, energy_calibration, evaluation)})
    with np.load(args.native_record.parent/"scores.npz", allow_pickle=False) as frozen:
        for name, previous in source["models"].items():
            native = tf.keras.models.load_model(previous["checkpoint"], safe_mode=True)
            reference = predict(native, test.x, config["training"]["batch_size"])
            if np.max(np.abs(reference-frozen[name])) > 1e-6:
                raise ValueError("checkpoint predictions differ from frozen source scores")
            records = {}
            for mode in stress["runtime_modes"]:
                if mode == "native":
                    score_function = lambda frames: predict(native, frames, config["training"]["batch_size"])
                elif mode == "int8":
                    exported = previous["exports"][mode]
                    path = Path(exported["path"])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != exported["sha256"]:
                        raise ValueError("exported artifact identity changed")
                    score_function = lambda frames: tflite_scores(str(path), frames)[0]
                else:
                    raise ValueError("unsupported runtime mode")
                calibration_scores = score_function(calibration_x)
                runtime_calibration = calibrate(calibration_scores)
                records[mode] = {"calibration_distinct_scores": len(np.unique(calibration_scores)), "conditions": []}
                for drift in stress["receiver_gain_drift_db"]:
                    scores = score_function(test.x * 10**(drift/20))
                    raw[f"{name}_{mode}_gain_{drift}"] = scores
                    observation = {"gain_drift_db": drift,
                                   "runtime_calibrated": describe_scores(scores, test.y_binary, test.snr, runtime_calibration, evaluation)}
                    if mode != "native" and drift == 0:
                        observation["native_threshold_transferred"] = describe_scores(scores, test.y_binary, test.snr, previous["calibration"], evaluation)
                        observation["decision_changes_at_native_threshold"] = int(np.count_nonzero((scores >= previous["calibration"]["threshold"]) != (reference >= previous["calibration"]["threshold"])))
                    records[mode]["conditions"].append(observation)
            result["models"][name] = records
            print(json.dumps({"model": name, "native_pfa_by_gain": [item["runtime_calibrated"]["h0"]["pfa"] for item in records["native"]["conditions"]],
                              "int8_pfa_by_gain": [item["runtime_calibrated"]["h0"]["pfa"] for item in records["int8"]["conditions"]]}), flush=True)
    np.savez_compressed(args.output_directory/"scores.npz", **raw)
    (args.output_directory/"record.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
