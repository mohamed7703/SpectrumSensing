"""Run the prespecified synthetic GAP receiver calibration benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_recovery_native import predict, tflite_scores
from src.math_models.energy_detection import compute_energy_statistic
from src.math_models.threshold_calibration import calibrate_score_threshold
from src.models import get_model
from src.training.dataset_loader import SyntheticRadioMLGenerator
from src.training.keras_trainer import KerasSensingTrainer


def frames(rng, n: int, label: str, snr: float = 0.) -> np.ndarray:
    """Draw n independent synthetic baseband frames using the supplied generator."""
    return np.stack([rng.generate_single_frame(label, snr) for _ in range(n)])


def fit_rules(cal: dict, nominal: np.ndarray, cfg: dict, budget: int, gain: float) -> dict:
    """Fit the five predeclared thresholds using only labeled calibration noise."""
    gains = np.array(cfg["gains_db"])
    nearest = int(np.argmin(np.abs(gains - gain)))

    def boundary(values: np.ndarray) -> float:
        return calibrate_score_threshold(values, cfg["target_pfa"], cfg["calibration_confidence"]).threshold

    ordinary = [boundary(cal[g]["int8"][:budget]) for g in gains]
    order_scores = np.nextafter(ordinary, -np.inf)
    slope, intercept = np.polyfit(gains, order_scores, 1)
    linear = np.nextafter(float(np.round(slope*gain + intercept, 12)), np.inf)
    return {"linear": linear, "ordinary": ordinary[nearest], "fixed": boundary(nominal[:len(gains)*budget]),
            "fp32": boundary(cal[gains[nearest]]["fp32"][:budget]),
            "energy": boundary(cal[gains[nearest]]["energy"][:budget])}


def export_int8(model, representative: np.ndarray, path: Path) -> Path:
    """Freeze trained inference variables before calibrating the integer graph."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    serving = tf.function(lambda x: model(x, training=False))
    frozen = convert_variables_to_constants_v2(serving.get_concrete_function(
        tf.TensorSpec([None, 1024, 2], tf.float32)))
    np.testing.assert_allclose(frozen(tf.constant(representative[:8]))[0].numpy(),
                               model(representative[:8], training=False).numpy(), atol=1e-6)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([frame[None]] for frame in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())
    return path


def run_block(index: int, cfg: dict, out: Path, model_dir: Path, writer: csv.DictWriter) -> dict:
    """Train one independent model and evaluate its three paired synthetic sessions."""
    import tensorflow as tf
    seed = cfg["model_seeds"][index]
    base = cfg["seed_base"] + index*10000
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    def partition(offset: int, key: str):
        return SyntheticRadioMLGenerator(base+offset).generate_dataset(cfg[key], ["QPSK"], cfg["snrs_db"], .5)
    train = partition(0, "training_frames_per_snr")
    val = partition(1, "validation_frames_per_snr")
    trainer = KerasSensingTrainer(get_model("gap").to_keras(), cfg["optimizer"], seed)
    folder = model_dir/str(seed)
    trainer.fit(train.x, train.y_binary, val.x, val.y_binary, run_directory=folder,
                epochs=cfg["epochs"], patience=cfg["epochs"], min_delta=0, batch_size=cfg["batch_size"])
    rep = np.random.default_rng(base+2).permutation(len(train.x))[:cfg["representative_frames"]]
    path = export_int8(trainer.model, train.x[rep], folder/"gap_int8.tflite")
    def score(x: np.ndarray) -> dict:
        return {"int8": tflite_scores(str(path), x)[0], "fp32": predict(trainer.model, x, cfg["batch_size"]),
                "energy": compute_energy_statistic(x)}
    gains = cfg["gains_db"]
    max_budget = max(cfg["calibration_budgets_per_gain"])
    for session in range(cfg["sessions_per_model"]):
        seed_cal = base + 100 + session*100
        rng = SyntheticRadioMLGenerator(seed_cal)
        cal = {g: score(frames(rng, max_budget, "NOISE") * 10**(g/20)) for g in gains}
        nominal = tflite_scores(str(path), frames(rng, len(gains)*max_budget, "NOISE"))[0]
        archive = {f"cal_{g}_{k}": v for g, d in cal.items() for k, v in d.items()}
        archive["nominal_cal"] = nominal
        conditions = [(g, 0) for g in [*gains, cfg["transfer_gain_db"]]] + [(0, cfg["noise_stress_db"])]
        for ci, (gain, noise_db) in enumerate(conditions):
            gen = SyntheticRadioMLGenerator(seed_cal+ci+1)
            nh, ns = cfg["test_h0_frames_per_condition"], cfg["test_h1_frames_per_snr"]
            pieces = [frames(gen, nh, "NOISE")]
            for snr in cfg["snrs_db"]:
                pieces.append(frames(gen, ns, "QPSK", snr-noise_db))
            scores = score(np.concatenate(pieces) * 10**((gain+noise_db)/20))
            for k, values in scores.items():
                archive[f"test_{ci}_{k}"] = values
            for b in cfg["calibration_budgets_per_gain"]:
                thresholds = fit_rules(cal, nominal, cfg, b, gain)
                for method, threshold in thresholds.items():
                    runtime = method if method in ("energy","fp32") else "int8"
                    d = scores[runtime] >= threshold
                    for s, snr in enumerate(cfg["snrs_db"]):
                        hits = d[nh+s*ns:nh+(s+1)*ns].sum()
                        writer.writerow(dict(seed=seed, session=session, test_seed=seed_cal+ci+1, gain=gain,
                                             noise_db=noise_db, budget=b, snr=snr, method=method, threshold=threshold,
                                             fa=d[:nh].sum(), h0=nh, hits=hits, h1=ns))
        np.savez_compressed(out/f"scores_{seed}_{session}.npz", **archive)
    return {"seed":seed, "train_seed":base, "validation_seed":base+1, "representative_seed":base+2,
            "best_epoch":trainer.best_epoch, "int8_path":str(Path(path).resolve().relative_to(ROOT)),
            "int8_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256((folder/"best.keras").read_bytes()).hexdigest()}


def main() -> None:
    """Execute a fresh exclusive run from a frozen YAML and record model identities."""
    import tensorflow as tf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.models.exists():
        parser.error("choose two new output directories")
    args.out.mkdir(parents=True)
    args.models.mkdir(parents=True)
    cfg = yaml.safe_load(args.config.read_text())
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    info = {"config":cfg,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"tensorflow":tf.__version__, "blocks":[]}
    with (args.out/"counts.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames="seed session test_seed gain noise_db budget snr method threshold fa h0 hits h1".split())
        writer.writeheader()
        for index in range(len(cfg["model_seeds"])):
            info["blocks"].append(run_block(index,cfg,args.out,args.models,writer))
            file.flush()
            (args.out/"manifest.json").write_text(json.dumps(info,indent=2))
            print("Completed model", index+1, flush=True)


if __name__ == "__main__":
    main()
