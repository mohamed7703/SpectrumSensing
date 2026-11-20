"""Train detector v2 candidates and apply the frozen selection rule.

Subcommands, all driven by ``experiments/configs/detector_v2.yaml``:

* ``lr-search``: pick the SGD learning rate on validation BCE with development seeds.
* ``train``: train candidates (five AWGN-only, five impaired) and score the selection set.
* ``select``: apply the frozen rule mechanically and write the report.

Evidence is synthetic CPU. Outputs go to new paths; existing candidates are never overwritten.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.calibration_benchmark import export_int8, frames
from scripts.validate_recovery_native import predict, tflite_scores
from src.math_models.energy_detection import compute_energy_statistic
from src.math_models.threshold_calibration import calibrate_score_threshold
from src.models import get_model
from src.preprocessing.receiver_impairments import ImpairmentRanges, impair_in_sessions
from src.training.dataset_loader import SyntheticRadioMLGenerator
from src.training.keras_trainer import KerasSensingTrainer

RUNTIMES = ("int8", "fp32", "energy")


def load_config(path: Path, require_frozen: bool = True) -> dict:
    """Load the YAML config; train and select refuse a draft or an unset learning rate."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if require_frozen:
        if cfg.get("status") != "frozen":
            raise ValueError("config status is not 'frozen'; commit the frozen config before training")
        if cfg["optimizer"].get("learning_rate") is None:
            raise ValueError("optimizer learning_rate is not set; run lr-search and freeze the config first")
    return cfg


def data_seeds(cfg: dict) -> list[int]:
    """List every integer data seed the script uses, so the layout can be checked for collisions."""
    search, sel = cfg["learning_rate_search"], cfg["selection_set"]
    seeds = [spec["base"] + offset for spec in candidate_specs(cfg) for offset in range(5)]
    seeds += [search["data_seed_base"] + shift + offset for shift in (0, 10) for offset in range(4)]
    seeds += [sel["seed_base"] + 10 * i for i in range(sel["sessions"])]
    return seeds


def check_seed_layout(cfg: dict) -> None:
    """Raise ValueError if any data seed repeats or falls in the v1 (821000-869999) or E4 (900000-999999) ranges."""
    seeds = data_seeds(cfg)
    if len(set(seeds)) != len(seeds):
        raise ValueError("data seeds collide")
    if any(821000 <= s < 870000 or 900000 <= s < 1000000 for s in seeds):
        raise ValueError("data seeds overlap the v1 or E4 ranges")


def impairment_ranges(cfg: dict) -> ImpairmentRanges:
    """Build the impairment ranges from the config's ``impairments`` block."""
    block = cfg["impairments"]
    return ImpairmentRanges(dc_power_db=tuple(block["dc_power_db"]), iq_gain_db_max=block["iq_gain_db_max"],
                            iq_phase_deg_max=block["iq_phase_deg_max"], cutoff_fraction=tuple(block["cutoff_fraction"]),
                            filter_order=block["filter_order"], adc_bits=block["adc_bits"],
                            adc_full_scale=block["adc_full_scale"])


def candidate_specs(cfg: dict) -> list[dict]:
    """List the candidates in order (AWGN seeds, then impaired seeds) with their data seed bases."""
    specs = []
    for kind, block in cfg["candidates"].items():
        for seed in block["model_seeds"]:
            index = len(specs)
            specs.append({"name": f"{kind}-{seed}", "kind": kind, "seed": int(seed),
                          "impaired": bool(block["impaired_training"]),
                          "base": int(cfg["data_seed_base"]) + 1000 * index})
    return specs


def build_training_partition(cfg: dict, base: int, frames_per_snr: int, impaired: bool,
                             session_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one labelled partition, shuffled, optionally passed through receiver sessions.

    Frames are shuffled before being cut into sessions so each session mixes both
    hypotheses and all SNRs; otherwise a session-level artifact would identify the label.

    Args:
        cfg: Detector config.
        base: Seed of the frame generator.
        frames_per_snr: H1 frames per SNR (H0 is added to reach ``h0_ratio``).
        impaired: Apply session-level receiver impairments when true.
        session_seed: Seed for the shuffle and the session draws.

    Returns:
        Float32 frames (N, 1024, 2) and int32 binary labels.
    """
    data = SyntheticRadioMLGenerator(base).generate_dataset(frames_per_snr, ["QPSK"], cfg["snrs_db"], cfg["h0_ratio"])
    order = np.random.default_rng([session_seed, 0]).permutation(len(data.x))
    x, y = data.x[order], data.y_binary[order]
    if impaired:
        x, _ = impair_in_sessions(x, np.random.default_rng([session_seed, 1]), impairment_ranges(cfg),
                                  cfg["frames_per_session"])
    return x.astype(np.float32), y.astype(np.int32)


def build_selection_session(cfg: dict, seed: int, impaired: bool) -> dict:
    """Generate one selection session: H0 calibration, H0 test and H1 test frames.

    The impaired variant passes all three groups through a single receiver session,
    as one lab recording would; the clean variant is the same frames without impairments.
    """
    sel = cfg["selection_set"]
    gen = SyntheticRadioMLGenerator(seed)
    calibration = frames(gen, sel["calibration_h0_frames"], "NOISE")
    test_h0 = frames(gen, sel["test_h0_frames"], "NOISE")
    test_h1 = np.concatenate([frames(gen, sel["test_h1_frames"], "QPSK", float(snr)) for snr in sel["h1_snrs_db"]])
    if impaired:
        n_cal, n_h0 = len(calibration), len(test_h0)
        joined = np.concatenate([calibration, test_h0, test_h1])
        joined, _ = impair_in_sessions(joined, np.random.default_rng([seed, 1]), impairment_ranges(cfg), len(joined))
        calibration, test_h0, test_h1 = joined[:n_cal], joined[n_cal:n_cal + n_h0], joined[n_cal + n_h0:]
    return {"calibration": calibration, "test_h0": test_h0, "test_h1": test_h1}


def detection_counts(calibration: np.ndarray, test_h0: np.ndarray, test_h1: np.ndarray,
                     target_pfa: float, confidence: float) -> dict:
    """Calibrate on H0 scores (strict order statistic) and count false alarms and detections."""
    threshold = calibrate_score_threshold(calibration, target_pfa, confidence).threshold
    return {"threshold": threshold, "fa": int(np.sum(test_h0 >= threshold)), "h0": len(test_h0),
            "hits": int(np.sum(test_h1 >= threshold)), "h1": len(test_h1)}


def score_frames(model, int8_path: Path, x: np.ndarray, batch_size: int) -> tuple[dict, dict]:
    """Score frames with the INT8 file, the FP32 graph and the energy statistic."""
    int8, info = tflite_scores(str(int8_path), x)
    return {"int8": int8, "fp32": predict(model, x, batch_size), "energy": compute_energy_statistic(x)}, info


def evaluate_selection(cfg: dict, model, int8_path: Path, impaired: bool) -> dict:
    """Score every selection session and pool counts; returns metrics tagged by variant."""
    sel, tag = cfg["selection_set"], "impaired" if impaired else "clean"
    pooled = {rt: {"fa": 0, "h0": 0, "hits": 0, "h1": 0} for rt in RUNTIMES}
    agree = total = clipped = values = 0
    levels = 0
    for i in range(sel["sessions"]):
        session = build_selection_session(cfg, sel["seed_base"] + 10 * i, impaired)
        sizes = [len(session[k]) for k in ("calibration", "test_h0", "test_h1")]
        scores, info = score_frames(model, int8_path, np.concatenate(list(session.values())), cfg["batch_size"])
        clipped += info["clipped_input_values"]
        values += info["total_input_values"]
        decisions = {}
        for rt in RUNTIMES:
            cal, h0, h1 = np.split(scores[rt], np.cumsum(sizes)[:2])
            counts = detection_counts(cal, h0, h1, sel["target_pfa"], sel["calibration_confidence"])
            for key in pooled[rt]:
                pooled[rt][key] += counts[key]
            decisions[rt] = np.concatenate([h0, h1]) >= counts["threshold"]
            if rt == "int8":
                levels = max(levels, len(np.unique(cal)))
        agree += int(np.sum(decisions["int8"] == decisions["fp32"]))
        total += len(decisions["int8"])
    result = {f"int8_fp32_agreement_{tag}": agree / total, f"int8_clipped_input_fraction_{tag}": clipped / values}
    for rt, c in pooled.items():
        result.update({f"{rt}_pd_{tag}": c["hits"] / c["h1"], f"{rt}_hits_{tag}": c["hits"],
                       f"{rt}_fa_{tag}": c["fa"] / c["h0"], f"{rt}_false_alarms_{tag}": c["fa"]})
    result[f"h1_total_{tag}"] = pooled["int8"]["h1"]
    result[f"h0_total_{tag}"] = pooled["int8"]["h0"]
    if impaired:
        result["int8_distinct_h0_levels"] = levels
    return result


def select_candidate(rows: list[dict], rule: dict) -> dict:
    """Apply the frozen rule: highest metric, then each tie-break (higher wins), then lowest seed."""
    if not rows:
        raise ValueError("no candidates to select from")
    keys = [rule["metric"], *rule.get("tie_breaks", [])]
    for row in rows:
        if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in keys):
            raise ValueError(f"candidate {row.get('seed')} has a missing or non-finite selection metric")
    return min(rows, key=lambda r: ([-r[k] for k in keys], r["seed"]))


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_min(values: list[float]) -> float:
    """Return the smallest finite value, or infinity when none is finite."""
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else math.inf


def train_candidate(cfg: dict, spec: dict, out: Path, models: Path, provenance: dict | None = None) -> dict:
    """Train one candidate, export INT8, score the selection set and write its record.

    Args:
        cfg: Detector config.
        spec: One entry of ``candidate_specs``.
        out: Output root; the record goes to ``out/candidates``.
        models: Model root; this candidate gets its own folder.
        provenance: Extra fields stored in the record (config and script hashes, threads).

    Raises:
        FileExistsError: If this candidate's model folder or record already exists.
    """
    import tensorflow as tf
    folder = Path(models) / spec["name"]
    record_path = Path(out) / "candidates" / f"{spec['name']}.json"
    if folder.exists() or record_path.exists():
        raise FileExistsError(f"candidate {spec['name']} already exists")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(spec["seed"])
    base, impaired = spec["base"], spec["impaired"]
    x_train, y_train = build_training_partition(cfg, base, cfg["training_frames_per_snr"], impaired, base + 2)
    x_val, y_val = build_training_partition(cfg, base + 1, cfg["validation_frames_per_snr"], impaired, base + 3)
    trainer = KerasSensingTrainer(get_model("gap").to_keras(), cfg["optimizer"], spec["seed"])
    trainer.fit(x_train, y_train, x_val, y_val, run_directory=folder, epochs=cfg["epochs"],
                batch_size=cfg["batch_size"], patience=cfg["patience"], min_delta=cfg["min_delta"])
    train_seconds = time.perf_counter() - started
    chosen = np.random.default_rng(base + 4).permutation(len(x_train))[:cfg["representative_frames"]]
    int8_path = export_int8(trainer.model, x_train[chosen], folder / "gap_int8.tflite")
    _, info = tflite_scores(str(int8_path), x_train[:1])
    record = {**(provenance or {}), "tensorflow": tf.__version__,
              "name": spec["name"], "kind": spec["kind"], "seed": spec["seed"], "impaired_training": impaired,
              "data_seed_base": base, "best_epoch": trainer.best_epoch, "epochs_run": len(trainer.history),
              "best_validation_loss": min(h["val_loss"] for h in trainer.history),
              "parameter_count": int(trainer.model.count_params()),
              "int8_sha256": sha256(int8_path), "checkpoint_sha256": sha256(folder / "best.keras"),
              "int8_input_dtype": info["input_dtype"], "int8_output_dtype": info["output_dtype"],
              "int8_input_quantization": [float(v) for v in info["input_quantization"]],
              "int8_output_quantization": [float(v) for v in info["output_quantization"]]}
    for impaired_eval in (True, False):
        record.update(evaluate_selection(cfg, trainer.model, int8_path, impaired_eval))
    record["wall_seconds"] = round(time.perf_counter() - started, 1)
    record["train_seconds"] = round(train_seconds, 1)
    with record_path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    return record


def lr_search(cfg: dict, out: Path) -> dict:
    """Train development models over the learning-rate grid and choose by mean best validation BCE."""
    import tensorflow as tf
    search = cfg["learning_rate_search"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for rate in search["grid"]:
        for kind, seed in search["dev_model_seeds"].items():
            impaired = kind == "impaired"
            base = search["data_seed_base"] + (10 if impaired else 0)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            x, y = build_training_partition(cfg, base, search["training_frames_per_snr"], impaired, base + 2)
            xv, yv = build_training_partition(cfg, base + 1, search["validation_frames_per_snr"], impaired, base + 3)
            started = time.perf_counter()
            trainer = KerasSensingTrainer(get_model("gap").to_keras(), {**cfg["optimizer"], "learning_rate": rate}, seed)
            try:
                history = trainer.fit(x, y, xv, yv, run_directory=out / "runs" / f"{kind}-lr{rate}",
                                      epochs=search["epochs"], batch_size=cfg["batch_size"],
                                      patience=search["epochs"], min_delta=0.0)
                best, final, diverged = finite_min([h["val_loss"] for h in history]), history[-1]["val_loss"], False
            except FloatingPointError:
                best, final, diverged = math.inf, math.inf, True
            rows.append({"learning_rate": rate, "kind": kind, "best_validation_loss": best,
                         "final_validation_loss": final, "diverged": diverged,
                         "seconds": round(time.perf_counter() - started, 1)})
            print(rows[-1], flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    means = {r: float(np.mean([row["best_validation_loss"] for row in rows if row["learning_rate"] == r]))
             for r in search["grid"]}
    chosen = min(sorted(means), key=means.get)
    result = {"rule": search["rule"], "mean_best_validation_loss": means, "chosen_learning_rate": chosen, "rows": rows}
    (out / "lr_search.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_selection(cfg: dict, config_path: Path, out: Path, models: Path) -> dict:
    """Apply the frozen rule to all recorded candidates and write CSV, manifest and report."""
    out = Path(out)
    rows = []
    for spec in candidate_specs(cfg):
        path = out / "candidates" / f"{spec['name']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate record {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    winner = select_candidate(rows, cfg["selection_rule"])
    if len({(r["energy_pd_impaired"], r["energy_pd_clean"]) for r in rows}) != 1:
        raise ValueError("energy reference differs across candidates; selection frames were not shared")
    current = {"config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__))}
    for r in rows:
        if any(r.get(k) != v for k, v in current.items()):
            raise ValueError(f"{r['name']} was trained under a different config or script than the current files")
        for key, name in (("int8_sha256", "gap_int8.tflite"), ("checkpoint_sha256", "best.keras")):
            if sha256(Path(models) / r["name"] / name) != r[key]:
                raise ValueError(f"{r['name']} {name} differs from the file that was evaluated")
    if len({r["tensorflow"] for r in rows}) != 1:
        raise ValueError("candidates were trained under different TensorFlow versions")
    fields = ["name", "seed", "impaired_training", "best_epoch", "epochs_run", "best_validation_loss",
              "int8_pd_impaired", "fp32_pd_impaired", "energy_pd_impaired", "int8_fa_impaired",
              "int8_fp32_agreement_impaired", "int8_pd_clean", "fp32_pd_clean", "energy_pd_clean", "int8_fa_clean",
              "int8_fp32_agreement_clean", "int8_distinct_h0_levels", "wall_seconds", "selected"]
    with (out / "selection.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected": row["name"] == winner["name"]})
    selected_dir = out / "selected"
    selected_dir.mkdir(exist_ok=False)
    for name in ("gap_int8.tflite", "best.keras"):
        (selected_dir / name).write_bytes((Path(models) / winner["name"] / name).read_bytes())
    import tensorflow as tf
    manifest = {"config": cfg, **current, "threads": sorted({r["threads"] for r in rows}),
                "tensorflow": tf.__version__, "evidence": "synthetic CPU", "selected": winner["name"],
                "selected_int8_sha256": sha256(selected_dir / "gap_int8.tflite"),
                "selected_checkpoint_sha256": sha256(selected_dir / "best.keras"),
                "candidates": [{k: r[k] for k in ("name", "seed", "int8_sha256", "checkpoint_sha256", "wall_seconds")}
                               for r in rows]}
    for name, text in (("manifest.json", json.dumps(manifest, indent=2)), ("REPORT.md", render_report(cfg, rows, winner))):
        with (out / name).open("x", encoding="utf-8") as file:
            file.write(text)
    return winner


def render_report(cfg: dict, rows: list[dict], winner: dict) -> str:
    """Render the selection report from recorded candidate rows."""
    sel, rule = cfg["selection_set"], cfg["selection_rule"]
    ref = rows[0]
    lines = ["# Detector v2 selection report", "",
             "Evidence setting: synthetic frames, CPU, native Keras and TFLite INT8. Not a hardware result.", "",
             f"Rule (frozen before training): {rule['metric_definition']}. Ties: {rule['agreement_definition']}; "
             f"then {rule['final_tie_break']}.", "",
             f"Selection set: {sel['sessions']} sessions, {sel['calibration_h0_frames']} H0 calibration frames, "
             f"{sel['test_h0_frames']} H0 test frames and {sel['test_h1_frames']} QPSK frames at each of "
             f"{sel['h1_snrs_db']} dB per session ({ref['h1_total_impaired']} H1 and {ref['h0_total_impaired']} H0 test "
             f"frames pooled). Target Pfa {sel['target_pfa']}, confidence {sel['calibration_confidence']}. "
             "The impaired variant draws one receiver session per selection session; the clean variant is reported only.", "",
             "Caveats. SNR is nominal at the antenna port: the receive filter acts on signal and noise together, so "
             "in-band SNR of impaired frames is higher than the label and impaired and clean Pd are not comparable at the "
             "same label. Renormalising by the filter's power gain fixes mean noise power only; coloured noise has fewer "
             "effective degrees of freedom, so per-frame energy varies more and the energy detector is weaker under "
             "impairments. Each session calibrates and tests inside one receiver draw. The winner's Pd here is optimistically "
             "biased by selection; claims need the separate benchmark.", "",
             f"Energy detector on the same frames: Pd {ref['energy_pd_impaired']:.3f} impaired, "
             f"{ref['energy_pd_clean']:.3f} clean.", "",
             "| candidate | trained on | epochs | INT8 Pd imp. | FP32 Pd imp. | INT8 FA imp. | agreement imp. | "
             "INT8 Pd clean | FP32 Pd clean | INT8 levels | selected |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {'impaired' if r['impaired_training'] else 'AWGN'} | {r['epochs_run']} | "
                     f"{r['int8_pd_impaired']:.3f} | {r['fp32_pd_impaired']:.3f} | {r['int8_fa_impaired']:.3f} | "
                     f"{r['int8_fp32_agreement_impaired']:.3f} | {r['int8_pd_clean']:.3f} | {r['fp32_pd_clean']:.3f} | "
                     f"{r['int8_distinct_h0_levels']} | {'**yes**' if r['name'] == winner['name'] else ''} |")
    lines += ["", f"Selected: `{winner['name']}` (seed {winner['seed']}).", "",
              "INT8 levels is the largest number of distinct INT8 scores among one session's calibration frames.", ""]
    return "\n".join(lines)


def configure_tensorflow(threads: int) -> None:
    """Fix thread counts and enable deterministic ops, as the v1 benchmark does."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def main() -> None:
    """Command-line entry point for lr-search, train and select."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lr-search", "train", "select"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--only", nargs="*", help="candidate names to train (default: all)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config, require_frozen=args.command != "lr-search")
    check_seed_layout(cfg)
    configure_tensorflow(args.threads)
    if args.command == "lr-search":
        print(json.dumps(lr_search(cfg, args.out)["chosen_learning_rate"]))
        return
    if args.models is None:
        parser.error("--models is required")
    if args.command == "train":
        names = {spec["name"] for spec in candidate_specs(cfg)}
        if args.only and not set(args.only) <= names:
            parser.error(f"unknown candidate names: {sorted(set(args.only) - names)}")
        provenance = {"config_sha256": sha256(args.config), "script_sha256": sha256(Path(__file__)), "threads": args.threads}
        for spec in candidate_specs(cfg):
            if args.only and spec["name"] not in args.only:
                continue
            record = train_candidate(cfg, spec, args.out, args.models, provenance)
            print(spec["name"], "int8 Pd (impaired)", record["int8_pd_impaired"], "seconds", record["wall_seconds"], flush=True)
    else:
        print("selected", write_selection(cfg, args.config, args.out, args.models)["name"])


if __name__ == "__main__":
    main()
