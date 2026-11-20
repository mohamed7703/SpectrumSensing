"""Tests for the detector v2 training and selection script."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.train_detector_v2 import (
    build_selection_session,
    build_training_partition,
    candidate_specs,
    check_seed_layout,
    data_seeds,
    detection_counts,
    finite_min,
    impairment_ranges,
    load_config,
    select_candidate,
    train_candidate,
)
from src.preprocessing.receiver_impairments import ImpairmentRanges

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "configs" / "detector_v2.yaml"
V1_SEEDS = range(821000, 870000)
E4_SEEDS = range(900000, 1000000)


@pytest.fixture
def cfg() -> dict:
    """Return the frozen config with tiny sizes so tests stay fast."""
    config = load_config(CONFIG, require_frozen=False)
    config.update(snrs_db=[-10, 0, 10], training_frames_per_snr=20, validation_frames_per_snr=10,
                  frames_per_session=40, epochs=1, patience=1, representative_frames=8, batch_size=32)
    config["optimizer"] = {**config["optimizer"], "learning_rate": 0.01}
    config["selection_set"] = {**config["selection_set"], "sessions": 1, "calibration_h0_frames": 80,
                               "test_h0_frames": 40, "test_h1_frames": 40, "h1_snrs_db": [-10, -6]}
    return config


def test_ten_candidates_with_the_planned_seeds() -> None:
    """Five AWGN and five impaired candidates carry the pre-registered model seeds."""
    specs = candidate_specs(load_config(CONFIG, require_frozen=False))
    assert [s["seed"] for s in specs if not s["impaired"]] == [29031, 29032, 29033, 29034, 29035]
    assert [s["seed"] for s in specs if s["impaired"]] == [39031, 39032, 39033, 39034, 39035]
    assert len({s["name"] for s in specs}) == 10


def test_every_seed_is_disjoint_from_v1_and_e4() -> None:
    """Model and data seeds avoid the v1 benchmark range, the planned E4 range and each other."""
    config = load_config(CONFIG, require_frozen=False)
    data = data_seeds(config)
    models = [s["seed"] for s in candidate_specs(config)]
    assert len(set(data)) == len(data)
    for seed in data + models:
        assert seed not in V1_SEEDS and seed not in E4_SEEDS
    assert not set(models) & {19031, 19032, 19033, 19034, 19035}
    check_seed_layout(config)


def test_seed_layout_check_catches_collisions_and_reserved_ranges() -> None:
    """A selection seed base inside a candidate's block, or inside the E4 range, is rejected."""
    config = load_config(CONFIG, require_frozen=False)
    clash = {**config, "selection_set": {**config["selection_set"], "seed_base": config["data_seed_base"]}}
    with pytest.raises(ValueError, match="collide"):
        check_seed_layout(clash)
    reserved = {**config, "data_seed_base": 950000}
    with pytest.raises(ValueError, match="overlap"):
        check_seed_layout(reserved)


def test_selection_rule_is_frozen_in_the_config() -> None:
    """The rule names its metric, tie-breaks and final tie-break before any training."""
    rule = load_config(CONFIG, require_frozen=False)["selection_rule"]
    assert rule["metric"] == "int8_pd_impaired"
    assert rule["tie_breaks"] == ["int8_fp32_agreement_impaired"]
    assert "lowest model seed" in rule["final_tie_break"]


def test_training_refuses_a_draft_or_unset_learning_rate(tmp_path: Path) -> None:
    """Only a frozen config with a learning rate can train; the committed file satisfies both once frozen."""
    config = load_config(CONFIG, require_frozen=False)
    if config["status"] != "frozen":
        with pytest.raises(ValueError, match="frozen"):
            load_config(CONFIG)
    draft = tmp_path / "unset.yaml"
    draft.write_text("status: frozen\noptimizer: {name: SGD, learning_rate: null}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(draft)
    if config["status"] == "frozen":
        assert config["optimizer"]["learning_rate"] in config["learning_rate_search"]["grid"]


def test_finite_min_ignores_nan_and_reports_infinity_when_all_diverge() -> None:
    """A diverged learning-rate run cannot win through NaN ordering."""
    assert finite_min([float("nan"), 0.4, 0.3]) == 0.3
    assert finite_min([float("nan"), float("inf")]) == float("inf")


def test_impairment_ranges_follow_the_config() -> None:
    """The YAML block becomes the ranges object."""
    ranges = impairment_ranges(load_config(CONFIG, require_frozen=False))
    assert isinstance(ranges, ImpairmentRanges) and ranges.adc_bits == 12 and ranges.filter_order == 4


def test_impaired_sessions_mix_both_hypotheses(cfg: dict) -> None:
    """Shuffling before chunking keeps a session from carrying a single label."""
    x, y = build_training_partition(cfg, 5000, cfg["training_frames_per_snr"], impaired=True, session_seed=5002)
    assert x.shape == (120, 1024, 2) and x.dtype == np.float32
    for start in range(0, len(y), cfg["frames_per_session"]):
        chunk = y[start:start + cfg["frames_per_session"]]
        assert 0 < chunk.sum() < len(chunk)


def test_training_and_validation_partitions_do_not_share_frames(cfg: dict) -> None:
    """Different seeds give disjoint frames."""
    a, _ = build_training_partition(cfg, 5000, 20, impaired=False, session_seed=5002)
    b, _ = build_training_partition(cfg, 5001, 10, impaired=False, session_seed=5003)
    hashes = lambda arr: {hashlib.sha256(f.tobytes()).hexdigest() for f in arr}
    assert not hashes(a) & hashes(b)


def test_clean_noise_frames_have_unit_power_and_impaired_noise_stays_near_it(cfg: dict) -> None:
    """The energy cue is intact after impairments."""
    for impaired in (False, True):
        x, y = build_training_partition(cfg, 5000, 20, impaired=impaired, session_seed=5002)
        noise = x[y == 0].astype(np.float64)
        assert np.mean(np.sum(noise ** 2, axis=-1)) == pytest.approx(1.0, rel=0.05)


def test_selection_session_is_deterministic_and_sized(cfg: dict) -> None:
    """Same seed, same frames; clean and impaired variants differ."""
    a = build_selection_session(cfg, 7000, impaired=True)
    b = build_selection_session(cfg, 7000, impaired=True)
    clean = build_selection_session(cfg, 7000, impaired=False)
    assert a["calibration"].shape[0] == 80 and a["test_h0"].shape[0] == 40 and a["test_h1"].shape[0] == 80
    np.testing.assert_array_equal(a["test_h1"], b["test_h1"])
    assert not np.array_equal(a["test_h1"], clean["test_h1"])


def test_detection_counts_use_the_strict_order_statistic_threshold() -> None:
    """The threshold sits just above the rank-th calibration score, so ties are rejected as a block."""
    calibration = np.repeat(np.arange(100, dtype=float), 1)
    result = detection_counts(calibration, test_h0=np.array([10.0, 96.0, 98.0, 99.5]),
                              test_h1=np.array([50.0, 97.0, 99.9]), target_pfa=0.05, confidence=0.95)
    assert result["threshold"] > 0.0 and result["h0"] == 4 and result["h1"] == 3
    assert result["fa"] == int(np.sum(np.array([10.0, 96.0, 98.0, 99.5]) >= result["threshold"]))
    assert result["hits"] == int(np.sum(np.array([50.0, 97.0, 99.9]) >= result["threshold"]))


def test_detection_counts_with_tied_scores_reject_the_whole_block() -> None:
    """A coarse lattice cannot admit part of a tied block."""
    calibration = np.array([0.0] * 90 + [1.0] * 10)
    result = detection_counts(calibration, test_h0=np.array([0.0, 1.0]), test_h1=np.array([1.0, 2.0]),
                              target_pfa=0.05, confidence=0.95)
    assert result["fa"] == 0 and result["hits"] == 1


def row(seed: int, pd: float, agreement: float) -> dict:
    """Return a minimal candidate row."""
    return {"seed": seed, "int8_pd_impaired": pd, "int8_fp32_agreement_impaired": agreement}


RULE = {"metric": "int8_pd_impaired", "tie_breaks": ["int8_fp32_agreement_impaired"]}


def test_select_candidate_takes_the_highest_pd() -> None:
    """The metric decides first."""
    rows = [row(1, 0.60, 0.99), row(2, 0.70, 0.90), row(3, 0.65, 0.95)]
    assert select_candidate(rows, RULE)["seed"] == 2


def test_select_candidate_breaks_pd_ties_by_agreement_then_lowest_seed() -> None:
    """Agreement breaks a Pd tie; an exact double tie goes to the lowest seed."""
    assert select_candidate([row(5, 0.7, 0.90), row(4, 0.7, 0.95)], RULE)["seed"] == 4
    assert select_candidate([row(8, 0.7, 0.9), row(6, 0.7, 0.9), row(7, 0.7, 0.9)], RULE)["seed"] == 6


def test_select_candidate_rejects_empty_and_nonfinite_rows() -> None:
    """No silent selection from missing or NaN metrics."""
    with pytest.raises(ValueError):
        select_candidate([], RULE)
    with pytest.raises(ValueError):
        select_candidate([row(1, float("nan"), 0.9), row(2, 0.5, 0.9)], RULE)


def test_train_candidate_smoke_writes_hashed_models_and_refuses_overwrite(cfg: dict, tmp_path: Path) -> None:
    """One tiny candidate trains, exports INT8, evaluates the selection set and records hashes."""
    spec = candidate_specs(cfg)[0]
    out, models = tmp_path / "out", tmp_path / "models"
    record = train_candidate(cfg, spec, out, models, {"config_sha256": "c" * 64, "script_sha256": "s" * 64, "threads": 2})
    assert record["config_sha256"] == "c" * 64 and record["threads"] == 2 and record["tensorflow"]
    folder = models / spec["name"]
    assert (folder / "best.keras").is_file() and (folder / "gap_int8.tflite").is_file()
    assert record["int8_sha256"] == hashlib.sha256((folder / "gap_int8.tflite").read_bytes()).hexdigest()
    assert record["checkpoint_sha256"] == hashlib.sha256((folder / "best.keras").read_bytes()).hexdigest()
    saved = json.loads((out / "candidates" / f"{spec['name']}.json").read_text())
    for key in ("int8_pd_impaired", "int8_fp32_agreement_impaired", "int8_pd_clean", "fp32_pd_impaired",
                "energy_pd_impaired", "int8_distinct_h0_levels", "wall_seconds", "parameter_count"):
        assert key in saved
    with pytest.raises(FileExistsError):
        train_candidate(cfg, spec, out, models)


def fake_candidates(cfg: dict, out: Path, models: Path, *, winner_seed: int) -> None:
    """Write ten consistent fake candidate records and model files for selection tests."""
    import scripts.train_detector_v2 as module
    provenance = {"config_sha256": module.sha256(CONFIG), "script_sha256": module.sha256(Path(module.__file__)),
                  "threads": 4, "tensorflow": "x"}
    (out / "candidates").mkdir(parents=True)
    for spec in candidate_specs(cfg):
        folder = models / spec["name"]
        folder.mkdir(parents=True)
        (folder / "gap_int8.tflite").write_bytes(spec["name"].encode() + b"int8")
        (folder / "best.keras").write_bytes(spec["name"].encode() + b"keras")
        pd = 0.9 if spec["seed"] == winner_seed else 0.5
        record = {**provenance, "name": spec["name"], "kind": spec["kind"], "seed": spec["seed"],
                  "impaired_training": spec["impaired"], "best_epoch": 3, "epochs_run": 9, "best_validation_loss": 0.3,
                  "int8_sha256": module.sha256(folder / "gap_int8.tflite"),
                  "checkpoint_sha256": module.sha256(folder / "best.keras"), "wall_seconds": 1.0,
                  "int8_pd_impaired": pd, "fp32_pd_impaired": 0.6, "energy_pd_impaired": 0.7, "int8_fa_impaired": 0.04,
                  "int8_fp32_agreement_impaired": 0.9, "int8_pd_clean": 0.5, "fp32_pd_clean": 0.5,
                  "energy_pd_clean": 0.8, "int8_fa_clean": 0.04, "int8_fp32_agreement_clean": 0.9,
                  "int8_distinct_h0_levels": 8, "h1_total_impaired": 100, "h0_total_impaired": 100}
        (out / "candidates" / f"{spec['name']}.json").write_text(json.dumps(record), encoding="utf-8")


def test_write_selection_applies_the_rule_and_copies_hashed_files(cfg: dict, tmp_path: Path) -> None:
    """The winner by the frozen rule is copied, and every output file is created exclusively."""
    from scripts.train_detector_v2 import write_selection
    out, models = tmp_path / "out", tmp_path / "models"
    fake_candidates(cfg, out, models, winner_seed=39033)
    winner = write_selection(cfg, CONFIG, out, models)
    assert winner["name"] == "impaired-39033"
    assert (out / "selected" / "gap_int8.tflite").read_bytes() == b"impaired-39033int8"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["selected"] == "impaired-39033" and (out / "REPORT.md").is_file()
    with pytest.raises(FileExistsError):
        write_selection(cfg, CONFIG, out, models)


def test_write_selection_rejects_changed_models_and_changed_scripts(cfg: dict, tmp_path: Path) -> None:
    """A file replaced after scoring, or a record from another script version, blocks selection."""
    from scripts.train_detector_v2 import write_selection
    out, models = tmp_path / "a" / "out", tmp_path / "a" / "models"
    fake_candidates(cfg, out, models, winner_seed=29031)
    (models / "awgn-29031" / "gap_int8.tflite").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="differs from the file that was evaluated"):
        write_selection(cfg, CONFIG, out, models)
    out2, models2 = tmp_path / "b" / "out", tmp_path / "b" / "models"
    fake_candidates(cfg, out2, models2, winner_seed=29031)
    record_path = out2 / "candidates" / "awgn-29032.json"
    record = json.loads(record_path.read_text())
    record["script_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="different config or script"):
        write_selection(cfg, CONFIG, out2, models2)
