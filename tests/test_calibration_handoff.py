"""Regression checks for scientific boundaries in the pre-USRP handoff."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibration_benchmark import fit_rules
from scripts.usrp_calibration_replay import digest, fixture, process, recordings, score_key
from scripts.validate_recovery_native import tflite_scores
from src.math_models.threshold_calibration import calibrate_score_threshold

ROOT = Path(__file__).resolve().parents[1]
INT8_MODEL = ROOT / "docs/calibration/usrp-2920/gap-seed19031-int8.tflite"
BUDGET = 100
METHOD_RUNTIMES = dict(ordinary="int8", linear="int8", fixed="int8", fp32="fp32",
                       int8_copied_fp32="int8", energy="energy")


def test_constant_quantized_scores_preserve_strict_ties() -> None:
    """Both fitted rules must reject an entire noise score tie."""
    scores=np.full(100,.5234375)
    cfg=dict(gains_db=[-1,0,1],target_pfa=.05,calibration_confidence=.95)
    cal={g:{rt:scores for rt in ["int8","energy","fp32"]} for g in cfg["gains_db"]}
    rules=fit_rules(cal,np.tile(scores,3),cfg,100,0)
    assert all(t>scores[0] for t in rules.values())


@pytest.fixture
def sessions(tmp_path: Path) -> Path:
    """Supply fixture metadata and tiny labeled I/Q under an isolated path."""
    folder=tmp_path/"iq"
    fixture(folder)
    return folder


@pytest.mark.parametrize("field,value", [("dropped_samples",1),("sha256","bad"),("sample_count",3)])
def test_reject_corrupt_capture(sessions: Path, field: str, value) -> None:
    """Reject recorded packet loss, hash mismatch, and truncated I/Q."""
    path=sessions/"test.json"
    meta=json.loads(path.read_text())
    meta["recordings"][0][field]=value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        list(recordings(path))


def test_reject_overlapping_ground_truth(sessions: Path) -> None:
    """Labels covering the same sample twice invalidate the capture."""
    path=sessions/"test-9.csv"
    path.write_text(path.read_text().replace("32768,65536","32767,65536"))
    with pytest.raises(ValueError,match="overlapping"):
        list(recordings(sessions/"test.json"))


def test_reject_reused_day(sessions: Path) -> None:
    """An acquisition on the calibration day cannot establish cross-day transfer."""
    meta=json.loads((sessions/"test.json").read_text())
    lock=dict(model_sha256=None, session_id="different",day=meta["started_utc"][:10])
    with pytest.raises(ValueError,match="later independent day"):
        process(sessions/"test.json","replay",None,lock,100)


@pytest.fixture(scope="module")
def native_models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Save two untrained FP32 GAP graphs; these test plumbing, not detector quality."""
    tf = pytest.importorskip("tensorflow")
    from src.models import get_model
    folder = tmp_path_factory.mktemp("native")
    paths = []
    for seed in (7, 8):
        tf.keras.utils.set_random_seed(seed)
        path = folder / f"gap{seed}.keras"
        get_model("gap").to_keras().save(path)
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture
def fitted(sessions: Path, native_models: tuple[Path, Path]) -> tuple[Path, dict]:
    """Fit all six rules on the fixture calibration day and return the lock."""
    lock = process(sessions/"calibration.json", "fit", INT8_MODEL, None, BUDGET, native_model=native_models[0])
    return sessions, lock


def test_lock_has_every_method_with_runtime_tags(fitted: tuple[Path, dict], native_models: tuple[Path, Path]) -> None:
    """The lock names all six methods, their scoring runtime and their H0 cost."""
    _, lock = fitted
    assert lock["lock_format"] == 2 and lock["nominal_gain_db"] == 10
    assert lock["native_sha256"] == digest(native_models[0]) and lock["model_sha256"] == digest(INT8_MODEL)
    assert set(lock["thresholds"]) == {"9.0", "10.0", "11.0"}
    for table in lock["thresholds"].values():
        assert set(table) == set(METHOD_RUNTIMES)
        for method, entry in table.items():
            assert entry["runtime"] == METHOD_RUNTIMES[method]
            assert entry["h0_frames_charged"] == 3*BUDGET


def test_fixed_rule_uses_only_nominal_gain_frames(fitted: tuple[Path, dict]) -> None:
    """One INT8 threshold from 3B nominal-gain frames applies at every gain."""
    sessions, lock = fitted
    iq = np.fromfile(sessions/"calibration-10.iq", dtype="<c8")[:3*BUDGET*1024]
    scores, _ = tflite_scores(str(INT8_MODEL), iq.view("<f4").reshape(-1, 1024, 2)*8)
    expected = calibrate_score_threshold(scores).threshold
    assert {t["fixed"]["threshold"] for t in lock["thresholds"].values()} == {expected}


def test_copied_native_threshold_is_applied_to_int8_scores(fitted: tuple[Path, dict], native_models: tuple[Path, Path],
                                                           tmp_path: Path) -> None:
    """The copied rule is the FP32 threshold compared with INT8 scores, as in claim C14."""
    sessions, lock = fitted
    for table in lock["thresholds"].values():
        assert table["int8_copied_fp32"]["threshold"] == table["fp32"]["threshold"]
    out = process(sessions/"test.json", "replay", INT8_MODEL, lock, BUDGET, native_model=native_models[0],
                  scores_out=tmp_path/"scores.npz")
    scores = np.load(tmp_path/"scores.npz")
    rows = [r for r in out["decisions"] if r["method"] == "int8_copied_fp32"]
    assert rows
    for row in rows:
        values = scores[score_key("int8", row["gain_db"], row["label"], row["snr_db"])]
        threshold = lock["thresholds"][str(row["gain_db"])]["fp32"]["threshold"]
        assert row["positive"] == int((values >= threshold).sum()) and row["frames"] == len(values)


def test_replay_records_score_sidecar_hash(fitted: tuple[Path, dict], native_models: tuple[Path, Path],
                                           tmp_path: Path) -> None:
    """The per-frame score archive is hashed so ties can be replayed later."""
    sessions, lock = fitted
    out = process(sessions/"test.json", "replay", INT8_MODEL, lock, BUDGET, native_model=native_models[0],
                  scores_out=tmp_path/"scores.npz")
    assert out["scores_sha256"] == digest(tmp_path/"scores.npz")
    assert {r["method"] for r in out["decisions"]} == set(METHOD_RUNTIMES)


def test_replay_rejects_other_native_model(fitted: tuple[Path, dict], native_models: tuple[Path, Path]) -> None:
    """A different FP32 model cannot be replayed against the locked thresholds."""
    sessions, lock = fitted
    with pytest.raises(ValueError, match="native model"):
        process(sessions/"test.json", "replay", INT8_MODEL, lock, BUDGET, native_model=native_models[1])


def test_energy_thresholds_follow_receiver_gain(fitted: tuple[Path, dict]) -> None:
    """Energy scales with power, so +1 dB gain raises the threshold by about 1.26."""
    _, lock = fitted
    energy = {float(g): t["energy"]["threshold"] for g, t in lock["thresholds"].items()}
    for low, high in ((9.0, 10.0), (10.0, 11.0)):
        assert energy[high]/energy[low] == pytest.approx(10**((high-low)/10), rel=.05)


def test_fit_requires_native_model_and_nominal_gain(sessions: Path, native_models: tuple[Path, Path]) -> None:
    """Fitting needs the FP32 model, a declared nominal gain and 3B frames at that gain."""
    path = sessions/"calibration.json"
    with pytest.raises(ValueError, match="native model"):
        process(path, "fit", INT8_MODEL, None, BUDGET)
    with pytest.raises(ValueError, match="nominal gain"):
        process(path, "fit", INT8_MODEL, None, 2*BUDGET, native_model=native_models[0])
    meta = json.loads(path.read_text())
    del meta["nominal_gain_db"]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="nominal_gain_db"):
        process(path, "fit", INT8_MODEL, None, BUDGET, native_model=native_models[0])
    meta["nominal_gain_db"] = 10.5
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a recorded gain"):
        process(path, "fit", INT8_MODEL, None, BUDGET, native_model=native_models[0])


def first_int8_scores(folder: Path, gain: int, frames: int) -> np.ndarray:
    """Score the first frames of a fixture calibration recording with the INT8 model."""
    iq = np.fromfile(folder/f"calibration-{gain}.iq", dtype="<c8")[:frames*1024]
    return tflite_scores(str(INT8_MODEL), iq.view("<f4").reshape(-1, 1024, 2)*8)[0]


def test_charge_follows_the_number_of_gains(tmp_path: Path, native_models: tuple[Path, Path]) -> None:
    """With four gains every method is charged four budgets and fixed uses four."""
    folder = tmp_path/"iq4"
    fixture(folder, gains=(9, 10, 11, 12))
    lock = process(folder/"calibration.json", "fit", INT8_MODEL, None, BUDGET, native_model=native_models[0])
    assert set(lock["thresholds"]) == {"9.0", "10.0", "11.0", "12.0"}
    for table in lock["thresholds"].values():
        assert {e["h0_frames_charged"] for e in table.values()} == {4*BUDGET}
    expected = calibrate_score_threshold(first_int8_scores(folder, 10, 4*BUDGET)).threshold
    assert {t["fixed"]["threshold"] for t in lock["thresholds"].values()} == {expected}


def test_fit_uses_only_the_first_budget_frames(tmp_path: Path, native_models: tuple[Path, Path]) -> None:
    """Surplus frames are ignored: B per gain and 3B at the nominal gain."""
    folder = tmp_path/"iq"
    fixture(folder, calibration_frames=150)
    lock = process(folder/"calibration.json", "fit", INT8_MODEL, None, BUDGET, native_model=native_models[0])
    for gain in (9, 11):
        expected = calibrate_score_threshold(first_int8_scores(folder, gain, BUDGET)).threshold
        assert lock["thresholds"][f"{gain}.0"]["ordinary"]["threshold"] == expected
    assert lock["thresholds"]["10.0"]["fixed"]["threshold"] == calibrate_score_threshold(
        first_int8_scores(folder, 10, 3*BUDGET)).threshold


def test_gain_split_over_intervals_gives_the_same_lock(fitted: tuple[Path, dict],
                                                      native_models: tuple[Path, Path]) -> None:
    """Two noise intervals at one gain are pooled exactly like a single interval."""
    sessions, lock = fitted
    (sessions/"calibration-9.csv").write_text(
        "start_sample,stop_sample,class,snr_db,ground_truth\n0,51200,H0,,synthetic noise\n"
        "51200,102400,H0,,synthetic noise\n")
    again = process(sessions/"calibration.json", "fit", INT8_MODEL, None, BUDGET, native_model=native_models[0])
    assert again["thresholds"] == lock["thresholds"]


def test_replay_counts_match_archive_for_every_method(fitted: tuple[Path, dict], native_models: tuple[Path, Path],
                                                      tmp_path: Path) -> None:
    """Each method's positives equal its locked runtime scores compared with its locked threshold."""
    sessions, lock = fitted
    out = process(sessions/"test.json", "replay", INT8_MODEL, lock, BUDGET, native_model=native_models[0],
                  scores_out=tmp_path/"scores.npz")
    archive = np.load(tmp_path/"scores.npz")
    assert len(out["decisions"]) == 6*3*2
    for row in out["decisions"]:
        entry = lock["thresholds"][str(row["gain_db"])][row["method"]]
        values = archive[score_key(entry["runtime"], row["gain_db"], row["label"], row["snr_db"])]
        assert row["positive"] == int((values >= entry["threshold"]).sum())
    h0 = {rt: archive[score_key(rt, 10.0, "H0", None)] for rt in ("int8", "fp32", "energy")}
    assert not np.array_equal(h0["fp32"], h0["int8"]) and not np.array_equal(h0["energy"], h0["int8"])


def test_replay_rejects_nonfinite_scores(fitted: tuple[Path, dict], native_models: tuple[Path, Path],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN score would silently count as a missed detection, so replay must stop."""
    import scripts.usrp_calibration_replay as replay
    sessions, lock = fitted
    monkeypatch.setattr(replay, "predict", lambda model, x, size: np.full(len(x), np.nan))
    with pytest.raises(ValueError, match="nonfinite score"):
        process(sessions/"test.json", "replay", INT8_MODEL, lock, BUDGET, native_model=native_models[0])
