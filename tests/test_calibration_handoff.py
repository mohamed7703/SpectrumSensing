"""Regression checks for scientific boundaries in the pre-USRP handoff."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibration_benchmark import fit_rules
from scripts.usrp_calibration_replay import fixture, process, recordings


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
