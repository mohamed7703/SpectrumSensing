"""Independently recount frozen recovery scores and pairwise-rank AUCs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def recount(scores: np.ndarray, labels: np.ndarray, snrs: np.ndarray, expected: dict) -> dict:
    """Recalculate specified counts and AUC directly, without production metrics code."""
    decisions = scores >= expected["calibration"]["threshold"]
    false_alarms = int(np.count_nonzero(decisions[labels == 0]))
    if false_alarms != expected["h0"]["false_alarms"]:
        raise AssertionError("false alarm count disagrees with stored scores")
    detections = []
    for item in expected["h1_by_snr"]:
        selected = (labels == 1) & (snrs == item["snr_db"])
        count = int(np.count_nonzero(decisions[selected]))
        if count != item["detections"] or int(selected.sum()) != item["frames"]:
            raise AssertionError("condition count disagrees with stored scores")
        detections.append({"snr_db": item["snr_db"], "detections": count})
    positive, negative = scores[labels == 1], scores[labels == 0]
    # This is an intentionally small, independent fixed-fixture oracle.
    auc = float(np.mean((positive[:, None] > negative) + .5*(positive[:, None] == negative)))
    if not np.isclose(auc, expected["auc"], atol=1e-12, rtol=0):
        raise AssertionError("AUC disagrees with independent pairwise ranking")
    return {"false_alarms": false_alarms, "h0_frames": len(negative), "h1": detections, "independent_auc": auc}


def main() -> None:
    """Read supplied run records and write an exclusive independent replay record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-record", type=Path, required=True)
    parser.add_argument("--runtime-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing recount")
    source = json.loads(args.native_record.read_text(encoding="utf-8"))
    stress = json.loads(args.runtime_record.read_text(encoding="utf-8"))
    checks = {}
    with np.load(args.native_record.parent/"scores.npz", allow_pickle=False) as frozen:
        for name, metrics in source["models"].items():
            checks[f"{name}_native_smoke"] = recount(frozen[name], frozen["labels"], frozen["snr"], metrics)
        checks["energy_native_smoke"] = recount(frozen["energy"], frozen["labels"], frozen["snr"], source["energy_detector"])
    with np.load(args.runtime_record.parent/"scores.npz", allow_pickle=False) as frozen:
        for name, runtimes in stress["models"].items():
            for mode, values in runtimes.items():
                for condition in values["conditions"]:
                    drift = condition["gain_drift_db"]
                    key = f"{name}_{mode}_gain_{drift}"
                    checks[key] = recount(frozen[key], frozen["labels"], frozen["snr"], condition["runtime_calibrated"])
                    if "native_threshold_transferred" in condition:
                        checks[key+"_transferred_threshold"] = recount(frozen[key], frozen["labels"], frozen["snr"], condition["native_threshold_transferred"])
                        native = frozen[f"{name}_native_gain_0"]
                        threshold = source["models"][name]["calibration"]["threshold"]
                        changes = int(np.count_nonzero((native >= threshold) != (frozen[key] >= threshold)))
                        if changes != condition["decision_changes_at_native_threshold"]:
                            raise AssertionError("paired decision-change count disagrees")
                        checks[key+"_transferred_threshold"]["paired_decision_changes"] = changes
    result = {"scope": "independent arithmetic replay of fixed synthetic arrays, not new experiments",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                               [args.native_record, args.runtime_record, args.native_record.parent/"scores.npz", args.runtime_record.parent/"scores.npz"]},
              "checks": checks}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Independent counts and pairwise AUC agree for {len(checks)} recorded conditions.")


if __name__ == "__main__":
    main()
