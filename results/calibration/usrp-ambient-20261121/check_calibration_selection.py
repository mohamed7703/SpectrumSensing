"""Check disjoint alternative calibration samples against the original held-out scores."""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis"
REPO = BASE.parents[2]
sys.path.insert(0, str(REPO))
from scripts.validate_recovery_native import tflite_scores
from src.math_models.threshold_calibration import calibrate_score_threshold


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    lock = json.loads((OUT / "early_lock.json").read_text())
    iq = np.memmap(BASE / "ampient_turned_off.iq", dtype="<c8", mode="r")
    original = set(lock["calibration_sample_offsets"])
    # One frame per eighth of each second per panel; sample without replacement
    # across panels and exclude the original 200 frames.
    rng = np.random.default_rng(20261124)
    offsets = []
    for second in range(5, 30):
        frames = second * 1_000_000 + np.arange(976) * 1024
        for stratum in np.array_split(frames, 8):
            eligible = [int(i) for i in stratum if i not in original]
            offsets.append(rng.choice(eligible, size=20, replace=False))
    panels = np.asarray(offsets).T
    assert np.min(np.diff(np.sort(panels.ravel()))) >= 1024
    model_dir = REPO / "docs/calibration/usrp-2920"
    fp = tf.keras.models.load_model(model_dir/"gap-seed19031-fp32.keras", compile=False)
    raw = np.stack([iq[i:i+1024] for i in panels.ravel()])
    x = raw.view("<f4").reshape(-1,1024,2)*np.float32(lock["scale"])
    scores = {"fp32":np.concatenate([fp(x[i:i+64], training=False).numpy().ravel()
                                     for i in range(0,len(x),64)]).reshape(20,200),
              "int8":tflite_scores(str(model_dir/"gap-seed19031-int8.tflite"),x)[0].reshape(20,200)}
    np.savez_compressed(OUT/"selection_calibration_scores.npz", start_sample=panels, **scores)
    with np.load(OUT/"all_test_scores.npz") as test, (OUT/"selection_sensitivity.csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panel","model","strict_cutoff","triggers","test_frames"])
        for name, groups in scores.items():
            held = test[name]
            rates = []
            cutoffs = []
            for panel, group in enumerate(groups,1):
                threshold = calibrate_score_threshold(group).threshold
                count = np.count_nonzero(held >= threshold)
                cutoff = np.nextafter(threshold,-np.inf)
                writer.writerow([panel,name,cutoff,count,len(held)])
                cutoffs.append(cutoff)
                rates.append(100*count/len(held))
            print(name,"min/median/max trigger %",np.percentile(rates,[0,50,100]),
                  "distinct cutoffs",len(set(cutoffs)),"same as original",sum(c==np.nextafter(lock[name],-np.inf) for c in cutoffs))


if __name__=="__main__":
    main()
