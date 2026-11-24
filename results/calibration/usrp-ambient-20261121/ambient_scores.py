"""Exploratory within-capture comparison; no H0/H1 labels are assumed."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
sys.path.insert(0, str(REPO))
from scripts.validate_recovery_native import tflite_scores
from src.math_models.threshold_calibration import calibrate_score_threshold

RATE = 1_000_000
FRAME = 1024
MODEL = REPO / "docs/calibration/usrp-2920"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", action="store_true")
    parser.add_argument("--lock-only", action="store_true",
                        help="Freeze the full-window calibration before reading any held-out frame.")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    output = BASE / ("prefix_analysis" if args.prefixes else "analysis")
    output.mkdir(exist_ok=True)
    total = 231970431
    if args.prefixes:
        def window(t):
            pos = t * RATE * 8
            stop = pos + 524288
            f = BASE / "ranges" / f"{pos}-{stop}.bin"
            if f.stat().st_size != stop-pos:
                raise ValueError(f"Incomplete range {pos}")
            return np.memmap(f, mode="r", dtype="<c8")
    elif args.lock_only:
        def window(t):
            a=t*RATE*8
            b=a+RATE*8
            split=a+524288
            bounds=[(a,split)]+[(s,min(s+2_000_000,b)) for s in range(split,b,2_000_000)]
            chunks=[]
            for s,e in bounds:
                block=(BASE/"ranges"/f"{s}-{e}.bin").read_bytes()
                assert len(block)==e-s
                chunks.append(block)
            return np.frombuffer(b"".join(chunks),dtype="<c8")
    else:
        iq = np.memmap(BASE/"ampient_turned_off.iq", mode="r", dtype="<c8")
        assert iq.size == total
        def window(t):
            return iq[t*RATE:(t+1)*RATE]
    engineering_power = np.mean([np.mean(np.abs(window(t).astype(np.complex128))**2)
                                 for t in range(1,5)])
    scale = float(1/np.sqrt(engineering_power))
    chunks, starts = [], []
    for t in range(5,30 if args.lock_only else 232):
        w = window(t)
        n = 8 if t < 30 else 32
        rows = w[:w.size//FRAME*FRAME].view("<f4").reshape(-1, FRAME, 2)
        indices = np.floor((np.arange(n)+.5)*len(rows)/n).astype(int)
        chunks.append(rows[indices])
        starts.extend(t*RATE + indices*FRAME)
    raw = np.concatenate(chunks)
    frames = raw*np.float32(scale)
    model = tf.keras.models.load_model(MODEL/"gap-seed19031-fp32.keras", compile=False)
    integer_path = MODEL/"gap-seed19031-int8.tflite"
    fp = np.concatenate([model(segment[i:i+64], training=False).numpy().ravel()
                         for segment in (frames[:200],frames[200:])
                         for i in range(0,len(segment),64)]).astype(float)
    integer, details = tflite_scores(str(integer_path), frames)
    assert np.all(np.isfinite(fp)) and np.all(np.isfinite(integer))
    lock={"scale":scale,"fp32":calibrate_score_threshold(fp[:200]).threshold,
          "int8":calibrate_score_threshold(integer[:200]).threshold,
          "calibration_sample_offsets":starts[:200]}
    lock["calibration_sample_offsets"]=list(map(int,lock["calibration_sample_offsets"]))
    lock_file=output/"early_lock.json"
    if args.lock_only:
        with lock_file.open("x") as f:
            json.dump(lock,f,indent=2)
        print({k:v for k,v in lock.items() if k!="calibration_sample_offsets"})
        return
    if lock_file.exists():
        assert json.loads(lock_file.read_text())==lock, "Early lock mismatch"
    raw_q = np.rint(raw/details["input_quantization"][0]+details["input_quantization"][1])
    raw_codes = np.unique(raw_q)
    df = dict(start_sample=starts, second=np.array(starts)//RATE,
                           raw_power_db=10*np.log10(np.mean(raw.astype(np.float64)**2,axis=(1,2))*2),
                           fp32=fp,int8=integer)
    cal = df["second"] < 30
    assert cal.sum()==200 and (np.diff(starts)>=FRAME).all()
    summary = {"scale":scale, "engineering_samples":sum(len(window(t)) for t in range(1,5)),
               "tflite":details,"unscaled_int8_input_codes":raw_codes.tolist(),
               "fp32_sha256":hashlib.sha256((MODEL/"gap-seed19031-fp32.keras").read_bytes()).hexdigest(),
               "int8_sha256":hashlib.sha256(integer_path.read_bytes()).hexdigest(), "models":{}}
    for name in ["fp32","int8"]:
        fit = calibrate_score_threshold(df[name][cal])
        df[name+"_trigger"] = (df[name] >= fit.threshold).astype(int)
        held = df[name][~cal]
        vals,counts = np.unique(held,return_counts=True)
        summary["models"][name] = dict(threshold=fit.threshold,rank=fit.order_rank,
            calibration_triggers=int(df[name+"_trigger"][cal].sum()),test_n=len(held),
            triggers=int(df[name+"_trigger"][~cal].sum()),calibration_distinct=len(np.unique(df[name][cal])),
            test_distinct=len(vals),modal_test_score=float(vals[counts.argmax()]),
            modal_test_count=int(counts.max()))
    with (output/"scores.csv").open("w",newline="") as target:
        writer=csv.writer(target)
        writer.writerow(df)
        writer.writerows(zip(*df.values()))
    (output/"inference.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
