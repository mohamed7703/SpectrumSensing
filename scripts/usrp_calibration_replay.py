"""Check and replay labeled NI USRP-2920 host I/Q recordings without a radio."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_recovery_native import predict, tflite_scores
from src.math_models.energy_detection import compute_energy_statistic
from src.math_models.threshold_calibration import calibrate_score_threshold
from src.training.dataset_loader import SyntheticRadioMLGenerator

FRAME = 1024
CONTEXT = ["device_model", "serial", "frequency_hz", "sample_rate_hz", "bandwidth_hz", "antenna_port",
           "model_scale", "full_scale", "sample_format", "driver", "setup_id", "evidence_type"]


def digest(path: Path) -> str:
    """Hash a recording or model in bounded blocks."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def recordings(session_file: Path, role: str | None = None):
    """Validate a session and yield framed, labeled intervals from mapped I/Q."""
    meta = json.loads(session_file.read_text(encoding="utf-8-sig"))
    for key in [*CONTEXT, "session_id", "started_utc", "role", "gain_control"]:
        if key not in meta or not str(meta[key]).strip():
            raise ValueError(f"missing {key}")
    dt = datetime.fromisoformat(meta["started_utc"].replace("Z","+00:00"))
    if dt.utcoffset() is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("use UTC timestamps")
    if meta["role"] not in ("calibration","test") or (role and meta["role"] != role):
        raise ValueError("incorrect session role")
    if (meta["sample_format"] != "cf32_le" or meta["gain_control"] != "manual"
            or meta["device_model"] != "NI USRP-2920" or meta["evidence_type"] not in ("synthetic_fixture","hardware")):
        raise ValueError("unexpected format, device or acquisition mode")
    for name in ("sample_rate_hz","frequency_hz","bandwidth_hz","model_scale","full_scale"):
        if not np.isfinite(meta[name]) or meta[name] <= 0:
            raise ValueError(f"invalid {name}")
    if not meta["recordings"]:
        raise ValueError("empty session")
    seen = set()
    for entry in meta["recordings"]:
        p = session_file.parent/entry["iq_file"]
        gain = float(entry["gain_db"]) + 0.0
        if not np.isfinite(gain) or entry["overflows"] != 0 or entry["dropped_samples"] != 0:
            raise ValueError("invalid gain or discontinuous capture")
        if p.stat().st_size != entry["sample_count"]*8 or entry["sample_count"] < FRAME:
            raise ValueError("I/Q byte count disagrees with recorded complex sample count")
        checksum = digest(p)
        if checksum != entry["sha256"] or checksum in seen:
            raise ValueError("I/Q hash mismatch or duplicated recording")
        seen.add(checksum)
        iq = np.memmap(p, mode="r", dtype="<c8")
        last = 0
        with (session_file.parent/entry["labels_file"]).open(encoding="utf-8-sig",newline="") as f:
            labels = list(csv.DictReader(f))
        if not labels:
            raise ValueError("empty labels")
        for label in labels:
            start, stop = int(label["start_sample"]), int(label["stop_sample"])
            if not last <= start < stop <= len(iq) or stop-start < FRAME:
                raise ValueError("overlapping, out-of-range or too-short label")
            if label["class"] not in ("H0","H1") or not label["ground_truth"].strip():
                raise ValueError("unknown label or missing ground truth")
            if meta["role"] == "calibration" and label["class"] != "H0":
                raise ValueError("fit accepts confirmed noise-only intervals")
            snr = float(label["snr_db"]) if label["class"] == "H1" else None
            if snr is not None and not np.isfinite(snr):
                raise ValueError("missing measured H1 SNR")
            last = stop
            n = (stop-start)//FRAME
            rows = iq[start:start+n*FRAME].view("<f4").reshape(n,FRAME,2)
            yield meta, checksum, gain, label["class"], snr, rows, (stop-start)%FRAME


LOCK_FORMAT = 2
RUNTIMES = ("int8", "fp32", "energy")


def score_key(runtime: str, gain: float, label: str, snr: float | None) -> str:
    """Name one per-frame score array in the replay archive."""
    return f"{runtime}__g{float(gain)}__{label}__snr{snr if snr is None else float(snr)}"


def nominal_gain_of(meta: dict) -> float:
    """Return the nominal receiver gain that a calibration session declares."""
    if "nominal_gain_db" not in meta:
        raise ValueError("session lacks nominal_gain_db")
    value = float(meta["nominal_gain_db"])
    if not np.isfinite(value):
        raise ValueError("invalid nominal_gain_db")
    return value


def score_runtimes(batch: np.ndarray, scale: float, model: Path, native) -> tuple[dict, dict]:
    """Score one raw batch with INT8, FP32 and energy; also return INT8 input-clipping counts."""
    x = np.asarray(batch*scale, dtype=np.float32)
    int8, details = tflite_scores(str(model), x)
    return dict(int8=np.asarray(int8, dtype=np.float64), fp32=np.asarray(predict(native, x, len(x)), dtype=np.float64),
                energy=np.asarray(compute_energy_statistic(x), dtype=np.float64)), details


def fit_thresholds(calibration: dict, nominal_gain: float, budget: int) -> dict:
    """Fit the six rules from confirmed noise-only scores; each is charged budget frames at every gain."""
    gains = sorted(calibration)
    if len(gains) < 3 or nominal_gain not in calibration:
        raise ValueError("need at least three gains including the nominal gain")
    charged = len(gains)*budget
    if len(calibration[nominal_gain]["int8"]) < charged:
        raise ValueError("need gains*budget frames at the nominal gain")
    if min(len(calibration[g]["int8"]) for g in gains) < budget:
        raise ValueError("need the chosen budget at every gain")
    def boundary(values: list) -> float:
        return float(calibrate_score_threshold(np.array(values)).threshold)
    local = {rt: [boundary(calibration[g][rt][:budget]) for g in gains] for rt in RUNTIMES}
    slope, intercept = np.polyfit(gains, np.nextafter(local["int8"], -np.inf), 1)
    fixed = boundary(calibration[nominal_gain]["int8"][:charged])
    def entry(runtime: str, threshold: float) -> dict:
        return dict(runtime=runtime, threshold=float(threshold), h0_frames_charged=charged)
    return {str(g): dict(
        ordinary=entry("int8", local["int8"][i]),
        linear=entry("int8", np.nextafter(np.round(g*slope+intercept, 12), np.inf)),
        fixed=entry("int8", fixed), fp32=entry("fp32", local["fp32"][i]),
        int8_copied_fp32=entry("int8", local["fp32"][i]), energy=entry("energy", local["energy"][i]))
        for i, g in enumerate(gains)}


def process(path: Path, mode: str, model: Path | None, lock: dict | None, budget: int,
            native_model: Path | None = None, scores_out: Path | None = None) -> dict:
    """Check labeled windows and fit locked thresholds or replay on a separate day."""
    role = {"fit":"calibration","replay":"test"}.get(mode)
    summary, calibration, counts, hashes, archive = [], {}, {}, set(), {}
    identity = digest(model) if model else None
    native_id = digest(native_model) if native_model else None
    if lock and identity != lock["model_sha256"]:
        raise ValueError("model identity differs from locked calibration")
    if mode == "fit" and native_model is None:
        raise ValueError("fit requires the native model")
    if mode == "replay" and scores_out is not None and scores_out.exists():
        raise ValueError("choose a new score archive")
    native = None
    if native_model is not None and mode in ("fit", "replay"):
        import tensorflow as tf
        native = tf.keras.models.load_model(native_model, compile=False, safe_mode=True)
    nominal_gain, n_gains = None, 0
    for meta, checksum, gain, label, snr, rows, tail in recordings(path, role):
        hashes.add(checksum)
        if mode == "fit" and n_gains == 0:
            nominal_gain = nominal_gain_of(meta)
            recorded = {float(e["gain_db"]) + 0.0 for e in meta["recordings"]}
            if nominal_gain not in recorded:
                raise ValueError("nominal gain is not a recorded gain")
            n_gains = len(recorded)
        if lock:
            if (meta["session_id"] == lock["session_id"] or meta["started_utc"][:10] <= lock["day"]
                    or checksum in lock["iq_hashes"] or any(meta[k] != lock["context"][k] for k in CONTEXT)):
                raise ValueError("replay requires a later independent day with matching acquisition context")
            if lock.get("lock_format") != LOCK_FORMAT:
                raise ValueError(f"lock format {LOCK_FORMAT} required; refit the calibration day")
            if native_id is None or native_id != lock["native_sha256"]:
                raise ValueError("native model differs from locked calibration")
            if str(gain) not in lock["thresholds"]:
                raise ValueError("test gain absent from locked table")
        power, near, total = 0., 0, 0
        for offset in range(0,len(rows),128):
            batch = rows[offset:offset+128]
            if not np.isfinite(batch).all():
                raise ValueError("nonfinite raw I/Q")
            power += float(np.square(batch,dtype=np.float64).sum())
            near += int((np.abs(batch) >= .99*meta["full_scale"]).sum())
            total += batch.size
            if mode == "fit":
                sample = calibration.setdefault(gain, {rt: [] for rt in RUNTIMES})
                take = min((n_gains*budget if gain == nominal_gain else budget)-len(sample["int8"]), len(batch))
                if take:
                    for rt, values in score_runtimes(batch[:take], meta["model_scale"], model, native)[0].items():
                        sample[rt].extend(values)
            if mode == "replay":
                scores, details = score_runtimes(batch, meta["model_scale"], model, native)
                if not all(np.isfinite(values).all() for values in scores.values()):
                    raise ValueError("nonfinite score; a NaN would count as a silent non-detection")
                for rt, values in scores.items():
                    archive.setdefault(score_key(rt, gain, label, snr), []).extend(values)
                for method, entry in lock["thresholds"][str(gain)].items():
                    row = counts.setdefault((gain, label, snr, method), [0,0,0,0])
                    row[0] += int((scores[entry["runtime"]] >= entry["threshold"]).sum())
                    row[1] += len(batch)
                    row[2] += details["clipped_input_values"] if entry["runtime"] == "int8" else 0
                    row[3] += details["total_input_values"]
        if power <= 0:
            raise ValueError("all-zero interval")
        summary.append(dict(gain_db=gain,label=label,snr_db=snr,frames=len(rows),tail_discarded=tail,
                            mean_complex_power=2*power/total,near_full_scale_fraction=near/total))
    output = dict(session_id=meta["session_id"],day=meta["started_utc"][:10], context={k:meta[k] for k in CONTEXT},
                  iq_hashes=sorted(hashes), model_sha256=identity, native_sha256=native_id, intervals=summary)
    if mode == "fit":
        output.update(lock_format=LOCK_FORMAT, nominal_gain_db=nominal_gain, budget_per_gain=budget,
                      thresholds=fit_thresholds(calibration, nominal_gain, budget))
    if mode == "replay":
        for gain in lock["thresholds"]:
            if {s["label"] for s in summary if s["gain_db"] == float(gain)} != {"H0","H1"}:
                raise ValueError("test requires both labels at every locked gain")
        output["decisions"] = [dict(gain_db=g,label=l,snr_db=s,method=m,positive=v[0],frames=v[1],
                                    input_clipped=v[2],scalar_inputs=v[3]) for (g,l,s,m),v in counts.items()]
        if scores_out is not None:
            with scores_out.open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in archive.items()})
            output["scores_sha256"] = digest(scores_out)
    return output


def fixture(folder: Path, calibration_frames: int = 100, gains: tuple = (9, 10, 11)) -> None:
    """Create two labeled synthetic sessions; the nominal-gain (10 dB) calibration recording holds one budget per gain."""
    folder.mkdir(parents=True,exist_ok=False)
    for day,role in enumerate(["calibration","test"],1):
        meta = dict(device_model="NI USRP-2920",serial="FIXTURE",frequency_hz=915e6,sample_rate_hz=1e6,
                    bandwidth_hz=1e6,antenna_port="fixture",model_scale=8,full_scale=1,sample_format="cf32_le",
                    driver="synthetic generator",setup_id="fixture-v1",evidence_type="synthetic_fixture",
                    session_id=f"fixture-{day}",started_utc=f"2026-01-0{day}T12:00:00Z",role=role,
                    gain_control="manual",nominal_gain_db=10,recordings=[])
        gen = SyntheticRadioMLGenerator(70100+day)
        for gain in gains:
            base = folder/f"{role}-{gain}"
            n0,n1 = (calibration_frames*(len(gains) if gain==10 else 1),0) if day==1 else (32,32)
            x = np.stack([gen.generate_single_frame("NOISE" if i<n0 else "QPSK",0) for i in range(n0+n1)])
            iq = (x[:,:,0]+1j*x[:,:,1])*10**((gain-10)/20)/8
            iq.astype("<c8").tofile(base.with_suffix(".iq"))
            with base.with_suffix(".csv").open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(["start_sample","stop_sample","class","snr_db","ground_truth"])
                w.writerow([0,n0*FRAME,"H0","","synthetic noise"])
                if n1:
                    w.writerow([n0*FRAME,(n0+n1)*FRAME,"H1",0,"synthetic QPSK"])
            meta["recordings"].append(dict(iq_file=base.name+".iq",labels_file=base.name+".csv",gain_db=gain,
                       overflows=0,dropped_samples=0,sample_count=(n0+n1)*FRAME,sha256=digest(base.with_suffix(".iq"))))
        (folder/f"{role}.json").write_text(json.dumps(meta,indent=2))


def main() -> None:
    """Run fixture generation, capture checks, threshold fitting or sealed replay."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["fixture","check","fit","replay"])
    p.add_argument("path",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--native-model",type=Path)
    p.add_argument("--lock",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--budget",type=int,default=200)
    args=p.parse_args()
    if args.mode=="fixture":
        fixture(args.path)
        return
    if (args.out is None or (args.mode in ["fit","replay"] and (args.model is None or args.native_model is None))
            or (args.mode=="replay" and args.lock is None)):
        p.error("check requires --out; fit/replay also require --model and --native-model; replay requires --lock")
    if args.out.exists():
        p.error("choose a new output file")
    scores_out=args.out.with_name(args.out.stem+"-scores.npz") if args.mode=="replay" else None
    data=process(args.path,args.mode,args.model,json.loads(args.lock.read_text()) if args.lock else None,args.budget,
                 native_model=args.native_model,scores_out=scores_out)
    if args.lock:
        data["lock_sha256"]=digest(args.lock)
    with args.out.open("x") as f:
        json.dump(data,f,indent=2,allow_nan=False)


if __name__=="__main__":
    main()
