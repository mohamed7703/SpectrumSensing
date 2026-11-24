"""Replay every disjoint complete frame after 30 s with the two frozen thresholds."""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import numpy as np
import tensorflow as tf

BASE=Path(__file__).resolve().parent
OUT=BASE/"analysis"
REPO=BASE.parents[2]
sys.path.insert(0,str(REPO))
from scripts.validate_recovery_native import tflite_scores


def main():
    run=json.loads((OUT/"inference.json").read_text())
    model_dir=REPO/"docs/calibration/usrp-2920"
    fp=tf.keras.models.load_model(model_dir/"gap-seed19031-fp32.keras",compile=False)
    predict=tf.function(lambda x:fp(x,training=False),
                        input_signature=[tf.TensorSpec([None,1024,2],tf.float32)])
    iq=np.memmap(BASE/"ampient_turned_off.iq",dtype="<c8",mode="r")
    start=30_000_000
    n,tail=divmod(len(iq)-start,1024)
    raw=iq[start:start+n*1024].view("<f4").reshape(n,1024,2)
    first=start+np.arange(n)*1024
    sec=first//1_000_000
    scores={k:np.empty(n,dtype="f8") for k in ["fp32","int8","energy_db"]}
    clipped=0
    for i in range(0,n,512):
        sl=slice(i,min(i+512,n))
        frames=raw[sl]*np.float32(run["scale"])
        scores["fp32"][sl]=np.concatenate([predict(frames[j:j+64]).numpy().ravel()
                                          for j in range(0,len(frames),64)])
        scores["int8"][sl],details=tflite_scores(str(model_dir/"gap-seed19031-int8.tflite"),frames)
        clipped+=details["clipped_input_values"]
        scores["energy_db"][sl]=10*np.log10(np.mean(raw[sl].astype(float)**2,axis=(1,2))*2)
    np.savez_compressed(OUT/"all_test_scores.npz",start_sample=first,**scores)
    thresholds={k:run["models"][k]["threshold"] for k in ["fp32","int8"]}
    decisions={k:scores[k]>=v for k,v in thresholds.items()}
    summary=dict(frames=n,unused_tail_samples=tail,clipped_input_values=clipped,
                 input_values=n*2048,both_trigger=int((decisions["fp32"]&decisions["int8"]).sum()),
                 models={})
    quart=np.searchsorted(np.quantile(scores["energy_db"],[.25,.5,.75]),scores["energy_db"])
    rows=[]
    for t in np.unique(sec):
        m=sec==t
        rows.append(dict(start_s=t,frames=m.sum(),fp32=decisions["fp32"][m].sum(),int8=decisions["int8"][m].sum()))
    for name,flags in decisions.items():
        v,c=np.unique(scores[name],return_counts=True)
        equal=scores[name]==np.nextafter(thresholds[name],-np.inf)
        summary["models"][name]=dict(triggers=int(flags.sum()),distinct=len(v),modal_score=float(v[c.argmax()]),
                  modal_count=int(c.max()),equal_cutoff=int(equal.sum()),
                  energy_quartile_counts=np.bincount(quart).tolist(),
                  energy_quartile_triggers=np.bincount(quart,weights=flags).astype(int).tolist(),
                  triggered_seconds=int(len(np.unique(sec[flags]))))
    with (OUT/"all_test_per_second.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT/"all_test_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
