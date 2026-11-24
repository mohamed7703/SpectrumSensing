"""Recount saved scores and draw within-session summaries from both result tables."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent/"analysis"


def read(name):
    return np.genfromtxt(BASE/name, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    q = read("quality_per_second.csv")
    d = read("scores.csv")
    summary = json.loads((BASE/"inference.json").read_text())
    capture = json.loads((BASE/"capture.json").read_text())
    assert q["samples"].sum()==capture["samples"] and q["start_s"].tolist()==list(range(232))
    np.testing.assert_array_equal(d["second"], d["start_sample"]//1_000_000)
    test = d[d["second"]>=30]
    # All bins below contain ten seconds, except the final two-second bin.
    edges=np.r_[np.arange(30,232,10),232]
    bins=np.searchsorted(edges,test["second"],side="right")-1
    metrics={}
    fp_color,int_color="#1e6388","#b74821"
    fig,(ax,bx,cx)=plt.subplots(3,1,figsize=(10,8.7),layout="constrained")
    q1=q[1:]
    ax.plot(q1["start_s"],q1["mean_power_db"],color=fp_color,lw=1.1)
    ax.set(title="Mean I/Q power after the first second",ylabel="Digital power (dB)")
    bx.plot(q["start_s"],q["median_psd_db_hz"],color="#367757",lw=1.1)
    bx.set(title="Broadband background across the capture",ylabel="Median PSD (dB/Hz)")
    for a in (ax,bx,cx):
        a.grid(alpha=.2)
        a.ticklabel_format(useOffset=False,axis="y")
        a.set_xlim(0,233)
    table=[]
    for name,color in [("fp32",fp_color),("int8",int_color)]:
        m=summary["models"][name]
        triggers=test[name]>=m["threshold"]
        assert int(triggers.sum())==m["triggers"]
        np.testing.assert_array_equal(triggers,test[name+"_trigger"])
        counts=np.bincount(bins,weights=triggers,minlength=len(edges)-1).astype(int)
        n=np.bincount(bins,minlength=len(edges)-1)
        rates=100*counts/n
        cx.stairs(rates,edges,baseline=None,label=name.upper(),color=color,lw=1.7)
        cut=np.nextafter(m["threshold"],-np.inf)
        energy_quartiles=np.quantile(test["raw_power_db"],[.25,.5,.75])
        quart=np.searchsorted(energy_quartiles,test["raw_power_db"])
        metrics[name]=dict(cutoff_score=cut,eval_equal_cutoff=int((test[name]==cut).sum()),
                triggered_seconds=int(len(np.unique(test["second"][triggers]))),
                energy_quartiles_db=energy_quartiles.tolist(),
                energy_quartile_triggers=np.bincount(quart,weights=triggers).astype(int).tolist(),
                energy_quartile_counts=np.bincount(quart).tolist(),window_triggers=counts.tolist(),
                window_sizes=n.tolist())
        for a,b,k,den in zip(edges[:-1],edges[1:],counts,n):
            table.append(dict(model=name,start_s=a,end_s_exclusive=min(b,capture["samples"]/1e6),
                              sampled_frames=den,triggers=k))
    metrics["paired_both"]=int(np.sum((test["fp32_trigger"]==1)&(test["int8_trigger"]==1)))
    with (BASE/"trigger_windows.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (BASE/"checks.json").write_text(json.dumps(metrics,indent=2))
    cx.legend(ncols=2)
    cx.set(title="Ambient trigger rate in successive test windows",xlabel="Time in capture (s)",
           ylabel="Sampled frames (%)")
    fig.savefig(BASE/"session_summary.png",dpi=160)
    plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
