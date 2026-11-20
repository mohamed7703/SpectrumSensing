"""Independently recount fresh decisions and summarize the frozen primary endpoint."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main(folder: Path) -> None:
    """Validate a complete run, summarize by condition, and compute blocked uncertainty."""
    manifest=json.loads((folder/"manifest.json").read_text())
    c=manifest["config"]
    rows=np.atleast_1d(np.genfromtxt(folder/"counts.csv",delimiter=",",names=True,dtype=None,encoding="utf8"))
    seeds=c["model_seeds"]
    gains=c["gains_db"]
    conds=[(g,0) for g in [*gains,c["transfer_gain_db"]]]+[(0,c["noise_stress_db"])]
    methods=["linear","ordinary","fixed","fp32","energy"]
    expected=len(seeds)*c["sessions_per_model"]*len(conds)*len(c["snrs_db"])*len(methods)*len(c["calibration_budgets_per_gain"])
    assert len(rows)==expected and [b["seed"] for b in manifest["blocks"]]==seeds
    keys=set()
    for seed in seeds:
        for session in range(c["sessions_per_model"]):
            group=rows[(rows["seed"]==seed)&(rows["session"]==session)]
            with np.load(folder/f"scores_{seed}_{session}.npz",allow_pickle=False) as data:
                for r in group:
                    ci=conds.index((r["gain"],r["noise_db"]))
                    key=(seed,session,ci,r["snr"],r["budget"],r["method"])
                    assert key not in keys
                    keys.add(key)
                    rt=r["method"] if r["method"] in ("fp32","energy") else "int8"
                    d=data[f"test_{ci}_{rt}"]>=r["threshold"]
                    nh,ns=c["test_h0_frames_per_condition"],c["test_h1_frames_per_snr"]
                    start=nh+c["snrs_db"].index(r["snr"])*ns
                    assert (d[:nh].sum(),d[start:start+ns].sum(),nh,ns)==(r["fa"],r["hits"],r["h0"],r["h1"])
    groupkeys=["gain","noise_db","snr","budget","method"]
    summary=[]
    for key in sorted({tuple(r[k] for k in groupkeys) for r in rows}):
        mask=np.ones(len(rows),dtype=bool)
        for k,v in zip(groupkeys,key):
            mask &= rows[k]==v
        r=rows[mask]
        summary.append([*key,*[r[k].sum() for k in ["fa","h0","hits","h1"]]])
    with (folder/"summary.csv").open("w",newline="") as f:
        csv.writer(f).writerows([[*groupkeys,"fa","h0","hits","h1"],*summary])
    primary=rows[(rows["budget"]==c["primary_budget"])&(rows["snr"]==c["primary_snr_db"])&
                 (rows["noise_db"]==0)&np.isin(rows["gain"],gains)]
    def rates(method: str, gain=None) -> np.ndarray:
        sel=primary[primary["method"]==method]
        if gain is not None:
            sel=sel[sel["gain"]==gain]
        return np.array([[sum(s["hits"])/sum(s["h1"]),sum(s["fa"])/sum(s["h0"])]
            for seed in seeds for s in [sel[sel["seed"]==seed]]])
    diff=rates("linear")[:,0]-rates("ordinary")[:,0]
    radius=stats.t.ppf(.975,len(seeds)-1)*stats.sem(diff)
    bounds={}
    for method in ["linear","ordinary"]:
        for g in gains:
            r=primary[(primary["method"]==method)&(primary["gain"]==g)]
            p=rates(method,g)[:,1]
            alpha=(1-c["calibration_confidence"])/(2*len(gains))
            exact=stats.binomtest(int(sum(r["fa"])),int(sum(r["h0"])),alternative="less").proportion_ci(1-alpha).high
            upper=min(1,max(exact,p.mean()+stats.t.ppf(1-alpha,len(seeds)-1)*stats.sem(p)))
            bounds[f"{method}_{g}"]=dict(observed=p.mean(),upper=upper)
    result=dict(verified_rows=len(keys),paired_model_deltas=diff.tolist(),mean_difference=diff.mean(),
                interval=[diff.mean()-radius,diff.mean()+radius],pfa=bounds,
                detection={m:rates(m)[:,0].mean() for m in methods},
                recommend_superiority_pilot=bool(diff.mean()-radius>0 and all(v["upper"]<=c["target_pfa"] for v in bounds.values())))
    (folder/"primary.json").write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for m in methods:
        axes[0].plot(gains,[100*rates(m,g)[:,0].mean() for g in gains],"o-",label=m)
        axes[1].plot(gains,[100*rates(m,g)[:,1].mean() for g in gains],"o-",label=m)
    axes[1].axhline(100*c["target_pfa"],color="black",ls="--",lw=1,label="False-alarm limit")
    for ax,title in zip(axes,["Detection at 0 dB SNR (%)","False alarms (%)"]):
        ax.set(xlabel="Gain drift (dB)",ylabel=title,xticks=gains)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fresh synthetic QPSK · 600 calibration frames · five model blocks")
    fig.savefig(folder/"primary.png",dpi=160)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    main(parser.parse_args().folder)
