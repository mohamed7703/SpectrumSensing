"""Summarize every complex sample and calculate a comparable Welch PSD per second."""
import hashlib
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

BASE = Path(__file__).resolve().parent
RATE = 1_000_000


def db(power):
    return 10*np.log10(np.maximum(power,1e-300))


def main():
    output = BASE / "analysis"
    output.mkdir(exist_ok=True)
    iq = np.memmap(BASE/"ampient_turned_off.iq", mode="r",dtype="<c8")
    assert iq.size == 231970431
    digest = hashlib.sha256()
    rows = []
    total_zero = 0
    longest_zero = run = 0
    baseline = np.mean(np.abs(iq[RATE:5*RATE].astype(np.complex128))**2)
    startup = iq[:10000].astype(np.complex128)
    above = np.flatnonzero(np.abs(startup) > .001)
    first_power = (abs(startup)**2).reshape(-1,100).mean(1)
    for t,start in enumerate(range(0,len(iq),RATE)):
        src = iq[start:start+RATE]
        digest.update(src.tobytes())
        x = src.astype(np.complex128)
        p = np.abs(x)**2
        f,s = welch(x,fs=RATE,nperseg=16384,noverlap=8192,window="hann",
                    detrend=False,return_onesided=False)
        nonzero = np.flatnonzero(p)
        if nonzero.size:
            longest_zero = max(longest_zero,run+nonzero[0],np.diff(nonzero).max(initial=1)-1)
            run = len(x)-1-nonzero[-1]
        else:
            run += len(x)
        longest_zero = max(longest_zero,run)
        total_zero += len(x)-len(nonzero)
        # Peak full one-millisecond block, plus its timestamp; the incomplete final
        # millisecond still contributes to the per-second summaries and hash.
        ms = p[:len(p)//1000*1000].reshape(-1,1000).mean(1)
        rows.append(dict(start_s=t,samples=len(x),mean_power_db=db(p.mean()),
                 median_psd_db_hz=db(np.median(s)),
                 earlier_tone_band_pct=100*s[abs(f+26733.4)<=200].sum()/s.sum(),
                 center_1khz_pct=100*s[abs(f)<=1000].sum()/s.sum(),
                 strongest_bin_hz=f[np.argmax(s)],max_ms_power_db=db(ms.max()),
                 max_ms_at_s=t+np.argmax(ms)/1000,ms_over_baseline_6db=int(sum(ms>baseline*10**.6)),
                 nonfinite=int(sum(~np.isfinite(x))),near_rail=int(sum((abs(x.real)>=.99)|(abs(x.imag)>=.99))),
                 peak_component=float(max(abs(x.real).max(),abs(x.imag).max()))))
    with (output/"quality_per_second.csv").open("w",newline="") as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta=dict(bytes=iq.nbytes,samples=iq.size,sha256=digest.hexdigest(),
              total_exact_zeros=int(total_zero),longest_zero_run_samples=int(longest_zero),
              startup_above_001_indices=above.tolist(),startup_first_three_100us_db=db(first_power[:3]).tolist(),
              sample_at_60s_sha256=hashlib.sha256(iq[60*RATE:61*RATE].tobytes()).hexdigest())
    (output/"capture.json").write_text(json.dumps(meta,indent=2))
    print(meta)
    for field in rows[0]:
        print(field,np.percentile([row[field] for row in rows[1:]],[0,50,100]))


if __name__=="__main__":
    main()
