"""Create the report and figures from the completed capture and saved replay outputs."""
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
OUT=BASE/"analysis"


def load(name):
    return json.loads((OUT/name).read_text())


def read(name):
    return np.genfromtxt(OUT/name,delimiter=",",names=True,dtype=None,encoding="utf8")


def main():
    q=read("quality_per_second.csv")
    rows=read("all_test_per_second.csv")
    fit=load("inference.json")
    cap=load("capture.json")
    dense=load("all_test_summary.json")
    data=np.load(OUT/"all_test_scores.npz")
    excerpt=BASE/"diagnostics"/"sample_at_60s.iq"
    sample_sha=hashlib.sha256(excerpt.read_bytes()).hexdigest()
    assert sample_sha==cap["sample_at_60s_sha256"]
    assert dense["frames"]*1024+dense["unused_tail_samples"]+30_000_000==cap["samples"]
    for name in ["fp32","int8"]:
        assert np.isfinite(data[name]).all()
        count=int((data[name]>=fit["models"][name]["threshold"]).sum())
        assert count==rows[name].sum()==dense["models"][name]["triggers"]
    post=q[1:]
    n=dense["frames"]
    fmt=lambda value:f"{value:,}"
    sec=cap["samples"]/1e6
    int_m=dense["models"]["int8"]
    fp_m=dense["models"]["fp32"]
    cut=np.nextafter(fit["models"]["int8"]["threshold"],-np.inf)
    edges=np.r_[np.arange(30,232,10),232]
    bins=np.searchsorted(edges,rows["start_s"],side="right")-1
    sizes=np.bincount(bins,weights=rows["frames"])
    full_window_rates={}
    fig,(ax,bx,cx)=plt.subplots(3,1,figsize=(10,8.7),layout="constrained")
    ax.plot(post["start_s"],post["mean_power_db"],color="#1e6388",lw=1.15)
    ax.set(title="Mean I/Q power after the first second",ylabel="Digital power (dB)")
    bx.plot(q["start_s"],q["median_psd_db_hz"],color="#367757",lw=1.15)
    bx.set(title="Broadband background across the capture",ylabel="Median PSD (dB/Hz)")
    for name,color in [("fp32","#1e6388"),("int8","#b74821")]:
        rate=100*np.bincount(bins,weights=rows[name])/sizes
        full_window_rates[name]=rate[:-1]
        cx.stairs(rate,edges,baseline=None,label=name.upper(),color=color,lw=1.6)
    cx.legend(ncols=2)
    cx.set(title="Ambient trigger rate across all held-out frames",xlabel="Time in capture (s)",
           ylabel="Frames triggering (%)")
    for a in (ax,bx,cx):
        a.grid(alpha=.2)
        a.ticklabel_format(useOffset=False,axis="y")
        a.set_xlim(0,233)
    fig.savefig(OUT/"full_session.png",dpi=160)
    plt.close(fig)

    table=[]
    for name in ["fp32","int8"]:
        m=dense["models"][name]
        f=fit["models"][name]
        rates=full_window_rates[name]
        table.append(f"| {name.upper()} | {np.nextafter(f['threshold'],-np.inf):.9g} | {f['calibration_triggers']}/200 | {m['triggers']:,}/{n:,} ({100*m['triggers']/n:.2f}%) | {m['distinct']:,} | {rates.min():.2f}–{rates.max():.2f}% |")
    start_bins=cap["startup_first_three_100us_db"]
    energetic=post["max_ms_power_db"].argmax()
    scale=fit["scale"]
    floor=post["median_psd_db_hz"]
    peak_second=int(post["start_s"][floor.argmax()])
    peak_row=rows[rows["start_s"]==peak_second][0]
    cutoff_inclusive=int_m["triggers"]+int_m["equal_cutoff"]
    text=f"""# USRP ambient baseline and within-session threshold replay

The full generator-off antenna recording supports a comparison of background stability and detector decisions during one session at the reported 10 dB gain. The measured broadband floor varied by {np.ptp(post['median_psd_db_hz']):.2f} dB across the retained one-second summaries. FP32 triggered on {100*fp_m['triggers']/n:.2f}% of held-out frames and INT8 triggered on {100*int_m['triggers']/n:.2f}%. The recording has no verified signal labels.

## Capture and signal checks

The [source recording](https://drive.google.com/file/d/1_pmfCVqJa41YR3L2EwPTCnhy2a_BL8Po/view) contains {fmt(cap['bytes'])} bytes, or {fmt(cap['samples'])} little-endian complex float32 samples. At the reported 1 MS/s this corresponds to {sec:.6f} seconds. The reported receiver settings were 915 MHz, RX2 and 10 dB gain, with the generator RF output disabled. These settings come from the lab discussion. The raw format contains no timestamps, gain readbacks or overflow log. Acquisition continuity and RF power in dBm remain unverified.

The SHA-256 of the assembled file is `{cap['sha256']}`. Its second at 60 s matches the entire separately supplied 8 MB excerpt byte for byte (SHA-256 `{sample_sha}`). Transfer ranges were checked against their requested offsets and source length.

- Across {len(q)} successive time windows, including the final partial second, {int(q['nonfinite'].sum())} samples had a nonfinite component and {int(q['near_rail'].sum())} had a component at or above 0.99 in absolute value. The longest run of paired I/Q zeros was {cap['longest_zero_run_samples']} samples.
- The opening 100 microseconds averaged {start_bins[0]:.2f} dB in digital power. The next two 100-microsecond bins measured {start_bins[1]:.2f} and {start_bins[2]:.2f} dB. Within the opening 10 ms the magnitude exceeded 0.001 at sample indices {cap['startup_above_001_indices']}. This describes the high-amplitude portion at the file boundary. Its physical onset and origin are unknown. The analysis reserved the entire first second for startup.
- From 1 s onward, mean power per window ranged from {post['mean_power_db'].min():.2f} to {post['mean_power_db'].max():.2f} dB. Median spectral density across frequency bins ranged from {post['median_psd_db_hz'].min():.2f} to {post['median_psd_db_hz'].max():.2f} digital dB/Hz. The power share within ±200 Hz of the earlier −26.7334 kHz tone reached at most {post['earlier_tone_band_pct'].max():.3f}%. The earlier tone-dominated capture had about 99.99% in that band.
- The median share within ±1 kHz of the tuned center was {np.median(post['center_1khz_pct']):.2f}%. A center-frequency feature and frequency-dependent noise are present. Across complete 1 ms windows after the first second, {int(post['ms_over_baseline_6db'].sum())} exceeded the engineering baseline mean power by 6 dB. The strongest averaged {post['max_ms_power_db'][energetic]:.2f} dB at {post['max_ms_at_s'][energetic]:.3f} s. These are descriptive digital-power measures.

![Session power, spectral floor and ambient trigger rates](full_session.png)

The middle 90% of the retained window-level spectral floors lay between {np.percentile(floor,5):.2f} and {np.percentile(floor,95):.2f} digital dB/Hz. The highest floor occurred at {peak_second} s. Among frames starting in that second, FP32 triggered {peak_row['fp32']}/{peak_row['frames']} times and INT8 triggered {peak_row['int8']}/{peak_row['frames']} times. The full floor excursion at fixed gain is comparable in magnitude to the planned 1 dB gain adjustment. This motivates repeat measurements at the reference gain.

## Model setup and time split

Both models use the project's matching GAP seed 19031 native Keras checkpoint and INT8 TFLite export. Multiply all raw I/Q values by one fixed factor of **{scale:.5f}** before inference. This maps average complex power during 1–5 s to one. Those four seconds were used only to choose the scale. The setup preserves later changes in amplitude. The scaled dense test clipped {dense['clipped_input_values']:,} of {dense['input_values']:,} scalar values during INT8 input conversion. Without scaling, all scalar values in the systematically sampled calibration/evaluation frames mapped to code {fit['unscaled_int8_input_codes']}. This experimental scale was established on this capture.

From 5–30 s, select eight evenly spaced, disjoint 1,024-sample frames per second. Use the same 200 frames to fit each model's own cutoff. Apply the project's nominal 5% / 95% order-statistic rule, selecting rank 196 of 200 and triggering strictly above that score. From 30 s to the end, run every complete contiguous frame with those thresholds frozen. Discard the final {dense['unused_tail_samples']} samples, fewer than one frame. Time separation prevents shared samples, though consecutive frames can still be dependent.

| Model | Strict cutoff | Calibration triggers | Held-out ambient triggers | Distinct held-out scores | Range across full 10 s test windows |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

At the INT8 boundary, {int_m['equal_cutoff']:,} test frames had exactly the cutoff score ({cut:.8f}). Including that single tied group would raise the ambient trigger rate to {100*cutoff_inclusive/n:.2f}%. The most frequent INT8 score was {int_m['modal_score']:.8f}, shared by {int_m['modal_count']:,} frames. Both detectors triggered on {dense['both_trigger']:,} of the same frames.

FP32 triggered in {fp_m['triggered_seconds']} of {len(rows)} held-out second bins and INT8 in {int_m['triggered_seconds']}. Dividing frames by their raw mean power into four equal-sized groups, from lowest to highest, the FP32 trigger percentages were {', '.join(f'{100*a/b:.2f}%' for a,b in zip(fp_m['energy_quartile_triggers'],fp_m['energy_quartile_counts']))}. INT8 gave {', '.join(f'{100*a/b:.2f}%' for a,b in zip(int_m['energy_quartile_triggers'],int_m['energy_quartile_counts']))}. These are associations within the capture. They do not identify a transmitter.

## What this establishes and what to measure next

This is a single-receiver, single-session ambient replay. The finite output resolution of INT8 has a measurable consequence for threshold selection on these data. A lower trigger rate alone cannot establish better sensing. An antenna can receive unknown emissions, and the capture contains no controlled signal-present intervals to measure missed detections. Neither a false-alarm guarantee nor cross-day repeatability follows from the number of adjacent frames.

On the next lab visit, repeat the generator-off receive-gain sequence **10 → 9 → 11 → 10 dB**, 11 seconds each at the same tuning and sample rate. Log actual gain, port, timestamps, source state and UHD overflow messages. Compare the two 10 dB runs for drift, then replay the other gains with the scale and thresholds above held fixed. To establish false-alarm and detection performance, collect independently labeled noise-only and known signal-present intervals and evaluate on a separate day. Testing incoming broadband noise variation separately requires fixed receive gain and a controlled source.

## Reproducibility

The [signal summaries](quality_per_second.csv), [model identities and calibration thresholds](inference.json), [calibration and systematically sampled scores](scores.csv), [all held-out scores with sample offsets](all_test_scores.npz), [per-second trigger counts](all_test_per_second.csv), and [dense replay totals](all_test_summary.json) accompany this report. The Welch spectra use a 16,384-sample Hann window, 50% overlap, and no detrending. The paths below are relative to this report.

With the capture in the parent directory and the project environment active, run `ambient_quality.py`, `ambient_scores.py`, `all_frame_replay.py`, `plot_and_check.py`, and `finish_report.py` from that parent directory. The final report generation also compares the local original 60 s excerpt. The dated source recording and model hashes identify the inputs.
"""
    (OUT/"REPORT.md").write_text(text,encoding="utf-8")
    print("Verified report and figure",OUT)


if __name__=="__main__":
    main()
