# NI USRP-2920 two-day capture and replay

> **Update, 2026-11-20 (pre-USRP v2, branch `codex/pre-usrp-v2`).** `fit` and `replay` now lock and replay six methods (ordinary, linear, fixed nominal INT8, FP32, INT8 with the FP32 threshold copied, and gain-aware energy). They require `--native-model <FP32 .keras>` in addition to `--model`, and the calibration session must declare `nominal_gain_db` with at least 3x the budget of noise frames at that gain. Locks from earlier runs (format 1) cannot be replayed. The command list below shows the original two-method workflow; the v2 protocol will replace it.

**Pilot question.** Does a small threshold shift admit a block of tied INT8 noise scores on this receiver, and does ordinary recalibration transfer to another day? The fresh synthetic benchmark found no improvement from the linear rule. Use this pilot to characterize the failure. The USRP acquires I/Q and inference runs on the host.

## Setup

Record the serial, Ethernet interface, driver version, input port, source, clock, attenuation and cabling. Use the lab's supported receive application. The model is confirmed. The available reference source or second SDR, attenuation and center frequency are pending local confirmation. Check actual readbacks and input limits for the lab instrument. [NI USRP-2920 specifications](https://www.ni.com/docs/en-US/bundle/usrp-2920-specs/page/specs.html)

Use a controlled cabled QPSK source through suitable attenuation. For H0, disable the signal while preserving impedance and noise path, and log the source state. A matched termination gives an additional instrument-noise control. For H1, log source state and measure received signal and noise in the same bandwidth and gain setting. For stable matched on/off measurements compute SNR as 10 log10((P_on−P_off)/P_off) using linear powers. Mark SNR unresolved if the difference is nonpositive. Exclude transitions and settling from labeled intervals.

Choose a nominal manual gain with headroom and three supported gain settings around it, preferably center ±1 dB. Record actual **absolute** gains. A starting sample rate is 1 MS/s with QPSK at 125 ksymbols/s for eight samples per symbol. Record actual pulse shaping and clock settings. Freeze one input scale for all gains and both days. One option is to map mean nominal-gain noise power to one using a separate day-one engineering recording. Record that frame count as shared setup cost. Preserve unnormalized raw I/Q.

## Run sheet

| Session | Acquire | Deliver |
|---|---|---|
| Day 1, calibration | Confirmed H0 at three gains. Capture at least 400 usable 1024-sample frames per gain after settling. | Check the raw data, label source, gain, timestamps, expected sample counts and overflow counters. Fit on 200 frames per gain. Lock thresholds and model identity. |
| Day 2, test | Independent H0 and known QPSK H1 at each gain. Aim for 300 H0 and 100 H1 frames per gain and measured SNR near −10, 0 and +10 dB. | Run quality checks and replay with the day-one lock. Record detections, false alarms and clipping for each condition. |

Use separate session IDs and UTC dates. A single test day characterizes the observed transfer. More independent days are needed for cross-day uncertainty.

## File format

Copy [session.template.json](session.template.json) for each session and [labels.template.csv](labels.template.csv) for each recording. Write little-endian float32 I then Q, equivalent to little-endian complex64, eight bytes per complex sample. State the true full-scale component amplitude, acquisition-side sample count, overflow and dropped-sample counters, and SHA-256. Unknown loss or discontinuity calls for reacquisition. Labels use zero-based half-open sample ranges. Gaps are allowed for settling. The checker reports unused partial-frame tails, raw mean complex power and near-full-scale component fraction.

The checker verifies file consistency. Label and SNR ground truth come from the lab setup. Keep hardware captures under `data/raw/`. The supplied model is the first prespecified synthetic seed, 19031. Its SHA-256 is `c9dac2c86285ebc371ab55c464d83dd2082f520833afe458086bdb646d57320d`.

## Offline commands

Run from the repository root in the research Python environment. Use new output names. The fixture command creates two labeled example sessions.

```powershell
$tool = "scripts/usrp_calibration_replay.py"
$model = "docs/calibration/usrp-2920/gap-seed19031-int8.tflite"
$dir = "results/local-runs/2920-example"
python $tool fixture $dir
python $tool check "$dir/calibration.json" --out "$dir/quality-day1.json"
python $tool check "$dir/test.json" --out "$dir/quality-day2.json"
python $tool fit "$dir/calibration.json" --model $model --budget 100 --out "$dir/lock.json"
python $tool replay "$dir/test.json" --model $model --lock "$dir/lock.json" --out "$dir/decisions.json"
```

For hardware use the real session paths, mark evidence type as hardware and fit with `--budget 200`. Replay requires a later UTC day, matching receiver context, the locked model, distinct recordings, and both labels at every calibrated gain. Return the raw files or their approved storage location, sidecars, label CSVs, quality records, lock and counts. [Completed synthetic dry run](../../../results/calibration/2920-dry-run-v1/)
