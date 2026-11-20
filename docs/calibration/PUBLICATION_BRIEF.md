# Publication brief | Reliable spectrum sensing on edge receivers

**Decision for the supervisor, 18 November 2026.** Pursue a receiver-calibration paper and use the *Journal on Wireless Communications and Networking* as the initial target. The research question is practical: after a neural spectrum detector is converted to 8-bit inference and the receiver gain changes, can a short noise-only calibration keep false alarms below a chosen limit while preserving weak-signal detection?

### Why this question is worth testing

In one two-epoch synthetic pilot, the full-precision detector raised 5 false alarms in 300 noise-only trials (1.67%). Reusing its threshold after INT8 conversion raised that to 60/300 (20%). Calibrating the INT8 threshold separately reduced nominal false alarms to 0/300, but at 0 dB it detected only 69/100 signals compared with 97/100 for full precision. In a separate gain test, a +1 dB change at unchanged input signal-to-noise ratio raised full-precision false alarms to 170/300 (56.67%). These observations come from one lightly trained model. Independent USRP sessions and adequately trained models are the next evidentiary step. [Project results](https://github.com/mohamed7703/SpectrumSensing/blob/main/docs/recovery/RECOVERY_PROGRESS.md)

The original 2024 paper already studied deep learning for spectrum sensing. A 2025 *IEEE Communications Letters* paper deployed a quantization-aware detector on a Sony Spresense. An October 2026 study compared classical and neural detectors under receiver impairments using common constant-false-alarm thresholding in simulation. The candidate contribution is therefore quite narrow: determine how many **confirmed noise-only observations**, collected under which receiver settings, suffice to recover a usable operating point on an **actual quantized receiver**, and establish the limits on independent capture days. Priority remains open pending a full-text literature comparison. [Original paper](https://doi.org/10.3390/s24247907) · [Edge quantization study](https://doi.org/10.1109/LCOMM.2025.3582680) · [Impairment benchmark](https://doi.org/10.1007/s13369-026-11704-1)

### Three possible contributions

| Direction | What the paper would contribute | What it requires |
|---|---|---|
| **1. Calibration after quantization and gain changes — recommended** | A calibration rule using the deployed model's discrete scores, receiver gain and a limited set of noise-only observations. Report the detection cost and which conditions exceed the false-alarm limit. | One strong model, simple competing calibration rules, an energy detector, repeat USRP sessions and an executable INT8 model. |
| **2. A reproducible receiver benchmark** | Release labeled multi-day captures, capture metadata, fixed splits and an evaluation protocol. Show which conclusions about detectors change between simulation and hardware. | More capture effort, sharable data and clear coverage beyond existing impairment benchmarks. A strong choice if several calibration methods perform similarly. |
| **3. Sensing errors and network access** | Determine when sensing errors persist long enough to harm the primary user, and test an access rule that reduces interference at a measured throughput cost. | Time-ordered receiver traces, matched traffic, receiver-level interference validation and a network baseline. Longest route from the current evidence. |

### Where to submit

Publisher pages checked 18 November 2026. JIF means Journal Impact Factor. The paper descriptions below are our assessment of fit from the journals' published scopes.

| Journal | Publisher-reported JIF | Manuscript that fits |
|---|---:|---|
| **Journal on Wireless Communications and Networking**, formerly *EURASIP Journal on Wireless Communications and Networking* | **2.5 (2025)** | First choice for direction 1 or 2. A focused wireless signal-processing study with a clear receiver method, classical baselines, and repeatable over-the-air or controlled RF measurements. [Springer Nature](https://link.springer.com/journal/13638) |
| **IEEE Open Journal of the Communications Society** | **6.1**, attributed by IEEE ComSoc to **JCR 2025** | Ambitious target for direction 1 if the method adds a clear performance advantage over strong alternatives and transfers across independent receiver conditions. A second radio would strengthen the claim. The society page identifies the JCR edition without separately labeling the metric year. [IEEE ComSoc](https://www.comsoc.org/publications/journals/ieee-ojcoms) |
| **Telecommunication Systems** | **2.2 (2025)** | Best fit for direction 3 if receiver errors are quantitatively connected to primary-user interference and secondary-user throughput. [Springer Nature](https://link.springer.com/journal/11235) |

### Workflow for direction 1

1. **Check the gap before training.** Read the closest papers in full and make a one-page comparison covering numeric precision, gain changes, calibration budget, independent capture sessions, false-alarm control and detection cost. Look specifically for studies combining these elements. If one already answers the question, narrow the claim or choose direction 2.
2. **Write the experiment on one page.** Choose the operating limit in advance, for example a 5% noise-only false-alarm probability. Define the primary outcome as detection probability at specified low SNRs when the false-alarm limit is met. Agree how much detection loss is acceptable and where a deployed receiver would obtain verified noise-only samples.
3. **Establish a credible baseline.** Train one full-precision model over multiple seeds. RadioML signal-present examples can support this stage. Prepare noise-only data separately. Keep training, model selection, threshold calibration and final testing distinct. Give every detector the same held-out trials.
4. **Run a small hardware pilot first.** Use controlled labeled signals and a known noise-only input, record receiver gain and measured SNR, and repeat acquisition on different days. Compare full precision, INT8 with the copied threshold, INT8 with ordinary independent recalibration, and gain-aware energy detection. Check whether gain drift or coarse INT8 scores still cause a meaningful loss. Decide whether to scale up from that result.
5. **Test the proposed rule on sealed sessions.** Calibrate with a limited noise budget at known gain settings. Specify how tied INT8 scores are handled and what happens when no threshold meets both goals. Vary calibration size and gain separately. Report false alarms, weak-signal detections, uncertainty across capture sessions, and the best simple competitor. Adjacent frames from one recording do not count as independent capture sessions.
6. **Write around the decisive result.** The main figure should show detection versus gain at the stated false-alarm limit and flag settings where that limit is missed. A second figure should show how much calibration data is needed. If ordinary recalibration matches the proposed method, report the measured trade-off and assess whether the dataset and benchmark justify direction 2. Add edge latency or energy claims only after measuring them on the named device.

**Immediate decision.** Approve direction 1, assign the prior-work comparison, and reserve two short USRP sessions on different days for a feasibility pilot. The large RadioML download can wait until the question and capture protocol are fixed. The current software and synthetic results support running this study. Its scientific outcome and publication readiness remain open.
