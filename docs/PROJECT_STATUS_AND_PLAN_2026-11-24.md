# SpectrumSensing research progress and future plan

**Status as of 24 November 2026 · Supervisor and lab handoff**

We began by checking whether a passing software suite reproduced a published spectrum-sensing study. The audit exposed gaps in the evaluation, and the corrected experiments led us to a narrower research question: **how reliably does a threshold transfer when a neural detector is converted to INT8 and the receiver gain or capture day changes?**

We now have a real USRP recording that shows why this question matters. In one session, **19,820 of 197,236** held-out frames received exactly the INT8 cutoff score. Counting those ties as triggers changed the observed trigger rate from **0.43% to 10.48%**. That establishes a threshold effect on real receiver data. Its consequences for missed signals and false alarms still require controlled signal labels.

## 1. Where we started

The starting materials were the 2024 *Sensors* paper on deep-learning spectrum sensing, the project's comprehensive verification report, and a repository extending the work across mathematics, radio reception, network simulation and machine learning. The initial review preserved the original inputs, inspected the implementations and configurations, reran all **205 tests** and executed all **19 experiment scripts**. All ran successfully. A separate assessment of what the 205 collected cases actually supported classified **121 as Verified, 73 as Partially Verified, 11 as Failed and 0 as Inconclusive**. Those counts apply to the individual test cases. Broader paper and deployment claims could still lack evidence. [Original paper](https://doi.org/10.3390/s24247907) · [Technical audit](../docs/recovery/baseline-audit/TECHNICAL_AUDIT.md) · [Individual test assessments](../docs/recovery/baseline-audit/TEST_BY_TEST_AUDIT.md)

| Layer | What the review found and what we changed |
|---|---|
| Mathematical foundations | Corrected ROC/AUC calculations, handling of equal scores, gain and SNR conventions, and uncertainty reporting. The original checks could pass while giving an incorrect operating point. |
| Physical radio and DSP | Corrected mock receiver scaling and buffer handling and separated calibration noise from evaluation noise. Later USRP captures established that the real acquisition and host replay path works under the tested settings. |
| Network simulation | Corrected packet timing and interference accounting, including primary-user activity starting during a packet. The measured networking results came from the project's Python simulator. A physical network or independent NS-3 validation remains open. |
| AI/ML and model export | The historical NumPy inference path used a four-feature classifier. We verified native neural-network training, checkpoint restoration and nine executable TensorFlow Lite exports. Some earlier export assertions had checked only that files existed. |

The corrected suite reached **331 passing software tests**, including six lab-workflow checks. The full RadioML evaluation from the paper was never independently rerun. The audit establishes limitations in the project's reproduction evidence and leaves the original authors' results open. [Recovery record](../docs/recovery/RECOVERY_PROGRESS.md)

## 2. Why we chose calibration

Each model scores a frame of 1,024 complex I/Q samples. A threshold turns that score into a signal-present decision. **Detection probability (Pd)** requires known signal-present frames. **False-alarm probability (Pfa)** requires confirmed noise-only frames. For passive antenna recordings with unknown emissions, we report the **ambient trigger rate**, the fraction of frames the detector calls signal-present.

In the first synthetic pilot, reusing the full-precision (FP32) threshold after 8-bit integer conversion (INT8) produced **60 false alarms in 300 noise frames**. Fitting a separate INT8 threshold gave **0/300**. We then swept every attainable threshold on the saved scores. At nominal gain and 0 dB SNR, under an empirical 5% false-alarm limit, the best INT8 threshold detected **69/100** signals and the best FP32 threshold detected **98/100**. These are retrospective limits for that checkpoint and sample. [Frozen threshold sweep](../results/calibration/frozen-sweep-v1/REPORT.md)

We tested a candidate calibration rule that fits a straight line through thresholds at three gains. The fresh experiment used **five independently trained models**, **15 synthetic calibration/test session pairs**, and an equal budget of **200 noise frames per gain**. At 0 dB SNR, both the linear rule and ordinary per-gain recalibration detected **2,121/3,600 signals (58.92%)**.

| Gain relative to nominal | False alarms with linear rule | False alarms with ordinary per-gain recalibration |
|---|---:|---:|
| −1 dB | 902/4,500 (20.04%) | 0/4,500 (0%) |
| 0 dB | 41/4,500 (0.91%) | 4/4,500 (0.09%) |
| +1 dB | 189/4,500 (4.20%) | 16/4,500 (0.36%) |

The candidate failed its prespecified 5% limit at −1 dB and delivered no detection gain in the primary comparison. We recounted **3,375 result rows** from archived scores. The paper direction therefore shifted to measuring when quantized scores and limited calibration data make threshold transfer unreliable. The closest literature comparison remains provisional because only the abstract of one closely related IEEE paper was available. [Benchmark](../results/calibration/benchmark-v1/REPORT.md) · [Frozen protocol](../docs/calibration/FROZEN_PROTOCOL.md) · [Literature comparison](../literature/novelty-comparison.md)

## 3. Moving from simulation to the lab

We organized the research repo into documentation, literature, data, notebooks, source, experiment configurations, results, manuscript inputs, external tools, scripts and tests. The project includes domain-specific research guidance, a supervisor brief, two interactive presentations, and a USRP handoff with label templates, model identity and offline replay commands. The research layout and pre-USRP evidence were merged into GitHub at commit `d7061aa`. The later hardware measurements are archived alongside this report. [Research commit](https://github.com/mohamed7703/SpectrumSensing/commit/d7061aa62c7bcaafeccd7df87b0816a8292b5cb1) · [Progress presentation](presentations/research-validation-progress/output/Research_Validation_Progress.html) · [Calibration workflow](presentations/receiver-calibration-workflow/Receiver_Calibration_Workflow.html) · [Lab handoff](../docs/calibration/usrp-2920/README.md)

The lab brought up an **NI USRP-2920** in GNU Radio Companion over Gigabit Ethernet at `192.168.10.3`. The initial startup error occurred while UHD waited for a pulse-per-second signal. Selecting **No Sync** allowed the standalone capture to run. The reported settings were **915 MHz**, **1 MS/s**, **RX2**, and **10 dB receive gain**. A Head block controlled the sample count. The File Sink saved little-endian complex float32 I/Q, eight bytes per complex sample.

| Capture | Measured result | Use in the study |
|---|---|---|
| `noise_only.iq` | Mean digital power about −81.87 dB in a 1.024 s capture. | Early receiver diagnostic. The filename does not establish noise-only ground truth. |
| `open_rx_check.iq` | About 99.99% of spectral power fell within ±200 Hz of a tone 26.73 kHz below the tuned center. | Strong-tone acquisition check. The change was consistent with the lab generator producing a carrier. |
| `ampient_turned_off.iq` | 1,855,763,448 bytes, 231,970,431 samples, or 231.970431 s at 1 MS/s. The separately supplied 8 MB excerpt at 60 s matched byte for byte. | Full generator-off ambient baseline at one gain on one receiver. |

The photographed RF generator was operating in carrier mode. Varying its carrier level changes signal power. Whether it can generate controlled broadband RF noise remains unconfirmed. The raw capture has no embedded gain readbacks, timestamps or UHD overflow log. Those details matter for future replication. [Shared recording](https://drive.google.com/file/d/1_pmfCVqJa41YR3L2EwPTCnhy2a_BL8Po/view) · [Capture identity and checksum](../results/calibration/usrp-ambient-20261121/analysis/capture.json)

## 4. What the complete recording showed

- **Capture quality.** Every sample was finite, and no I or Q component reached 0.99 of nominal full scale. The opening eight samples contained a large transient. We excluded the first second before setting the model scale or thresholds.
- **Background stability.** After startup, mean digital power per window ranged from **−81.91 to −81.15 dB**. The middle 90% of broadband floor readings spanned about **0.31 dB**, while the full range was **1.16 dB**. The highest floor occurred at **62 s**, even with gain held fixed. The earlier tone's frequency band contained at most **0.088%** of window power after startup. These are relative digital measurements. Absolute RF power in dBm remains unknown.
- **Input scaling.** Without scaling, every raw component in the sampled calibration/evaluation frames mapped to one INT8 input code. We used seconds **1–5** to set one fixed amplitude multiplier of **12,278.65135**. Preserving that multiplier lets later gain or background changes reach the detector.
- **Held-out testing.** We fitted separate thresholds for the matching seed-19031 FP32 and INT8 models on **200 frames from seconds 5–30**. We froze both thresholds and tested **every complete frame from 30 s onward**. The 197,236 evaluation frames shared no samples with calibration. They belong to one continuous session and can be correlated. [Hardware report and figure](../results/calibration/usrp-ambient-20261121/analysis/REPORT.md) · [Per-second quality measurements](../results/calibration/usrp-ambient-20261121/analysis/quality_per_second.csv)

| Model | Strict score cutoff, rounded | Held-out ambient triggers | Distinct held-out scores |
|---|---:|---:|---:|
| FP32 | 0.527542651 | 3,854/197,236 (1.95%) | 88,805 |
| INT8 | 0.527343750 | 855/197,236 (0.43%) | 7 |

The trigger rule was **score strictly above cutoff**. INT8 assigned exactly 0.52734375 to **19,820** held-out frames. Including those ties raises its trigger count from **855 to 20,675**, or **10.48%**. During the second with the highest broadband floor, FP32 triggered on **187/977** frames and INT8 on **51/977**. The observed association does not identify the source of the change. [Frozen thresholds](../results/calibration/usrp-ambient-20261121/analysis/early_lock.json) · [Dense replay totals](../results/calibration/usrp-ambient-20261121/analysis/all_test_summary.json) · [Per-second trigger counts](../results/calibration/usrp-ambient-20261121/analysis/all_test_per_second.csv)

### Would another choice of 200 calibration frames change the result?

We drew **20 additional disjoint sets of 200 frames** from the same early calibration interval, excluding the original calibration frames. Each set covered all 25 seconds, used the same scale and was evaluated on the same 197,236 later frames.

| Model | Result across 20 alternative calibration sets |
|---|---|
| FP32 | All 20 cutoffs differed. Ambient trigger rates ranged from **0.98% to 5.70%**, median **3.39%**. |
| INT8 | All 20 reproduced the original cutoff, giving **855 triggers (0.43%)** every time. |

The INT8 cutoff was repeatable across these frame selections, yet including one additional score level changed decisions for about one in ten held-out frames. These 20 selections provide a within-session sensitivity check. Independent capture days are still required to measure transfer. [Sensitivity results](../results/calibration/usrp-ambient-20261121/analysis/selection_sensitivity.csv) · [Reproducible selection script](../results/calibration/usrp-ambient-20261121/check_calibration_selection.py)

## 5. Future plan, in priority order

| Priority | Experiment or deliverable | Decision it enables |
|---|---|---|
| 1 · Next lab visit | With generator RF **off**, use the same antenna position, 915 MHz and 1 MS/s. Set the Head block to **11,000,000 items**. Record **10 → 9 → 11 → 10 dB**, 11 s per file. Use unique names such as `ambient_g10_before.iq`, `ambient_g09.iq`, `ambient_g11.iq`, `ambient_g10_after.iq`. Each should be **88,000,000 bytes**. Retain all raw samples and analyze the final 10 s per file. | Compare the two 10 dB runs first to estimate drift. Replay all four with the existing scale and thresholds. If same-gain drift is comparable to the apparent gain effect, repeat with changed order before attributing the change to gain. |
| 2 · Establish labels | Arrange a controlled RF path with verified source-off noise-only intervals (H0) and known source-on QPSK intervals (H1). Preserve impedance, bandwidth and signal path between the two states. Log source transitions and exclude settling. For direct generator-to-USRP cabling, obtain the **30 dB input attenuator** specified by NI. | Enables measured false-alarm and detection probabilities. Keep the antenna-only files as environmental measurements. |
| 3 · Test incoming noise variation | Verify access to an RF source that produces broadband noise across the capture band. Hold receive gain fixed, vary the noise source, and measure the actual change in received broadband power. Start with nominal and approximately ±1 dB if the setup can resolve those levels. | Separates the effect of changes in incoming noise power from the effect of adjusting receiver gain. Report achieved levels and repeatability. |
| 4 · Run the labeled multi-day protocol | On calibration day, collect at least **400 usable H0 frames per gain** and fit on **200 per gain** at 9, 10 and 11 dB. Freeze model identities, scale, thresholds and tie rule. On a later day, target at least **300 H0 frames per gain** and **100 H1 frames per gain and measured SNR** near −10, 0 and +10 dB. Repeat on additional independent days to assess variability. | Compare FP32, ordinary INT8 recalibration, the linear candidate and an energy detector under matched conditions. Report Pd, Pfa, missed signals, counts and uncertainty by day, gain and SNR. Record any condition exceeding the 5% false-alarm target. |
| 5 · Prepare the paper | Update the literature comparison with the closest full texts. Archive scripts, labeled data or shareable subsets, capture metadata, locked splits, model hashes, thresholds and score tables. Write the synthetic failure and the hardware evidence together. The publication brief identified the *Journal on Wireless Communications and Networking* as the initial target. | A submission is justified when independent labeled sessions support a precise claim about threshold transfer and its cost in detection. Scope the conclusion to the receivers and conditions actually tested. |

For **every new capture**, save the GNU Radio flowgraph, USRP identity and driver version, UTC time, gain readback, center frequency, sample rate, bandwidth, antenna or cable path, source state, expected and actual sample count, any overflow or dropped-sample messages, and the file checksum. A second at 1 MS/s contains almost a thousand 1,024-sample frames, but additional frames from one run cannot substitute for independent sessions. [Lab run sheet and templates](../docs/calibration/usrp-2920/README.md) · [NI connection guide](https://www.farnell.com/datasheets/4148463.pdf) · [Publication brief](../docs/calibration/PUBLICATION_BRIEF.md)

## 6. Publication readiness

**Established:** the four-layer audit and correction record, a failed synthetic gain-aware candidate, a working USRP acquisition and host-inference pipeline, and a directly counted INT8 threshold discontinuity on one real ambient recording.

**Still open:** verified hardware Pd and Pfa, weak-signal detection cost, transfer across gains and days, response to controlled broadband noise changes, and the originality of the final contribution against the closest prior work. The evidence supports a supervisor progress report and a focused hardware study. The next milestone for a paper is a labeled, independent-day result showing whether the observed threshold behavior helps or harms sensing in practice.
