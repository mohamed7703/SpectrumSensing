# SpectrumSensing project summary

*21 November 2026*

We started by auditing a published spectrum-sensing study and its verification project. We corrected the research code, discovered that thresholds transfer poorly to some INT8 models, and tested a gain-aware calibration rule. That rule gave the same detection rate as ordinary recalibration and caused more false alarms. The NI USRP‑2920 lab packet is ready to test whether the failure we observed in simulation also occurs on a real receiver.

## 1. What the project does

A cognitive radio listens to a channel and decides whether a primary user is transmitting. The receiver processes in-phase and quadrature (I/Q) samples and assigns each frame a score. A decision threshold turns that score into a signal-present or noise-only decision.

- **Detection probability (Pd)** is the fraction of signal-present frames correctly detected.
- **False-alarm probability (Pfa)** is the fraction of noise-only frames incorrectly classified as signals.
- **Calibration** uses confirmed noise-only frames to set the decision threshold. Receiver gain changes can move scores relative to that threshold. Converting a model from FP32 to INT8 can also make many frames share the same score. Moving the threshold across one of those scores changes the decisions for the entire group.

## 2. What we audited and fixed

The original project passed **205 software tests**. We checked which research claims those tests supported and found issues across all four layers.

| Layer | What we changed |
|---|---|
| **Mathematics** | Corrected ROC and AUC calculations, tied-score handling, SNR and gain conventions, and uncertainty reporting for detection and false alarms. |
| **Physical / SDR** | Fixed mock receiver scaling and buffer handling. Used separate noise samples for calibration and testing. |
| **Network** | Fixed packet timing and interference accounting so a primary-user transmission that starts mid-packet is counted. The checks used the project’s Python simulator. |
| **AI / ML** | Identified that the historical NumPy path used a four-feature classifier. Added and verified native CNN training, checkpoint restoration, and nine executable TensorFlow Lite exports. |

The updated suite passed **331 software tests** with 18 upstream deprecation warnings. The original paper’s full RadioML evaluation and hardware performance remain unverified. [Technical audit](recovery/baseline-audit/TECHNICAL_AUDIT.md) · [Recovery report](recovery/RECOVERY_PROGRESS.md)

## 3. Why we focused on calibration

In the original synthetic GAP pilot, reusing the full-precision threshold after INT8 conversion produced **60 false alarms in 300 noise-only trials (20%)**. Setting a new INT8 threshold reduced that to **0/300** and lowered weak-signal detection.

We swept every attainable deterministic threshold on the saved scores. At nominal gain and **0 dB SNR**, the best INT8 threshold within the **5% empirical false-alarm limit** detected **69/100 signals**. The best FP32 threshold detected **98/100**. For this checkpoint and sample, threshold adjustment alone could not recover FP32 detection. [Threshold sweep](../results/calibration/frozen-sweep-v1/REPORT.md)

Our literature review found published work on quantized spectrum sensing, noise-only threshold calibration, and USRP data collection. Our narrower question is **how tied INT8 scores affect detection and false alarms when thresholds transfer across receiver gains and capture days under a fixed calibration budget**. We could access only the abstract of the closest IEEE edge paper, so the novelty assessment is provisional. [Literature comparison](../literature/novelty-comparison.md)

## 4. What the fresh experiment showed

We trained **five independent GAP models** and evaluated them on **15 synthetic calibration and test session pairs**. Ordinary INT8 recalibration sets a threshold independently at each gain. The candidate rule fits a straight line through the thresholds measured at three gains. Both methods received **200 noise-only calibration frames per gain**, for **600 frames total**. [Frozen experiment protocol](calibration/FROZEN_PROTOCOL.md)

At **0 dB SNR**, both methods detected **2,121 of 3,600 signals (58.92%)**. Their false-alarm rates differed.

| Receiver gain drift | Linear gain-aware rule | Ordinary INT8 recalibration |
|---|---:|---:|
| −1 dB | 902/4,500 (20.04%) | 0/4,500 (0.00%) |
| 0 dB | 41/4,500 (0.91%) | 4/4,500 (0.09%) |
| +1 dB | 189/4,500 (4.20%) | 16/4,500 (0.36%) |

At −1 dB gain drift, the linear rule exceeded the **5% false-alarm limit**. In one session, a small downward shift admitted **208 of 300 noise frames** sharing one INT8 score and detected the same 80 signals as ordinary recalibration. The candidate gave no detection gain on the primary comparison. We independently recounted all **3,375 benchmark rows** from the saved scores. [Benchmark report, figures, and run records](../results/calibration/benchmark-v1/REPORT.md)

## 5. Where the project stands

- **Research materials are on GitHub `main`.** The research layout organizes the audit, protocols, literature notes, scripts, score archives, and lab materials. The interactive workflow explains the study and lets you replay the original synthetic pilot. [Research commit `d7061aa`](https://github.com/mohamed7703/SpectrumSensing/commit/d7061aa62c7bcaafeccd7df87b0816a8292b5cb1) · [Interactive workflow](presentations/receiver-calibration-workflow/Receiver_Calibration_Workflow.html)
- **The NI USRP‑2920 handoff is ready.** It includes a two-day run sheet, capture and label templates, an INT8 model, a data-quality checker, and offline replay commands. The workflow passed a labeled synthetic dry run. The lab still needs to confirm the RF source and cabled setup. [USRP‑2920 lab packet](calibration/usrp-2920/README.md)
- **The next experiment is to measure threshold transfer on real hardware.** On day one, collect confirmed noise-only data at three manual gain settings and lock the thresholds. On day two, capture independent noise-only and known-signal data and replay it using those thresholds. The result will show whether tied scores cause the same false-alarm jumps on the USRP. [Supervisor brief](calibration/SUPERVISOR_BRIEF.md)
