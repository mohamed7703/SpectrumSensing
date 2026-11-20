# Pre-USRP decision brief

**Decision.** Revise the linear gain-aware threshold before making a superiority claim. A short NI USRP-2920 pilot is justified if its purpose is to measure score ties and cross-day threshold transfer on real captures.

The archived synthetic checkpoint revealed a resolution limit. At nominal gain and 0 dB SNR, the best deterministic INT8 threshold meeting the sample false-alarm limit detected 69/100 signals. Native scores allowed 98/100. [Threshold sweep](../../results/calibration/frozen-sweep-v1/REPORT.md)

In the prespecified fresh benchmark, five independent GAP models and three session pairs per model compared the proposed rule with ordinary recalibration. At 200 noise frames per gain, both detected 58.92% of 0 dB signals. At −1 dB gain the proposed rule raised 902 false alarms in 4,500 noise frames. Ordinary recalibration raised zero there. Small shifts of the fitted line across a discrete noise score admitted many tied frames. [Benchmark and uncertainty](../../results/calibration/benchmark-v1/REPORT.md)

Published work already covers quantization-aware edge sensing, held-out noise-only threshold calibration, and USRP datasets. A narrower question concerns the attainable detection loss and false-alarm behavior of discrete INT8 thresholds under gain and day changes at specified calibration budgets. Novelty remains provisional, including details unavailable from the closest IEEE paper. [Literature comparison](../../literature/novelty-comparison.md)

The handoff includes a two-day capture protocol, labeled data format, first-seed model, quality checker, lock-and-replay commands, and a completed synthetic dry run. The team has identified the receiver. The RF source, cabling, bandwidth, manual gains and supported driver settings still require lab confirmation. A hardware result requires actual recordings. [USRP-2920 lab packet](usrp-2920/README.md)
