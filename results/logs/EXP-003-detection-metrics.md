# Experiment 003: Detection Metrics, Confusion Matrix, and ROC/AUC Analysis
- **Date:** 2026-10-28
- **Phase:** 1 - Math Foundations
- **Status:** Completed
- **Task:** P1-003

## ⚙️ Configuration
- **Sample Length ($L$):** 1024
- **Noise Variance ($\sigma_w^2$):** 1.0
- **Modulation:** QPSK
- **SNR Levels Analyzed:** -16 dB, -13 dB, -10 dB, -7 dB (plus full sweep from -20 dB to 0 dB)
- **Thresholds Evaluated per Curve:** 150 - 300 discrete thresholds
- **Monte Carlo Sample Count:** 4,000 to 6,000 frames per SNR point

## 📊 Results
| SNR (dB) | Theoretical AUC | Empirical AUC | $P_d$ @ $P_{\text{FA}}=0.05$ | Notes |
|---|---|---|---|---|
| -16.0 dB | 0.697 | 0.695 | 0.200 | Weak separability |
| -13.0 dB | 0.854 | 0.851 | 0.490 | Intermediate transition |
| -10.0 dB | 0.978 | 0.976 | 0.921 | High detection reliability |
| -7.0 dB | 0.999 | 1.000 | 0.999 | Approaching near-perfect detection |
| -4.0 dB | 1.000 | 1.000 | 1.000 | Fully saturated ROC |

## 💾 Artifacts
- Source code: [`src/math_models/detection_metrics.py`](file:///s:/Projects/SpectrumSensing/src/math_models/detection_metrics.py)
- Unit tests: [`tests/test_detection_metrics.py`](file:///s:/Projects/SpectrumSensing/tests/test_detection_metrics.py) (9/9 passed, 36/36 overall)
- Verification script: [`scripts/verify_detection_metrics.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_detection_metrics.py)
- Generated figure: [`results/plots/p1_003_roc_curves.png`](file:///s:/Projects/SpectrumSensing/results/plots/p1_003_roc_curves.png)

## 📝 Observations
- ROC curves confirm that at $L=1024$, an energy detector requires at least $-10$ dB SNR to achieve the standard IEEE 802.22 sensing benchmarks ($P_d \ge 0.90$ with $P_{\text{FA}} \le 0.10$).
- The AUC curve shows a steep sigmoid response transitioning between $-16$ dB and $-8$ dB, validating why the paper set its noise filtering threshold at $-8$ dB for training deep learning models.
