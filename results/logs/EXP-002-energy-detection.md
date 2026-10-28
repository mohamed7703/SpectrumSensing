# Experiment 002: Energy Detection Decision Statistic & CFAR Thresholding
- **Date:** 2026-10-28
- **Phase:** 1 - Math Foundations
- **Status:** Completed
- **Task:** P1-002

## ⚙️ Configuration
- **Sample Length ($L$):** 1024
- **Degrees of Freedom ($2L$):** 2048
- **Noise Variance ($\sigma_w^2$):** 1.0
- **Target $P_{\text{FA}}$:** 0.05 (CFAR rule)
- **Threshold Setting ($\gamma$):** $1.0520$ (Exact Chi-Square PPF)
- **Trials:** 10,000 frames under $H_0$, 10,000 frames under $H_1$, 15-point SNR sweep (-18 dB to -4 dB)

## 📊 Results
| Metric | Theoretical | Empirical | Notes |
|---|---|---|---|
| Threshold $\gamma$ (Exact $\chi^2$) | 1.0520 | 1.0520 | Analytically derived |
| Threshold $\gamma$ (CLT Gaussian) | 1.0514 | - | 0.06% difference from exact |
| False Alarm Rate ($P_{\text{FA}}$) | 0.0500 | 0.0483 | Within 95% confidence interval |
| $P_d$ @ SNR = -10.0 dB | 0.9213 | 0.9214 | 0.01% error |
| $P_d$ @ SNR = -16.0 dB | 0.2002 | 0.2010 | Low-SNR transition regime |
| $P_d$ @ SNR = -4.0 dB | 1.0000 | 1.0000 | Full detection saturation |

## 💾 Artifacts
- Source code: [`src/math_models/energy_detection.py`](file:///s:/Projects/SpectrumSensing/src/math_models/energy_detection.py)
- Unit tests: [`tests/test_energy_detection.py`](file:///s:/Projects/SpectrumSensing/tests/test_energy_detection.py) (11/11 passed)
- Verification script: [`scripts/verify_energy_detection.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_energy_detection.py)
- Validation figure: [`results/plots/p1_002_energy_detection.png`](file:///s:/Projects/SpectrumSensing/results/plots/p1_002_energy_detection.png)

## 📝 Observations
- Excellent alignment between non-central Chi-square survival function and Monte Carlo empirical iterations.
- At $L=1024$, the detector transitions from $P_d \approx P_{\text{FA}}$ at $\le -20$ dB to $P_d > 0.9$ around $-10$ dB, and saturates at $P_d = 1.0$ by $-4$ dB.
