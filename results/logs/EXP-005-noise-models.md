# Experiment 005: Noise Models, Calibrated AWGN, and the SNR Wall
- **Date:** 2026-10-28
- **Phase:** 1 - Math Foundations
- **Status:** Completed
- **Task:** P1-005

## ⚙️ Configuration
- **Calibrated Standard Deviation:** $\sigma_w = 0.708$ ($\sigma_w^2 \approx 0.5013$) matching the Sensors 2024 paper
- **Uncertainty Bounds ($B_{\text{dB}}$):** 0.1 dB, 0.25 dB, 0.4 dB, 0.5 dB, 1.0 dB
- **Sample Length ($L$):** 1024
- **CFAR Target $P_{\text{FA}}$:** 0.05
- **Target $P_d$:** 0.90 (IEEE 802.22 standard)

## 📊 Results
| Uncertainty $B_{\text{dB}}$ | Linear Factor $\rho$ | $\text{SNR}_{\text{wall}}$ (Linear) | $\text{SNR}_{\text{wall}}$ (dB) | Practical Consequence |
|---|---|---|---|---|
| 0.10 dB | 1.0233 | 0.0461 | -13.36 dB | Minimal hardware drift |
| 0.25 dB | 1.0593 | 0.1153 | -9.38 dB  | High-precision SDR frontend |
| 0.40 dB | 1.0965 | 0.1845 | -7.34 dB  | Matches transition cutoff (~-8 dB) |
| 0.50 dB | 1.1220 | 0.2308 | -6.37 dB  | Low-cost SDR (e.g. RTL-SDR Blog V3) |
| 1.00 dB | 1.2589 | 0.4646 | -3.33 dB  | Severe thermal / AGC drift |

## 💾 Artifacts
- Source code: [`src/math_models/noise_models.py`](file:///s:/Projects/SpectrumSensing/src/math_models/noise_models.py)
- Unit tests: [`tests/test_noise_models.py`](file:///s:/Projects/SpectrumSensing/tests/test_noise_models.py) (8/8 passed, 56/56 overall)
- Verification script: [`scripts/verify_noise_models.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_noise_models.py)
- Generated figure: [`results/plots/p1_005_snr_wall.png`](file:///s:/Projects/SpectrumSensing/results/plots/p1_005_snr_wall.png)

## 📝 Observations & Key Architectural Takeaways
1. **The Asymptotic Barrier**:
   - As SNR decreases toward $\text{SNR}_{\text{wall}}$, the required sample size $L$ explodes to infinity ($L \to \infty$). Below $\text{SNR}_{\text{wall}}$, no classical energy detector can guarantee $P_d \ge 90\%$ and $P_{\text{FA}} \le 5\%$, regardless of observation duration.
2. **Justification for Deep Learning**:
   - For an uncertainty of $B \approx 0.4\text{ dB}$, the wall is at $-7.34\text{ dB}$. This mathematically explains why both the paper and deployment plan set their SNR filter at $-8\text{ dB}$: conventional energy detection is fundamentally blind below this boundary under realistic RF hardware noise drift.
   - Deep learning architectures (1D-CNN, Bi-GRU) circumvent the SNR wall by classifying structural modulation waveforms and temporal correlations rather than relying strictly on uncalibrated power thresholds.
