# Experiment 004: Channel Fading Distributions and Multipath Degradation
- **Date:** 2026-10-28
- **Phase:** 1 - Math Foundations
- **Status:** Completed
- **Task:** P1-004

## ⚙️ Configuration
- **Distributions Evaluated:** Rayleigh, Rician ($K=4$), Nakagami-$m$ ($m=2.0$), AWGN baseline
- **Sample Length ($L$):** 1024
- **Channel Power ($\Omega$):** 1.0 ($\mathbb{E}[|h|^2] = 1.0$)
- **SNR Sweep Range:** -16 dB to +8 dB
- **Monte Carlo Sample Count:** 30,000 samples for envelope PDFs, 1,500 frames per SNR point for sensing evaluation

## 📊 Results
| SNR (dB) | AWGN $P_d$ (Baseline) | Rician ($K=4$) $P_d$ | Rayleigh $P_d$ (NLOS) | Fading Penalty (Rayleigh vs AWGN) |
|---|---|---|---|---|
| -16.0 dB | 0.200 | 0.245 | 0.339 | Divergence at noise floor |
| -10.0 dB | 0.921 | 0.771 | 0.620 | **-30.1%** penalty |
| -8.0 dB  | 0.999 | 0.865 | 0.738 | **-26.1%** penalty |
| -4.0 dB  | 1.000 | 0.952 | 0.887 | **-11.3%** penalty |
| 0.0 dB   | 1.000 | 0.985 | 0.947 | **-5.3%** penalty |
| +8.0 dB  | 1.000 | 0.997 | 0.989 | **Recovers to >95%** |

## 💾 Artifacts
- Source code: [`src/math_models/channel_fading.py`](file:///s:/Projects/SpectrumSensing/src/math_models/channel_fading.py)
- Unit tests: [`tests/test_channel_fading.py`](file:///s:/Projects/SpectrumSensing/tests/test_channel_fading.py) (12/12 passed, 48/48 overall)
- Verification script: [`scripts/verify_channel_fading.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_channel_fading.py)
- Generated figure: [`results/plots/p1_004_channel_fading.png`](file:///s:/Projects/SpectrumSensing/results/plots/p1_004_channel_fading.png)

## 📝 Observations
- In the transition regime (-12 dB to -6 dB), Rayleigh fading creates a severe detection degradation (dropping $P_d$ by up to 30 percentage points) due to deep destructive multipath fades.
- Line-of-Sight propagation (Rician $K=4$) significantly dampens envelope fluctuations, outperforming Rayleigh fading by ~15 percentage points at -10 dB.
- By +8 dB SNR, the average signal energy provides sufficient fading margin to restore detection to $>98\%$, matching the research plan's channel characterization.
