# Experiment 007: Physical-Layer I/Q Preprocessing, Impairment Correction, and Augmentations
- **Date:** 2026-10-28
- **Phase:** 2 - Physical / SDR Layer
- **Status:** Completed
- **Task:** P2-002

## ⚙️ Configuration
- **Sample Length:** 2048 complex baseband samples
- **Simulated Hardware Impairments:**
  - DC Bias (Carrier Leakage): $0.40 + 0.30j$
  - Gain Imbalance: $1.25$ (+25% on Q channel)
  - Phase Mismatch: $20.0^\circ$ non-orthogonality
- **Target Normalized Power:** 1.0000
- **Framing Specification:** $1024 \times 2$ (RadioML format)
- **Augmentations Evaluated:** Carrier Frequency Offset (+200 kHz), Phase rotation, Cyclic time roll

## 📊 Results
| Preprocessing Stage | Metric | Distorted Frontend | Corrected Signal | Target / Improvement |
|---|---|---|---|---|
| DC Offset Removal | Mean DC Bias Magnitude | 0.500 | $< 10^{-12}$ | 100% DC cancellation |
| Gram-Schmidt Correction | I/Q Cross-Covariance | 0.2032 | -0.0000 | Orthogonality completely restored |
| Gain Equalization | $\sigma_I^2 / \sigma_Q^2$ | 0.6400 | 1.0000 | Amplitude rails matched |
| Power Normalization | Frame Average Energy | 1.834 | 1.0000 | Exact unit energy scaling |
| Frame Builder | Tensor Dimensionality | Stream | $(N, 1024, 2)$ | Strict RadioML compatibility |
| Noise Injection | Empirical vs Target SNR | - | Error $< 0.15\text{ dB}$ | Calibrated noise addition |

## 💾 Artifacts
- Source code:
  - [`src/preprocessing/iq_normalizer.py`](file:///s:/Projects/SpectrumSensing/src/preprocessing/iq_normalizer.py)
  - [`src/preprocessing/frame_builder.py`](file:///s:/Projects/SpectrumSensing/src/preprocessing/frame_builder.py)
  - [`src/preprocessing/noise_injector.py`](file:///s:/Projects/SpectrumSensing/src/preprocessing/noise_injector.py)
  - [`src/preprocessing/augmentation.py`](file:///s:/Projects/SpectrumSensing/src/preprocessing/augmentation.py)
- Unit tests: [`tests/test_preprocessing.py`](file:///s:/Projects/SpectrumSensing/tests/test_preprocessing.py) (12/12 passed, 80/80 overall)
- Verification script: [`scripts/verify_iq_preprocessing.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_iq_preprocessing.py)
- Generated figure: [`results/plots/p2_002_iq_preprocessing.png`](file:///s:/Projects/SpectrumSensing/results/plots/p2_002_iq_preprocessing.png)

## 📝 Observations
- In direct-conversion SDR receivers (e.g., RTL-SDR, LimeSDR, USRP), analog imperfections skew the constellation into an ellipse with an off-center offset.
- The preprocessing pipeline successfully removes the DC center offset and rotates/scales the constellation back to a clean symmetric square constellation prior to downstream detection or deep learning feature extraction.
