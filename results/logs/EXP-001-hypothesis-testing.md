# Experiment 001: Binary Hypothesis Testing Validation
- **Date:** 2026-10-28
- **Phase:** 1 - Math Foundations
- **Status:** Completed
- **Task:** P1-001

## ⚙️ Configuration
- **Samples per frame (L):** 1024
- **Noise variance ($\sigma_w^2$):** 1.0
- **Modulation:** QPSK
- **SNR:** 3.0 dB ($10^{0.3} \approx 1.9953$)
- **Config File:** `/configs/math_config.yaml`
- **Frames generated:** 2,000 (1,000 $H_0$, 1,000 $H_1$)

## 📊 Results
| Metric | Theoretical | Empirical | Error |
|---|---|---|---|
| $H_0$ Noise Variance ($\sigma_w^2$) | 1.0000 | 0.9994 | -0.06% |
| $H_1$ Signal+Noise Variance ($\sigma_w^2(1 + \text{SNR})$) | 2.9953 | 2.9946 | -0.02% |
| Frame output tensor shape | $(2000, 1024, 2)$ | $(2000, 1024, 2)$ | Exact |
| Prior $P(H_1)$ | 0.5000 | 0.4985 | 0.30% |

## 💾 Artifacts
- Source code: [`src/math_models/hypothesis_testing.py`](file:///s:/Projects/SpectrumSensing/src/math_models/hypothesis_testing.py)
- Unit tests: [`tests/test_hypothesis_testing.py`](file:///s:/Projects/SpectrumSensing/tests/test_hypothesis_testing.py) (16/16 passed)
- Verification script: [`scripts/verify_hypothesis_testing.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_hypothesis_testing.py)
- Verification plot: [`results/plots/p1_001_hypothesis_testing.png`](file:///s:/Projects/SpectrumSensing/results/plots/p1_001_hypothesis_testing.png)

## 📝 Notes
- Model accurately reproduces both complex baseband and RadioML compatible $(N, L, 2)$ tensors.
- BPSK, QPSK, sinusoid, and complex Gaussian modulation schemes all conserve specified signal power.
