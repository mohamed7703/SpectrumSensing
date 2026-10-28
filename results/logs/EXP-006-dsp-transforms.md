# Experiment 006: Digital Signal Processing (DSP) Transforms
- **Date:** 2026-10-28
- **Phase:** 2 - Physical / SDR Layer
- **Status:** Completed
- **Task:** P2-001

## ⚙️ Configuration
- **Sampling Frequency ($f_s$):** 2.0 MHz (SDR standard)
- **FFT Sizes ($N_{\text{fft}}$):** 64, 128, 256, 512, 1024
- **Window Functions:** Hann, Hamming, Blackman, Rectangular (Boxcar)
- **STFT Segment Length ($N_{\text{perseg}}$):** 64 - 128 samples with 75% overlap
- **Filter Families:** 6th-order digital Butterworth (IIR), Matched Filter template

## 📊 Results
| Component | Metric | Value | Verification Status |
|---|---|---|---|
| FFT / IFFT | Parseval Energy Relative Error | $< 10^{-12}$ | Energy strictly conserved |
| FFT Peak Detection | Frequency Accuracy | $< 488\text{ Hz}$ ($< 0.5$ bin) | Exact single-tone recovery |
| Welch PSD | White Noise Power Density | Matches $\sigma_w^2 / f_s$ within $8\%$ | Flat noise floor verified |
| Welch PSD | Peak Prominence over Noise | $> 20\text{ dB}$ | Narrowband tone cleanly separated |
| 2D STFT Spectrogram | Dynamic Range | $80\text{ dB}$ clipped floor | Frequency hops clearly distinguished |
| Butterworth Filter | Stopband Attenuation | $> 25\text{ dB}$ | Out-of-band noise suppressed |
| Matched Filter | Peak Synchronization Alignment | $\pm 0\text{ samples}$ | Optimal peak correlation |

## 💾 Artifacts
- Source code:
  - [`src/dsp/fft_utils.py`](file:///s:/Projects/SpectrumSensing/src/dsp/fft_utils.py)
  - [`src/dsp/psd_estimator.py`](file:///s:/Projects/SpectrumSensing/src/dsp/psd_estimator.py)
  - [`src/dsp/spectrogram.py`](file:///s:/Projects/SpectrumSensing/src/dsp/spectrogram.py)
  - [`src/dsp/filters.py`](file:///s:/Projects/SpectrumSensing/src/dsp/filters.py)
- Configuration: [`configs/dsp_config.yaml`](file:///s:/Projects/SpectrumSensing/configs/dsp_config.yaml)
- Unit tests: [`tests/test_dsp.py`](file:///s:/Projects/SpectrumSensing/tests/test_dsp.py) (12/12 passed, 68/68 overall)
- Verification script: [`scripts/verify_dsp_transforms.py`](file:///s:/Projects/SpectrumSensing/scripts/verify_dsp_transforms.py)
- Multi-panel figure: [`results/plots/p2_001_dsp_transforms.png`](file:///s:/Projects/SpectrumSensing/results/plots/p2_001_dsp_transforms.png)

## 📝 Observations
- The STFT spectrogram matrix generator provides the exact 2D feature representations required for convolutional spatial feature extraction and the jamming defense models referenced in the deployment plan.
- Zero-phase bidirectional filtering (`filtfilt`) prevents phase group delay distortion across I and Q baseband rails.
