---
id: P2-001
title: "Implement Digital Signal Processing (DSP) Transform Library"
phase: 2 - Physical / SDR Layer
status: done
priority: critical
depends_on: [P1-001, P1-002]
tags: [dsp, fft, psd, stft, spectrogram]
---

## Objective
Implement core DSP transforms for RF I/Q time-series signals: Fast Fourier Transform (FFT),
Power Spectral Density (PSD) via Welch's averaged periodogram, Short-Time Fourier Transform
(STFT) spectrogram generation for 2D/1D CNN inputs, and digital bandpass/lowpass filter banks.

## Acceptance Criteria
- [x] FFT/IFFT with windowing (Hann, Hamming, Blackman) and frequency shifting in `/src/dsp/fft_utils.py`
- [x] Welch PSD estimation with configurable segment overlap and resolution in `/src/dsp/psd_estimator.py`
- [x] STFT spectrogram generator producing 2D time-frequency power matrices in `/src/dsp/spectrogram.py`
- [x] FIR/IIR digital filter design (Butterworth, Chebyshev, Matched filter) in `/src/dsp/filters.py`
- [x] Unit test suite in `/tests/test_dsp.py` verifying Parseval's energy theorem and frequency recovery
- [x] Verification script and plots in `/results/plots/p2_001_dsp_transforms.png`
