---
id: P2-002
title: "Implement I/Q Signal Preprocessing & Normalization Pipeline"
phase: 2 - Physical / SDR Layer
status: backlog
priority: critical
depends_on: [P2-001]
tags: [preprocessing, iq, normalization, framing]
---

## Objective
Implement physical-layer I/Q data preprocessing: DC offset removal, I/Q imbalance correction,
unit energy / variance normalization, dynamic frame chunking into (1024, 2) arrays, and calibrated
noise injection.

## Acceptance Criteria
- [ ] DC offset estimation and carrier leakage mitigation in `/src/preprocessing/iq_normalizer.py`
- [ ] Power normalization ensuring $\mathbb{E}[I^2 + Q^2] = 1.0$ or unit variance
- [ ] Frame builder converting continuous stream into $1024 \times 2$ arrays in `/src/preprocessing/frame_builder.py`
- [ ] Synthetic noise injector adding calibrated AWGN across target SNR levels in `/src/preprocessing/noise_injector.py`
- [ ] Signal augmentation (frequency offset, phase jitter, fractional delay) in `/src/preprocessing/augmentation.py`
- [ ] Unit tests in `/tests/test_preprocessing.py`
