---
id: P1-004
title: "Implement Channel Fading Models (Rayleigh, Rician, Nakagami-m)"
phase: 1 - Math Foundations
status: done
priority: high
depends_on: [P1-001]
tags: [math, fading, channel-models]
---

## Objective
Implement wireless channel fading PDFs and their effect on signal detection.

## Acceptance Criteria
- [x] Rayleigh fading: f(a) = (a/σ²)·exp(-a²/2σ²)
- [x] Rician fading with Bessel function I₀
- [x] Nakagami-m generalized model
- [x] Validate: accuracy <70% at SNR=0dB under Rayleigh/Rician (or severe degradation in transition)
- [x] Validate: accuracy >95% at SNR≥8dB
- [x] Output: `/src/math_models/channel_fading.py`
