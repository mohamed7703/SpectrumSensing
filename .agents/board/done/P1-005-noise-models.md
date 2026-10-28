---
id: P1-005
title: "Implement Noise Models and SNR Wall Analysis"
phase: 1 - Math Foundations
status: done
priority: high
depends_on: [P1-002]
tags: [math, noise, snr-wall]
---

## Objective
Implement AWGN generation and demonstrate the SNR wall phenomenon.

## Acceptance Criteria
- [x] AWGN generator with configurable σ (paper uses σ=0.708)
- [x] SNR wall demonstration: show detection failure at -12 to -8 dB
- [x] Noise uncertainty impact analysis
- [x] Motivational plot: energy detection vs ML-based sensing at low SNR
- [x] Output: `/src/math_models/noise_models.py`
