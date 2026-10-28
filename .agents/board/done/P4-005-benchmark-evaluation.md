---
id: P4-005
title: "Implement Benchmark Evaluation Suite (ROC, Pd vs SNR, Energy Detector Comparison)"
phase: 4 - AI/ML Sensing
status: done
priority: high
depends_on: [P4-004]
tags: [evaluation, benchmarks, roc-curves, pd-snr, comparisons]
---

## Objective
Implement evaluation suite comparing deep learning models against the classical Energy Detector
baseline across the full SNR span (-20 dB to +30 dB), plotting ROC curves, detection probability
versus SNR, and confusion matrices.

## Acceptance Criteria
- [x] Evaluation engine in `/src/evaluation/evaluator.py`
- [x] Direct comparison scripts reproducing paper Figure 6 (ROC curves) and Figure 7 (Pd vs SNR)
- [x] Verification of deep learning advantage below the SNR wall (-8.2 dB)
- [x] Unit tests in `/tests/test_evaluation.py`
