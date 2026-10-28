---
id: P4-002
title: "Implement 1D-CNN Spectrum Sensing Architectures (Baseline, BatchNorm, GAP)"
phase: 4 - AI/ML Sensing
status: done
priority: high
depends_on: [P4-001]
tags: [models, cnn, baseline, batchnorm, gap, tensorflow, keras]
---

## Objective
Implement all three 1D-CNN architectures analyzed in the Abdelbaset et al. (Sensors 2024) paper:
1. Baseline CNN (~2,060,000 parameters)
2. BatchNorm CNN (~23,000 parameters, 99% parameter reduction)
3. Global Average Pooling (GAP) CNN (~15,000 parameters, maximum parameter efficiency)

## Acceptance Criteria
- [x] Baseline CNN model builder in `/src/models/cnn_baseline.py`
- [x] BatchNorm CNN model builder in `/src/models/cnn_batchnorm.py`
- [x] GAP CNN model builder in `/src/models/cnn_gap.py`
- [x] Architectural config definitions in `/configs/model_config.yaml`
- [x] Exact parameter counts and layer topology validation matching the paper
- [x] Unit tests in `/tests/test_cnn_models.py`
