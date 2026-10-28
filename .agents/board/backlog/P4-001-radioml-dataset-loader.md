---
id: P4-001
title: "Implement RadioML 2018.01A Dataset Ingestion & Batch Loader"
phase: 4 - AI/ML Sensing
status: backlog
priority: high
depends_on: [P3-004]
tags: [dataset, radioml, hdf5, loader, data-generator]
---

## Objective
Implement dataset ingestion pipeline for RadioML 2018.01A (2.55M frames of 1024x2 I/Q samples across
24 modulation schemes and SNR from -20 to +30 dB), with SNR filtering (-8 dB to +30 dB per paper),
binary hypothesis labelling (H0 vs H1), synthetic dataset fallback, and batched streaming data generator.

## Acceptance Criteria
- [ ] RadioML HDF5 dataset loader with out-of-core memory-mapped access in `/src/training/dataset_loader.py`
- [ ] Synthetic RadioML dataset generator for offline testing without the 21.5 GB file
- [ ] Binary hypothesis label mapper (Noise only = H0, Modulated signals = H1)
- [ ] SNR filtering range configuration in `/configs/training_config.yaml`
- [ ] Unit tests in `/tests/test_dataset_loader.py`
