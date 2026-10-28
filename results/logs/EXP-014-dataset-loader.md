# Experiment Log: EXP-014 — RadioML 2018.01A Dataset Ingestion & Loader

- **Task**: P4-001: Implement RadioML 2018.01A Dataset Ingestion & Batch Loader
- **Date**: 2026-10-29
- **Status**: PASSED

## Objectives
1. Implement out-of-core memory-mapped ingestion for the RadioML 2018.01A HDF5 dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`, 2,555,904 frames of shape $1024 \times 2$).
2. Implement high-fidelity synthetic fallback generator (`SyntheticRadioMLGenerator`) for offline CI/CD and environments without the 21.5 GB dataset file.
3. Map modulation types to binary hypothesis sensing labels: $y = 0$ ($\mathcal{H}_0$ idle noise) vs $y = 1$ ($\mathcal{H}_1$ active signals).
4. Enforce SNR threshold filtering ($-8.0\text{ dB}$ to $+30.0\text{ dB}$) per the paper to avoid training degeneration below the $-8.2\text{ dB}$ SNR wall.
5. Provide stratified train / validation / test partitioning ($70\% / 15\% / 15\%$) and mini-batch generator.

## Results
- **Unit Test Coverage**: 7/7 tests passed in `tests/test_dataset_loader.py`; **163/163 unit tests passed across entire repository**.
- **Dataset Verification**:
  - Ingested 1,920 balanced frames of exact shape $(1920, 1024, 2)$ float32.
  - Successfully mapped 960 $\mathcal{H}_0$ noise frames and 960 $\mathcal{H}_1$ modulated signal frames.
  - Stratified partitions: Train = 1,344 ($70\%$), Val = 288 ($15\%$), Test = 288 ($15\%$).
  - Batch generator verified yielding mini-batches of shape $(B, 1024, 2)$ and labels $(B,)$.
- **Plot Saved**: `results/plots/p4_001_dataset_loader.png`.
