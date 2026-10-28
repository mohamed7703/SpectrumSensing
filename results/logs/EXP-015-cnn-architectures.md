# Experiment Log: EXP-015 — 1D-CNN Spectrum Sensing Architectures

- **Task**: P4-002: Implement 1D-CNN Spectrum Sensing Architectures (Baseline, BatchNorm, GAP)
- **Date**: 2026-10-29
- **Status**: PASSED

## Objectives
1. Implement the 3 deep learning 1D-CNN spectrum sensing architectures from Abdelbaset et al. (Sensors 2024):
   - **Baseline 1D-CNN** (Table 1 from paper): Standard convolutional model with Flatten and Dense layers.
   - **BatchNorm 1D-CNN** (Table 3 from paper): Batch-Normalized regularized model with 99% parameter reduction.
   - **GAP 1D-CNN** (Table 4 from paper): Global Average Pooling model with 99.3% parameter reduction.
2. Validate layer topologies, spatial tensor propagation from $(1024, 2) \to (1,)$, and parameter math against the published paper.
3. Benchmark single-frame forward inference latency against the $2.3\text{ ms}$ processing budget.
4. Establish unified model factory (`get_model()`) and YAML architectural configurations (`configs/model_config.yaml`).

## Results
- **Unit Test Coverage**: 12/12 tests passed in `tests/test_cnn_models.py`; **175/175 unit tests passed across entire repository**.
- **Architectural Parameter Verification**:
  - **Baseline 1D-CNN**: **2,087,457 parameters** (Exact match with Table 1).
  - **BatchNorm 1D-CNN**: **23,009 parameters** (22,817 trainable, 192 non-trainable; exact match with Table 3).
  - **GAP 1D-CNN**: **14,881 parameters** (Exact match with Table 4).
- **Latency Benchmarks**:
  - Baseline CNN: $12.33\ \mu\text{s}$/frame
  - BatchNorm CNN: $40.60\ \mu\text{s}$/frame
  - GAP CNN: $52.95\ \mu\text{s}$/frame
  - All models execute in $<0.06\text{ ms}$ per frame, utilizing $<2.5\%$ of the $2.3\text{ ms}$ budget!
- **Plot Saved**: `results/plots/p4_002_cnn_architectures.png`.
