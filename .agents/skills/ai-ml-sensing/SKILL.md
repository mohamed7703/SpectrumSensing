---
name: ai-ml-sensing
description: >-
  Deep learning model development for spectrum sensing: 1D-CNN architectures
  (Baseline 2M params, BatchNorm 23K params, GAP 15K params), Bi-GRU temporal
  models, training on RadioML 2018.01A dataset, Binary Cross-Entropy loss,
  SGD optimization, and model export for edge deployment. Use when building,
  training, evaluating, or deploying the AI spectrum sensing models.
---

# AI/ML Spectrum Sensing Skill

## Scope
Implements the three CNN architectures from the paper plus extensions from the plan.
This is the final phase — it depends on math foundations and DSP being complete.

## Workflow

### Step 1: Dataset Pipeline
Implement in `/src/training/`:
- `dataset_loader.py` — Load RadioML 2018.01A (HDF5 format)
  - 2,555,904 frames, shape (1024, 2), 24 modulation types
  - Split by modulation type + SNR level (no data leakage)
  - Training / Validation / Test partitions
- `data_augmentation.py` — SNR filtering (-8 to +30 dB), noise injection

### Step 2: Model Architecture — Baseline (Table 1 from Paper)
Implement in `/src/models/cnn_baseline.py`:
```
Conv1D(k=3, f=32) → MaxPool(2) → Conv1D(k=3, f=64) → MaxPool(2)
→ Flatten → Dense(128) → Dropout(50%) → Dense(1, sigmoid)
Total: 2,087,457 parameters
```

### Step 3: Model Architecture — Batch Normalization (Table 3)
Implement in `/src/models/cnn_batchnorm.py`:
```
Input(1024,2) → Conv1D(32) → BatchNorm → MaxPool → Dropout
→ Conv1D(64) → BatchNorm → MaxPool → Dropout
→ Flatten → Dense(1, sigmoid)
Total: 23,009 trainable parameters
```

### Step 4: Model Architecture — GAP (Table 4)
Implement in `/src/models/cnn_gap.py`:
```
Conv1D(k=3, f=32) → MaxPool(2) → Conv1D(k=3, f=64) → MaxPool(2)
→ GlobalAveragePooling1D → Dense(128) → Dropout → Dense(1, sigmoid)
Total: 14,881 parameters
```

### Step 5: Loss Function & Optimizer
Implement in `/src/training/`:
- `loss_functions.py` — Binary Cross-Entropy (BCE)
  - `LogLoss = -(1/n) Σ Σ y_ij · log(p_ij)`
- `train.py` — Training loop with:
  - SGD: lr=0.001, momentum=0, weight_decay=0.0001
  - Batch sizes: 128 → 2048 (scaled for GPU)
  - Epochs: 10 (baseline), 32 (optimized), 40 (with BN)

### Step 6: Evaluation Metrics
Implement in `/src/evaluation/`:
- `metrics.py` — Pd, Pfa, ROC curves, AUC
- `benchmark.py` — Compare against eigenvalue-based and entropy-based methods
- `snr_analysis.py` — Accuracy vs SNR plots per modulation type

### Step 7: Advanced Architectures (Plan Extensions)
- `/src/models/bigru_temporal.py` — Bidirectional GRU (128 neurons each direction)
  - Update gate: z_t, Reset gate: r_t (sigmoid)
  - `h_t = (1 - z_t) · h̃_t + z_t · h_{t-1}`
- `/src/models/attention_hybrid.py` — ACNet / SPA-ResNet style attention
- `/src/models/mobilenet1d.py` — MobileNetV2 adapted for 1D signals (~11.6 MB)

### Step 8: Model Export
- `/saved_models/checkpoints/` — Training checkpoints (.h5)
- `/saved_models/exported/` — TFLite, ONNX for edge deployment

## Expected Results (from Paper)
| Config | Accuracy | Epochs |
|--------|----------|--------|
| No noise | 100% | 2 |
| AWGN, batch 128 | 87.5% | 10 |
| AWGN + BN, batch 2048 | 97.5% | 40 |
| AWGN, SNR -8→+30, GAP | >98% | 32 |

## Verification
- Tests in `/tests/test_models.py`
- Model output shapes must match paper's tables exactly
- Training on clean signals must reach 100% by epoch 2
- Accuracy under AWGN must reach >97.5% with BN architecture
