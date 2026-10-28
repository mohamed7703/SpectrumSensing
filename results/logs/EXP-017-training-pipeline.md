# Experiment Log: EXP-017 — Deep Learning Training Pipeline & Optimization

## Metadata
- **Date**: 2026-10-29
- **Phase**: Phase 4: AI/ML Sensing
- **Milestone / Task**: P4-004 — Implement Training Pipeline (BCE Loss, SGD Optimizer, Schedulers)
- **Script**: `scripts/verify_trainer.py`
- **Plot**: `results/plots/p4_004_training_pipeline.png`
- **Status**: SUCCESS

---

## 1. Objective
Implement and verify a complete, scalable deep learning training pipeline for cognitive radio spectrum sensing matching the specifications in Abdelbaset et al. (Sensors 2024). The pipeline includes:
- Binary Cross-Entropy (BCE) loss with numerical stability safeguards and L2 weight decay ($\lambda = 10^{-4}$).
- Stochastic Gradient Descent (SGD) optimizer with learning rate $\eta = 10^{-3}$ and momentum $\mu = 0$.
- Early stopping callback monitoring validation loss (patience = 5).
- Model checkpointing saving optimal parameters upon validation loss improvement.
- Comprehensive cognitive radio metric tracking ($P_d$, $P_{fa}$, ROC-AUC, Binary Accuracy, Precision, F1-score).

---

## 2. Mathematical Formulation
1. **Regularized Binary Cross-Entropy Loss**:
   $$\mathcal{L}_{\text{total}}(\mathbf{y}, \hat{\mathbf{p}}, \mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{\lambda}{2} \|\mathbf{W}\|_2^2$$
   where $\hat{p}_i = \sigma(z_i)$ and $\epsilon = 10^{-15}$ avoids logarithmic singularities.
2. **Analytical Gradient with Respect to Logits**:
   $$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z_i} = \frac{1}{N} (\hat{p}_i - y_i)$$
3. **SGD Parameter Update with Weight Decay**:
   $$\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t) + \lambda \boldsymbol{\theta}_t$$
   $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_{t+1}$$

---

## 3. Training Dynamics & Results

- **Dataset**: Balanced synthetic RadioML 2018.01A frames filtered to $\text{SNR} \ge -8.0\text{ dB}$ (avoiding the SNR wall).
  - Train: 806 frames
  - Validation: 172 frames
  - Test: 174 frames
- **Model**: `BatchNorm 1D-CNN` (23,009 parameters).
- **Optimization**: Mini-batch SGD (batch size = 64, learning rate = 0.001, weight decay = 0.0001).

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val $P_d$ (%) | Val ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | 0.4412 | 96.53% | 0.4314 | 95.35% | 98.82% | 0.9961 |
| 4 | 0.4314 | 96.77% | 0.4214 | 97.09% | 97.65% | 0.9959 |
| 8 | 0.4194 | 96.77% | 0.4089 | 98.26% | 96.47% | 0.9955 |
| 12 | **0.4083** | **96.53%** | **0.3974** | **98.26%** | **96.47%** | **0.9953** |

### Held-Out Test Set Performance (Epoch 12 Best Checkpoint):
- **Test Loss**: $0.4101$
- **Test Accuracy**: $94.83\%$
- **Detection Probability ($P_d$)**: $\mathbf{92.39\%}$ ($\ge 90\%$ IEEE 802.22 target satisfied)
- **False Alarm Rate ($P_{fa}$)**: $\mathbf{2.44\%}$ ($\le 10\%$ IEEE 802.22 limit satisfied)
- **ROC Area Under Curve (AUC)**: $\mathbf{0.9916}$

---

## 4. Verification Conclusion
The training pipeline successfully stabilizes and converges in 12 epochs with monotonic validation loss descent and exceptional discriminative accuracy ($\text{AUC} > 0.99$). All cognitive radio regulatory constraints are satisfied on unseen test data.

---

## 5. Artifacts Produced
- `src/training/loss_functions.py`: BCE loss, L2 penalty, analytical gradient.
- `src/training/trainer.py`: `SpectrumSensingTrainer`, `SGDOptimizer`, `EarlyStopping`, `ModelCheckpoint`, `compute_binary_metrics`.
- `tests/test_trainer.py`: 11 unit tests covering all loss, optimizer, and callback functions.
- `scripts/verify_trainer.py`: End-to-end verification and visualization script.
- `results/plots/p4_004_training_pipeline.png`: 4-panel training dynamics plot.
