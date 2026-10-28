# Experiment Log: EXP-016 — Bidirectional GRU (Bi-GRU) Temporal Architecture

## Metadata
- **Date**: 2026-10-29
- **Phase**: Phase 4: AI/ML Sensing
- **Milestone / Task**: P4-003 — Implement Bi-GRU Recurrent Temporal Architecture
- **Script**: `scripts/verify_bigru.py`
- **Plot**: `results/plots/p4_003_bigru_architecture.png`
- **Status**: SUCCESS

---

## 1. Objective
Design and implement a Bidirectional Gated Recurrent Unit (Bi-GRU) neural network model to capture long-range temporal dependencies, carrier phase coherence, and digital modulation symbol transitions across baseband I/Q frames $(1024 \times 2)$. Benchmark architectural parameter complexity, recurrent state propagation, and forward inference latency against 1D-CNN architectures (Baseline, BatchNorm, GAP).

---

## 2. Mathematical Formulation
For an input sequence $\mathbf{X} \in \mathbb{R}^{T \times 2}$ ($T = 1024$ samples):
1. **Forward GRU Recurrence**:
   $$\mathbf{z}_t^f = \sigma(\mathbf{W}_{xz}^f \mathbf{x}_t + \mathbf{W}_{hz}^f \mathbf{h}_{t-1}^f + \mathbf{b}_z^f)$$
   $$\mathbf{r}_t^f = \sigma(\mathbf{W}_{xr}^f \mathbf{x}_t + \mathbf{W}_{hr}^f \mathbf{h}_{t-1}^f + \mathbf{b}_r^f)$$
   $$\tilde{\mathbf{h}}_t^f = \tanh(\mathbf{W}_{xh}^f \mathbf{x}_t + \mathbf{r}_t^f \odot (\mathbf{W}_{hh}^f \mathbf{h}_{t-1}^f + \mathbf{b}_{hr}^f) + \mathbf{b}_{hx}^f)$$
   $$\mathbf{h}_t^f = (1 - \mathbf{z}_t^f) \odot \tilde{\mathbf{h}}_t^f + \mathbf{z}_t^f \odot \mathbf{h}_{t-1}^f$$
2. **Backward GRU Recurrence**:
   Symmetric propagation traversing time backwards ($t = T \to 1$) producing $\mathbf{h}_t^b$.
3. **State Fusion & Decision Head**:
   $$\mathbf{h}_{\text{merged}} = [\mathbf{h}_T^f \,\|\, \mathbf{h}_1^b] \in \mathbb{R}^{256}$$
   $$\mathbf{y}_1 = \text{ReLU}(\mathbf{W}_{d1} \mathbf{h}_{\text{merged}} + \mathbf{b}_{d1}) \in \mathbb{R}^{64}$$
   $$\hat{y} = \sigma(\mathbf{w}_{d2}^\top \mathbf{y}_1 + b_{d2}) \in [0, 1]$$

---

## 3. Parameter Accounting & Complexity Benchmark

| Architecture | Layer Topology | Total Params | Trainable | Non-Trainable | Inference Latency | Real-time Budget ($< 2.3$ ms) |
|---|---|---|---|---|---|---|
| **Baseline 1D-CNN** | Conv(32)-Pool-Conv(64)-Pool-Flatten-Dense(128)-Dense(1) | 2,087,457 | 2,087,457 | 0 | 42.6 µs | PASS (< 2% budget) |
| **BatchNorm 1D-CNN** | Conv(32)-BN-Pool-Drop-Conv(64)-BN-Pool-Drop-Dense(1) | 23,009 | 22,817 | 192 | 53.2 µs | PASS (< 3% budget) |
| **GAP 1D-CNN** | Conv(32)-Pool-Conv(64)-Pool-GAP-Dense(128)-Dense(1) | 14,881 | 14,881 | 0 | 59.3 µs | PASS (< 3% budget) |
| **Bi-GRU Temporal** | Bi-GRU(128)-Dense(64)-Drop(0.5)-Dense(1) | **117,889** | **117,889** | 0 | **94.3 µs** | **PASS (< 5% budget)** |

### Parameter Breakdown for Bi-GRU:
- **Bidirectional GRU Layer**:
  - Forward GRU: $2 \times 3 \times 128 + 128 \times 3 \times 128 + 2 \times 3 \times 128 = 768 + 49,152 + 768 = 50,688$
  - Backward GRU: 50,688
  - Total GRU: $101,376$
- **Dense Projection Layer (64 units)**: $256 \times 64 + 64 = 16,448$
- **Dropout (0.50)**: 0
- **Dense Output Layer (1 unit)**: $64 \times 1 + 1 = 65$
- **Total**: $117,889$ parameters.

---

## 4. Verification Results
- **Unit Tests**: 9/9 passing in `tests/test_bigru.py` (total repository suite: 184/184 tests passing).
- **Recurrent State Propagation**: Full sequential recurrence and subsampled verification completed with bounded hidden activations $[-1, 1]$ and finite state norm $\|\mathbf{h}_{\text{merged}}\|_2 \approx 0.81$.
- **Latency Verification**: Forward inference completes in $94.3\ \mu\text{s}$, well within the $2.3\text{ ms}$ real-time SDR frame budget (utilizing only 4.1% of maximum allowable budget).

---

## 5. Artifacts Produced
- Implementation: `src/models/bigru_model.py` and `src/models/bigru_temporal.py`
- Factory Registration: `src/models/__init__.py` (`get_model('bigru')`)
- Model Configuration: `configs/model_config.yaml`
- Test Suite: `tests/test_bigru.py`
- Benchmark Script: `scripts/verify_bigru.py`
- Plot: `results/plots/p4_003_bigru_architecture.png`
