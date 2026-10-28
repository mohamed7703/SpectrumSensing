# Experiment Log: EXP-019 — Model Export, Edge Quantization & Latency Profiling

## Metadata
- **Date**: 2026-10-29
- **Phase**: Phase 4: AI/ML Sensing
- **Milestone / Task**: P4-006 — Implement Model Export & Quantization for Edge SDR Deployment
- **Script**: `scripts/verify_model_export.py`
- **Plot**: `results/plots/p4_006_model_edge_export.png`
- **Status**: SUCCESS

---

## 1. Objective
Implement an edge model export and post-training quantization pipeline supporting:
- Serialization to NumPy NPZ (`.npz`), TensorFlow Lite FlatBuffers (`.tflite`), and Open Neural Network Exchange (`.onnx`).
- Uniform affine INT8 quantization ($4.0\times$ compression) and FP16 half-precision ($2.0\times$ compression).
- Hardware-level latency and throughput profiling against the $2.3\text{ ms}$ real-time SDR frame budget (for embedded deployment on Raspberry Pi 4/5 and NVIDIA Jetson).

---

## 2. Mathematical Formulation
1. **Affine INT8 Quantization**:
   $$S = \frac{\max(\mathbf{W}) - \min(\mathbf{W})}{255}, \quad Z = \text{round}\left( -\frac{\min(\mathbf{W})}{S} \right)$$
   $$\mathbf{Q} = \text{clip}\left(\text{round}\left(\frac{\mathbf{W}}{S}\right) + Z, 0, 255\right)$$
   $$\hat{\mathbf{W}} = S \cdot (\mathbf{Q} - Z)$$
2. **Reconstruction Distortion**:
   $$\text{MSE} = \frac{1}{K} \sum_{k=1}^K (w_k - \hat{w}_k)^2, \quad \text{MAE} = \max_k |w_k - \hat{w}_k|$$

---

## 3. Quantization Precision & Compression Results

| Quantization Mode | Data Type | Parameter Storage | Compression Ratio | Mean Squared Error (MSE) | Max Absolute Error (MAE) |
|---|---|---|---|---|---|
| **FP32 (Original)** | `float32` | 4 bytes / param | **1.0x** (Baseline) | $0.0$ | $0.0$ |
| **FP16 (Half)** | `float16` | 2 bytes / param | **2.0x** | $4.37 \times 10^{-8}$ | $1.91 \times 10^{-4}$ |
| **INT8 (Affine)** | `uint8` | 1 byte / param | **4.0x** | $8.56 \times 10^{-5}$ | $1.95 \times 10^{-2}$ |

---

## 4. Edge Latency & Real-Time SDR Compliance Benchmark
- Real-time frame deadline at $446.4\text{ kSps}$ ($1024$ samples): $\mathbf{2.3\text{ ms}} = 2,300\ \mu\text{s}$.

| Architecture | Model Footprint (INT8) | Mean Latency | P95 Latency | Throughput (FPS) | SDR Budget Utilization | Status |
|---|---|---|---|---|---|---|
| **Baseline 1D-CNN** | 2,038.5 KB | 112.0 µs | 156.4 µs | 8,926 FPS | 4.87% | **COMPLIANT** |
| **BatchNorm 1D-CNN** | 22.5 KB | 120.0 µs | 184.8 µs | 8,330 FPS | 5.22% | **COMPLIANT** |
| **GAP 1D-CNN** | **14.5 KB** | **118.8 µs** | **184.9 µs** | **8,415 FPS** | **5.17%** | **OPTIMAL EDGE CANDIDATE** |
| **Bi-GRU Temporal** | 115.1 KB | 126.4 µs | 229.8 µs | 7,911 FPS | 5.50% | **COMPLIANT** |

---

## 5. Verification Summary
All four deep learning architectures have been successfully exported and quantized. With INT8 quantization, the `GAP 1D-CNN` shrinks to just **14.5 KB** while sustaining **8,415 frames per second** (mean latency $118.8\ \mu\text{s}$), using barely $5\%$ of the $2.3\text{ ms}$ real-time budget.

---

## 6. Artifacts Produced
- Exporter module: `src/models/model_export.py`
- Test suite: `tests/test_model_export.py` (6 unit tests)
- Verification script: `scripts/verify_model_export.py`
- Visual benchmark: `results/plots/p4_006_model_edge_export.png`
- Exported models directory: `saved_models/exported/` (.npz, .tflite, .onnx)
