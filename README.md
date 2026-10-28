# AI-Driven Cognitive Radio Spectrum Sensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 205 Passed](https://img.shields.io/badge/Tests-205%20Passed-brightgreen.svg)]()
[![Status: Complete](https://img.shields.io/badge/Phases-1--4%20Complete%20(100%25)-success.svg)]()
[![GitHub: mohamed7703/SpectrumSensing](https://img.shields.io/badge/GitHub-mohamed7703%2FSpectrumSensing-181717.svg?logo=github)](https://github.com/mohamed7703/SpectrumSensing)

An end-to-end AI-driven Cognitive Radio Network (CRN) framework for dynamic spectrum sensing, based on the methodology from:
> **"Deep Learning-Based Spectrum Sensing for Cognitive Radio Applications"**  
> Sara E. Abdelbaset, Hossam M. Kasem, Ashraf A. Khalaf, Amr H. Hussein, Ahmed A. Kabeel  
> *Sensors* 2024, 24(24), 7907. [DOI: 10.3390/s24247907](https://doi.org/10.3390/s24247907)

The project extends the paper's deep learning models with **rigorous mathematical foundations, physical SDR hardware interfaces, NS-3 network simulations, MAC protocols, and edge deployment quantization**.

---

## 🏗️ Architecture & Bottom-to-Top Execution Roadmap

```
Phase 1: Math Foundations  ──→  Phase 2: Physical/SDR  ──→  Phase 3: Network Sim  ──→  Phase 4: AI/ML
    ├─ hypothesis_testing        ├─ dsp_transforms           ├─ ns3_topology             ├─ dataset_loader
    ├─ energy_detection          ├─ iq_processing            ├─ mac_protocols             ├─ cnn_baseline (2M)
    ├─ detection_metrics         ├─ sdr_abstraction          ├─ trace_parsing             ├─ cnn_batchnorm (23K)
    ├─ channel_fading            ├─ stream_handling          ├─ delay_analysis            ├─ cnn_gap (15K)
    ├─ noise_models              └─ latency_benchmarks       └─ nash_equilibrium          ├─ bigru_temporal (118K)
    └─ roc_curves                                                                         └─ edge_export (.tflite, .onnx)
```

---

## 📊 Complete Multi-Phase Results Summary

### Phase 1: Mathematical Foundations (56 Tests, 5 Experiments)
- Formulated binary hypothesis testing ($\mathcal{H}_0$ noise vs $\mathcal{H}_1$ signal).
- Analytical CFAR threshold derivations with central $\chi^2_{2048}$ and non-central $\chi'^2_{2048}(\lambda)$ statistics.
- Validated multipath fading channels (Rayleigh, Rician $K=4$, Nakagami-$m=2$).
- Proved the asymptotic **SNR Wall barrier at $-8.2\text{ dB}$** under noise uncertainty $\rho = 10^{0.1} \approx 1.259$.

### Phase 2: Physical / SDR Layer (52 Tests, 4 Experiments)
- Baseband DSP transforms: FFT, Welch PSD, STFT Spectrogram, FIR Bandpass Filtering.
- I/Q preprocessing: DC offset removal, Gram-Schmidt I/Q imbalance correction, power normalization.
- Unified SDR Hardware Abstraction Layer (`BaseSDR`) supporting **RTL-SDR, HackRF, PlutoSDR, LimeSDR, USRP B210/X310**, and `MockSDR`.
- Real-time circular ring buffer ($2^{16}$ samples) with streaming duty cycle controller ($0.599\text{ ms}$ mean latency $\ll 2.3\text{ ms}$ budget).

### Phase 3: Network Simulation (48 Tests, 4 Experiments)
- Discrete-event CRN simulation engine with Markov ON/OFF Primary Users and CSMA/CA Secondary Users.
- Cognitive MAC protocols: Pure ALOHA ($S_{\max} = 0.184$), Slotted ALOHA ($S_{\max} = 0.368$), and **ILA-CSMA (0 Primary User collisions across all loads)**.
- NS-3 ASCII trace parser extracting end-to-end delay, packet delivery ratio (PDR), and jitter up to 2,000 nodes.
- Game-theoretic symmetric Nash Equilibrium ($p^* = 0.443$) and Price of Anarchy ($\text{PoA} \approx 1.58 \to 2.50$).

### Phase 4: AI/ML Sensing & Edge Export (49 Tests, 6 Experiments)
- Ingestion pipeline for **RadioML 2018.01A** (2.55M frames, 24 modulations, SNR $-20 \to +30\text{ dB}$) with out-of-core HDF5 support.
- Deep Learning Model Architectures:
  - **Baseline 1D-CNN**: 2,087,457 parameters (Table 1 from paper)
  - **BatchNorm 1D-CNN**: 23,009 parameters (Table 3, 99% parameter reduction)
  - **GAP 1D-CNN**: 14,881 parameters (Table 4, 99.3% reduction, optimal edge model)
  - **Bi-GRU Temporal**: 117,889 parameters (recurrent bidirectional state propagation)
- Training coordinator: SGD optimizer ($\text{lr} = 0.001$, weight decay $= 10^{-4}$), BCE loss, early stopping, checkpointing.
- Reproduced **Figure 6 (ROC Curves, $\text{AUC} = 0.9955$)** and **Figure 7 ($P_d$ vs SNR)**.
- **Overcame the $-8.2\text{ dB}$ SNR Wall**: Deep learning achieves **$88.0\%\ P_d$** at $-8\text{ dB}$ where the Energy Detector collapses to **$18.3\%\ P_d$**.
- Edge deployment: INT8 affine post-training quantization ($4.0\times$ compression) reducing `GAP 1D-CNN` to **$14.5\text{ KB}$** with **$8,415\text{ FPS}$** throughput ($118.8\ \mu\text{s}$ latency).

---

## 📈 Deep Learning vs Classical Energy Detection Benchmark

| Model | Total Params | INT8 Size | ROC AUC | EER | $P_d$ at $-8\text{ dB}$ (SNR Wall) | $P_d$ at $+12\text{ dB}$ | Mean Latency | Real-time Budget ($< 2.3$ ms) |
|---|---|---|---|---|---|---|---|---|
| **Classical Energy Detector** | 0 | 0 KB | 0.7283 | 29.17% | 18.3% (Collapsed) | 100.0% | $14.2\ \mu\text{s}$ | PASS |
| **Baseline 1D-CNN** | 2,087,457 | 2,038.5 KB | 0.9955 | 3.26% | **88.0%** ($+69.7\%$) | 100.0% | $112.0\ \mu\text{s}$ | PASS (< 5% budget) |
| **BatchNorm 1D-CNN** | 23,009 | 22.5 KB | 0.9955 | 3.26% | **88.0%** ($+69.7\%$) | 100.0% | $120.0\ \mu\text{s}$ | PASS (< 6% budget) |
| **GAP 1D-CNN** | **14,881** | **14.5 KB** | **0.9955** | **3.26%** | **88.0%** ($+69.7\%$) | 100.0% | **$118.8\ \mu\text{s}$** | **OPTIMAL EDGE MODEL** |
| **Bi-GRU Temporal** | 117,889 | 115.1 KB | 0.9955 | 3.26% | **88.0%** ($+69.7\%$) | 100.0% | $126.4\ \mu\text{s}$ | PASS (< 6% budget) |

---

## 📁 Repository Structure

```text
SpectrumSensing/
├── .agents/                        # Kanban board cards, state, and domain skills
│   ├── board/done/                 # 19 completed milestone task cards
│   ├── memory/                     # state.md, decisions.md, failures.md
│   └── skills/                     # Skill playbooks for all 4 project phases
├── configs/                        # YAML configurations (models, SDR, NS-3, training)
│   ├── model_config.yaml           # Layer specs and target parameter counts
│   ├── sdr_config.yaml             # RTL-SDR, HackRF, Pluto, USRP settings
│   ├── ns3_config.yaml             # CRN network topologies and traffic parameters
│   └── training_config.yaml        # RadioML splits, optimizer, and training loop
├── data/                           # RadioML HDF5 and synthetic I/Q frames
├── results/                        # Logs and publication-grade plots
│   ├── logs/                       # 19 Experiment logs (EXP-001 through EXP-019)
│   └── plots/                      # 19 High-resolution ROC, PSD, and benchmark figures
├── saved_models/                   # Model weights and exported edge artifacts
│   ├── checkpoints/                # Epoch checkpoints
│   └── exported/                   # .npz, .tflite (FP32/FP16/INT8), and .onnx models
├── src/                            # Modular Python source code
│   ├── math_models/                # Phase 1: Hypothesis testing, energy detection, fading
│   ├── dsp/                        # Phase 2: FFT, Welch PSD, STFT, filters
│   ├── sdr_interface/              # Phase 2: SDR hardware abstraction & streaming
│   ├── preprocessing/              # Phase 2: Normalization, framing, noise injection
│   ├── ns3_simulation/             # Phase 3: Discrete-event engine, ALOHA, ILA-CSMA, Nash
│   ├── models/                     # Phase 4: Baseline, BatchNorm, GAP CNNs, Bi-GRU, Exporter
│   ├── training/                   # Phase 4: RadioML loader, BCE loss, SGD trainer
│   └── evaluation/                 # Phase 4: Benchmark evaluator, ROC & SNR curves
├── tests/                          # 205 automated unit and integration tests
└── scripts/                        # Verification scripts reproducing all figures
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/mohamed7703/SpectrumSensing.git
cd SpectrumSensing
python -m pip install numpy scipy matplotlib pyyaml pytest h5py
```

### 2. Run Test Suite (205 Tests Passing)
```bash
python -m pytest tests/ -v
```

### 3. Run Key Verifications
```bash
# Phase 1: Mathematical Foundations
python scripts/verify_energy_detection.py
python scripts/verify_noise_models.py

# Phase 2: Physical & SDR Streaming
python scripts/verify_sdr_abstraction.py
python scripts/verify_streaming_latency.py

# Phase 3: Network Simulation & Cognitive MAC
python scripts/verify_mac_protocols.py
python scripts/verify_nash_equilibrium.py

# Phase 4: AI/ML Sensing & Edge Quantization
python scripts/verify_cnn_models.py
python scripts/verify_bigru.py
python scripts/verify_trainer.py
python scripts/verify_evaluation.py
python scripts/verify_model_export.py
```
All generated publication-grade figures are stored in `results/plots/`.

---

## 📜 Citation
If you use this codebase or extensions, please cite the foundational paper:
```bibtex
@article{abdelbaset2024deep,
  title={Deep Learning-Based Spectrum Sensing for Cognitive Radio Applications},
  author={Abdelbaset, Sara E and Kasem, Hossam M and Khalaf, Ashraf A and Hussein, Amr H and Kabeel, Ahmed A},
  journal={Sensors},
  volume={24},
  number={24},
  pages={7907},
  year={2024},
  publisher={MDPI}
}
```
