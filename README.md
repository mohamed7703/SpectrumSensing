# AI-Driven Cognitive Radio Spectrum Sensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 56 Passed](https://img.shields.io/badge/Tests-56%20Passed-brightgreen.svg)]()
[![Phase: Phase 1 Complete](https://img.shields.io/badge/Phase-1%20Complete%20(Math)-blue.svg)]()

An end-to-end AI-driven Cognitive Radio Network (CRN) framework for dynamic spectrum sensing, based on the methodology from:
> **"Deep Learning-Based Spectrum Sensing for Cognitive Radio Applications"**  
> Sara E. Abdelbaset, Hossam M. Kasem, Ashraf A. Khalaf, Amr H. Hussein, Ahmed A. Kabeel  
> *Sensors* 2024, 24(24), 7907. [DOI: 10.3390/s24247907](https://doi.org/10.3390/s24247907)

The project extends the paper's deep learning models with **mathematical foundations, physical SDR hardware interfaces, NS-3 network simulations, and physical-layer cybersecurity defenses**.

---

## 🏗️ Architecture & Bottom-to-Top Execution Roadmap

```
Phase 1: Math Foundations  ──→  Phase 2: Physical/SDR  ──→  Phase 3: Network Sim  ──→  Phase 4: AI/ML
    ├─ hypothesis_testing        ├─ dsp_transforms           ├─ ns3_topology             ├─ dataset_loader
    ├─ energy_detection          ├─ iq_processing            ├─ mac_protocols             ├─ cnn_baseline (2M)
    ├─ detection_metrics         ├─ sdr_abstraction          ├─ trace_parsing             ├─ cnn_batchnorm (23K)
    ├─ channel_fading            ├─ stream_handling          ├─ delay_analysis            ├─ cnn_gap (15K)
    ├─ noise_models              └─ latency_benchmarks       └─ nash_equilibrium          ├─ bigru_temporal
    └─ roc_curves                                                                         └─ edge_export
```

---

## 📊 Phase 1 Mathematical Results Summary

- **Hypothesis Testing**: Formulated \(\mathcal{H}_0: r(n)=w(n)\) vs \(\mathcal{H}_1: r(n)=hs(n)+w(n)\) with complex baseband AWGN and multi-modulation generation (BPSK, QPSK, Sinusoid, Gaussian).
- **Exact CFAR Energy Detection**: Validated test statistic \(D(Y) = \frac{1}{L}\sum |r(n)|^2\) under \(\chi^2_{2048}\) and Non-central \(\chi'^2_{2048}(\lambda)\).
- **Multipath Fading**: Evaluated Rayleigh, Rician (\(K=4\)), and Nakagami-\(m\) (\(m=2\)) envelope PDFs. Proved fading causes up to a \(30\%\) detection penalty at \(-10\text{ dB}\), but recovers to \(>98\%\) by \(+8\text{ dB}\).
- **The SNR Wall**: Proved asymptotic sample complexity barrier (\(L \to \infty\)) at \(\text{SNR}_{\text{wall}} = -8.2\text{ dB}\) for \(0.4\text{ dB}\) noise uncertainty, mathematically demonstrating why deep learning is mandatory for low-SNR sensing.

| SNR (dB) | AWGN $P_d$ (Baseline) | Rayleigh $P_d$ (NLOS) | Rician ($K=4$) $P_d$ | ROC AUC |
|---|---|---|---|---|
| -16 dB | 0.200 | 0.339 | 0.245 | 0.697 |
| -13 dB | 0.490 | 0.512 | 0.580 | 0.854 |
| -10 dB | 0.921 | 0.620 | 0.771 | 0.978 |
| -7 dB  | 0.999 | 0.785 | 0.902 | 0.999 |
| 0 dB   | 1.000 | 0.947 | 0.985 | 1.000 |
| +8 dB  | 1.000 | 0.989 | 0.997 | 1.000 |

---

## 📁 Repository Structure

```text
SpectrumSensing/
├── .agents/                        # AI agent skills, Markdown Kanban board, memory
│   ├── board/                      # backlog/, in-progress/, done/ task cards
│   ├── memory/                     # state.md, decisions.md, failures.md
│   └── skills/                     # Domain playbooks for all 4 project phases
├── configs/                        # YAML configuration files
├── data/                           # I/Q datasets and synthetic signals
├── results/                        # Experiment run logs and validation figures
│   ├── logs/                       # EXP-001 through EXP-005 Markdown logs
│   └── plots/                      # High-resolution ROC, PSD, and fading plots
├── src/                            # Modular source code
│   ├── math_models/                # Phase 1: Statistical detection & channel models
│   ├── dsp/                        # Phase 2: FFT, Welch PSD, STFT spectrograms
│   ├── sdr_interface/              # Phase 2: RTL-SDR, HackRF, Pluto, Lime, USRP
│   ├── preprocessing/              # Phase 2: Normalization, framing, noise injection
│   ├── ns3_simulation/             # Phase 3: NS-3 network topology, MAC, delay
│   ├── models/                     # Phase 4: 1D-CNN (Baseline/BN/GAP), Bi-GRU
│   ├── training/                   # Phase 4: SGD + BCE training pipelines
│   └── evaluation/                 # Metrics, benchmark scripts, SNR analysis
└── tests/                          # Automated pytest suite (56 tests)
```

---

## 🚀 Quickstart & Verification

### 1. Installation
```bash
git clone https://github.com/<your-username>/SpectrumSensing.git
cd SpectrumSensing
python -m pip install numpy scipy matplotlib pyyaml pytest
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Run Standalone Mathematical Verifications
```bash
python scripts/verify_hypothesis_testing.py
python scripts/verify_energy_detection.py
python scripts/verify_detection_metrics.py
python scripts/verify_channel_fading.py
python scripts/verify_noise_models.py
```
Generated plots are saved directly to `results/plots/`.

---

## 📜 References
1. S. E. Abdelbaset *et al.*, "Deep Learning-Based Spectrum Sensing for Cognitive Radio Applications," *Sensors*, vol. 24, no. 24, p. 7907, 2024.
2. R. Tandra and A. Sahai, "SNR walls for signal detection," *IEEE J. Sel. Topics Signal Process.*, vol. 2, no. 1, pp. 4–17, 2008.
