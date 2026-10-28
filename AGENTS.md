# AGENTS.md — AI-Driven Cognitive Radio Spectrum Sensing

## Project Overview
This project implements an AI-driven Cognitive Radio Network for dynamic spectrum sensing.
It follows the methodology from the paper "Deep Learning-Based Spectrum Sensing for Cognitive
Radio Applications" (Abdelbaset et al., Sensors 2024) and extends it with SDR hardware,
NS-3 network simulation, and cybersecurity capabilities per the deployment plan.

## Execution Order (Bottom → Top)
1. **Phase 1: Mathematical Foundations** — Binary hypothesis testing, detection metrics, noise models, channel fading PDFs
2. **Phase 2: Physical / SDR Layer** — I/Q signal processing, SDR hardware interfaces, DSP transforms
3. **Phase 3: Network Simulation** — NS-3 topology, MAC protocols (ILA-CSMA), end-to-end delay analysis
4. **Phase 4: AI/ML Sensing** — 1D-CNN architectures (Baseline/BN/GAP), Bi-GRU, training on RadioML 2018.01A

## Architecture & Layout
- `/src/math_models/` — Hypothesis testing, Pd/Pfa derivations, threshold optimization, ROC curves
- `/src/dsp/` — Digital signal processing transforms (FFT, PSD, Welch, STFT, wavelets)
- `/src/sdr_interface/` — SDR hardware abstraction (RTL-SDR, HackRF, PlutoSDR, LimeSDR, USRP)
- `/src/preprocessing/` — I/Q normalization, spectrogram generation, noise injection
- `/src/models/` — CNN architectures (baseline, batch-norm, GAP), Bi-GRU, attention models
- `/src/training/` — Training pipeline, dataset loaders, loss functions, hyperparameter configs
- `/src/evaluation/` — Metrics (Pd, Pfa, ROC, AUC), benchmark comparison scripts
- `/src/ns3_simulation/` — NS-3 scenario configs, trace parsers, MAC protocol implementations
- `/data/` — Raw I/Q captures, processed arrays, synthetic datasets (RadioML 2018.01A)
- `/notebooks/` — Jupyter notebooks for exploration and visualization
- `/configs/` — YAML configs for models, SDR hardware, simulation parameters
- `/saved_models/` — Checkpoints and exported models (.h5, .tflite, .onnx)
- `/tests/` — Unit and integration tests
- `/results/` — Experiment logs, plots, benchmark tables
- `/docs/` — Documentation, figures, references

## State & Memory
- **ALWAYS** read `.agents/memory/state.md` at the start of every session.
- **ALWAYS** update `.agents/memory/state.md` after completing a milestone.
- Check `.agents/board/` for current task status before starting new work.
- Log failures in `.agents/memory/failures.md` to avoid repeating dead-end approaches.

## Code Style & Guardrails
- **Python 3.10+** with type hints throughout
- **NumPy/SciPy** for mathematical models and DSP
- **TensorFlow/Keras** as primary DL framework (matching the paper)
- **Matplotlib/Plotly** for visualization
- All functions must include docstrings with parameter descriptions
- Every module must have a corresponding test in `/tests/`
- Use YAML configs in `/configs/` — never hardcode hyperparameters

## Key Constants
- Dataset: RadioML 2018.01A (2,555,904 frames, 1024×2 I/Q, 24 modulation types, SNR -20 to +30 dB)
- Loss function: Binary Cross-Entropy (BCE)
- Optimizer: SGD (lr=0.001, momentum=0, weight_decay=0.0001)
- Batch sizes: 128 (baseline) → 2048 (scaled)
- SNR filtering: -8 dB to +30 dB for best results
