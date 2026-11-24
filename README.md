# Cognitive Radio Spectrum Sensing

Research implementation and evidence for mathematical detection, physical/mock SDR processing, network simulation and neural spectrum sensing.

**Status on 24 November 2026.** The corrected suite has a recorded 331 passing tests. On one USRP-2920 ambient capture, 19,820 of 197,236 held-out frames landed exactly on the INT8 cutoff. Including that tied group increased the ambient trigger rate from 0.43% to 10.48%. Measuring detection and false alarms requires independent labeled captures. [Project history and future plan](docs/PROJECT_STATUS_AND_PLAN_2026-11-24.md) · [Hardware evidence and replay instructions](results/calibration/usrp-ambient-20261121/README.md)

The work studies the methodology in [Deep Learning-Based Spectrum Sensing for Cognitive Radio Applications](https://doi.org/10.3390/s24247907), Sensors 2024, 24, 7907. This repository currently supports an audited implementation and limited local experiments. It does not claim an independently reproduced paper result.

## Start here

- [Pre-USRP decision brief and measured calibration results](docs/calibration/SUPERVISOR_BRIEF.md)
- [NI USRP-2920 lab handoff and offline replay](docs/calibration/usrp-2920/README.md)
- [Prior-work and novelty comparison](literature/novelty-comparison.md)
- [Interactive calibration workflow and synthetic replay](docs/presentations/receiver-calibration-workflow/Receiver_Calibration_Workflow.html)
- [Current recovery report and prioritized next steps](docs/recovery/RECOVERY_PROGRESS.md)
- [Claim-by-claim evidence status](docs/recovery/CLAIM_LEDGER.md)
- [Measurement and statistical conventions](docs/recovery/MEASUREMENT_CONTRACT.md)
- [Eight-slide animated presentation](docs/presentations/research-validation-progress/README.md), with offline HTML, Framer Motion source and SVG diagrams
- [Initial technical audit](docs/recovery/baseline-audit/TECHNICAL_AUDIT.md), [per-test audit](docs/recovery/baseline-audit/TEST_BY_TEST_AUDIT.md) and [recovery plan](docs/recovery/baseline-audit/RECOVERY_AND_VALIDATION_PLAN.md)

GitHub displays the presentation HTML as source. Download [the self-contained HTML](docs/presentations/research-validation-progress/output/Research_Validation_Progress.html) and open it in a browser. It works offline. Left/Right changes slides, F opens fullscreen and N opens presenter notes.

## What the evidence currently supports

| Area | Established locally | Remaining gap |
|---|---|---|
| Mathematical foundations | Exact ROC/AUC and ties, explicit SNR/gain conventions, specified detector laws and conservative sample bounds | Broader noise/channel assumptions and statistical operating conditions |
| Physical/mock processing | Independent H0 calibration, specified mock detection trials and buffer invariants | Labelled real captures, acquisition timing and receiver calibration |
| Network model | Known packet-overlap, completion and observation-horizon cases | Receiver SINR/capture, sensing error/latency, fair scalability comparisons |
| Native CNNs | Actual Keras gradients, changed convolutional weights and restorable checkpoints | Adequate training, independent full-data evaluation and model-seed variation |
| TFLite exports | Nine executable FP32/FP16/INT8 files with measured sizes | Sensing equivalence, runtime calibration and named edge-device measurements |

Passing software tests does not validate all scientific claims. The latest [pre-push test record](results/recovery/pre-push-2026-11-08.json) records 325 passing cases and 18 upstream Keras/NumPy deprecation warnings.

## Counterexamples found during recovery

The new synthetic GAP experiments identify concrete operating limits:

- Transferring the native FP32 threshold to INT8 produces **60/300 false alarms (20%)**, compared with **5/300 (1.67%)** for native FP32.
- Independent INT8 calibration lowers detection at 0 dB from **97/100 native to 69/100 INT8** on the same active frames.
- A **+1 dB common signal/noise gain shift**, with nominal calibration held fixed, raises native GAP false alarms to **170/300 (56.67%)** while preserving input SNR.

These results concern a fixed synthetic smoke checkpoint. They do not establish failure of the original paper's differently trained models. See the [runtime-stress record](results/recovery/R005-runtime-stress/record.json) and [independent score recount](results/recovery/R007-independent-recount.json).

## Install and test

The verified environment uses Python **3.10.11**, TensorFlow **2.20.0** and Keras **3.12.4**. From the repository root:

```sh
python -m venv .venv
```

Activate it on Windows with `.\.venv\Scripts\Activate.ps1`, or on Linux/macOS with `source .venv/bin/activate`, then run:

```sh
python -m pip install -r requirements-recovery.txt
python -m pytest -q
```

TensorFlow must be installed to execute the native neural checks. A run that skips those checks does not reproduce the complete 325-case validation. Only a CPU was visible in the recorded local environment.

## Run the corrected local experiments

The following commands use synthetic inputs and do not download RadioML. Each run requires a new output path and preserves existing records.

```sh
python scripts/validate_recovery_foundations.py --config experiments/configs/recovery_foundations.yaml --output results/local-runs/foundations.json

python scripts/validate_recovery_native.py --config experiments/configs/recovery_native_smoke.yaml --output-directory results/local-runs/native --model-directory saved_models/checkpoints/local-native
```

The first command checks specified theory, calibrated mock sensing and known network intersections. The second performs small native CNN training/checkpoint/export smoke runs. Its AUCs are not RadioML performance estimates or reliable architecture rankings.

Stored [native experiment records](results/recovery/R003b-native-smoke/record.json), [runtime-stress records](results/recovery/R005-runtime-stress/record.json) and score arrays allow independent numerical inspection. Use `scripts/replay_recovery_scores.py --help` for the recount command. Local experiment outputs and large checkpoint binaries stay out of Git by default.

The pre-USRP threshold sweep, targeted literature comparison and fresh synthetic calibration-budget benchmark are available in the decision brief above. At the primary budget, the linear rule equaled ordinary recalibration in detection and exceeded the false-alarm limit. The next hardware question is whether discrete noise-score ties and threshold transfer behave similarly across real USRP-2920 sessions.

## Repository layout

| Path | Purpose |
|---|---|
| `src/math_models/` | Hypothesis testing, detection metrics, fading and threshold calibration |
| `src/sdr_interface/`, `src/dsp/`, `src/preprocessing/` | SDR interfaces, mock streams and signal processing |
| `src/ns3_simulation/` | Lightweight Python discrete-event network model and trace analysis |
| `src/models/`, `src/training/` | Architecture descriptions, explicit feature baseline, native Keras training and export |
| `experiments/configs/`, `experiments/logs/` | Experiment settings and execution records |
| `tests/` | Software checks and adversarial regression fixtures |
| `results/recovery/`, `results/calibration/` | Archived validation records and new calibration measurements |
| `data/`, `notebooks/` | Local datasets, split manifests, and exploratory analysis |
| `docs/calibration/`, `literature/` | Study protocol, source papers, and novelty review |
| `docs/recovery/` | Current assessment and historical audit |
| `docs/presentations/` | Progress deck and interactive calibration workflow |
| `paper/raw_materials/`, `paper/generated/` | Curated manuscript inputs and local generated drafts |
| `external/` | [Upstream tool guidance](external/README.md) |

The network model produces NS-3-style traces. This does not establish independent validation against the ns-3 simulator.

## Legacy work and historical evidence

The [previous repository README](docs/recovery/baseline-audit/LEGACY_README.md) preserves the old completion narrative. Historical reports under `docs/COMPREHENSIVE_VERIFICATION_REPORT.*`, `results/logs/`, `results/plots/` and `saved_models/exported/` describe the pre-audit implementation. They are not certificates for the corrected source.

The historical NumPy `predict_proba` path and `SpectrumSensingTrainer` execute a shared four-feature logistic baseline. Declared CNN/GRU layer metadata does not make that path neural. Use a native `to_keras()` graph and `KerasSensingTrainer` for actual convolutional training.

The `validate_recovery_*.py` scripts are the current experiment entry points. Legacy `scripts/verify_*.py` scripts preserve original assumptions, export interfaces, and configuration paths. Reproduce those scripts from the checkout that supplied the corresponding historical evidence. Optional Bi-GRU, game-theory and ONNX deployment claims need separate validation.

Historical provenance records contain paths from the machine where experiments ran. Those paths identify original inputs and locally retained model binaries; they are not required to open the presentation or recount the committed score arrays.
