# Fresh synthetic benchmark | Revise the calibration rule

At the primary 0 dB SNR and 600 total calibration frames, the fitted gain line and independent INT8 recalibration each detected **2,121/3,600 signals (58.92%)**. The line exceeded the 5% false-alarm limit at −1 dB gain. This result does not support a hardware trial claiming improved detection from this rule.

| Gain | Candidate detections | Ordinary detections | Candidate false alarms | Ordinary false alarms |
|---:|---:|---:|---:|---:|
| −1 dB | 702/1,200 | 702/1,200 | 902/4,500 (20.04%) | 0/4,500 |
| 0 dB | 708/1,200 | 708/1,200 | 41/4,500 (0.91%) | 4/4,500 (0.09%) |
| +1 dB | 711/1,200 | 711/1,200 | 189/4,500 (4.20%) | 16/4,500 (0.36%) |

**Uncertainty.** All five model-block differences were zero. The prespecified paired t interval therefore collapses to [0, 0] on this sample. Five fitted models and a discrete score lattice give limited information about changes in a new population. The candidate's adjusted upper false-alarm bounds were 74.80%, 4.10%, and 18.88% across the three gains. The wide bounds reflect substantial between-model variation. Ordinary recalibration cleared the prespecified upper-bound criterion at all three primary gains. [Primary statistics](primary.json)

**Failure mechanism.** For model 19035, session 0 at −1 dB, the ordinary rule excluded the 0.5390625 score block. The fitted line lowered the threshold to 0.53841146, admitting 208/300 noise frames while both methods detected 80/80 signals. Model 19033, session 1, admitted 270/300 noise frames while both methods detected 0/80 signals. These cases explain why a smooth threshold in score space can change many decisions at once.

**Other conditions.** At 0 dB SNR, gain-specific FP32 detected 60.0% across the primary gains, fixed nominal INT8 detected 53.0%, and gain-aware energy detected 100% on these stationary synthetic conditions. At unseen +0.5 dB gain, candidate and ordinary false alarms reached 494/4,500 and 457/4,500. Under +1 dB noise power at fixed signal power and nominal gain, candidate and ordinary false alarms rose to 1,459/4,500 and 939/4,500. Energy then raised 4,500/4,500 false alarms. In the noise stress rows SNR identifies the **nominal signal-power condition**; the actual SNR is 1 dB lower. The full table includes all three budgets and SNRs. [Condition summary](summary.csv) · [Figure](primary.png)

**Evidence boundary.** Each of five GAP models was trained independently on QPSK/AWGN for up to eight epochs, then evaluated on three independently generated session pairs. Ordinal seed/session identifiers in the score archives tie back to the configuration and run manifest. The primary comparison used a selected-validation checkpoint, actual INT8 inference, a shared calibration budget, and matched independent test frames. These are simulated sessions without RF acquisition impairments. Weak native performance and substantial seed sensitivity limit extrapolation to stronger detectors.

**Reproduction.** From the repository root, use the research Python environment and new output/model directories. The summary command verifies 3,375 condition rows against the saved score arrays. It can run on the archived benchmark without retraining.

```sh
python scripts/calibration_benchmark.py --config experiments/configs/calibration_benchmark_v1.yaml --out results/local-runs/new-benchmark --models saved_models/checkpoints/new-benchmark
python scripts/calibration_summary.py results/local-runs/new-benchmark
python scripts/calibration_sweep.py --out results/local-runs/new-sweep
```

The initial conversion attempt stopped on an uninitialized TensorFlow Lite resource variable before fresh test scoring. The successful conversion froze trained inference variables, checked outputs on eight training-side representative frames to 1e−6 absolute tolerance, and enforced integer operators and I/O. The evaluation settings and statistical criterion were frozen before the fresh runs. [Config and model identities](manifest.json) · [Per-session thresholds and counts](counts.csv) · [Protocol](../../../docs/calibration/FROZEN_PROTOCOL.md)
