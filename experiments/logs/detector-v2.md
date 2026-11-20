# Detector v2 (E2) experiment log

Program: pre-USRP v2 (Decision 004), card R009. Evidence setting: synthetic frames, CPU, native Keras and TFLite INT8. Repo skills applied: `ai-ml-sensing`, `math-foundations`, `physical-sdr`.

## 2026-11-20: before any candidate trains

### Impairment module
`src/preprocessing/receiver_impairments.py`: session-level Butterworth-coloured noise (circular, zero phase), IQ imbalance, DC offset and a 12-bit ADC. Noise power is renormalised to one after the filter and the imbalance, so noise-only frames carry mean power `1 + |dc|^2`. No random gain is applied. 17 tests, written first and confirmed red.

Properties to state in any write-up:
- SNR is nominal at the antenna port. The receive filter acts on signal and noise together, so in-band SNR of impaired frames is higher than the label; impaired and clean Pd are not comparable at the same label. The pilot below shows the shift: energy Pd at -12 dB is 0.85 impaired and 0.50 clean.
- Renormalising fixes mean noise power only. Coloured noise has fewer effective degrees of freedom, so per-frame energy varies more and the energy detector is weaker under impairments.
- At 12 bits with full scale 8 per component the ADC step is 0.0039, far below the noise, so its effect is negligible; it is kept for faithfulness.
- Training frames are shuffled before being cut into sessions of 50, so every session mixes both hypotheses and all SNRs (tested). Without the shuffle a session would carry one label.

### Review (ecc:python-reviewer)
No maths defects. Fixed before freezing: the frozen status was not enforced or recorded (now `status: frozen` is required, and every candidate record stores config hash, script hash, TensorFlow version and thread count, and `select` rejects any mismatch); the learning-rate search aborted on divergence (a diverged run now counts as infinite loss, results are written incrementally, ties go to the smaller rate); the selection metric would have saturated (below); the winner's files are verified against recorded hashes before copying; all outputs are created exclusively; a seed-layout guard runs at start-up; `--only` rejects unknown names. Noted in the report: calibration and test share one receiver draw; the winner's Pd is optimistically biased by selection; AWGN and impaired candidates use different data draws.

### Learning-rate search (development seeds 29030 and 39030, validation BCE only)
Grid 0.003, 0.01, 0.03, 0.1; SGD momentum 0.9, weight decay 1e-4, batch 128, 500 training and 250 validation frames per SNR, 12 epochs. Mean best validation BCE over the two classes: 0.003 -> 0.0882, 0.01 -> 0.0232, **0.03 -> 0.0122**, 0.1 -> 0.0140. Chosen: **0.03**. At 0.1 the impaired run ended at 0.104 against a best of 0.018 (unstable). Per run: AWGN 0.0875, 0.0170, 0.0105, 0.0101; impaired 0.0888, 0.0294, 0.0140, 0.0179. Each run took 96 to 261 s. Raw output: `results/local-runs/e2-lr-search/` (git-ignored). The search ran before the divergence handling was added; no run diverged, so that code path was not exercised.

### Plan deviation: selection SNRs
The plan chose the winner by INT8 Pd at 0 dB. An energy-only pilot (dev seeds 1027000+, four sessions, 300 H1 frames per SNR, same impairment pipeline, no neural model) showed saturation:

| H1 SNR (dB) | energy Pd impaired | energy Pd clean |
|---|---|---|
| -16 | 0.255 | 0.117 |
| -14 | 0.502 | 0.237 |
| -12 | 0.848 | 0.499 |
| -10 | 0.994 | 0.879 |
| -8 | 1.000 | 0.996 |
| 0 | 1.000 | 1.000 |

At 0 dB many candidates would score about 1.0, so ties would decide the winner and the final tie-break (lowest seed) always favours the AWGN class. The selection metric is therefore pooled INT8 Pd over H1 SNRs -16, -14, -12 and -10 dB. This was fixed from the energy detector alone, before any neural candidate was scored. The hardware pilot still uses -10, 0 and +10 dB.

### Frozen settings
`experiments/configs/detector_v2.yaml` with `status: frozen`: learning rate 0.03, 40 epochs maximum, patience 6, batch 128, 1,000 training and 250 validation frames per SNR (11 SNRs, -10 to +10 dB), 256 representative frames from training data, selection set of four sessions with 500 H0 calibration, 500 H0 test and 300 H1 test frames per SNR. Candidates: AWGN seeds 29031-29035 and impaired seeds 39031-39035. Throughput measured at about 1,100 frames per second on the i5-9300H.
