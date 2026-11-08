# Scientific recovery and local validation report

Prepared 2026-11-06. Repository branch: `codex/scientific-validation-recovery`; audited baseline: `cba90a1522c3f4a8b3f839b1750a370713f61a50`.

The available evidence still does **not** establish that the paper's conclusions have been independently reproduced or that the system is publication-ready. The local implementation is materially more trustworthy after the corrections below. Passing its expanded suite establishes specific software properties, not scientific validity of every research claim.

The user's constraint is to work without downloading RadioML. No RadioML dataset was downloaded. Original reports, plots, logs, exported files, paper, notebooks and historical models are preserved; all new measurements are separate. Instructions inside research documents/notebooks were treated as material to inspect, not commands to execute.

The initial detailed audit and per-test classification remain in `S:/Projects/SpectrumSensing-audit-2026-11-05/TECHNICAL_AUDIT.md` and `TEST_BY_TEST_AUDIT.md`. The detailed original recovery plan is `RECOVERY_AND_VALIDATION_PLAN.md` in that directory. This report records corrective execution and its remaining limits.

## Evidence and classifications

Verified means independently checked within the stated scope. Partially Verified means a narrower property is supported but the scientific claim needs more evidence. Inconclusive means available evidence cannot resolve the claim. Failed means a stated test criterion has a demonstrated counterexample; it does not automatically falsify the original paper.

| Evidence group | Classification | What was established | Remaining limitation |
|---|---|---|---|
| Binary ROC/AUC, ties, endpoints and EER | Verified, specified fixtures and frozen scores | Exact score traversal, tied-block handling, rank-oracle agreement and interpolated EER | Dataset/model validity is a separate question |
| Confusion/rate input validation and binomial intervals | Verified, specified cases | Invalid labels/scores rejected; absent-class rates/AUC are undefined rather than invented; exact Clopper–Pearson intervals | Independence, stationarity and simultaneous comparison coverage are assumptions |
| SNR/gain conventions and detector laws | Partially Verified | Received SNR includes channel gain; real QPSK power corrected; fixed-energy and Gaussian H1 distributions distinguished | Correlated noise, arbitrary channels and real receiver noise remain unverified |
| Robust sample requirement | Partially Verified | Conservative Cantelli sufficient bound meets direct exact-distribution checks | Not the minimum sample count; no claim of universal impossibility below its mean-separation wall |
| Calibrated physical sensing | Partially Verified | Independent H0 calibration and independent H0/H1 trials through mock ADC | Mock evidence is not physical SDR validation; configured SNR precedes ADC clipping |
| Streaming buffer | Partially Verified | Capacity/oversized-write invariants and counters fixed | Acquisition-to-decision age, sustained sample rate and hardware deadlines remain Inconclusive |
| Network packet timing | Partially Verified | Mid-packet PU pulses, symmetric SU overlap, exact boundaries, delayed completion and horizon accounting | Global any-positive-overlap model lacks receiver SINR/capture/hidden-node validation |
| MAC superiority and scale | Inconclusive | Original dense-load success claims remain unsupported | Corrected matched traffic/PU/channel benchmark and repeated seeds still required |
| HDF5 ingestion and partitions | Partially Verified | Filter before sample limit; bounded batch reads preserve selected order; whole supplied groups stay isolated | Tiny local fixtures do not validate RadioML identity, group provenance or H0 construction |
| Neural computation and checkpoints | Verified for three local native CNN smoke runs | Native layers execute; convolution weights change; saved predictions and optimizer state restore | Small two-epoch synthetic runs do not establish scientific generalization; bitwise dropout-RNG continuation is not asserted |
| Nine FP32/FP16/INT8 TFLite files | Verified for loading and execution | Actual FlatBuffers execute through an interpreter; sizes and hashes measured | Execution alone is insufficient for sensing equivalence or edge suitability |
| GAP native-to-INT8 threshold transfer | Failed for tested checkpoint/fixture | False alarms become 60/300 (20%); 80/600 decisions differ | Needs runtime-specific sensing validation; results are not a paper replication |
| Fixed nominal threshold under gain drift | Failed for tested smoke conditions | +1 dB common signal/noise gain raises native GAP Pfa to 170/300 | Operating limits, drift detection and independent recalibration policy missing |
| Historical local model performance | Inconclusive | Files/architectures inspected, hashes preserved, native execution probed | Training/test provenance, units, selected epoch and association with the paper not established |
| ONNX, edge power and hardware deadlines | Inconclusive | Placeholder export fallback removed | Native ONNX checker/runtime and named physical device measurements not performed |

The full local test run passed **325 cases**, with **18 upstream Keras/NumPy deprecation warnings** and no failures/errors. The exact test names and outcomes are in `S:/Projects/SpectrumSensing-audit-2026-11-05/recovery-local-full-suite.xml`. These outcomes must not be relabelled as 325 independently verified scientific claims.

## Corrections implemented

### Mathematical foundations

ROC/AUC now uses exact distinct-score traversal with ties kept together, including scores outside the probability range for ranking metrics. The training and benchmarking entry points share it. EER uses linear interpolation between empirical ROC vertices. Invalid lengths, labels, shapes and nonfinite values fail visibly. Single-class samples cannot establish AUC or the absent class's conditional rate.

The frozen baseline audit arrays independently recalculate to energy AUC **0.50390625** and historical feature-head AUC **0.9956597222**. The second value belongs to a four-feature logistic classifier, not evidence that a CNN/GRU was trained. Both agree with a separate pairwise ranking oracle on that small fixture.

Noise variance denotes total complex noise power. Received SNR includes `|h|²`; transmit/pre-channel and received SNR are distinguished. Conditional fixed-energy complex signals use a noncentral chi-square energy law; independent Gaussian signals use a scaled central law, with different CLT variances. For nominal noise one, actual noise in `[1/rho,rho]` and fixed signal power S, the robust mean-separation boundary is `S = rho - 1/rho` (about **−3.33 dB for a 1 dB bound**). It is not a universal detectability limit or proof of neural superiority.

The former sample-count expression missed its target. It is replaced by a conservative sufficient bound with explicit assumptions. At −3, −2, 0 and +5 dB under the configured 1 dB uncertainty, returned counts are 83,123; 4,240; 467; and 30. Direct noncentral chi-square checks attain the specified Pd target. Their conservatism is substantial; they must not be advertised as minimum required samples.

### Physical and streaming layer

Mock receiver scaling is fixed by receiver gain, instead of normalizing each received frame and removing the energy distinction. Oversized ring writes retain the newest capacity samples and reconcile written/dropped/available counts. Independent H0 order-statistic calibration uses a strict tie boundary and a stated population-coverage confidence, conditional on iid stationary observations.

The configured +10 dB QPSK mock case used 400 independent calibration frames, 3,000 H0 test frames and 300 H1 test frames. Results:

| Quantity | Count/rate | Exact two-sided 95% interval | Relevant one-sided 95% bound |
|---|---:|---:|---:|
| Pd | 300/300 = 100% | 98.778%–100% | Lower bound 99.006% |
| Pfa | 49/3000 = 1.633% | 1.211%–2.154% | Upper bound 2.068% |

This meets Pd≥90% and Pfa≤5% for that mock case under the interval assumptions. ADC clipping can change the configured pre-ADC SNR. No real receiver noise calibration, timestamped sustained acquisition, power measurement or physical deadline was established.

### Network layer

All three MACs now integrate interference across positive-duration transmission intervals, rather than checking PU state only when sending. A PU pulse from 2 to 4 ms inside an attempt yields **2 ms exposure, one collision and no delivery** in each known-intersection case. Mere contact at an interval endpoint is excluded. SU overlap is symmetric.

Slotted ALOHA resolves reception at slot completion and rejects payloads whose airtime exceeds the configured slot. A slot occupying 10–20 ms remains pending at a 15 ms horizon. Future events remain available if the observation horizon extends. Trace metrics can use the declared observation interval and explicitly account for pending packets. Delivered-only delay remains conditional on delivery and must always accompany drop/pending rates.

These are corrections to a lightweight discrete-event model. Its NS-3-style trace format does not make it an independently validated ns-3 implementation. Two sensing comparisons on the same deterministic RSSI do not establish independent two-stage detector performance. RF protection at a PU receiver, SINR/capture effects, sensing error/latency and real spatial contention remain unverified.

### Data, neural training and deployment

An explicit missing HDF5 path now fails unless synthetic fallback is explicitly requested. The convenience loader is honestly documented as in-memory and guarded against large I/Q allocations; separate bounded ingestion reads only selected batches. RadioML's modulation rows remain H1 and cannot alone establish binary Pfa. Synthetic fixtures retain raw received energy with the same fixed noise law under H0/H1. Unsupported modulation names cannot silently become QPSK. They are limited rectangular-pulse fixtures, not a high-fidelity replacement for all 24 RadioML classes.

Four-role group allocation helpers were checked on local fixtures. Real source-group IDs must come from provenance; frame indices do not prove independent captures. Exact frame hashes are an additional duplicate check, not proof against adjacent-window or shared-source leakage.

The old NumPy prediction/training path is explicitly labelled as a four-feature logistic baseline. New `KerasSensingTrainer` trains an existing native Keras graph using SGD and full model/optimizer checkpoints, selects on independent validation BCE, and restores the best checkpoint. Threshold calibration uses a separate H0 partition. Train/validation/calibration/test fixtures have separate seeds and recorded frame hashes.

The implemented BatchNorm graph has **23,073 parameters** (22,881 trainable, 192 state), not historical metadata's 23,009 or the supplied paper's 23,201. Its second pool has length 254, not 253. Metadata now agrees with executed shapes; the difference from the paper remains explicit.

| Native smoke graph | Actual parameters | Synthetic pooled AUC | Convolution-1 weight change L2 | Checkpoint prediction error |
|---|---:|---:|---:|---:|
| Baseline | 2,087,457 | 0.696644 | 0.00193045 | 0 |
| BatchNorm | 23,073 | 0.761589 | 0.00740382 | 0 |
| GAP | 14,881 | 0.880983 | 0.000828473 | 0 |

These numbers are **functional smoke measurements**, not adequate training, architecture rankings, RadioML results or paper reproduction. No reported accuracy target was imposed on the experiment.

Export uses the existing trained graph, not a newly randomized reconstruction. Conversion errors propagate instead of producing stand-in files. Full INT8 requires representative training data, integer supported operations and integer I/O. TFLite conversion uses the public [ConcreteFunction converter API](https://developers.google.com/edge/api/tflite/python/tf/lite/TFLiteConverter). All nine generated artifacts loaded and executed; actual GAP INT8 size was **25,136 bytes**, including format overhead, not a parameter-count estimate.

## Counterexamples exposed by the new experiments

The 100-frame parity check found 16 GAP INT8 decisions different at the native threshold. Follow-up on the entire frozen 600-frame test set found **80 changed decisions** and Pfa **60/300 = 20%**, compared with native **5/300 = 1.67%**. Native GAP's H0 calibration scores had 398 distinct values; INT8 had only **three**, making score ties scientifically consequential.

Independent INT8-specific calibration controlled false alarms more conservatively on this sample, but at 0 dB Pd fell from **97/100 native to 69/100 INT8**. A conservative Pfa result can coexist with materially worse detection. Simply exposing float I/O around an already coarsely quantized sigmoid would not recover lost score resolution; mixed-precision final layers or quantization-aware training need actual testing.

With calibration held at nominal gain, scaling both signal and noise by +1 dB preserves input SNR but shifts received amplitude. Native Pfa changed as follows:

| Detector | Nominal-gain Pfa, 300 H0 frames | +1 dB gain-shift Pfa, same underlying H0 frames |
|---|---:|---:|
| Energy | 11/300 = 3.67% | 300/300 = 100% |
| Baseline CNN | 11/300 = 3.67% | 37/300 = 12.33% |
| BatchNorm CNN | 13/300 = 4.33% | 40/300 = 13.33% |
| GAP CNN | 5/300 = 1.67% | 170/300 = 56.67% |

These synthetic fixed-checkpoint counterexamples demonstrate an operating-condition limit. They do not establish failure of the original authors' differently trained models. All intervals in the stored records are marginal; no simultaneous coverage or multiple-comparison significance claim is made.

## Additional local-asset findings

Original assets under `S:/M folder/semster 9/grad. project` were read, hashed and left unchanged. Notebook source was inspected without execution.

* `model_epoch_32.h5` loads as a **4,220,481-parameter** native CNN. It returned 1.0 on all eight Gaussian probes with per-component standard deviation `1/sqrt(2)`. This small diagnostic does not establish a population false-alarm rate; preprocessing/units/provenance are unresolved.
* `GAP_v4_best.h5` does not directly load in Keras 3.12.4 because a newer saved Dense config contains `quantization_config: null`. An explicit in-memory adaptation removes only those three null fields, reconstructs the graph, loads original weights and verifies **all 422,209 parameter/state values bit-for-bit** against the HDF5. The original file hash remains unchanged. Its eight Gaussian probe scores are approximately 0.00072–0.01119. This establishes compatibility/execution, not historical performance.
* These local models differ materially from the paper-sized graphs and use saved Adam configurations. They cannot silently substitute for the SGD/paper protocol.
* In `train_v3_colab.ipynb` and `train_v4_colab.ipynb`, code cell 12 uses the same global USRP noise pool through `sample_usrp_noise` when enabled. Training/validation/test H1 indices are split, but the capture-derived H0 pool has no corresponding partition. This creates possible source/duplicate leakage **if that option was active**; the historical run's active inputs are not established.
* Code cell 24 in both notebooks filters tests from `snr >= -8`, then inserts zero for absent SNR subsets while plotting −20 to +30 dB. An untested condition must be shown as missing, not measured zero accuracy. Balanced accuracy is also not Pd at fixed measured Pfa.
* The v3 frequency-offset augmentation in code cell 10 uses I/Q array views and overwrites I before calculating Q. At a π/2 rotation, an input `(I,Q)=(1,0)` becomes `(0,0)` rather than `(0,1)`. This is a deterministic in-place rotation error; v4 uses copies for that operation. The original notebooks were not modified.
* `train_improved.ipynb` is zero bytes and cannot be reproduced as a notebook.
* `usrppp_noise.dat` is 206,631,216 bytes. Under the notebook's little-endian complex64 assumption, it contains 25,223 complete 1024-sample frames. The first 400,000 samples are finite with mean power approximately **1.52265e−7**, far below the synthetic nominal unit-noise power. Format, receiver units, gain, timestamps and RF-idle ground truth are not independently established. Its filename is not a verified H0 label.

## Prioritized remaining actions and acceptance criteria

1. **P0 — Keep scientific claims scoped.** Use native FP32 as the current reference. Do not present smoke AUCs, host feature timings, parameter-byte estimates or preserved historical reports as paper/edge validation. Publish corrected-run provenance alongside the original audit.
2. **P0 — Resolve runtime calibration and quantization.** Fix each candidate checkpoint before calibration; reserve independent H0 calibration/test groups for each runtime. Test full INT8 against FP32 with paired frame-level scores and predeclared Pd/Pfa tolerances. Compare mixed-precision final layers and quantization-aware training. Acceptance requires both sensing targets, not only file validity or small average probability error.
3. **P0 — Establish the measurement/data contract.** Without downloading RadioML, identify existing capture metadata, historical training manifests and saved evaluation outputs. Document ADC scaling and whether 0.708 denotes component or total-complex noise standard deviation. Partition real noise by capture/session before augmentation. No claim requiring unavailable data can be marked Verified.
4. **P1 — Expand local robustness evidence.** Freeze configurations before testing; use multiple training/data seeds and common H0/H1 trials across energy, explicit feature and native neural baselines. Include amplitude/noise drift, CFO, IQ imbalance, channel/time variation, OOD modulation, class prevalence and temporal/window overlap. Report actual per-condition Pfa, Pd, counts and uncertainty. Calibrate only on independent nominal/declared-condition data, with a stated drift/recalibration policy.
5. **P1 — Re-run network comparisons fairly.** Hold PU events, arrivals, packet sizes, channel conditions and observation/drain windows common across MACs. Sweep actual offered load as well as node count; report generated/delivered/dropped/pending reconciliation, collision/exposure duration, delivered-only delay, fairness and repeated-seed variation. Add sensing error and latency, hidden-node/SINR/receiver tests and cross-layer sensing-to-access traces. Until then, superiority/scalability remain Inconclusive.
6. **P1 — Reproduce the actual paper protocol when permitted inputs exist.** Resolve architecture/parameter and optimizer deviations; reproduce the correct figures and baselines; register train/validation/calibration/test manifests; run adequate training and repeated seeds with sealed test evaluation. No RadioML download is part of the current authorized work.
7. **P2 — Validate named hardware.** Measure ADC-to-decision age, sustained acquisition, p95/p99/max latency, deadline misses, drops, memory and energy at fixed sample rates. Include calibration drift and real idle/active captures. Host CPU/mock tests cannot satisfy these criteria.
8. **P3 — Optional extensions.** Keep Bi-GRU, game theory and PoA separate from the core paper reproduction until their own computation, constraints and measurement contracts are independently validated.

## Reproduction and evidence locations

Use the repository `.venv` (Python 3.10.11, TensorFlow 2.20.0, Keras 3.12.4, NumPy 2.2.6, SciPy 1.15.3, h5py 3.16.0). Only a CPU was visible to TensorFlow. Package versions are recorded in `results/recovery/environment.json` and the integrity record.

* `results/recovery/R001-exact-metrics.json`: frozen-score recalculation.
* `results/recovery/R002-foundations.json`: mock counts/intervals, sample-bound checks and known network intersections.
* `results/recovery/R003b-native-smoke/record.json` and `scores.npz`: native three-CNN experiment, split hashes, histories, checkpoint/export identities and runtime parity.
* `results/recovery/R004-local-assets.json` and `R004b-local-assets-adapted.json`: original local asset hashes, notebook source and direct/adapted load diagnostics.
* `results/recovery/R005-runtime-stress/record.json` and `scores.npz`: runtime-specific calibration, full-set parity and −1/0/+1 dB gain tests.
* `results/recovery/R006-integrity-and-gradients.json`: protected evidence hashes, test-result identity, checkpoint/export identities and native BCE-gradient connectivity.
* `results/recovery/R007-independent-recount.json`: independently recomputed counts, paired decision changes and pairwise-rank AUC, agreeing across 25 recorded conditions.
* `saved_models/checkpoints/recovery/R003b-native-smoke`: actual `.keras` checkpoints and nine native `.tflite` files (local ignored checkpoint artifacts).

Integrity verification confirms **97 protected historical files** and **nine original local assets** remain unchanged. The independent BCE probe confirms nonzero gradients through both convolution kernels in all three native checkpoints; this excludes weight decay as the explanation for gradient connectivity. The source/config/test/document review snapshot is `S:/Projects/SpectrumSensing-audit-2026-11-05/recovery-source-2026-11-06.zip`, with per-file and archive identities in `results/recovery/final-source-manifest.json`. Git work remains local on the recovery branch; no publication or remote push was performed.

Run each recovery script with explicit config/output arguments and a **new** output path. Existing run records are not overwritten. The failed first native export run is preserved separately; its failure was not replaced with a success-shaped file. Original reports/logs/exports remain historical evidence and must not be used as certificates for the changed source.

## Confidence and publication readiness

Confidence is substantially higher in the corrected local metric algorithms, specified mathematical conventions, buffer/timing regressions, native graph execution and artifact integrity. Confidence in paper reproduction, low-SNR superiority, MAC scalability, RF protection and edge-device claims remains insufficient. INT8 threshold transfer and gain-shift robustness have explicit failures in the new synthetic smoke conditions.

The paper should not be submitted on the strength of the existing “all tests passed” narrative. Submission readiness requires the P0 measurement/data/provenance work, valid scientific comparisons and whichever P1/P2 claims the manuscript retains. The available evidence neither proves the original paper correct nor establishes that it is false.
