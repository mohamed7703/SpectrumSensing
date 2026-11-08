# Recovery and scientific validation plan

Prepared 5 November 2026 from the independent audit of local commit `cba90a1522c3f4a8b3f839b1750a370713f61a50`.

**Recommendation:** preserve the useful mathematical/DSP components, repair the measurement and implementation defects, and establish one trustworthy sensing experiment before expanding the system. More epochs, more synthetic frames, and more passing tests will not resolve the current model, metric, and integration problems by themselves.

This document is a proposed execution plan. No source changes, dependency installations, data downloads, hardware purchases, or changes to historical results were performed to create it. Existing audit artifacts and original repository files remain preserved.

**Recommended research scope**

Use two explicitly separated tracks:

| Track | Question | Recommended scope |
|---|---|---|
| A: sensing reproduction | Can an actual implementation of the supplied paper's CNN reproduce its reported behavior under documented assumptions? | Correct metrics; credible H0/H1 data; Baseline CNN, BatchNorm CNN and GAP CNN; energy, feature-based and paper-specific conventional baselines. Resolve architecture ambiguities openly. |
| B: system extension | Does the validated detector improve network access and execute within the chosen hardware deadline? | Measured detector outputs feeding a corrected MAC simulation, followed by executable model export and target-device measurements. |

Start Track A with the Baseline CNN and a classical energy detector. Retain the existing four-feature logistic classifier as an honestly named additional baseline. Add BatchNorm and GAP only after the first neural implementation is demonstrably real and correctly evaluated. Treat Bi-GRU, Nash/PoA, broad multi-device support and energy-saving claims as optional extensions with their own evidence requirements. They should not delay a sound sensing result.

The repository's four-layer organization can remain. Correct mathematical foundations first, then physical interfaces and simulator internals, then actual neural learning, and finally close the loop between the layers. Repairing a network primitive early does not require making end-to-end network claims before the detector is validated.

**Execution rules**

1. Write down each scientific claim and its acceptance conditions before running its final evaluation.
2. Turn demonstrated defects into regression cases that fail on the current implementation before applying the relevant fix.
3. Keep historical results immutable. New results receive new run identifiers and supersede old claims through an explicit correction record.
4. Never silently substitute synthetic data, mock hardware, metadata exports, or an untrained model in a publication experiment.
5. Separate a successful software execution from an accepted scientific result. Record both.
6. Keep final test data untouched during architecture selection, threshold tuning and debugging. If test results influence a redesign, treat that set as development data and establish a genuinely new final evaluation.
7. Report failures and operating limits. Do not adjust targets until the desired result appears.

**Step 0 — Freeze the evidence and define the deliverable**

Actions:

- Preserve the audited commit, original report, logs, figures, model files and SHA-256 manifest. Keep the audit directory outside the implementation checkout.
- Begin implementation work on an identified branch in a suitable checkout, with the baseline commit recorded. Do not overwrite the audit's reproduction snapshot.
- Create a claim ledger. Suggested columns: claim ID, precise wording, supporting paper figure/table, implementation path, data/config/run IDs, acceptance criterion, current status, and known limitations.
- Replace blanket project-status language in the working documentation with specific evidence statuses when implementation work begins. Preserve the original document as history.
- Decide whether the immediate deliverable is a faithful reproduction, an improved sensing study, or an end-to-end extension. Recommended immediate deliverable: a corrected sensing reproduction with transparent deviations.
- Inventory available original-author code, notebooks, trained weights, RadioML file/version, GPU resources, physical SDRs and target edge board. Missing items become explicit limitations, not guessed configurations.

Deliverables: frozen baseline manifest; claim ledger; experiment scope; resource inventory; correction/change log.

Exit criterion: every intended conclusion has an identifiable experiment and a defined evidence requirement. Unsupported claims of zero collisions, wall-breaking, trained CNN performance and edge deployment are marked as pending in working drafts.

**Step 1 — Repair the independent measurement tools**

This is the first implementation priority because every subsequent result depends on it.

Actions:

- Consolidate ROC/AUC calculation or ensure both existing implementations use the same validated definition. Traverse all distinct score thresholds in order, including endpoints, and handle tied scores correctly. Raw detector scores must not be restricted to [0,1].
- Compare AUC against both an independent pairwise ranking calculation and a maintained reference implementation such as [scikit-learn's ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). Record the exact reference version.
- Define behavior for empty data, one missing class, invalid labels, NaN/Inf, unequal lengths and malformed shapes. Undefined scientific quantities must not quietly become a reassuring zero or 0.5.
- Validate confusion matrices, class denominators, Pd, Pfa, accuracy, precision and F1 against manually calculable examples.
- Define EER and interpolation explicitly; use Pd at a controlled Pfa as the primary sensing comparison, rather than relying on EER or pooled AUC alone.
- Add exact or justified interval estimates for detection, false alarms and collision rates. For correlated captures, use capture/session-level resampling or another justified dependence-aware analysis.
- Define packet accounting: generated, attempted, successful, dropped and pending. Do not force PDR and explicit drop rate to sum to one when pending packets exist.

Required regression examples:

| Case | Required behavior |
|---|---|
| Labels [0,0,1,1], scores [1.1,1.2,2,3] | AUC=1; no clipping of score range. |
| Labels [0,0,1,1], scores [0.5001,0.5002,0.5003,0.5004] | AUC=1; fine score separation is retained. |
| All scores tied with both classes present | AUC=0.5 with correct tie handling. |
| Perfectly reversed ranking | AUC=0. |
| Benchmark's frozen energy scores | Exact rank result agrees with recalculated ROC AUC, approximately 0.50390625 for the audit dataset. |
| One delivered packet and one pending packet | PDR=50%, explicit drop rate=0%, pending rate=50%. |

Deliverables: validated metric module; regression cases; numerical comparison transcript; explicit metric definitions.

Exit criterion: independent calculations and the implementation agree to an appropriate documented numerical tolerance on analytical fixtures and saved score arrays. These tests must not use the implementation under test to generate their expected answers.

**Step 2 — Make the mathematics and signal conventions consistent**

Actions:

- Define whether noise variance means total complex power or variance per I/Q component. State the conversion explicitly everywhere.
- Define SNR before or after fading. Choose a reference convention and preserve it in generation, theoretical helpers, labels, plots and network mappings.
- Correct the channel-gain statistics helper and real-QPSK power behavior, or remove unsupported real-signal operation from claimed scope.
- Distinguish fixed-energy/deterministic signal vectors from random complex Gaussian signals under H1. Use the corresponding exact distribution and CLT variance consistently.
- Validate detector Pd in the transition region, not only at saturated high Pd. Sweep N, SNR, modulation and channel gain.
- Recalculate noise-uncertainty walls using one declared uncertainty interval and one SNR reference. Label robust/worst-case statements separately from average-case simulation results.
- Replace or qualify the finite-sample requirement approximation. Check that its returned N actually attains the requested Pd/Pfa using an independent exact calculation and Monte Carlo where applicable.
- Replace assumed drift Pfa values with measured or independently derived values for the specified detector and noise drift.
- Resolve which parts of the source paper are ambiguous rather than silently selecting favorable interpretations.

Concrete regression anchors from the audit:

- N=1024, total complex noise variance=1, Pfa=0.05: exact energy threshold approximately 1.051950849.
- Under the report's bounded-noise convention, B=1 dB gives wall approximately -3.3292 dB, not -8.2 dB.
- Configured reference SNR=0 dB, noise power=1 and h=0.1 imply received mean total power 1.01 when transmit power is held fixed.
- The current sample-complexity N=6397 at B=1 dB, nominal SNR=-3 dB fails the stated Pd=0.9 target in the audit's exact worst-case calculation. A corrected implementation must explain or remove that discrepancy.

Deliverables: mathematical specification; corrected helpers; analytical/Monte Carlo agreement tables; documented applicability limits.

Exit criterion: theory, generation and measurements use the same definitions and agree within justified numerical/statistical error. A model-specific detection limitation is not described as a universal impossibility theorem.

**Step 3 — Repair the Physical/SDR layer's correctness**

Actions:

- Establish a signal contract for complex/IQ shape, dtype, sample rate, timestamps, power units, gain/AGC state and preprocessing version.
- Correct the mismatch between MockSDR gain normalization and the detector's noise power/threshold. Obtain threshold calibration from independent H0 records in the same units as H1.
- Evaluate both H0 and H1 through the complete preprocessing and detection path. Avoid a fix that detects every frame as busy and merely transfers the failure to false alarms.
- Test DC correction and IQ correction on noncircular modulations and low-SNR signals. Quantify whether legitimate information is removed.
- Repair oversized ring-buffer writes, preserving consistent counters and the documented retention policy even on errors. Test capacity boundaries, wraparound, stalled consumers and restart behavior.
- Add frame sequence numbers and acquisition timestamps so dropped samples, stale frames and queueing delay can be distinguished from computation time.
- Derive each throughput/deadline budget from the actual sample rate, frame stride, decimation and scheduling policy. At 2.4 MS/s, nonoverlapping 1024-sample frames arrive every approximately 0.4267 ms; a 2.3 ms cycle needs an explicit explanation.
- Distinguish a mock test from hardware-in-the-loop. Mark incomplete real drivers as unsupported until they can configure, stream, stop and release actual devices.
- Make duty-cycle actuation and its measurement explicit. Changing a control variable alone is not an energy experiment.

Deliverables: calibrated sensing pipeline; robust buffer; frame-age/drop telemetry; supported-device matrix; corrected latency definitions.

Exit criterion: the audited active +10 dB QPSK scenario no longer produces 0/300 detections, and fresh H0 data meet the predeclared false-alarm requirement. Detection acceptance is based on confidence bounds under specified conditions, not simply on improving that one counterexample. Buffer state and sample-order invariants hold across the stress cases.

**Step 4 — Repair network-simulation internals before comparing protocols**

Actions:

- Label the current implementation as a custom Python discrete-event simulator. If an NS-3 claim is retained, define and execute an actual NS-3 scenario with recorded version/build/configuration, then validate correspondence with the custom simulator where feasible.
- Track complete transmission intervals. A PU entering during an SU transmission must update the relevant overlap/collision/exposure records.
- Define the physical meaning of collision. If claiming PU protection, represent PU receivers and a justified interference/SINR or exposure criterion; transmitter activity alone is not a complete harmful-interference model.
- Correct Slotted ALOHA completion scheduling and trace timestamps. Success must not be counted before the modeled receive event or beyond an excluded observation horizon.
- Specify event ordering for simultaneous start/end/PU-state transitions, and test boundary cases explicitly.
- Validate per-node queues, packet uniqueness, retry limits, dropped/pending state and delivered-only latency censoring.
- Avoid calling two thresholds on the same deterministic RSSI value distinct sensing stages unless they implement genuinely different observations or procedures.

Required checks include PU onset before DIFS, during DIFS, during backoff, during transmission and during sensing sleep; weak/hidden PUs; two overlapping SU transmissions; simultaneous events; transmission across the simulation horizon; retry exhaustion; and packet-object/trace/summary consistency.

Deliverables: simulator event specification; interval-based accounting; independent hand-calculable scenarios; consistent traces.

Exit criterion: all known interval intersections and completions are correctly accounted for under the stated simplified or physical model. The audit's mid-packet PU example and premature Slotted ALOHA completion must be caught by regression tests.

**Step 5 — Build a credible sensing dataset and freeze its partitions**

Actions:

- Verify actual RadioML file identity, schema, modulation order and metadata. Record hashes and source/version; do not rely on a filename alone.
- Define H0 generation or capture independently. The real RadioML modulation frames are H1; ensure the binary sensing dataset contains a credible H0 class with matched receiver/channel processing.
- Apply hypothesis-independent preprocessing. Normalization, filtering, gain, quantization, padding and dtype must not reveal the label through different processing branches.
- Implement chunked/lazy HDF5 ingestion if required by dataset size; filter and stream without materializing unnecessary full arrays.
- Partition by original capture/session/transmitter or generating realization wherever those identities exist. Keep augmented derivatives of the same source in one partition.
- Maintain separate training, model-selection/validation, threshold-calibration and final-test roles. The exact ratios should follow the available independent groups; a four-way 60/10/10/20 split is a possible starting allocation, not a universal requirement or a claimed reproduction of an unspecified paper split.
- Save immutable frame/group identifiers and hashes; check duplicate and common-origin leakage across every partition and across independently executed scripts.
- Keep synthetic generators as labeled debugging tools. Either implement the claimed modulation faithfully or label it as an approximation. Do not call eight default labels a 24-modulation validation.
- Predefine full-SNR testing, including conditions below the training cutoff. If studying the paper's SNR filter, make filtered versus unfiltered training a named ablation, rather than choosing a cutoff from final test performance.
- Create separate generalization evaluations with entire modulations, transmitters or channel conditions held out from training.

Deliverables: dataset card; raw-data manifest; H0/H1 construction specification; preprocessing contract; immutable grouped splits; leakage-check report.

Exit criterion: no identified frame or generating-source overlap crosses partitions; both hypotheses pass through the declared processing; a reviewer can identify exactly which frames and independent groups entered each experiment. Missing real data causes publication runs to stop, not silently fall back.

**Step 6 — Implement and prove actual neural learning**

Actions:

- Use executable TensorFlow/Keras computation for the neural models, consistent with the project's stated framework. Keep the four-feature logistic rule under a distinct baseline name.
- Start with the paper's Baseline CNN. Verify actual tensors, layer dimensions, trainable parameter arrays and optimizer connections.
- Resolve the BatchNorm table ambiguity with a documented architecture choice. Distinguish the paper's reported total/trainable counts from inferred corrected counts. Record deviations rather than changing dimensions only to match a number.
- Train using the frozen training split, validate model selection separately, and preserve complete configuration and random-state information.
- Save actual models/weights, optimizer state when needed, and preprocessing/configuration metadata. Use a real checkpoint mechanism; [Keras ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) supports saving a model or weights for later loading.
- Verify early stopping restores the selected epoch when that option is enabled.
- Ensure evaluation refuses to run a publication benchmark without an identified trained checkpoint. Record checkpoint hash in every result.
- Add BatchNorm and GAP once the Baseline pipeline passes its structural and learning checks. Add Bi-GRU only if it addresses a stated hypothesis beyond the CNN study.

Required evidence:

- Representative neural weights receive nonzero gradients and change during training on a suitable controlled batch.
- Controlled perturbation or removal of relevant neural computations changes predictions on appropriate nonsaturated inputs. Tests must avoid falsely requiring every weight perturbation to affect every input.
- A deliberately small, learnable training problem can be fitted, while evaluation remains separate.
- Loading a saved checkpoint reproduces its predictions within documented numerical tolerance.
- Shuffled-label training does not produce systematic held-out performance above chance; confirm this over independent realizations rather than imposing an arbitrary exact 0.5 result on one finite sample.
- Architecture-specific predictions actually execute architecture-specific operations; equal accuracy alone is neither expected nor evidence of a bug.

Deliverables: trained Baseline checkpoint; structural/gradient checks; load/restore parity; training history; later separate BN/GAP checkpoints.

Exit criterion: there is direct evidence that the advertised network, rather than the shared feature rule, is being trained and evaluated. Large training runs should wait for this checkpoint.

**Step 7 — Run a fair and statistically specified sensing benchmark**

Before final evaluation, write an experiment protocol specifying the primary endpoint, minimum meaningful effect, conditions, trial/group counts, comparison method, threshold rule and uncertainty analysis.

Actions:

- Evaluate energy detection, the existing feature classifier, the actual CNNs, and the paper's MME/entropy methods if their superiority is claimed.
- Feed every detector the same underlying held-out I/Q realizations and ground truth. Detector-appropriate processing is allowed, but must be disclosed, fixed without test leakage and based only on information available at deployment. Never compare one method under noise uncertainty with another receiving easier fixed-noise data without labeling that as a different experiment.
- Set each operating threshold using separate calibration H0 data. For worst-case uncertainty claims, define and validate the worst-case calibration policy explicitly.
- Measure test Pfa and its interval. Report failure to meet the nominal target rather than silently describing all results as being at that target.
- Report per-SNR and per-modulation Pd/Pfa; also stratify by important channel/receiver conditions. Pooled AUC across mostly easy conditions cannot establish low-SNR reliability.
- Repeat model training with multiple seeds. A practical initial plan is five independent training seeds, expanded if uncertainty or instability remains material; five is not a guarantee of adequate statistical power.
- Select test sample sizes using the precision or decision objective. Save counts, not only rounded percentages.
- Use paired comparisons on common frames, with capture/session grouping in the uncertainty method when relevant. Report effect sizes and intervals, not only p-values.
- Keep untouched final evaluation separate from exploratory sensitivity work. If the final evaluation prompts redesign, disclose that and create a new independent evaluation for the revised system.

Example acceptance formulation: for a predeclared modulation/channel/SNR region, require a one-sided 95% lower confidence bound for Pd above the chosen minimum and a one-sided upper bound for Pfa below the chosen maximum. The actual targets must be justified by the research question and chosen before looking at final results. Do not require all SNRs to pass an unrealistic universal threshold.

The prior 88% result was 22/25, with a wide interval; it cannot support a 90% lower-bound claim. For planning only, a simple independent-binomial normal approximation needs about 865 H1 trials to estimate Pd near 0.9 within approximately plus/minus 0.02 at 95%. Real capture dependence, multiple comparisons and model-training uncertainty can require more data. False-alarm trials require their own calculation.

Deliverables: frozen protocol; raw per-frame scores/labels; seed/run manifest; per-condition tables with uncertainty; paired comparison analysis; failure-region map.

Exit criterion: conclusions follow the measured effects and confidence limits under matched conditions. If a CNN does not outperform a conventional detector in a condition, report that condition. A credible negative or qualified result is preferable to an unsupported superiority claim.

**Step 8 — Close the sensing-to-network loop**

This step depends on both the corrected simulator and accepted sensing benchmark.

Actions:

- Replace oracle PU-state sensing in the integrated experiment with actual detector outputs from timestamped frames, or a clearly labeled stochastic detector model derived from held-out measurements.
- Include sensing duration, computation delay, observation age, threshold errors, missed detections, false alarms and PU onset between observations.
- Retain oracle sensing only as an explicitly labeled upper-bound comparison.
- Use identical topology, PU schedules, source traffic and channel realizations for paired MAC comparisons where possible.
- Run two different scalability protocols: constant total offered load while node count changes, and constant per-node offered load. Report actual generated and attempted loads after every cap or retry.
- Include independent simulator replications, sensitivity to PU duty cycle and sensing period, and warm-up/observation/drain policies.
- Report PU exposure/interference, SU useful throughput, generated-packet PDR, deadline delivery, drops, pending packets, queue delays and per-user fairness together.
- If retaining a claim of two-stage sensing, implement and evaluate the two observations independently and jointly, including their dependence.

Deliverables: integrated trace dataset; oracle-versus-measured-detector comparison; fair protocol benchmark; network sensitivity and uncertainty results.

Exit criterion: the detector used in the network is traceable to the validated checkpoint/calibration, and all reported success/failure events can be reconciled with underlying observations and intervals. Zero observed collisions must be stated as a finite-sample observation with an upper rate bound, not a universal guarantee.

**Step 9 — Validate export and actual target-device execution**

Actions:

- Convert a frozen trained checkpoint. Conversion failure must produce an explicit failed/unsupported result, with no file-shaped metadata fallback presented as a model.
- Load artifacts with the actual TFLite/LiteRT interpreter and ONNX checker/runtime as appropriate. Execute held-out test vectors, not just shape inspection.
- Begin with FP32 parity, then FP16, then INT8. For full integer quantization, use representative calibration data from the development/calibration allocation and explicitly inspect integer operator and I/O requirements. Follow the [official integer-quantization workflow](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant).
- Test constant nonzero, all-positive, all-negative, tiny-range and extreme-value tensors if custom quantization remains. Measure task effects of quantization on the held-out sensing protocol.
- Recalibrate the quantized detector's threshold using permitted calibration data if that is the declared deployment procedure. Evaluate both retained-threshold and recalibrated behavior when informative.
- Measure actual serialized bytes, peak resident memory, intermediate activation memory, load/startup time and steady-state inference time.
- Benchmark the chosen target device with batch=1, actual input processing, a controlled software environment, recorded clock/power/temperature settings and sustained operation. The hardware need not be every advertised board; begin with one available supported target.
- Measure ADC-to-decision age separately from model-call latency; include frame drops, queue buildup and deadline misses.
- Measure energy directly before making energy-efficiency or duty-cycle savings claims.

Deliverables: executable exports; runtime parity report; post-quantization Pd/Pfa; actual size/memory/timing distributions; target-device run manifest.

Exit criterion: the same trained system can be loaded and executed on the target, its task performance remains within predeclared tolerances, and its sustained end-to-end behavior meets the actual rate/deadline specification. No hardware availability means a simulation-only result with hardware claims deferred.

**Step 10 — Resolve optional game-theory and robustness extensions**

For the game-theory component, write down the exact payoff, information available to players, feasible strategies and individual/coupled protection constraints. Verify no feasible unilateral deviation improves utility at each claimed equilibrium. Optimize social welfare over the same feasible set, and handle zero/negative welfare explicitly. Test independent-player dynamics and asymmetric costs if claiming them. A clipped shared probability and a matching clipped iteration do not by themselves establish a Nash result.

For sensing/system robustness, predefine representative sweeps of:

- Noise uncertainty endpoints and drift, colored/impulsive noise and non-PU interferers.
- Correlated fading, frequency-selective multipath, mobility/coherence and deep fades.
- CFO, sampling-rate error, phase noise, IQ imbalance, DC offset, gain changes and clipping.
- Unseen modulation/transmitter/receiver sessions and samples-per-symbol or pulse-shaping changes.
- Weak/hidden PUs, multiple PUs, sudden arrivals, short bursts and long packets.
- Heavy traffic, unfair access, queue saturation, hardware disconnection and stalled consumers.

Start with physically justified ranges from the chosen dataset and hardware. Do not invent an enormous arbitrary parameter grid. Use coarse exploratory sweeps followed by focused independent confirmation at failure boundaries. Report where the system stops meeting its targets.

Deliverables: robustness protocol; operating-region/failure maps; sensitivity tables; optional game-theory proof and numerical checks.

Exit criterion: extension claims are supported independently. If they are not essential to the principal result, remove or defer them rather than weakening the entire study.

**Step 11 — Regenerate the report and commission an independent rerun**

Actions:

- Generate tables, plots and test inventories directly from machine-readable outputs. Do not manually type scientific numbers into the report.
- Every figure/table records experiment ID, code commit, effective configuration, data/split hashes, checkpoint hash, sample/group counts and analysis version.
- Include full relevant axes, uncertainty, failed conditions and units. Reconcile matrices, scalar metrics, log entries and figures automatically.
- Match paper references accurately: Figure 6 is training/validation history; Figure 7 includes OOK and QPSK; the paper's traditional-method comparison uses a different stated Pfa from the current benchmark. State which results were reproduced and which experiments are new.
- Separate implementation correctness, empirical validity, generalization and deployment evidence in the text. Avoid treating the number of passing unit tests as the scientific conclusion.
- Publish a reproducibility package with pinned dependencies, configuration files, dataset-access instructions, split manifests, trained checkpoints, raw predictions, analysis scripts and complete run transcripts, subject to data permissions.
- Have someone outside the implementation effort rerun the principal experiment from a clean environment using only that package. They should reconcile results within documented stochastic/numerical tolerance, not require bit-identical hardware timings.
- Update the claim ledger with supported, restricted, inconclusive or rejected outcomes. Rewrite conclusions around the final evidence.

Deliverables: corrected manuscript/report; versioned reproducibility package; independent-rerun record; completed claim ledger.

Exit criterion: an external reader can trace each central claim to an executable experiment and the underlying data, and the independent rerun supports the stated conclusion and its limits.

**Practical sequence and checkpoints**

| Checkpoint | Complete these steps | Decision before continuing |
|---|---|---|
| C0: scope frozen | 0 | Is the next research deliverable precise and achievable with available resources? |
| C1: measurements trustworthy | 1-2 | Do independent mathematical and metric checks agree? |
| C2: primitives correct | 3-4 | Are calibration, buffer state, event timing and collision accounting correct on known scenarios? |
| C3: one valid sensing result | 5-7, starting with Baseline CNN | Is actual learning demonstrated and held-out performance credibly measured? |
| C4: integrated extension | 8 | Does measured sensing behavior support the network conclusions? |
| C5: deployment evidence | 9, only if claimed | Do real exported models meet task and system requirements on the named target? |
| C6: publication review | Relevant parts of 10-11 | Are all retained claims supported and independently reproducible? |

Steps 3 and 4 can be developed independently after the measurement conventions are fixed. Dataset preparation can proceed while simulator repairs are underway, but no final benchmark should start before metric, data and model checks pass. Optional game theory can be separated from the main path. Do not start with a full four-model training campaign or an all-device benchmark.

**Suggested immediate work queue**

1. Freeze the current baseline and create the claim ledger.
2. Add failing ROC/AUC regression cases from the audit, then correct the metric routines.
3. Correct SNR/noise/gain definitions and validate exact detector calculations.
4. Add the 0/300 active-PU streaming regression and repair calibration while controlling false alarms.
5. Add PU-mid-packet and premature Slotted ALOHA completion regressions, then repair event accounting.
6. Build the immutable H0/H1 dataset split and prove one actual Baseline CNN trains, saves and reloads correctly.

This queue is the recommended first work package. It produces reviewable corrections and a credible starting point for larger experiments without promising a favorable final scientific result.

**Effort and resource planning**

Budget by engineering milestones rather than a guaranteed publication date. Metric and deterministic accounting corrections are bounded software tasks; real-data preparation, architecture clarification and a new neural-training pipeline are larger tasks; physical acquisition and sustained device benchmarking depend on hardware access and setup. Establish a small pilot to measure actual data throughput, training time and capture reliability before committing to a full experiment matrix. The first scheduling estimate should be revised after C1 and the Baseline pilot.

The work requires implementation, signal-processing/statistical review, ML experimentation and, only for deployment claims, hardware validation. One person may fill several roles, but the principal reproduction should receive independent review. If data, GPU or hardware resources are limited, narrow the claims transparently: a rigorous sensing-only or simulation-only study is a defensible deliverable. Substituting mocks while keeping hardware claims is not.

**Required evidence directory for each new run**

Suggested contents:

```text
run-ID/
  manifest.json             # commit, environment, input and checkpoint hashes
  effective-config.yaml     # parameters actually used after overrides
  split-manifest.json       # frame/group IDs and partition roles
  checkpoint/               # real trained model and relevant training state
  predictions.npz           # frame IDs, ground truth, scores, conditions
  network-events.*          # when a network experiment is run
  device-timings.*           # when a hardware/runtime experiment is run
  metrics.json              # numerators, denominators, estimates, intervals
  analysis-log.txt
  figures/
```

Store seeds separately for data, split, initialization, training and simulator/channel processes. Record any nondeterminism and the observation horizon. Use independent original records/groups as the statistical unit where frames are correlated. Keep exploratory runs distinct from the locked final evaluation.

The next scientific milestone should read: **one correctly implemented, calibrated, trained and independently evaluated sensing pipeline**. Whether its result confirms, qualifies or rejects the paper's claimed advantages must remain an outcome of the experiments.
