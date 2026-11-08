# Independent technical audit of SpectrumSensing

Review date: 5 November 2026. Audited local commit: `cba90a1522c3f4a8b3f839b1750a370713f61a50`.

**Verdict: the current testing evidence does not support the report's claims of a validated deep-learning implementation, reproduced paper results, reliable PU protection, or deployment-ready edge models. Substantial implementation and experimental corrections are required before these claims are publication-ready.**

This conclusion is based on source inspection, independent calculations, a rerun of every collected test, execution of all 19 experiment scripts in a separate source copy, inspection of stored figures and model artifacts, and explicit counterexamples. It is not based merely on missing documentation. Several central claims are directly contradicted by executable evidence.

The software-suite result itself is real: **205/205 collected cases passed in 7.73 seconds** in the audit environment. All **19/19 experiment scripts also exited successfully**. Passing these assertions is insufficient because important tests exercise metadata, bounds, synthetic durations, and file existence while omitting the scientific behavior asserted in the report.

The report and the supplied Sensors article must be distinguished. The article is a published 2024 source; the repository describes an implementation and extensions, including SDR, networking, and Bi-GRU. This audit finds that the supplied repository/report does not establish an independent reproduction or validation of that article. It does **not** establish that the original authors' unavailable experiments necessarily failed.

**Review scope and preserved evidence**

The four domains are Mathematical Foundations, Physical/SDR, Network Simulation, and AI/ML Sensing/Export. The supplied Desktop `.md` file is actually an HTML document embedding Markdown. After extracting and decoding that embedded text, it exactly matches `docs/COMPREHENSIVE_VERIFICATION_REPORT.md` in the local checkout. The two local copies of the paper have identical SHA-256 hashes:

`743f4819629d0bc206f0ca1c56ef0233acab2a43a09eabd6f74fda2070b2f0c1`.

The local repository has the origin URL `https://github.com/mohamed7703/SpectrumSensing.git`. Direct web retrieval of that repository was unavailable during review; no claim is made that the local commit is the latest remote revision. The paper was read locally, including all 22 pages; key architecture and figure pages were also visually inspected. All 19 test files, six YAML configurations, 19 experiment logs, associated implementations, and verification scripts were examined. Stored figures were checked against their generating code, with direct visual inspection of figures central to the findings.

No original source, test, configuration, report, plot, or exported model was edited. Hashes of the audited inputs were recorded before execution and checked afterward; no recorded input changed. The original Git working tree remained clean. New probes, fixtures, rerun outputs, and this report are in a separate sibling audit directory. In-memory perturbations of freshly instantiated models were used for counterexamples; these did not change stored weights or source files.

Audit environment: Windows x64, Python 3.10.11, NumPy 2.2.6, SciPy 1.15.3, h5py 3.16.0, pytest 9.2.0. TensorFlow, tf2onnx, ONNX, ONNX Runtime, and TFLite Runtime were unavailable. This limited native runtime validation but did not prevent identifying the exact non-model bytes in the stored exports. Real RadioML data and physical SDR/edge hardware were not available in the reviewed artifacts.

**How test classifications should be read**

| Classification | Meaning |
|---|---|
| Verified | The specific narrow unit behavior was supported by inspected assertions and an independent execution/calculation. This is not a certification of the surrounding scientific claim. |
| Partially Verified | Some behavior is supported, but the assumptions, coverage, or assertion strength are insufficient for the reported conclusion. |
| Inconclusive | The claimed scenario lacks identifiable executable evidence or required data/runtime/hardware. |
| Failed | An independently established contradiction invalidates the reported scientific interpretation. A weak pytest assertion may still pass. |

[The complete case-by-case appendix](S:/Projects/SpectrumSensing-audit-2026-11-05/TEST_BY_TEST_AUDIT.md) classifies all **205 actually collected cases**, and separately reconciles all **205 names listed in the report**. The actual cases comprise **199 distinct source functions** plus six extra parameterized instances. Their audit classifications are **121 Verified**, **73 Partially Verified**, and **11 Failed**; all 205 have a separate software execution status of PASSED. The report-name reconciliation contains **88 Verified**, **60 Partially Verified**, **41 Inconclusive**, and **16 Failed** entries. These are different inventories and must not be added together. Forty-six report names have no exact collected-name match after handling the CNN presentation labels; some have related tests under different names. Absence of a name alone is not proof that no related primitive was tested.

**Layer and experiment assessment**

| Milestone / log | Domain | Assessment | Basis and limits |
|---|---|---|---|
| P1-001 / EXP-001 | Hypothesis testing | Partially Verified | Complex AWGN moments and basic H0/H1 power behavior reproduce. `theoretical_statistics(h)` ignores h; real-QPSK power scaling and SNR conventions are inconsistent. |
| P1-002 / EXP-002 | Energy detection | Partially Verified | Exact complex-noise threshold and fixed-energy BPSK benchmark reproduce. Gaussian-signal versus fixed-energy H1 distributions and CLT variance are mixed; the Pd unit test uses a saturated operating point. |
| P1-003 / EXP-003 | Metrics and ROC | Partially Verified | Confusion counts work. The empirical ROC routine fails simple perfect-separation examples; some log AUC numbers disagree with the current theoretical routine. |
| P1-004 / EXP-004 | Fading | Partially Verified | Selected PDF identities, integration, power moments, and Rayleigh behavior pass. Tests do not establish temporal fading realism, complete claimed tolerances, or all-channel recovery above 98%. |
| P1-005 / EXP-005 | Noise and SNR wall | Failed | The core coded wall formula is sound under its stated model, but the -8.2 dB headline is inconsistent, finite sample requirements miss requested targets, and the advertised noise-drift experiment uses an assumed Pfa. |
| P2-001 / EXP-006 | DSP | Partially Verified | Useful FFT/PSD/filter unit checks pass. Reported frequencies, filter order/attenuation, PSD flatness, and time/frequency axis conventions exceed or differ from actual tests. |
| P2-002 / EXP-007 | Preprocessing | Partially Verified | DC subtraction, scaling, basic correction and augmentation work on fixtures. End-to-end test named in report is absent; signal preservation and sensing calibration are not established. |
| P2-003 / EXP-008 | SDR abstraction | Partially Verified | Simulated drivers and missing-dependency exceptions pass. No physical verification; HackRF, LimeSDR, and USRP physical receive paths are incomplete. |
| P2-004 / EXP-009 | Streaming/latency | Failed | Mock loop is fast, but its active-PU detector misses 300/300 tested frames. Ring-buffer oversize-write counterexample corrupts state. Timing does not include a trained neural network or establish field real-time behavior. |
| P3-001 / EXP-010 | Topology/DES/traffic | Partially Verified | Basic event ordering, geometry, path loss, and generation tests pass. This is a custom Python DES, not an executed NS-3 experiment; realistic PHY/MAC coupling is absent. |
| P3-002 / EXP-011 | MAC protocols | Failed | Formula checks and favorable scenarios pass. PU onset during a packet is uncounted; a weak active PU causes collisions; protocol comparison conditions are unequal. |
| P3-003 / EXP-012 | Trace/delay/scalability | Failed | Parser handles its own trace grammar, but report's >92% PDR through 500 nodes is contradicted by 15.8%; stored plot clips away low values. Completion timestamps also disagree for Slotted ALOHA. |
| P3-004 / EXP-013 | Nash/PoA | Failed | Implemented utility differs from report; a capped symmetric probability is presented as an equilibrium without the corresponding deviation constraints. Social optimum ignores the protection cap. |
| P4-001 / EXP-014 | Dataset loader | Failed | Synthetic schema smoke tests pass. Actual HDF5 path is eager, H1-only, and untested by suite; fallback is not a faithful 24-modulation reproduction; splits are not stratified. |
| P4-002 / EXP-015 | CNN models | Failed | Runtime classifier is a shared four-feature logistic rule. Parameter metadata cannot validate executed convolution; BatchNorm counts disagree with both paper and Keras path. |
| P4-003 / EXP-016 | Bi-GRU | Failed | Isolated random-weight recurrent computation exists but is disconnected from prediction/training; reported latency times the shared logistic rule. |
| P4-004 / EXP-017 | Training | Failed | Only five feature-head parameters are trained. Checkpoints save no weights; early stopping does not restore weights. Reported history is not deep-network training evidence. |
| P4-005 / EXP-018 | Benchmark | Failed | Headline numerical outputs can be regenerated, but they come from untrained identical feature classifiers, invalid baseline ROC handling, unequal experiments, and reused calibration samples. |
| P4-006 / EXP-019 | Edge export | Failed | Stored TFLite/ONNX files are metadata placeholders; quantization uses random numbers, and timing executes host NumPy rather than exported neural models. |

The milestone classifications judge the claimed scientific deliverable, not whether every low-level function inside a milestone fails.

**1. Mathematical foundations: supported results and inconsistencies**

For independent complex noise with total variance v, the average energy T has `2NT/v ~ chi-square(2N)`. Therefore the exact CFAR threshold `v * chi2.ppf(1-alpha, 2N)/(2N)` is appropriate. At N=1024, v=1, alpha=0.05, I obtained **1.051950849**; the CLT value is **1.051401676**. EXP-002's rounded 1.0520 and 1.0514 are correct. The original verification script reproduced empirical Pfa **0.0483**, Pd **0.9214** at -10 dB, and exact fixed-energy Pd **0.921332082**. These are useful positive results under their explicit assumptions.

The exact noncentral chi-square H1 distribution applies conditionally to a fixed signal vector/energy in AWGN, including constant-modulus symbols at fixed h. For independent complex Gaussian signal samples, the unconditional received signal is zero-mean Gaussian with larger variance, so energy has a **scaled central** chi-square distribution. It is not the same unconditional noncentral distribution. The report introduces Gaussian signals but presents the latter distribution as universal.

The fixed-energy complex-signal energy variance is `v^2(1+2*SNR)/N`; for independent complex Gaussian signals it is `v^2(1+SNR)^2/N`. [The CLT implementation](S:/Projects/SpectrumSensing/src/math_models/energy_detection.py:182) uses the second expression while its exact method uses the first model. These agree closely in some large-N/low-SNR regimes, so the audit does not claim that every published rounded Pd is materially wrong. They must nevertheless be labeled and validated against the same generative assumptions. The existing empirical-Pd unit test operates at -5 dB, where theoretical Pd is almost one, and can pass without testing the transition where model errors matter.

[Theoretical statistics](S:/Projects/SpectrumSensing/src/math_models/hypothesis_testing.py:287) ignore the `channel_gain` argument. At configured SNR=0 dB, v=1 and h=0.1, the generated signal has expected total power **1.01**, with audit empirical power **1.00546** over 200,000 samples; the helper reports **2.0**. The actual generator treats configured SNR as a pre-fading/reference SNR, while its documentation also describes post-channel SNR. Choose one definition and propagate it consistently. The real-valued QPSK branch also returns average power **0.5** for requested signal power 1; only the complex path is covered by the parameterized power test.

The reported SNR wall uses `rho=10^(B/10)` with noise power in `[v/rho,rho*v]`, and `SNR_wall=rho-1/rho` relative to nominal noise power. Independent recalculation gives:

| Noise uncertainty B | Recalculated wall | Report/log issue |
|---|---:|---|
| 0.2 dB | -10.3557 dB | Existing narrow unit test is consistent. |
| 0.4 dB | approximately -7.34 dB | Report says -8.23 dB in a test description and labels a plot -8.2 dB. |
| 0.5 dB | -6.3683 dB | EXP-005 is consistent. |
| 1.0 dB | **-3.3292 dB** | EXP-005 is consistent; headline claims repeatedly use **-8.2 dB** instead. |

The alternate formula in `evaluator.py`, `(rho-1)/sqrt(rho)`, gives **-6.3683 dB** at B=1 dB, also not -8.2 dB. The underlying wall concerns **robust distinguishability under uncertainty**, not a universal statement that every finite-sample experiment below it must have Pd equal to Pfa. It also does not prove that deep learning is mandatory or universally escapes uncertainty limits. See the primary [Tandra–Sahai analysis](https://people.eecs.berkeley.edu/~sahai/Papers/tandra_sahai_revised_submission.jstsp.final.pdf).

The advertised noise-drift verification is not the described Monte Carlo experiment: [the script](S:/Projects/SpectrumSensing/scripts/verify_noise_models.py:75) sets `pfa_drifted_up = 0.22` and feeds that number into an analytical curve. It does not measure that Pfa from +0.4 dB drift. Its wall line is positioned from the computed -7.34 dB value while its label says -8.2 dB. The report's four-panel discussion of multiple-N collapse/uncertainty ROC is not supported by this two-panel script.

The finite sample requirement `[(Qinv(alpha)-Qinv(Pd))/(SNR-wall)]^2` drops uncertainty-dependent variance factors and is not a reliable target guarantee. In an independent exact conditional fixed-energy calculation, B=1 dB and nominal SNR=-3 dB give **N=6397** from the code. Using a worst-case-H0 exact CFAR threshold and worst-case-H1 noise, this N gives **Pd=0.76025**, not requested 0.90, with Pfa=0.05. At -2 dB the code returns N=310 and the same calculation gives **Pd=0.73610**. These results do not modify the code; they test its returned requirement. Either derive and label an approximation with its domain of validity or solve the exact target numerically.

The Phase-1 empirical ROC method also sorts points by FPR without preserving their threshold traversal for equal FPR. For perfectly separated scores `[1.1,1.2,2,3]` and labels `[0,0,1,1]`, its AUC is **0.875**, although exact AUC is **1**. Its perfect-separation unit fixture happens to pass. EXP-003's current theoretical AUC routine gives **0.71264, 0.86581, 0.98462** at -16,-13,-10 dB; its log lists 0.697,0.854,0.978. Those differences exceed three-decimal rounding. Historical provenance for the log numbers is not supplied.

Fading PDF identities, integration on selected grids, and average-power checks are reasonably supported. They validate marginal distributions, not correlated physical multipath channels. Independently resampling h per sample is not a Doppler/coherence-time model, and a per-frame scalar does not model frequency-selective delay spread. The Rayleigh degradation test checks only `Pd<0.80` at -8 dB: an always-idle detector passes that assertion. The recovery test checks Rayleigh `Pd>=0.95`, not the report's all-channel `Pd>0.98`. Use two-sided accuracy or distribution agreement checks, and test complete curves with uncertainty intervals.

**2. Physical/SDR: timing success conceals sensing failure**

The FFT roundtrip, Parseval check, mean Welch density, a chirp localization check, selected Butterworth responses, and basic preprocessing identities are useful unit tests. Several descriptions overstate them: the tone recovery fixture is 250 kHz at 1 MS/s, not 2.5 MHz; the white-noise PSD test checks **mean density within 8%**, not frequency-bin variation below 0.1 dB; the filter test uses a **sixth-order** filter with a stop-tone requirement of **25 dB**, not fifth-order/40 dB. Spectrogram functions return `(time,frequency)` and batches `(batch,time,frequency)`, contrary to the report's reversed conventions.

DC subtraction necessarily removes legitimate DC-centered signal content too: a constant complex signal of unit power becomes zero. Gram–Schmidt correction enforces moments on the samples but cannot by itself prove that an originally noncircular modulation was distorted by hardware rather than intrinsically noncircular. The tests use Gaussian rails. Add BPSK/OOK and low-SNR signal-preservation checks. The standard complex-to-I/Q helper does not cast to float32, and its small roundtrip fixture does not prove arbitrary complex128-to-float32 losslessness. Integer `np.roll` is tested; fractional delay is not.

The abstraction tests use simulated backends. Calling this hardware-in-the-loop is unsupported when no hardware participates. RTL-SDR and Pluto have real read branches that remain unverified here. [HackRF](S:/Projects/SpectrumSensing/src/sdr_interface/hackrf_driver.py), [LimeSDR](S:/Projects/SpectrumSensing/src/sdr_interface/limesdr_driver.py), and [USRP](S:/Projects/SpectrumSensing/src/sdr_interface/usrp_driver.py) instead unconditionally raise a physical-stream error after their mock branch; their actual acquisition paths are incomplete. The 16-bit ADC test bypasses quantization by casting to complex64. An 8-bit sweep of [-1,1] produces **255** distinct levels, while the test checks only step size. The report also describes USRP bit depths inconsistently with its own configured B210=12 and X310=14 bits. No physical hardware spec is independently inferred from these mocks.

The most important cross-layer counterexample uses the streaming script's own settings: MockSDR, 12-bit quantization, 2.4 MS/s, QPSK, +10 dB, seed 42, and the same 4096-sample chunks/1024-sample frames. MockSDR normalizes each chunk to its gain-dependent power; default gain is 50 dB in this construction. Over 300 frames, measured energies were **0.58781–0.63767**. The detector uses nominal noise variance 1 and threshold **1.051950849**. Consequently:

**Active-PU detections: 0/300. Final duty-cycle control variable: 0.3.**

The loop can therefore finish quickly and reduce its sensing-duty variable while systematically declaring this active PU idle. The claimed latency result is not evidence of correct sensing. The independent check matches the deterministic sample-generation sequence; acquisition scheduling is irrelevant to this amplitude/threshold mismatch.

The rerun streaming script measured mean **0.415 ms**, P95 **0.729 ms**, no reported overruns, and no underruns in its 300-frame mock run. Those are observations on this host, not a reproduction of the exact historical 0.599 ms or a proof of a trained neural pipeline's deadline. The script's detection stage uses energy detection, not any CNN/GRU. The two unit tests for latency statistics feed in invented durations; they check arithmetic, not elapsed pipeline time.

The 2.3 ms deadline comes from approximately **446.4 kS/s**. The streaming benchmark is configured for **2.4 MS/s**, whose nonoverlapping 1024-sample arrival period is **0.4267 ms**; 10 MS/s gives 0.1024 ms and 16 MS/s gives 0.064 ms. A slower configured sensing/decimation schedule could be valid, but it must be explicit. Mock generation has no hardware-rate pacing. Buffer dequeue time does not measure sample age from an ADC timestamp. Long-duration queueing stability, frame age, deadline-miss probability, hardware buffering, and processing of every arriving sample remain unverified.

[The ring buffer](S:/Projects/SpectrumSensing/src/sdr_interface/stream_handler.py:90) uses a lock and condition variable, so the report's “lock-free” description is false. Capacity also varies across report and implementation (2^16 versus benchmark 2^20). Writing 25 samples to a capacity-10 buffer raises a broadcasting exception after mutating state, leaving **available=-15** and **fill=-150%**. Normal overrun tests do not cover this case. The duty controller changes a number; the verification loop never calls its computed sleep time. Energy savings and unchanged sensing QoS were not measured.

**3. Network layer: protection, fairness, and delay claims fail**

The implementation is a custom priority-queue Python discrete-event simulator. A directory named `ns3_simulation` and a trace string containing `/NodeList/` do not demonstrate an NS-3 execution. No NS-3 version/build/scenario output is established. The parser's accepted grammar is its own `pkt_id=...` format; arbitrary native NS-3 trace compatibility remains unverified.

[ILA-CSMA sensing](S:/Projects/SpectrumSensing/src/ns3_simulation/mac_ila_csma.py:93) directly examines the simulation's true `pu.is_active` and computes deterministic RSSI with fading and shadowing disabled. It uses neither Phase-1 detector error probabilities nor Phase-4 model predictions. Global knowledge of any active SU transmission marks the channel busy regardless of distance. This removes realistic hidden/exposed-terminal behavior. Both “stages” use the same RSSI number: thresholds -90/-85 dBm (or -95/-90 in the benchmark) with an OR rule are equivalent to a single comparison against the more sensitive threshold. Stage 2 adds no distinct evidence.

Collision status is captured at transmission start and passed as a Boolean to completion. There is no mechanism to revise it when a PU activates during the transmission. An audit packet sent from **0.000050 to 0.008242 s**, with PU activation at **0.004 s**, overlaps the active PU for **4.242 ms** yet is recorded as **one success, zero PU collisions**. This establishes incomplete overlap accounting; it does not measure physical interference power at a PU receiver, which the simulator does not model. Separately, an active 30 dBm PU at (500,500) and SU at (0,0) gives deterministic RSSI **-116.402 dBm**, is sensed idle, and produces **one recorded PU collision**. Thus even the simulator's own zero-collision claim has a counterexample.

The suite contains no test named `test_ila_csma_zero_pu_collisions`; its favorable PU-off scenario cannot establish protection for arbitrary PU return times, weak PUs, missed detections, or receiver geometry. The code considers any active PU a collision at transmission start, even without a modeled PU receiver or interference threshold. Both false negatives and false positives in collision classification are therefore possible relative to a physical interference criterion.

The MAC comparison uses unequal conditions: Pure ALOHA and ILA-CSMA have PU toggles, but Slotted ALOHA is run with **`primary_users=None`** and no equivalent PU interference. Empirical Pure ALOHA at G=0.5 is **S=0.061**, not its no-PU theoretical peak 0.184. The theoretical peaks `1/(2e)` and `1/e` are correct formulas; their formula tests are not empirical validation under a different workload. A finite sample Slotted ALOHA throughput above 1/e is not by itself a bug; replication and confidence intervals are needed.

Traffic is capped at 500 packets in two seconds, while x-axis load labels continue to G=3. At labeled G=3, actual generated load is **500*0.008192/2=2.048**. Computing plotted PDR as S/labeled-G instead of successful/generated packets is then wrong. Arrival times are sorted uniform draws conditional on a chosen count rather than a full unconstrained Poisson process. A single SU is used for each aggregate load curve. These choices need disclosure and do not validate multi-node contention/fairness.

The Slotted ALOHA simulator also records a receive event at slot start while setting `packet.receive_time` to slot end. With a 10 ms slot and simulation horizon 11 ms, a packet starts at 10 ms, has receive_time **20 ms**, is already counted successful, and its trace gives parser delay **10 ms** instead of actual packet-object delay **20 ms**. Completion after the horizon is counted prematurely. This undermines cross-protocol delay and horizon accounting.

The exact dense-scenario helper rerun at seed 42 and duration 1.5 s gives:

| SU nodes | ILA PDR | ILA mean delivered delay | Comparison PDR | Comparison mean delivered delay |
|---:|---:|---:|---:|---:|
| 10 | 40.0% | 18.64 ms | 34.0% | 10.16 ms |
| 50 | 40.0% | 19.00 ms | 33.33% | 9.94 ms |
| 100 | 41.0% | 15.93 ms | 36.0% | 10.13 ms |
| 500 | **15.8%** | 16.57 ms | 15.6% | 11.18 ms |
| 2000 | **4.2%** | 15.59 ms | 4.15% | 11.06 ms |

These results agree with the stored figure's data, while contradicting the report's PDR >92% through 500 nodes and narrative of 4.2-to-28.5 ms scaling versus ALOHA >250 ms. The figure actually compares two configurations of the **same ILA-CSMA class**, not an independently implemented standard CSMA/CA or ALOHA. [Its y-axis minimum of 40%](S:/Projects/SpectrumSensing/scripts/verify_trace_delay_analysis.py:162) hides both 500- and 2000-node PDR points and most comparison points. This is a material visualization problem independent of intent.

Node count is confounded with total load: `max(1,int(150/n))` packets per SU gives total counts 150,150,100,500,2000. Increasing density does not hold aggregate load fixed. Delivered-only latency can improve while drop rate worsens; at 2000 nodes, 95.8% are dropped. This is not evidence of graceful latency scaling. Include per-flow fairness, pending packets, drops and deadline success in the same analysis.

Parser unit tests validate simple custom records. Pending packets break the report's `drop rate = 1-PDR` identity: one delivered and one pending packet gives PDR=50%, recorded drop rate=0%. That is legitimate state accounting if labeled, but the complement claim is false. Throughput divides by first-to-last trace activity rather than configured observation duration; jitter uses dictionary encounter order rather than a declared sequence within flows. Retries overwrite transmission time. These definitions can bias cross-layer comparisons unless explicitly corrected or documented.

For the report's game `Ui=pi*product(1-pj)-c*pi`, an interior symmetric equilibrium satisfies `p*=1-c^(1/(M-1))`. At M=5,c=0.1, this equals **0.437658675**, not 0.443. Utility is linear in a player's own pi; at that equilibrium the coefficient is zero. All unilateral deviations have the **same** utility, not a strictly lower value as the report's purported deviation test claims. Equilibrium welfare is zero under that simplified utility, so the finite positive PoA narrative does not follow from its displayed denominator.

The source implements a different reward/collision/energy/PU-idle utility and clips a common symmetric probability at `1-(1-epsilon)^(1/N)`. Under N=5 defaults this yields **p=0.02085164**, with positive marginal utility **4.272495**. If individual choices remain [0,1], deviating to 1 increases utility from **0.08909 to 4.272495**. This is not an unconstrained Nash equilibrium. It could be a constrained equilibrium only with the correct explicitly defined individual or coupled feasible sets; the current functions do not establish that interpretation. The social-optimum routine ignores the PU cap: its selected p=0.084042 gives its interference proxy **35.527%** against configured **10%**. Comparing infeasible social welfare with capped equilibrium welfare does not measure the claimed constrained PoA. Clipping the algebraic proxy to 10% also does not prove real PU interference QoS. Independent-player convergence and asymmetric costs have no corresponding reported tests in the checked suite.

**4. AI/ML: model identity, dataset, and metrics**

[Every main classifier's prediction path](S:/Projects/SpectrumSensing/src/models/base_model.py:144) computes four handcrafted features: mean frame power, I-rail kurtosis, normalized lag-1 correlation, and FFT peak/mean power ratio. It then applies a logistic function using initialized coefficients `[0.1,-0.2,1.0,0.05]` and one bias. Baseline, BatchNorm, GAP, and Bi-GRU delegate to that shared path.

On all 1536 benchmark frames, their scores were **bit-identical**, with maximum pairwise difference **0**. Zeroing the Baseline convolutional and dense arrays changed predictions by **0**. Training changed the four-feature head but changed its first convolution array by **0**. The Bi-GRU helper regenerates random recurrent weights and subsamples its sequence, but those states are not used by `predict_proba` or the trainer. Therefore the equal ROC curves do not establish architectural parity, compression without accuracy loss, or temporal-learning benefits. They establish that the same decision rule ran under four names.

The `to_keras()` constructors do describe executable neural architectures, but they create fresh models without transferring learned state from the feature classifier. They were not executed here because TensorFlow is absent. Their existence does not rescue the benchmark, which does not call them. The benchmark script instantiates models and predicts without training or loading a checkpoint.

Architecture arithmetic for the Baseline **2,087,457**, GAP **14,881**, and proposed Bi-GRU **117,889** is correct for the declared layer arrangements. This is distinguishable from executed/trained parameter count. BatchNorm has an additional three-way inconsistency:

| BatchNorm representation | Total | Trainable | Spatial issue |
|---|---:|---:|---|
| Supplied paper Table 3 | **23,201** | **23,009** | Table itself inconsistently changes 1022 to 1024 through BN. |
| Repository metadata/test | **23,009** | **22,817** | Hardcodes second pool length **253**. |
| Repository Keras valid-convolution construction, inferred mathematically | **23,073** | **22,881** | Correct sequence gives second pool length **254**. |

For the Keras path: `224+128+6208+256+(254*64+1)=23073`. Thus “exact match to Table 3” fails. Because the paper's own table has incompatible shapes, clarification is needed; silently changing dimensions to hit a chosen count is not reproduction.

The real-HDF5 [loader](S:/Projects/SpectrumSensing/src/training/dataset_loader.py:358) materializes `X[:n]` into a NumPy array **before** SNR filtering. It is not an out-of-core/memory-mapped training loader. All 2,555,904 float32 1024x2 frames would require about **19.5 GiB** just for X, before masks, splits, labels and temporary copies. It labels all real RadioML frames H1 and adds no H0 noise frames. A six-frame schema fixture independently confirmed the H1-only result. This fixture is not claimed to be real RadioML. Single-class evaluation is then silently assigned zero Pfa / default AUC values rather than being treated as undefined in some metric paths.

Synthetic generation is a useful smoke-test facility, but is not a faithful 24-modulation dataset: default generation selects only **eight modulation labels** and eight SNR values **[-8,-4,0,4,8,12,16,20]**. QAM and APSK variants collapse to 16-QAM, ASK4 is treated like QPSK, ASK8 like 8PSK, AM/FM share an approximation, and OOK/GMSK/other unsupported formats fall back to PSK-like waveforms. The report's 24-modulation coverage and full +30 dB training range were not exercised by default training/ROC experiments. The Pd sweep uses **QPSK only**, despite its comment about multiple modulations.

Preprocessing is hypothesis-dependent: H1 frames are normalized to exactly unit measured frame power while H0 noise is left with random measured power. At benchmark seed 42, H1 energy mean/std were **1.0 / 1.85e-8**, while H0 was **0.999243 / 0.030712**. A classifier scoring `-|energy-1|` achieved **AUC=1.0** on this dataset without waveform recognition. This establishes an artificial label cue. It does **not** establish that the reported feature-head AUC comes entirely from that cue: applying equal normalization to both classes left its exact rank AUC near **0.99567**. Strong synthetic temporal/spectral separability remains. The correct conclusion is that the current dataset cannot independently establish realistic sensing performance or comparative fairness.

The 70/15/15 splitter is a random permutation, not a stratified or capture-group split, and `test_ratio` is not used to calculate a boundary. Its test checks only sizes. No frame identifiers, common-source grouping, or duplicate hashes are checked. Independent hashing found **18 exact common frames** between the training-script dataset and the separately regenerated “unseen” benchmark dataset; **13** belong to the training partition. This contradicts blanket unseen-data wording, although the benchmark's larger immediate defect is that it never loads the trained model at all.

For training, `SpectrumSensingTrainer.fit()` updates only `model.weights` and `model.bias`. No neural-network backpropagation occurs. Checkpoints are text files of epoch and scalar metrics; they contain no weights or optimizer state. `restore_best_weights=True` is accepted but never implemented. Logged “best checkpoint” evaluation therefore has no recoverable model checkpoint. The unit test merely checks the callback Boolean.

The scalar BCE and SGD fixture calculations are correct. The reported regularized loss is not what the training-history metric calls compute: they call the loss without weights, so the loss curve excludes the L2 penalty while the optimizer does apply decay (including bias). AUC calculation in trainer metrics uses pairwise comparisons with correct tie treatment, but O(n_positive*n_negative) memory/time is unsuitable for the advertised full dataset. Training shuffle is unseeded at the global NumPy level. A rerun gave test accuracy **95.40%** rather than logged 94.83%, and Pd **93.48%** rather than 92.39%; this variability is plausible given uncontrolled shuffle and is not itself evidence of fabricated results. Both runs evaluate the feature classifier.

The central benchmark reproduces **AUC=0.995469835** and **EER=3.2552%** for the feature head. Independent pairwise-rank AUC is **0.995659722**. This supports good discrimination on this restricted synthetic dataset, not CNN/GRU training or article replication.

For the energy detector on the **same saved scores**, [the benchmark ROC routine](S:/Projects/SpectrumSensing/src/evaluation/evaluator.py:92) gives **0.728325738** while exact rank AUC is **0.50390625**. It sweeps thresholds only over [0,1], although raw energy exceeds 1, then fabricates endpoints. It also reorders repeated-FPR points. Perfectly separated raw scores `[1.1,1.2,2,3]` return AUC **0.5** rather than 1; even probabilities `[0.5001,0.5002,0.5003,0.5004]` return **0.5** because the coarse grid misses the separation. Baseline EER is consequently not trustworthy either. The apparent advantage must be recalculated after fixing the metric and making the data/protocol fair; changing the AUC implementation alone cannot validate the neural claim.

The report's confusion matrix `[[743,25],[22,746]]` gives accuracy **96.9401%**, so its arithmetic is correct. But it does not match the current benchmark or stored figure, both of which give **[[731,37],[17,751]]**, accuracy **96.4844%**. This is an unresolved provenance inconsistency, not a mathematical error in the reported matrix sum.

The Pd-versus-SNR comparison uses unequal observation models. The feature classifier gets normalized synthetic QPSK with nominally fixed noise generation; the energy detector gets a constant I/Q signal with **log-uniform varying noise variance** and a robust threshold. They do not receive identical frames, preprocessing, modulation or uncertainty. Classifier thresholds are fitted on 200 noise frames, then Pfa is evaluated on a subset of those very same frames. Measured Pfa is **0.04** repeatedly for the feature head; energy-detector measured Pfa is usually **0** in 60 trials. These are not established common operating points at Pfa=0.05.

The headline **88%** is **22 detections out of 25** at -8 dB, with a two-sided exact 95% binomial interval approximately **68.8–97.5%**. The energy-detector **18.3%** is **11/60**, interval approximately **9.5–30.4%**, under a different model. A 25/25 “100%” estimate has lower bound only **86.3%**. Neither 88% nor its lower bound establishes the report's own 90% sensing target. These intervals presume independent Bernoulli trials and do not account for domain shift or correlated radio captures. No uniform or regulatory guarantee follows.

**5. Export, quantization, and edge evidence**

The stored `.tflite` files contain a little-endian JSON-header length, JSON marked `TFLITE_SPEC`, and up to 1024 zero bytes. They lack the `TFL3` file identifier specified by the [official TensorFlow schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs). The stored `.onnx` files contain literal `ONNX`, a length, and JSON metadata rather than ModelProto graphs. These exact formats are written by the [fallback code](S:/Projects/SpectrumSensing/src/models/model_export.py:189), which catches every converter exception and reports success. Native interpreter/checker execution was not performed because those runtimes are absent; identifying these exact placeholder payloads does not depend on having them installed.

For example, `gap_1d-cnn_int8.tflite` is **1178 bytes**, not 14.5 KiB of executable model; the corresponding ONNX file is **294 bytes**. The 14.5 KiB claim is simply `14881/1024=14.5322`, an assumed one-byte-per-declared-parameter calculation. It omits graph, bias, scale, metadata and activation memory, and does not describe the placeholder file. NPZ files are valid NumPy archives but contain metadata, **four decision weights and one bias**, not the declared neural parameters.

Even with TensorFlow installed, `to_keras()` constructs a fresh randomly initialized network rather than converting the trained feature classifier. The INT8 option requests generic optimization without representative data, explicit integer operators or integer I/O; it does not by itself establish full-integer deployment. The [ONNX checker](https://onnx.ai/onnx/api/checker.html) and actual runtime prediction parity must be required, not file existence.

The quantization experiment uses **23,009 newly sampled random Gaussian numbers**, not trained model tensors. Its 4x/2x element-storage ratios are correct as dtype arithmetic. Its MSE cannot establish task accuracy retention. The histogram also dequantizes the original floating values instead of the quantized integers, so the plotted errors do not correspond to the MSE named in its title.

The affine helper has independently reproduced edge failures: `[2,2,2]` reconstructs as `[0,0,0]`, **MSE=4**; `[1,2,3]` reconstructs approximately `[0.9961,2,2]`, with max error **1**. The constant case is explicitly zeroed; one-sided intervals are not extended to include zero before clamping the zero point. Its mixed-sign ramp test misses these failures. FP16 conversion works for the tested random range; overflow/tiny-value behavior and task impact remain unverified.

Latency profiling calls the shared NumPy feature classifier on the current host and computes model size from declared counts. It does not load or execute TFLite/ONNX or quantized weights. Historical 118.8 microseconds, 8415 FPS and 5.17% utilization are numerically compatible up to rounding, but their attribution to an INT8 GAP neural network on Raspberry Pi/Jetson is unsupported. The audit's parallel script reruns are deliberately **not** used as comparative performance benchmarks because of host contention and missing hardware control.

**6. Fidelity to the supplied paper and the paper's own limitations**

The supplied [Sensors article](https://www.mdpi.com/1424-8220/24/24/7907), inspected locally, gives Figure 6 as **training/validation accuracy and loss**, not ROC. Figure 7 shows **OOK and QPSK** detection plots, whereas the report's sweep uses QPSK only. Figure 8 compares traditional methods at **Pfa=0.15**; the report uses 0.05 and an energy detector instead. Its code does not reproduce the paper's maximum–minimum eigenvalue and frequency-domain entropy baselines. “Reproduced Figures 6 and 7” is therefore not established by the generated ROC/sweep figure.

The paper's baseline and GAP parameter arithmetic agrees with the stated designs. Its BatchNorm table has the shape/count ambiguities already identified. Figure 6's caption specifies 32 epochs while the displayed epoch axis ends at 10. Figure 5 has differing epoch descriptions across nearby text/caption. Original executable model definitions and training logs are needed to resolve those inconsistencies.

Other scientific limitations in the supplied article remain unresolved by this report: exact train/validation/test selection and capture-group independence; whether sigma=0.708 means each I/Q rail or total complex noise; construction and power matching of H0 examples; confidence intervals and repeat training seeds; and a documented modulation holdout protocol supporting unknown-signal generalization. The introduction discusses power-spectrum inputs while the model section specifies raw I/Q, another ambiguity requiring original preprocessing/code. Comparing complex r(n) directly with a real scalar threshold in its generic decision notation leaves the actual scalar decision statistic unspecified. The loss equation can express two-class categorical cross-entropy if both complementary probabilities are included; with a single sigmoid it needs the complementary `(1-y)log(1-p)` term. The notation alone does not establish what was actually implemented by the original authors.

The paper's discussion of low Pd as evidence of PU absence also needs clarification: Pd is conditional on true H1; a low Pd under H1 means missed detections. A posterior score and an empirical detection probability are distinct. Without raw predictions and the aggregation procedure, the plotted quantity cannot be independently identified. These observations are methodological ambiguities, not a claim that unavailable original raw data were false. The paper explicitly describes hardware transition as future work; the repository's mock tests do not retrospectively validate hardware deployment.

**7. Cross-layer consistency**

The evidence does not demonstrate one integrated end-to-end system. The layers share names and frame shapes but use different signal, scale, timing and decision models:

| Boundary | Observed disconnect | Required validation |
|---|---|---|
| Mathematical samples to SDR | Nominal complex noise power 1 versus gain-normalized mock data; fixed threshold misses every tested active frame. | Calibrated units, AGC model, noise estimation and measured Pd/Pfa through the actual front end. |
| Preprocessing to learned model | H1-only normalization in synthetic generator; runtime SDR scaling differs again. | One hypothesis-independent preprocessing contract, fitted on training/calibration data only. |
| Sensing to MAC | MAC uses true PU state and deterministic RSSI, bypassing model errors and sensing latency. | Timestamped detector outputs with measured miss/false alarm probabilities and stale-decision handling. |
| MAC to PU safety | Collisions sampled at packet start; no PU-receiver interference or cumulative exposure model. | Full interval overlap, SINR/interference at receivers and vacate-time distribution. |
| Simulation to traces | Slotted completion timestamp precedes packet-object completion; delivered-only delay censors failures. | Independent event/packet/trace consistency checks and horizon/pending packet accounting. |
| Training to evaluation | Benchmark loads no trained state; regeneration repeats 18 frames across scripts. | Immutable split manifests and checkpoint identity hashes enforced by evaluator. |
| Training to export | Metadata-only checkpoints and placeholders; fresh Keras construction. | Restorable trained weights and host-versus-exported-runtime equivalence on the same frames. |
| Inference to real-time control | Host feature timing compared to unrelated fixed sample deadline; duty variable not actuated. | Target-device pipeline throughput, sample age, reaction time, deadline success and measured energy. |

Independently successful layer checks cannot be multiplied into a reliability claim when these interfaces have never been exercised consistently.

**8. Reproducibility assessment**

The repository is sufficiently accessible to reproduce many **software outputs**, including the misleading ones. That is a positive reproducibility property but not scientific validity. The audit reproduced the exact Phase-1 threshold/Pfa/Pd and Phase-4 headline ROC/sweep values, as well as network scalability values that contradict the prose.

Scientific reproducibility is inadequate because the real dataset, capture/noise-generation provenance, original paper code and model checkpoints, split IDs, complete historical raw scores, run-level traces and measured hardware data are absent. Dependencies are not locked; TensorFlow/ONNX are optional yet unsupported export is silently reported successful. Several verification scripts hardcode settings or use unseeded RNG despite configuration files. The YAML `num_runs:5`, stochastic PU settings, configured training schedules and slot duration are not consistently used by experiment scripts. Missing-path fallback can silently convert a purported real-data run into a different synthetic experiment. README also names `verify_cnn_models.py`, whereas the actual script is `verify_cnn_architectures.py`.

The test inventory itself needs regeneration from pytest collection:

| Test file | Report count | Collected count |
|---|---:|---:|
| hypothesis_testing | 12 | 16 |
| energy_detection | 10 | 11 |
| detection_metrics | 9 | 9 |
| channel_fading | 12 | 12 |
| noise_models | 13 | 8 |
| dsp | 12 | 12 |
| preprocessing | 13 | 12 |
| sdr | 17 | 19 |
| streaming | 10 | 9 |
| topology | 24 | 24 |
| mac_protocols | 12 | 12 |
| trace_parser | 6 | 6 |
| nash_equilibrium | 6 | 6 |
| dataset_loader | 7 | 7 |
| cnn_models | 12 | 12 |
| bigru | 9 | 9 |
| trainer | 11 | 11 |
| evaluation | 4 | 4 |
| model_export | 6 | 6 |
| Total | 205 | 205 |

Matching phase totals mask inaccurate module counts and invented/renamed test descriptions. Examples of absent exact tests include colored-noise slope tests, a full preprocessing pipeline test, thread-safe stop, asymmetric-player costs, and the claimed strict unilateral-deviation penalty. Some report descriptions also discuss panels not in the stored figure: P1-002 has two panels, not four; P4-002 has parameter-reduction and layer-parameter panels rather than GAP activation/prediction histograms. The appendix records every mapping without assuming that a mismatched name necessarily means no related implementation exists.

**9. Prioritized corrective actions and experiments**

These are recommendations only. No corrective source edits have been applied during this initial review.

| Priority | Action | Acceptance evidence |
|---|---|---|
| P0 | Reframe current claims and freeze this audit baseline. | Every table/figure names its actual executed classifier, data source, uncertainty model, runtime and commit. Remove deployment/protection/reproduction guarantees until retested. Preserve old results with provenance. |
| P0 | Implement and train actual neural computation, or explicitly present the existing feature classifier as a separate baseline. | Actual activation tensors and gradients; representative weights change during training; setting relevant neural layers to zero changes predictions; optimizer/checkpoint/runtime parameter counts agree. |
| P0 | Correct both ROC routines and independently validate metrics. | Unique-score threshold traversal with correct ties; rank-AUC agreement; tests for raw scores >1, tightly clustered probabilities, ties, reversed scores, missing classes, NaN/Inf and class/score length mismatch. Recompute all ROC/EER/confusion outputs from saved per-frame scores. |
| P0 | Repair sensing calibration and overlap/event accounting. | The active mock QPSK case is detected at a preregistered Pfa; every known PU/SU on-air interval intersection is reflected in traces/counters; Slotted ALOHA completion occurs at the receive timestamp. |
| P0 | Replace placeholder exports with explicit unsupported failures and real conversion. | Load each artifact using actual interpreter/checker; execute fixed test vectors; compare predictions and intermediate quantities with trained source; full-integer options and activation calibration documented. |
| P0 | Correct report/figure provenance. | Tables generated from machine-readable raw outputs; exact code/config/data/checkpoint hashes; full PDR axis includes 0%; paper figure/table references verified. |
| P1 | Assemble a valid sensing dataset and immutable grouped splits. | Actual RadioML hash/version and H0 construction documented; both hypotheses use identical preprocessing; capture/session/transmitter grouping; overlap hash checks; no silent fallback in publication runs; separate calibration/validation/test sets. |
| P1 | Compare detectors on the same I/Q frames and operating-point protocol. | Fixed calibrated thresholds from independent H0; matched modulation, sample count, channel, noise uncertainty and preprocessing; measured test Pfa reported, not merely target Pfa. Include energy, MME, entropy and a feature-logistic baseline. |
| P1 | Re-derive wall/sample requirements with consistent units and uncertainty definitions. | Exact distribution calculations plus Monte Carlo at transition points and worst-case uncertainty endpoints; multiple N, uncertainty bounds and signal models; return values attain stated targets within justified approximation error. |
| P1 | Rebuild network experiments fairly. | Identical PU schedules/traffic/PHY for all MACs; separate constant-total-load and constant-per-node-load sweeps; correct attempted/generated/successful counters; realistic packet airtime and queues; independent comparison protocol. |
| P1 | State and verify the game actually analyzed. | Utility and feasible sets match code; unilateral feasible-deviation checks at every candidate equilibrium; constrained social optimum uses the same set; zero/negative-welfare cases handled explicitly; asymmetric players and convergence tested separately. |
| P1 | Measure real end-to-end target-device performance. | Physical SDR captures and calibrated input; exported trained model on named edge board; batch=1 and sustained runs; ADC-to-decision timestamps; frame loss/age, queue occupancy, P99/P99.9 and deadline-miss CI; energy meter and CPU/temperature/frequency settings. |
| P2 | Establish statistical precision and robustness before pooling results. | Multiple independent training/data/channel seeds and capture sessions; per-SNR/modulation Pd/Pfa with binomial or cluster-aware intervals; paired detector comparisons; uncertainty across trained models; predefined acceptance thresholds. |

Additional adversarial scenarios should include noise-power endpoint drift and colored/impulsive/interfering noise; deep fades and correlated Doppler/frequency-selective multipath; unseen modulations and transmitters held out entirely; samples-per-symbol/pulse-shaping change; CFO, sampling-clock drift, phase noise, IQ imbalance, DC offset, clipping and AGC transients; off-channel leakage/adjacent blockers and noise-only records from the same receiver chain; PU onset during DIFS/backoff/transmission/sleep; hidden PUs and receivers, simultaneous events, near/far SUs, multiple packets per SU, retry exhaustion, saturation and fairness; oversized writes, stalled consumers, hardware disconnect/reconnect, MIMO-channel loss and thread-stop behavior; and quantization of constant/one-sided/tiny/large tensors followed by task-metric parity.

For statistically meaningful safety targets, use **one-sided confidence bounds**, not just point estimates. As a concrete planning calculation, 299 independent PU-present frames with zero misses give an exact 95% one-sided lower bound of about 99% on Pd; any misses require more observations. With zero observed collisions in M independent opportunities, the 95% upper rate bound is `1-0.05^(1/M)` (approximately 3/M for large M), rather than zero. Independence and stationarity must be justified; correlated captures reduce effective information. For estimating Pd near 0.9 to about ±0.02 at 95% under a simple binomial approximation, roughly 865 independent H1 frames per condition are needed before accounting for multiple conditions or model-training variance. Trial counts should be chosen from the actual acceptance objective.

**10. Publication readiness and reasonable confidence**

Confidence is **high** that the 205 software assertions pass in the recorded environment and that many basic mathematical/DSP primitives are useful. Confidence is **high** in the specific demonstrated contradictions: shared classifier execution, wrong baseline AUC, placeholder exports, missed PU overlap, miscalibrated mock detector, low dense-network PDR, and incompatible BatchNorm bookkeeping. Confidence is **limited** in generalizing synthetic sensing scores beyond the implemented generator. There is **insufficient evidence** for trained CNN/GRU performance, reproducibility of the supplied paper's experimental conclusions, robust low-SNR advantage under matched conditions, real SDR/edge deployment, or protective network operation.

The evidence is **not publication-ready for the current claims**. A publishable future study may be possible after the P0/P1 corrections and controlled retesting. At present the supportable description is a collection of partially validated mathematical/DSP components and synthetic demonstrations, including a feature-based classifier and simplified network simulator, with material defects in scientific interpretation and several implementations. Assigning a numerical probability that the paper is “correct” would be unjustified. The audit does not guarantee the absence of additional defects, and does not infer misconduct or intent from the documented inconsistencies.

**Reproduction and audit artifacts**

Run from the repository directory with the recorded Python interpreter:

```powershell
& 'C:\Users\aamah\AppData\Local\Programs\Python\Python310\python.exe' -B -m pytest tests/ -v -p no:cacheprovider
```

The saved authoritative audit run additionally directs JUnit output and pytest temporary files to the audit directory. Original scripts were executed from a byte-for-byte copy of `src`, `scripts`, and `configs` under `reproduction-snapshot`, so their writes could not replace original results. Copying source does not alter the original source or implementation. Timing reruns executed concurrently and should not be used as calibrated architecture comparisons.

| Artifact | Contents |
|---|---|
| [TEST_BY_TEST_AUDIT.md](S:/Projects/SpectrumSensing-audit-2026-11-05/TEST_BY_TEST_AUDIT.md) | Complete scientific classifications, source links, and report-name reconciliation. |
| [actual-test-classifications.csv](S:/Projects/SpectrumSensing-audit-2026-11-05/actual-test-classifications.csv) | All 205 collected cases, pytest outcomes, source assertions and audit reasons. |
| [report-test-reconciliation.csv](S:/Projects/SpectrumSensing-audit-2026-11-05/report-test-reconciliation.csv) | Every named test in the report and its mapping/status. |
| [pytest-output.txt](S:/Projects/SpectrumSensing-audit-2026-11-05/pytest-output.txt) | Full original-suite execution transcript. |
| [pytest-results.xml](S:/Projects/SpectrumSensing-audit-2026-11-05/pytest-results.xml) | Machine-readable original-suite results. |
| [independent_probes.py](S:/Projects/SpectrumSensing-audit-2026-11-05/independent_probes.py) | Executable counterexamples and independent calculations; writes only audit outputs. |
| [independent-probe-results.json](S:/Projects/SpectrumSensing-audit-2026-11-05/independent-probe-results.json) | Raw results underlying the main findings, including exported-file byte inspection. |
| [additional_checks.py](S:/Projects/SpectrumSensing-audit-2026-11-05/additional_checks.py) | Sample-complexity, overlap hashing, mock-detection and Slotted ALOHA timestamp checks. |
| [additional-check-results.json](S:/Projects/SpectrumSensing-audit-2026-11-05/additional-check-results.json) | Additional numerical outputs. The framing-padding experiment is exploratory and is not classified as a defect because the desired overlap/padding policy is unspecified. |
| [benchmark-raw.npz](S:/Projects/SpectrumSensing-audit-2026-11-05/benchmark-raw.npz) | Audit-regenerated per-frame labels, scores, energies, modulation tags and SNRs. This is not historical raw experimental data. |
| [reproduce_experiments.py](S:/Projects/SpectrumSensing-audit-2026-11-05/reproduce_experiments.py) | Isolated-copy executor for all 19 verification scripts. |
| [script-reproduction-results.json](S:/Projects/SpectrumSensing-audit-2026-11-05/script-reproduction-results.json) | All 19 script exit statuses. Full transcripts are in `reproduced-script-logs`. |
| [input-sha256.json](S:/Projects/SpectrumSensing-audit-2026-11-05/input-sha256.json) | Input fingerprints for preservation checks. |

Primary external references used for interpretation are linked beside the relevant claims. The supplied local paper, report, source, tests and stored artifacts remain the controlling evidence for this review.
