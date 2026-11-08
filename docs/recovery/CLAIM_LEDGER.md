# Scientific recovery claim ledger

Baseline: `cba90a1522c3f4a8b3f839b1750a370713f61a50`; recovery branch `codex/scientific-validation-recovery`.
Started 2026-11-05. Historical reports, plots, experiment logs and exports describe the baseline and are preserved. They are not current validation certificates.

| ID | Claim / scope | Current evidence status | Acceptance requirement |
|---|---|---|---|
| C01 | Empirical binary ROC/AUC is correct | Verified within defined algorithm/fixture scope; full local suite now 325 passed | Exact ties/endpoints/ranking and EER interpolation; invalid data rejected; independent rank agreement on fixtures and frozen raw scores |
| C02 | Noise/SNR/gain theory matches generation | Partially Verified: corrected conditional laws, gain and sufficient sample bound | Broader distribution checks, correlated noise and stated fading assumptions; sufficient bound is not a minimum sample count |
| C03 | Physical sensing is calibrated | Partially Verified: independent calibrated mock case passes; original evidence preserved | Physical ADC/gain/receiver calibration and capture provenance remain unverified |
| C04 | Streaming is reliable and real-time | Partially Verified for buffer invariants; device timing Inconclusive | Acquisition timestamps, sustained throughput and deadline/drop statistics at actual sample rates |
| C05 | PU collisions/exposure are correctly accounted for | Partially Verified: known positive-duration overlap, ties and completion/horizon cases pass | Explicit receiver interference criterion if physical protection claimed; this is a simplified global temporal collision model |
| C06 | MAC superiority and scalability | Not established; report contradicts stored simulator results | Matched PU/traffic/channel conditions; actual offered load; drops/pending/delays/fairness and uncertainty |
| C07 | Dataset represents credible independent H0/H1 tests | Partially Verified for bounded ingestion/group isolation on local fixtures; real data Inconclusive | Verified source/H0 provenance; hypothesis-independent processing; real grouped immutable four-role manifests |
| C08 | Actual CNNs/GRU are trained and evaluated | Partially Verified: three native CNN smoke runs, real weights/checkpoints and executable graphs; historical feature path explicitly labelled | Full identified training/evaluation data and repeated model seeds; GRU extension remains unverified |
| C09 | Reproduction of the supplied paper | Inconclusive; BN graph deviation documented (23,073 parameters versus paper 23,201) | Matching or explicitly justified architecture/protocol deviations; correct figure references; conventional baselines |
| C10 | Low-SNR/uncertainty superiority | Inconclusive; fixed-calibration gain robustness fails in tested smoke conditions | Matched I/Q, fixed calibration, actual Pfa, per-condition intervals and model-seed variation |
| C11 | Executable quantized edge models | Partially Verified: nine real TFLite artifacts executed; INT8 sensing parity fails for GAP threshold transfer; ONNX unverified | Per-runtime held-out calibration and sensing criteria, native checker/runtime verification for ONNX, measured device memory |
| C12 | Hardware deadlines and energy savings | Not established | Named physical devices; ADC-to-decision age and sustained runtime; measured energy, rather than a mock control variable |
| C13 | Constrained Nash equilibrium / PoA | Deferred optional extension | Defined utility/feasible sets; no improving feasible unilateral deviation; social optimum over the same feasible set |
| C14 | GAP INT8 can reuse native sensing threshold | Failed for this synthetic checkpoint/fixture: Pfa 60/300 = 20%, 80/600 changed decisions | Separate calibration and quantization-aware/mixed-precision experiments; investigate output score ties |
| C15 | Nominal calibrated thresholds remain valid under +1 dB receiver gain shift | Failed for this smoke experiment: GAP native Pfa 170/300 = 56.67% | Gain/noise drift monitoring, operating limits and independent recalibration policy |

Immediate research deliverable: a reproducible Baseline CNN versus energy and explicitly named feature-classifier comparison. BN/GAP follow after its computation/data checks; Bi-GRU/game theory remain separate optional extensions. Original scientific conclusions are outcomes to be tested, not acceptance values to force.

Dataset source supplied by user: https://www.kaggle.com/datasets/pinxau1000/radioml2018 . User subsequently instructed work without downloading the dataset. No RadioML data was downloaded; full-data identity/schema/performance and H0 provenance remain unverified. Candidate local model files can be exercised (one through an explicit in-memory null-metadata compatibility mapping with bitwise weight checks), but are not validated reproductions.

Current detailed results, limits and next steps: `RECOVERY_PROGRESS.md`. Classifications describe specified evidence scopes, not an absolute guarantee of correctness.
