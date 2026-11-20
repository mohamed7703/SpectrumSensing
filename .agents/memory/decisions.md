# Decisions Log
> Architectural and design decisions made during the project.

## Format
```
### Decision [ID]: Title
- **Date:** YYYY-MM-DD
- **Context:** Why the decision was needed
- **Decision:** What was decided
- **Alternatives Considered:** Other options
- **Rationale:** Why this option was chosen
```

## Decisions

### Decision 001: Execution Order — Bottom to Top
- **Date:** 2026-10-28
- **Context:** Need to determine which phase to implement first
- **Decision:** Math Foundations → Physical/SDR → Network Sim → AI/ML
- **Alternatives Considered:** Top-down (AI first), parallel execution
- **Rationale:** User preference; math validates physics, physics validates network, network validates AI. Each layer builds on the one below.

### Decision 002: Framework Choice — TensorFlow/Keras
- **Date:** 2026-10-28
- **Context:** Paper used TensorFlow/Keras on Kaggle (Tesla P100)
- **Decision:** Use TensorFlow/Keras as primary DL framework
- **Alternatives Considered:** PyTorch
- **Rationale:** Direct compatibility with paper's code and architecture specifications


### Decision 003: Remote Repository Provisioning
- **Date:** 2026-10-28
- **Context:** User requested public GitHub tracking of codebase and mathematical verifications
- **Decision:** Created repository `mohamed7703/SpectrumSensing` on GitHub with `main` tracking branch
- **URL:** https://github.com/mohamed7703/SpectrumSensing


### Decision 004: Pre-USRP v2 program environment, scope and git workflow
- **Date:** 2026-11-20
- **Context:** Pre-USRP v1 showed the linear gain-aware INT8 rule gives no benefit; the replay script locks only two methods; the v1 detector (about 60% Pd at 0 dB) is weak. The documented Python 3.10 environment was absent on the working machine.
- **Decision:** Run the full pre-USRP v2 program (plan: replay comparators, detector v2, INT8 resolution study, frozen benchmark v2, hardware-sized dry run, USRP protocol v2). Use a uv-managed Python 3.10 venv at `.venv` with `requirements-recovery.txt` (TF 2.20.0, Keras 3.12.4). Work on branch `codex/pre-usrp-v2` and commit and push to `mohamed7703/SpectrumSensing` after each milestone (user-authorized 2026-11-20 for this branch only). Train both AWGN-only and impairment-augmented detectors; choose the lab model by a rule frozen in YAML before training.
- **Alternatives Considered:** Existing Python 3.12 (rejected: deviates from the recorded environment); minimum-before-lab scope (rejected: skips the resolution study and benchmark v2); local-only commits.
- **Rationale:** Keeps results comparable with the 331-test baseline, attacks the tie-lattice cause instead of only reporting it, and makes the lab day produce publishable numbers. No RadioML download is part of this program.