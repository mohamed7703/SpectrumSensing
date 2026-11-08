# R001: Correct measurement primitives

Status: software correction verified on 2026-11-05; scientific experiment claims remain pending.

- New audit regression suite first failed on 32/41 cases before correction.
- Exact tie-preserving ROC used by mathematical, benchmark and trainer metrics.
- Interpolated EER, validated labels/scores, undefined missing-class rates, scalable AUC.
- Exact binomial interval helpers with elementary endpoint checks.
- Targeted checks: 77 passed. Full suite: 258 passed.
- Evidence: external audit directory contains `metric-regressions-before.*`, `metrics-after.xml`, `recovery-metrics-full-suite.xml`.
- Next: source/signal conventions and detector target attainment; physical calibration and event accounting.
