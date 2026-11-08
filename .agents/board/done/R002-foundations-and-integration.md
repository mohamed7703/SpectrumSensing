# R002: Repair foundations and deterministic integration defects

Completed narrow implementation scope: 2026-11-05. Branch: codex/scientific-validation-recovery.

Acceptance: gain statistics, signal/noise conventions and detector distributions agree; sample requirements meet defined targets; mock calibration controls both detection and false alarms; oversized writes preserve buffer state; PU-mid-packet overlap and slotted completion are correctly accounted for.

Regression evidence must be captured before each fix. Historical reports/plots/logs remain unchanged. Full dataset source was supplied by the user; physical hardware/GPU availability is not yet established.

Evidence: 16 foundation regressions originally failed, four of six initial physical regressions failed, and ten initial network regressions failed. Corrected foundation/physical/network checks passed; full suite at this milestone was 298 passed. New `results/recovery/R002-foundations.json` records counts, confidence intervals, exact distribution calculations, source/config hashes and known PU overlap durations. This does not establish RF protection, real-time hardware, MAC scalability, or paper reproduction.
