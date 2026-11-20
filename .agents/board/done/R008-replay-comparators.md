# R008: Complete the replay comparators

Date opened: 2026-11-20. Part of the pre-USRP v2 program (Decision 004), experiment E1.

Goal: make `scripts/usrp_calibration_replay.py` lock every compared method on the calibration day: ordinary INT8, linear, fixed nominal INT8, FP32, INT8 with the FP32 threshold copied, and the gain-aware energy detector. Add `--native-model` and `nominal_gain_db`, a lock format 2, and a per-frame `scores.npz` sidecar.

Rules: tests first in `tests/test_calibration_handoff.py`; reuse `calibrate_score_threshold`, `tflite_scores`, `predict` and `compute_energy_statistic`; keep `results/calibration/2920-dry-run-v1/` immutable. Repo skills: `physical-sdr`, `ai-ml-sensing`.
## Result (2026-11-20)

Done. `fit` locks six methods per gain (`ordinary`, `linear`, `fixed`, `fp32`, `int8_copied_fp32`, `energy`) in lock format 2 with both model hashes and `nominal_gain_db`; each method is charged budget x gains noise-only frames (`h0_frames_charged`). `replay` scores INT8, FP32 and energy once per batch, rejects non-finite scores, and writes a per-frame score archive whose SHA-256 is recorded. `--native-model` is required. Format 1 locks are rejected.

Review: `ecc:python-reviewer` found one HIGH, a mis-charge when more than three gains are recorded (hard-coded 3x budget), fixed and covered by `test_charge_follows_the_number_of_gains` (verified to fail on the old behavior). Also added tests for surplus frames, a gain split over intervals, every method recounted from the archive, and the NaN guard. Tests: 343 passed (331 baseline plus 12 new).

Evidence setting: synthetic fixtures only. The replay tests pair an untrained FP32 graph with the committed seed-19031 INT8 model to test plumbing, so their detection counts are not scientific results. Not changed: `results/calibration/2920-dry-run-v1/` stays as the format 1 record. Open from review, deliberately skipped: archive keys do not record per-interval provenance.