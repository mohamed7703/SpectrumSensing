# R008: Complete the replay comparators

Date opened: 2026-11-20. Part of the pre-USRP v2 program (Decision 004), experiment E1.

Goal: make `scripts/usrp_calibration_replay.py` lock every compared method on the calibration day: ordinary INT8, linear, fixed nominal INT8, FP32, INT8 with the FP32 threshold copied, and the gain-aware energy detector. Add `--native-model` and `nominal_gain_db`, a lock format 2, and a per-frame `scores.npz` sidecar.

Rules: tests first in `tests/test_calibration_handoff.py`; reuse `calibrate_score_threshold`, `tflite_scores`, `predict` and `compute_energy_statistic`; keep `results/calibration/2920-dry-run-v1/` immutable. Repo skills: `physical-sdr`, `ai-ml-sensing`.