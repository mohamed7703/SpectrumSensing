# R009: Detector v2 with receiver impairments

Date opened: 2026-11-20. Part of the pre-USRP v2 program (Decision 004), experiment E2.

Goal: train a detector strong enough for a fair test, once on clean AWGN and once with session-level receiver impairments (DC offset, IQ imbalance, coloured noise renormalised to unit noise power, 12-bit ADC). Choose the lab model with a rule frozen in `experiments/configs/detector_v2.yaml` before any candidate trains.

Rules: tests first; no random gain augmentation (gain drift stays the experimental variable); seeds 29031-29035 (AWGN) and 39031-39035 (impaired) stay disjoint from v1 and from E4; every candidate gets a new output path; evidence is synthetic CPU.

Repo skills: `ai-ml-sensing`, `math-foundations`, `physical-sdr`.

Steps:
1. `src/preprocessing/receiver_impairments.py` with tests (`tests/test_receiver_impairments.py`).
2. Timing and learning-rate search on a development seed (29030, not a candidate), validation data only.
3. Freeze `detector_v2.yaml` and push before training any candidate.
4. `scripts/train_detector_v2.py`, train 10 candidates, apply the selection rule mechanically.
5. `results/calibration/detector-v2/` (REPORT.md, manifest.json, selection.csv) and `experiments/logs/detector-v2.md`.
