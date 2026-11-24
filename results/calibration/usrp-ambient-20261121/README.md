# USRP-2920 ambient capture and calibration replay

At the reported 915 MHz, 1 MS/s and 10 dB gain, 19,820 of 197,236 held-out frames shared the INT8 decision cutoff. This pilot uses one generator-off antenna recording. Unknown transmissions may be present. [Full report](analysis/REPORT.md) · [Twenty calibration selections](analysis/selection_sensitivity.csv) · [Next lab experiment](../../../docs/PROJECT_STATUS_AND_PLAN_2026-11-24.md)

## Data

The [GitHub release archive](https://github.com/mohamed7703/SpectrumSensing/releases/download/usrp-ambient-20261121/usrp-ambient-20261121-iq.zip) contains the complete original recording, losslessly compressed. Extract it into this directory for replay.

The original `ampient_turned_off.iq` is **1,855,763,448 bytes**. Retrieve it from the [original Drive file](https://drive.google.com/file/d/1_pmfCVqJa41YR3L2EwPTCnhy2a_BL8Po/view) into this directory, or run `python fetch_capture.py` here to resume a range-based download. Verify its SHA-256 against `9f637ae161606ecd6f17284902597867ebef877f48fae5a8172e70449fc34be1` before replay. GNU Radio wrote little-endian float32 I and Q values. The archived [capture record](analysis/capture.json) includes the sample count and excerpt hash.

The [diagnostic recordings](diagnostics/) include the original early receiver check, tone-dominated capture and independently supplied second at 60 s. The [prefix analysis](prefix_analysis/) records preliminary sampling during transfer. Use the [dense replay totals](analysis/all_test_summary.json) for the final test counts.

## Reproduce

Activate the repository's research Python environment and change to this directory. The scripts load the matching FP32 checkpoint and INT8 export from [the lab handoff](../../../docs/calibration/usrp-2920/). Their SHA-256 identities and quantization settings are in [inference.json](analysis/inference.json). The commands regenerate the tracked analysis outputs, so use a separate checkout to compare a rerun to this archive.

```sh
python ambient_quality.py
python ambient_scores.py
python all_frame_replay.py
python plot_and_check.py
python finish_report.py
python check_calibration_selection.py
```

The scoring script checks the recovered calibration against the archived [early lock](analysis/early_lock.json). The final report generator verifies the separately supplied 60 s excerpt and recounts decisions from the saved scores. The sensitivity script uses seed 20261124 and twenty disjoint sets of 200 frames. Its [sample indices and calibration scores](analysis/selection_calibration_scores.npz) allow independent checks of the frame selection.
