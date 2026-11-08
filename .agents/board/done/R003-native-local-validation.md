# R003: Native local validation without downloading RadioML

Completed local computation/validation scope on 2026-11-06. Scientific generalization and deployment criteria are explicitly unresolved; see backlog R004.

User explicitly instructed: work without downloading the dataset. No RadioML download is authorized under this constraint. Continue local fixtures and existing artifacts only.

Implemented native Keras SGD, convolutional weight-change checks, best-model selection and full model/optimizer checkpoints. Three CNN graphs executed on independent-seed synthetic train/validation/calibration/test fixtures. FP32/FP16/INT8 files loaded and executed with actual interpreters. `results/recovery/R003b-native-smoke/record.json` is the completed measurement record; first R003 export attempt failed on a Keras private-API incompatibility and is preserved separately.

Scientific result is not forced positive: GAP INT8 changed 16/100 initial parity decisions, then 80/600 full-set decisions at its native calibrated threshold, with 60/300 H0 false alarms. Separate INT8 calibration reduces false alarms but lowers 0 dB Pd from 97/100 to 69/100. Gain-shift tests are in R005: native GAP Pfa rises from 5/300 to 170/300 at +1 dB common gain. ONNX optional dependencies/runtime are not established; exporter now fails rather than writing stand-ins.

Local artifact inspection is in `results/recovery/R004-local-assets.json` and R004b. Original hashes/source are preserved without notebook execution. Explicit in-memory null-metadata adaptation permits the historical GAP probe; all 422,209 values match original HDF5 weights/state bitwise. R006 verifies 97 protected historical files, 9 original local assets, 325 passing test cases, 9 exports and nonzero BCE gradients in all six tested convolutional kernels. R007 independently recounts fixed scores/AUC for 25 conditions. Dataset/source-group provenance and physical SDR validation remain unverified.
