---
name: ai-ml-sensing
description: Train, evaluate, and export native spectrum sensing models. Use for I/Q dataset splits, CNN checkpoints, TFLite INT8 conversion, runtime score comparisons, and detector threshold calibration.
---

# AI/ML spectrum sensing

- Use native Keras models and `KerasSensingTrainer` for neural training. The historical NumPy feature classifier and `SpectrumSensingTrainer` serve as an explicit baseline. Record model config, actual parameter counts, seed, and checkpoint hash. The recovery records document architecture differences from the paper.
- Preserve source or capture-session groups across training, model selection, threshold calibration, and sealed test partitions. Modulation and SNR stratification can balance conditions within that split. Record unresolved group provenance and checks for duplicate frames. RadioML modulation frames are signal-present inputs. Document the independent noise-only source.
- Export the trained graph. Select representative data from the training side for post-training integer conversion. That data sets tensor ranges. Use separate confirmed H0 data to calibrate the deployed detector's operating threshold.
- Inspect interpreter operator support, integer input/output types, tensor scales, zero points, and saturation before labeling an export full INT8. Compare FP32 and dequantized INT8 scores and decisions on paired frames. Log each runtime's distinct score values, its threshold, and the comparator so ties can be replayed.
- Compare a fixed nominal INT8 threshold, ordinary independent INT8 calibration, and the proposed rule using the same sealed frames and total noise-only calibration budget. Report Pd and Pfa counts by SNR, gain, and session. Record uncertainty and score or decision changes. For the pre-USRP study design, see [Calibration novelty review](../calibration-novelty-review/SKILL.md).

Use [experiment configs](../../../experiments/configs/), the [recorded INT8 pilot](../../../results/recovery/R005-runtime-stress/), and the existing training, export, and data-protocol tests. Smoke runs establish that a model trains and an export executes. Publication claims require adequate independent evaluation.

References: [LiteRT quantization skill](https://github.com/google-ai-edge/LiteRT/blob/30da00083b65151ea64af4e22280d232603b15f3/litert_quantizer/litert_quantizer/experimental/skills/auto_quant/SKILL.md) and [LiteRT integer quantization guide](https://ai.google.dev/edge/litert/models/post_training_integer_quant).
