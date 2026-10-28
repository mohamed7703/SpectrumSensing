---
id: P4-006
title: "Implement Model Export & Quantization for Edge SDR Deployment"
phase: 4 - AI/ML Sensing
status: done
priority: high
depends_on: [P4-005]
tags: [export, quantization, tflite, onnx, edge, embedded]
---

## Objective
Implement model exporter and quantization tool converting trained models into TFLite (.tflite),
ONNX (.onnx), and Keras SavedModel (.h5) formats, with FP16 and INT8 post-training quantization
for embedded execution on Raspberry Pi and NVIDIA Jetson.

## Acceptance Criteria
- [x] Model exporter in `/src/models/model_export.py`
- [x] Post-training INT8 / FP16 quantization pipeline
- [x] Inference latency benchmarks across exported formats
- [x] Unit tests in `/tests/test_model_export.py`
