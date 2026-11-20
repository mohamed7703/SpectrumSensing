---
name: physical-sdr
description: Work on SDR acquisition, raw I/Q and DSP, mock receiver tests, streaming loss and latency, and independent USRP calibration and validation captures.
---

# Physical SDR and I/Q capture

- Use `src/sdr_interface/` for hardware abstraction and buffering and `src/dsp/` and `src/preprocessing/` for transformations. Keep the raw amplitude and verify scaling, endianness, I/Q order, and clipping. Disable automatic gain during fixed-gain comparisons or record its behavior as an experimental variable. Per-frame normalization can erase the energy changes this study measures.
- When collecting labeled data, document how each H0 interval was confirmed noise-only and how each H1 signal was generated or identified. Record center frequency, sample rate, bandwidth, gain, receiver model and firmware, timestamp, capture and session identifiers, and connection or RF conditions. Store raw I/Q with a sidecar such as SigMF and preserve ground-truth labels with the capture.
- For each gain and SNR, verify a short capture first. Check expected sample count, device timestamps, streaming overflows, host buffer drops, and ADC clipping. Document where SNR was measured and how gaps or dropped samples were handled before framing. Keep session groups intact when splitting frames.
- Calibrate using noise-only observations from independent calibration sessions. For cross-day transfer, lock the threshold before scoring the held-out day. Track receiver gain changes separately from input SNR.
- Report acquisition-to-decision latency with a defined clock and boundary, sustained sample rate, and dropped samples when making hardware timing claims. A mock run supports mathematical and software checks. Physical timing, energy, and robustness require measurements on a named device.

Use [receiver settings](../../../experiments/configs/sdr_config.yaml) and the [draft capture protocol](../../../docs/calibration/README.md). Run the existing SDR, DSP, streaming, and physical recovery tests for software changes.

References: [Software Defined Radio skill](https://github.com/briannasywa/sdr-skills/blob/f6aab2dfbe6d0b502ac30cba6f869684b7cfce6e/skills/software-defined-radio/SKILL.md), [Ettus UHD streaming guide](https://github.com/EttusResearch/uhd/blob/d1be63490e3f3df22ff2f4ea1315f9aada468562/host/docs/driver_usage/stream.dox), and [SigMF format](https://github.com/sigmf/SigMF).
