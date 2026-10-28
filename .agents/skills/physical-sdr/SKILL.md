---
name: physical-sdr
description: >-
  SDR hardware integration and physical layer signal processing: I/Q data
  capture and streaming, DSP transforms (FFT, PSD, STFT), SDR hardware
  abstraction for RTL-SDR/HackRF/PlutoSDR/LimeSDR/USRP, real-time buffering,
  and signal preprocessing pipelines. Use when working on hardware interfaces,
  signal processing, or transitioning from simulation to over-the-air testing.
---

# Physical / SDR Layer Skill

## Scope
Bridges the mathematical models with real RF hardware. Covers I/Q signal processing,
DSP transforms, and SDR hardware abstraction for the cognitive radio testbed.

## Workflow

### Step 1: DSP Transform Library
Implement core signal processing in `/src/dsp/`:
- `fft_utils.py` — FFT, IFFT, frequency domain analysis
- `psd_estimator.py` — Power Spectral Density via Welch's method
- `spectrogram.py` — STFT-based spectrogram generation for CNN input
- `wavelet_transforms.py` — Wavelet decomposition for multi-resolution analysis
- `filters.py` — Bandpass, lowpass, matched filters

### Step 2: I/Q Signal Processing
Implement in `/src/preprocessing/`:
- `iq_normalizer.py` — I/Q scaling, DC offset removal, power normalization
- `noise_injector.py` — Add calibrated AWGN at specific SNR levels
- `augmentation.py` — Signal augmentation (time shift, phase rotation, fading)
- `frame_builder.py` — Segment continuous I/Q streams into 1024×2 frames

### Step 3: SDR Hardware Abstraction
Implement in `/src/sdr_interface/`:
- `base_sdr.py` — Abstract base class for SDR devices
- `rtlsdr_driver.py` — RTL-SDR Blog V3 (8-bit, ~2.4 MHz BW)
- `hackrf_driver.py` — HackRF One (8-bit, 20 MHz BW)
- `plutosdr_driver.py` — ADALM-PLUTO (12-bit, 20 MHz BW)
- `limesdr_driver.py` — LimeSDR-USB (12-bit, 61.44 MHz BW, 2×2 MIMO)
- `usrp_driver.py` — USRP B210/X310 (12-14 bit, 56-160 MHz BW)
- `stream_handler.py` — Real-time buffering, chunking, overflow management

### Step 4: SDR Configuration
Create configs in `/configs/`:
- `sdr_config.yaml` — Center frequency, sample rate, gain, FFT bins per device
- `capture_config.yaml` — Duration, file format, buffer sizes

### Step 5: Latency Benchmarking
- Measure frame processing time per SDR platform
- Target: <2.3ms per 1M-sample frame on GPU
- Document: 5ms/frame CPU ceiling on Raspberry Pi + LimeSDR
- Implement duty cycle reduction via predictive scheduling

## Hardware Tier Mapping
| Tier | SDR | ADC | BW | Role |
|------|-----|-----|-----|------|
| Edge Sensor | RTL-SDR V3 | 8-bit | 2.4 MHz | Distributed cooperative sensing |
| Portable TX | HackRF One | 8-bit | 20 MHz | Jamming emulation, freq hopping |
| Mobile SU | PlutoSDR | 12-bit | 20 MHz | UAV tracking, SU emulation |
| MIMO Node | LimeSDR-USB | 12-bit | 61.44 MHz | Cellular emulation, PHY security |
| Lab Grade | USRP B210 | 12-bit | 56 MHz | Full-duplex MIMO, AI inference |
| Enterprise | USRP X310 | 14-bit | 160 MHz/ch | Wideband I/Q streaming to GPU |

## Verification
- Tests in `/tests/test_dsp.py` and `/tests/test_sdr.py`
- Synthetic I/Q → DSP → spectrogram pipeline must reproduce known signal patterns
- Hardware tests require physical SDR devices connected
