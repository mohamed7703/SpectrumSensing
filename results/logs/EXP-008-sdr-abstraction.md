# Experiment Log: EXP-008 — Unified SDR Hardware Abstraction Layer

- **Task**: P2-003: Implement Unified SDR Hardware Abstraction Layer
- **Date**: 2026-10-28
- **Status**: PASSED

## Objectives
1. Implement polymorphic `BaseSDR` abstract interface covering frontend lifecycle (`configure`, `start_rx`, `read_samples`, `stop_rx`, `close`).
2. Provide concrete drivers for all 6 SDR platforms specified in the deployment plan:
   - RTL-SDR Blog V3 (8-bit ADC, 2.4 MSPS BW, 24-1766 MHz)
   - HackRF One (8-bit ADC, 20 MSPS BW, 1-6000 MHz)
   - PlutoSDR ADALM-PLUTO (12-bit ADC, 20 MSPS BW, 325-3800 MHz)
   - LimeSDR-USB (12-bit ADC, 61.44 MSPS BW, 0.1-3800 MHz, 2x2 MIMO)
   - USRP B210 (12-bit ADC, 56 MSPS BW, 70-6000 MHz, 2x2 MIMO)
   - USRP X310 (14-bit ADC, 160 MSPS BW, 10-6000 MHz, 2 channels)
3. Implement high-fidelity `MockSDR` generator for Hardware-in-the-Loop (HITL) CI/CD testing without physical USB hardware.
4. Establish unified configuration management via `configs/sdr_config.yaml` and `SDRFactory`.

## Results
- **Unit Test Coverage**: 19/19 tests passed in `tests/test_sdr.py`; 99/99 tests passed across entire testbed.
- **Hardware Abstraction Verification**:
  - `RTLSDRDriver`: Accurately clamped requested gains to nearest discrete vendor step (`20.7 dB` for `21 dB`), enforced 2.4 MSPS ceiling and 24-1766 MHz tuning bounds.
  - `HackRFDriver`: Verified LNA (0-40 dB, 8 dB steps) and VGA (0-62 dB, 2 dB steps) split behavior.
  - `PlutoSDRDriver`: Configured 12-bit sample quantization and IIO buffer streaming.
  - `LimeSDRDriver`: Verified 2x2 MIMO dual-channel coherent streaming matrix shape `(2, 4096)`.
  - `USRPDriver`: Model differentiation between B210 (12-bit, 56 MSPS) and X310 (14-bit, 160 MSPS).
  - `MockSDR`: Emulated tone, QPSK, OFDM, and AWGN noise modes with Rayleigh and Rician fading.
- **Plot Saved**: `results/plots/p2_003_sdr_abstraction.png`.
