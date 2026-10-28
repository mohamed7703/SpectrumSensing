---
id: P2-003
title: "Implement Unified SDR Hardware Abstraction Layer"
phase: 2 - Physical / SDR Layer
status: backlog
priority: high
depends_on: [P2-002]
tags: [sdr, hardware, abstraction, rtl-sdr, hackrf, plutosdr, limesdr, usrp]
---

## Objective
Implement an object-oriented hardware abstraction layer (HAL) providing a polymorphic interface
across all 6 SDR platforms specified in the deployment plan, with hardware-in-the-loop (HITL)
mock simulation fallback.

## Acceptance Criteria
- [ ] Abstract base class `BaseSDR` with `configure()`, `start_rx()`, `read_samples()`, `stop_rx()` in `/src/sdr_interface/base_sdr.py`
- [ ] RTL-SDR Blog V3 driver (8-bit, 2.4 MHz IBW) in `/src/sdr_interface/rtlsdr_driver.py`
- [ ] HackRF One driver (8-bit, 20 MHz IBW) in `/src/sdr_interface/hackrf_driver.py`
- [ ] PlutoSDR ADALM-PLUTO driver (12-bit, 20 MHz IBW) in `/src/sdr_interface/plutosdr_driver.py`
- [ ] LimeSDR-USB driver (12-bit, 61.44 MHz IBW, 2x2 MIMO) in `/src/sdr_interface/limesdr_driver.py`
- [ ] USRP B210/X310 driver (UHD, 12-14 bit, up to 160 MHz IBW) in `/src/sdr_interface/usrp_driver.py`
- [ ] High-fidelity `MockSDR` allowing full HITL tests without physical USB hardware connected in `/src/sdr_interface/mock_sdr.py`
- [ ] Hardware YAML configurations in `/configs/sdr_config.yaml`
- [ ] Unit tests in `/tests/test_sdr.py`
