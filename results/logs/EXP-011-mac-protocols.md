# Experiment Log: EXP-011 — Cognitive MAC Protocols (Pure ALOHA, Slotted ALOHA, ILA-CSMA)

- **Task**: P3-002: Implement Cognitive MAC Protocols
- **Date**: 2026-10-28
- **Status**: PASSED

## Objectives
1. Implement baseline unslotted Pure ALOHA contention MAC (`PureAlohaMAC`) with vulnerable period $2 T_p$ and exponential backoff.
2. Implement synchronized Slotted ALOHA MAC (`SlottedAlohaMAC`) with discrete slot boundary alignment and vulnerable period $T_p$.
3. Implement Interference-Limited Adaptive CSMA (`ILaCsmaMAC`) with two-stage hybrid sensing:
   - Stage 1: Channel Activity Detection (CAD) during DIFS preamble.
   - Stage 2: Deep RSSI / Energy Detection against adaptive threshold $\gamma_{\text{th}}$.
   - CSMA-HS (Interference Screening): Immediate backoff upon PU detection.
   - CSMA-AB (Adaptive Backoff): Binary exponential contention window expansion.
4. Benchmark empirical throughput $S(G)$ against theoretical derivations and quantify Primary User collision protection.

## Results
- **Unit Test Coverage**: 12/12 tests passed in `tests/test_mac_protocols.py`; **144/144 unit tests passed across entire testbed**.
- **Throughput Performance**:
  - Pure ALOHA: Matches theoretical peak $S_{\max} = \frac{1}{2e} \approx 0.184$ at $G = 0.5$; collapses to $S < 0.02$ at $G \ge 2.0$.
  - Slotted ALOHA: Matches theoretical peak $S_{\max} = \frac{1}{e} \approx 0.368$ at $G = 1.0$ (exact $2\times$ gain over Pure ALOHA).
  - ILA-CSMA: Sustains high normalized throughput ($S > 0.43$) up to $G = 3.0$ through carrier sensing and density-adaptive backoff.
- **Primary User Protection**:
  - Pure ALOHA produced **255 PU collisions** at $G = 3.0$ (severe interference into licensed primary spectrum).
  - ILA-CSMA produced **0 PU collisions** across the entire offered load range $[0.1, 3.0]$, fully verifying CSMA-HS interference screening.
- **Plot Saved**: `results/plots/p3_002_mac_protocols.png`.
