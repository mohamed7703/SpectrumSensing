---
id: P3-002
title: "Implement Cognitive MAC Protocols (Pure ALOHA, Slotted ALOHA, ILA-CSMA)"
phase: 3 - Network Simulation
status: backlog
priority: high
depends_on: [P3-001]
tags: [mac, aloha, slotted-aloha, ila-csma, sensing, crn]
---

## Objective
Implement MAC access protocols for cognitive radio spectrum sharing: Pure ALOHA, Slotted ALOHA,
and ILA-CSMA (Interference-Limited Adaptive CSMA with 2-stage CAD + RSSI sensing).

## Acceptance Criteria
- [ ] Baseline Pure ALOHA random access MAC in `/src/ns3_simulation/mac_pure_aloha.py`
- [ ] Slotted ALOHA time-synchronized access MAC in `/src/ns3_simulation/mac_slotted_aloha.py`
- [ ] Two-stage ILA-CSMA MAC (CAD sensing + threshold comparison + CSMA-HS/AB) in `/src/ns3_simulation/mac_ila_csma.py`
- [ ] Throughput vs offered load $S(G)$ theoretical curves and empirical verification
- [ ] Unit tests in `/tests/test_mac_protocols.py`
