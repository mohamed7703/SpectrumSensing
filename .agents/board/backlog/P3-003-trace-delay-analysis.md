---
id: P3-003
title: "Implement Trace Parser & End-to-End Delay Analysis Pipeline"
phase: 3 - Network Simulation
status: backlog
priority: high
depends_on: [P3-002]
tags: [trace, ns3-tr, delay, packet-delivery-ratio, pdr, collisions]
---

## Objective
Implement ASCII trace parser (`.tr` event parser) for NS-3 / discrete-event traces, calculating
end-to-end packet delivery ratio (PDR), collision probability, and latency distribution percentiles.

## Acceptance Criteria
- [ ] NS-3 ASCII `.tr` event trace parser in `/src/ns3_simulation/trace_parser.py`
- [ ] Metric extraction: send/recv timestamps, per-packet E2E delay, packet drop reasons
- [ ] Performance benchmark comparing ILA-CSMA vs CSMA/CA across dense multi-node scenarios (target: +20% PDR improvement, delay reduction)
- [ ] Unit tests in `/tests/test_trace_parser.py`
