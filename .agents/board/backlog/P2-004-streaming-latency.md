---
id: P2-004
title: "Implement Real-Time Streaming Buffer & Latency Profiling"
phase: 2 - Physical / SDR Layer
status: backlog
priority: high
depends_on: [P2-003]
tags: [streaming, circular-buffer, latency, hitl, duty-cycle]
---

## Objective
Implement thread-safe circular ring buffering for asynchronous SDR I/Q streaming, overflow/underflow
tracking, and latency benchmarking against the research plan's 2.3 ms processing budget.

## Acceptance Criteria
- [ ] Lock-free / thread-safe circular buffer for high-throughput I/Q streaming in `/src/sdr_interface/stream_handler.py`
- [ ] Buffer overrun/underrun accounting and dropped frame telemetry
- [ ] Latency profiler measuring end-to-end sample capture -> DSP -> inference loop
- [ ] Benchmarking against the 2.3 ms budget for 1M-sample frames and 5 ms embedded CPU ceiling
- [ ] Predictive scheduling duty cycle simulation (reducing sensing duty cycle by up to 50%)
- [ ] Unit tests in `/tests/test_streaming.py`
