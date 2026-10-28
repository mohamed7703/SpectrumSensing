---
id: P2-004
title: "Implement Real-Time Streaming Buffer & Latency Profiling"
phase: 2 - Physical / SDR Layer
status: done
priority: high
depends_on: [P2-003]
tags: [streaming, circular-buffer, latency, hitl, duty-cycle]
---

## Objective
Implement thread-safe circular ring buffering for asynchronous SDR I/Q streaming, overflow/underflow
tracking, and latency benchmarking against the research plan's 2.3 ms processing budget.

## Acceptance Criteria
- [x] Lock-free / thread-safe circular buffer for high-throughput I/Q streaming in `/src/sdr_interface/stream_handler.py`
- [x] Buffer overrun/underrun accounting and dropped frame telemetry
- [x] Latency profiler measuring end-to-end sample capture -> DSP -> inference loop
- [x] Benchmarking against the 2.3 ms budget for 1M-sample frames and 5 ms embedded CPU ceiling
- [x] Predictive scheduling duty cycle simulation (reducing sensing duty cycle by up to 50%)
- [x] Unit tests in `/tests/test_streaming.py`
