# Experiment Log: EXP-009 — Real-Time Streaming Buffer & Latency Profiling

- **Task**: P2-004: Implement Real-Time Streaming Buffer & Latency Profiling
- **Date**: 2026-10-28
- **Status**: PASSED

## Objectives
1. Implement thread-safe circular ring buffer (`CircularRingBuffer`) with lock-free single/multi-chunk block transfer semantics.
2. Ingest asynchronous SDR I/Q streams via background polling worker (`StreamHandler`) and track overrun/underrun drop rates.
3. Benchmark end-to-end processing pipeline against the research plan's $2.3\text{ ms}$ frame budget and $5.0\text{ ms}$ embedded ceiling.
4. Simulate adaptive predictive scheduling (`DutyCycleController`) to reduce sensing duty cycle by up to $50\%$ during predictable PU idle/busy intervals.

## Results
- **Unit Test Coverage**: 9/9 tests passed in `tests/test_streaming.py`; **108/108 unit tests passed across entire testbed**.
- **Buffer Telemetry**:
  - Total Samples Ingested: 315,392
  - Total Samples Processed: 307,200 (300 frames of 1024 samples)
  - Buffer Overruns: 0
  - Buffer Underruns: 0
  - Dropped Samples: 0
- **Latency Benchmarks**:
  - **Mean E2E Latency**: $0.599\text{ ms}$
  - **Median (P50)**: $0.537\text{ ms}$
  - **95th Percentile (P95)**: $1.156\text{ ms}$ (Target Budget: $2.3\text{ ms}$) $\implies$ **MEETS BUDGET** ($\approx 50\%$ safety margin)
  - **Max Peak Latency**: $1.786\text{ ms}$ (Embedded Ceiling: $5.0\text{ ms}$) $\implies$ **PASS**
- **Duty-Cycle Optimization**:
  - Dynamically throttled sensing duty cycle down to $30\%$ during sustained idle periods, conserving battery power and cognitive radio compute resources without sacrificing sensing QoS.
- **Plot Saved**: `results/plots/p2_004_streaming_latency.png`.
