"""Verification Script for P2-004: Real-Time Streaming Buffer & Latency Profiling.

Executes continuous SDR streaming through circular ring buffer, profiles
end-to-end processing pipeline against 2.3 ms budget, and benchmarks
predictive duty-cycle modulation.
"""

from pathlib import Path
import sys
import time

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.sdr_interface.mock_sdr import MockSDR
from src.sdr_interface.stream_handler import StreamHandler, LatencyProfiler, DutyCycleController
from src.preprocessing.iq_normalizer import remove_dc_offset, normalize_power
from src.dsp.psd_estimator import estimate_psd_welch
from src.math_models.energy_detection import compute_energy_statistic, compute_theoretical_threshold


def run_streaming_latency_verification() -> None:
    """Benchmark end-to-end streaming latency and duty-cycle control."""
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "p2_004_streaming_latency.png"

    print("=" * 70)
    print("P2-004: Real-Time Streaming Buffer & Latency Profiling Verification")
    print("=" * 70)

    # Initialize Mock SDR frontend
    sdr = MockSDR(
        device_name="HITL Benchmark SDR",
        adc_bits=12,
        sample_rate=2.4e6,
        signal_type="qpsk",
        snr_db=10.0,
        random_seed=42,
    )

    # Initialize streaming handler with 1M-sample ring buffer
    handler = StreamHandler(sdr_device=sdr, buffer_capacity=1048576, chunk_size=4096)
    profiler = LatencyProfiler(target_budget_ms=2.3, embedded_ceiling_ms=5.0)
    controller = DutyCycleController(base_duty_cycle=1.0, min_duty_cycle=0.3)

    handler.start()

    num_frames = 300
    frame_size = 1024
    noise_var = 1.0
    gamma_th = compute_theoretical_threshold(pfa_target=0.05, num_samples=frame_size, noise_variance=noise_var)

    duty_history = []
    e2e_latencies_ms = []

    try:
        for idx in range(num_frames):
            # 1. Capture Stage
            t0 = time.perf_counter()
            frame = handler.get_frame(frame_size=frame_size, timeout_s=1.0)
            t1 = time.perf_counter()
            profiler.record("Capture", t1 - t0)

            # 2. Preprocessing Stage (DC Offset + Normalization)
            frame_clean = remove_dc_offset(frame)
            frame_norm = normalize_power(frame_clean, target_power=1.0)
            t2 = time.perf_counter()
            profiler.record("Preprocessing", t2 - t1)

            # 3. DSP Transform Stage (Welch PSD)
            estimate_psd_welch(frame_norm, sample_rate=2.4e6, nperseg=256)
            t3 = time.perf_counter()
            profiler.record("DSP Transform", t3 - t2)

            # 4. Energy Detection Stage
            stat = compute_energy_statistic(frame, noise_variance=noise_var)
            is_busy = bool(stat > gamma_th)
            t4 = time.perf_counter()
            profiler.record("Detection Decision", t4 - t3)

            total_ms = (t4 - t0) * 1000.0
            e2e_latencies_ms.append(total_ms)

            # 5. Adaptive Duty-Cycle
            duty = controller.update(is_channel_busy=is_busy)
            duty_history.append(duty)

    finally:
        handler.stop()

    telemetry = handler.get_telemetry()
    total_stats = profiler.get_statistics()

    print(f"\n[Buffer Telemetry] Total Written: {telemetry['total_written']:,} | Total Read: {telemetry['total_read']:,}")
    print(f"[Buffer Telemetry] Overruns: {telemetry['overrun_count']} | Underruns: {telemetry['underrun_count']}")
    print(f"[Latency Benchmark] Frames Evaluated: {total_stats['count']}")
    print(f"[Latency Benchmark] Mean Latency:   {total_stats['mean_ms']:.3f} ms")
    print(f"[Latency Benchmark] Median (P50):   {total_stats['p50_ms']:.3f} ms")
    print(f"[Latency Benchmark] 95th %ile (P95): {total_stats['p95_ms']:.3f} ms (Budget: {total_stats['target_budget_ms']} ms)")
    print(f"[Latency Benchmark] Max Latency:    {total_stats['max_ms']:.3f} ms")
    print(f"[Latency Benchmark] Target Budget Met:   {total_stats['meets_target_budget']}")
    print(f"[Latency Benchmark] Embedded Ceiling Met: {total_stats['meets_embedded_ceiling']}")

    # Generate Publication Verification Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: End-to-End Latency vs Budget
    ax = axes[0, 0]
    ax.plot(e2e_latencies_ms, color="darkblue", lw=1.2, label="Measured E2E Latency")
    ax.axhline(2.3, color="crimson", ls="--", lw=1.8, label="2.3 ms Target Budget")
    ax.axhline(5.0, color="orange", ls=":", lw=1.8, label="5.0 ms Embedded Ceiling")
    ax.set_title("End-to-End Sensing Loop Latency per Frame", fontweight="bold")
    ax.set_xlabel("Frame Index (1024 samples/frame)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Plot 2: Stage Breakdown CDF
    ax = axes[0, 1]
    stages = ["Capture", "Preprocessing", "DSP Transform", "Detection Decision"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for stage, col in zip(stages, colors):
        st = profiler.get_statistics(stage)
        raw_vals = np.sort(profiler._stage_timings[stage])
        cdf = np.linspace(0, 1, len(raw_vals))
        ax.plot(raw_vals, cdf, label=f"{stage} (mean={st['mean_ms']:.2f}ms)", color=col, lw=1.5)
    ax.set_title("Latency Cumulative Distribution Function (CDF)", fontweight="bold")
    ax.set_xlabel("Execution Time (ms)")
    ax.set_ylabel("CDF P(T <= t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Plot 3: Ring Buffer Fill State and Telemetry
    ax = axes[1, 0]
    bars = ["Written", "Read", "Overruns", "Underruns"]
    counts = [telemetry["total_written"] / 1000.0, telemetry["total_read"] / 1000.0, telemetry["overrun_count"], telemetry["underrun_count"]]
    ax.bar(bars, counts, color=["teal", "steelblue", "crimson", "darkorange"])
    ax.set_title("SDR Ring Buffer High-Throughput Telemetry", fontweight="bold")
    ax.set_ylabel("Count (kSamples or Events)")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Predictive Duty Cycle Modulation
    ax = axes[1, 1]
    ax.plot(np.array(duty_history) * 100.0, color="forestgreen", lw=1.8, label="Adaptive Duty Cycle (%)")
    ax.axhline(100.0, color="gray", ls="--", label="100% Continuous Sensing")
    ax.axhline(50.0, color="purple", ls=":", label="50% Target Reduction")
    ax.set_title("Predictive Scheduling Duty Cycle Optimization", fontweight="bold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Active Sensing Duty Cycle (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(20, 110)

    plt.suptitle("Real-Time Streaming Buffer & Latency Profiling Verification (P2-004)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Real-time streaming verification plot saved to {plot_path}")


if __name__ == "__main__":
    run_streaming_latency_verification()
