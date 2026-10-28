r"""Verification and Latency Benchmarking for Bi-GRU Architecture (P4-003).

Benchmarks parameter efficiency, recurrent sequence propagation, and forward inference latency
comparing Bi-GRU against 1D-CNN architectures (Baseline, BatchNorm, GAP).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import BaselineCNN, BatchNormCNN, GAPCNN, BiGRUModel, get_model


def run_bigru_benchmarks():
    print("=" * 70)
    print("Cognitive Radio Spectrum Sensing — Bi-GRU Architecture Verification")
    print("=" * 70)

    # 1. Instantiate Models
    models = {
        "Baseline 1D-CNN": BaselineCNN(),
        "BatchNorm 1D-CNN": BatchNormCNN(),
        "GAP 1D-CNN": GAPCNN(),
        "Bi-GRU Temporal": BiGRUModel(),
    }

    print("\n--- Model Parameter Comparison ---")
    param_counts = {}
    for name, model in models.items():
        counts = model.count_parameters()
        param_counts[name] = counts["total_params"]
        print(f"{name:<20}: {counts['total_params']:>10,} total params ({counts['trainable_params']:,} trainable)")

    # 2. Detailed Summary for Bi-GRU
    bigru = models["Bi-GRU Temporal"]
    print("\n" + bigru.summary() + "\n")

    # 3. Latency Benchmarking
    print("--- Inference Latency Benchmarks (Batch = 1, Sequence = 1024x2) ---")
    sample_frame = np.random.randn(1024, 2).astype(np.float32)
    latencies = {}

    # Warmup
    for model in models.values():
        for _ in range(20):
            _ = model.predict_proba(sample_frame)

    # Benchmark 200 iterations
    n_iters = 200
    for name, model in models.items():
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model.predict_proba(sample_frame)
        elapsed = (time.perf_counter() - t0) / n_iters * 1e6  # microseconds
        latencies[name] = elapsed
        print(f"{name:<20}: {elapsed:>8.2f} µs / frame")

    # 4. Bidirectional Recurrent Propagation Mechanics
    print("\n--- Bidirectional Recurrent Step & State Verification ---")
    seq = np.random.randn(1024, 2).astype(np.float32)
    t0 = time.perf_counter()
    merged_state = bigru.forward_recurrent_sequence(seq, subsample_step=4)
    rec_time = (time.perf_counter() - t0) * 1e3
    print(f"Subsampled recurrent forward pass (256 steps): {rec_time:.2f} ms")
    print(f"Forward state h_T shape: {bigru.gru_units}, Backward state h_0 shape: {bigru.gru_units}")
    print(f"Merged representation dimension: {merged_state.shape[0]} features")
    print(f"State norms: ||h_merged||_2 = {np.linalg.norm(merged_state):.4f}")

    # 5. Temporal Coherence & Modulation Sensitivity Test
    print("\n--- Temporal Sensitivity Analysis (BPSK vs AWGN) ---")
    t = np.arange(1024)
    # Pure AWGN (H0)
    h0_frames = np.random.randn(50, 1024, 2).astype(np.float32)
    # BPSK modulated carrier with phase transitions (H1)
    symbols = np.repeat(np.random.choice([-1.0, 1.0], size=128), 8)
    carrier_i = np.cos(2 * np.pi * 0.1 * t) * symbols
    carrier_q = np.sin(2 * np.pi * 0.1 * t) * symbols
    h1_clean = np.stack([carrier_i, carrier_q], axis=-1)
    h1_frames = np.array([h1_clean + 0.3 * np.random.randn(1024, 2) for _ in range(50)], dtype=np.float32)

    p_h0 = bigru.predict_proba(h0_frames)
    p_h1 = bigru.predict_proba(h1_frames)
    print(f"Mean P(H1 | H0 noise):   {np.mean(p_h0):.4f} (std: {np.std(p_h0):.4f})")
    print(f"Mean P(H1 | H1 signal):  {np.mean(p_h1):.4f} (std: {np.std(p_h1):.4f})")

    # 6. Generate Verification Plot
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    plot_path = os.path.join("results", "plots", "p4_003_bigru_architecture.png")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Parameter Count Comparison (Log scale)
    arch_names = list(param_counts.keys())
    counts_val = list(param_counts.values())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    bars = axs[0, 0].bar(arch_names, counts_val, color=colors, edgecolor="black", alpha=0.85)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel("Total Parameters (Log Scale)", fontsize=11)
    axs[0, 0].set_title("Architectural Parameter Complexity", fontsize=12, fontweight="bold")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5, which="both")
    axs[0, 0].tick_params(axis="x", rotation=15)
    for bar in bars:
        h = bar.get_height()
        axs[0, 0].annotate(f"{int(h):,}",
                           xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 4), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: Forward Inference Latency
    lat_vals = [latencies[m] for m in arch_names]
    bars_lat = axs[0, 1].bar(arch_names, lat_vals, color=colors, edgecolor="black", alpha=0.85)
    axs[0, 1].axhline(y=2300.0, color="r", linestyle="--", linewidth=1.5, label="2.3 ms SDR Real-time Limit")
    axs[0, 1].set_ylabel(r"Inference Latency per Frame ($\mu$s)", fontsize=11)
    axs[0, 1].set_title("Inference Latency Benchmark (1024 I/Q Samples)", fontsize=12, fontweight="bold")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].tick_params(axis="x", rotation=15)
    axs[0, 1].legend(loc="upper right")
    for bar in bars_lat:
        h = bar.get_height()
        axs[0, 1].annotate(f"{h:.1f} µs",
                           xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 4), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 3: Recurrent Feature Activation Dynamics (Forward vs Backward States)
    axs[1, 0].plot(merged_state[:128], label=r"Forward GRU $h_T^f$", color="#4C72B0", lw=1.5)
    axs[1, 0].plot(merged_state[128:], label=r"Backward GRU $h_1^b$", color="#C44E52", lw=1.5)
    axs[1, 0].axhline(y=0.0, color="gray", linestyle=":", alpha=0.7)
    axs[1, 0].set_xlabel("Hidden Unit Index", fontsize=11)
    axs[1, 0].set_ylabel("State Value", fontsize=11)
    axs[1, 0].set_title("Bi-GRU Recurrent Hidden State Activations", fontsize=12, fontweight="bold")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend()

    # Subplot 4: Detection Probability Distributions under H0 and H1
    bins = np.linspace(0.0, 1.0, 25)
    axs[1, 1].hist(p_h0, bins=bins, alpha=0.65, label=r"$\mathcal{H}_0$ (Noise Floor)", color="#4C72B0", edgecolor="black")
    axs[1, 1].hist(p_h1, bins=bins, alpha=0.65, label=r"$\mathcal{H}_1$ (BPSK Signal)", color="#55A868", edgecolor="black")
    axs[1, 1].axvline(x=0.5, color="red", linestyle="--", label="Decision Threshold $\eta=0.5$")
    axs[1, 1].set_xlabel(r"Estimated Posterior $P(\mathcal{H}_1 \mid \mathbf{X})$", fontsize=11)
    axs[1, 1].set_ylabel("Frequency", fontsize=11)
    axs[1, 1].set_title(r"Bi-GRU Posterior Distribution ($\mathcal{H}_0$ vs $\mathcal{H}_1$)", fontsize=12, fontweight="bold")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nVerification plot saved to: {plot_path}")

    return param_counts, latencies


if __name__ == "__main__":
    run_bigru_benchmarks()
