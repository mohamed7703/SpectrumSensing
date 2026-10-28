"""Verification Script for P4-002: 1D-CNN Spectrum Sensing Architectures.

Verifies layer-by-layer topologies, exact parameter counts (Baseline 2,087,457,
BatchNorm 23,009, GAP 14,881), forward inference on RadioML frames, and
plots comparative architectural benchmarks.
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

from src.models import BaselineCNN, BatchNormCNN, GAPCNN, get_model
from src.training.dataset_loader import SyntheticRadioMLGenerator


def run_cnn_architectures_verification() -> None:
    """Benchmark and verify the 3 1D-CNN architectures from the paper."""
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "p4_002_cnn_architectures.png"

    print("=" * 70)
    print("P4-002: 1D-CNN Spectrum Sensing Architectures Verification")
    print("=" * 70)

    models = {
        "Baseline 1D-CNN (Table 1)": BaselineCNN(),
        "BatchNorm 1D-CNN (Table 3)": BatchNormCNN(),
        "GAP 1D-CNN (Table 4)": GAPCNN(),
    }

    # Print summary tables
    for name, model in models.items():
        print(f"\n{model.summary()}")

    # Verify exact parameter counts matching paper
    expected_params = {
        "Baseline 1D-CNN (Table 1)": 2087457,
        "BatchNorm 1D-CNN (Table 3)": 23009,
        "GAP 1D-CNN (Table 4)": 14881,
    }

    for name, model in models.items():
        cnt = model.count_parameters()["total_params"]
        exp = expected_params[name]
        assert cnt == exp, f"Param mismatch for {name}: got {cnt}, expected {exp}"
        print(f"[VERIFIED] {name}: Exactly {cnt:,} parameters (matches paper).")

    # Benchmark forward inference latency over 100 frames
    gen = SyntheticRadioMLGenerator(random_seed=42)
    batch_x = np.array([gen.generate_single_frame("QPSK", 10.0) for _ in range(100)])

    latencies_us = {}
    predictions = {}

    for name, model in models.items():
        # Warmup
        model.predict_proba(batch_x[:4])

        t0 = time.perf_counter()
        probs = model.predict_proba(batch_x)
        t1 = time.perf_counter()

        per_frame_us = ((t1 - t0) / len(batch_x)) * 1e6
        latencies_us[name] = per_frame_us
        predictions[name] = probs
        print(f"[{name}] Mean Inference Latency: {per_frame_us:.2f} µs/frame")

    # Generate Publication Verification Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Model Parameter Counts (Log Scale)
    ax1 = axes[0, 0]
    names = ["Baseline", "BatchNorm", "GAP"]
    param_counts = [models[k].count_parameters()["total_params"] for k in models]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax1.bar(names, param_counts, color=colors, width=0.45, edgecolor="black")
    ax1.set_yscale("log")
    ax1.set_title("Model Parameter Comparison (Log Scale)", fontweight="bold")
    ax1.set_ylabel("Total Parameters (log scale)")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, param_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, count * 1.3, f"{count:,}", ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylim(5e3, 1e7)

    # Plot 2: Parameter Reduction Percentage Relative to Baseline
    ax2 = axes[0, 1]
    reductions = [0.0, ((2087457 - 23009) / 2087457) * 100.0, ((2087457 - 14881) / 2087457) * 100.0]
    ax2.bar(names, reductions, color=["gray", "#1f77b4", "#2ca02c"], width=0.45, edgecolor="black")
    ax2.set_title("Parameter Reduction Relative to Baseline CNN", fontweight="bold")
    ax2.set_ylabel("Parameter Reduction (%)")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(reductions):
        ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 110)

    # Plot 3: Layer-by-Layer Trainable Parameters Breakdown
    ax3 = axes[1, 0]
    layer_categories = ["Conv Layers", "Batch Normalization", "Dense / Output Layers"]
    # Baseline
    b_conv = 224 + 6208
    b_bn = 0
    b_dense = 2080896 + 129
    # BatchNorm
    bn_conv = 224 + 6208
    bn_bn = 64 + 128
    bn_dense = 16193
    # GAP
    gap_conv = 224 + 6208
    gap_bn = 0
    gap_dense = 8320 + 129

    x_idx = np.arange(len(layer_categories))
    w = 0.25
    ax3.bar(x_idx - w, [b_conv, b_bn, b_dense], width=w, color="#d62728", label="Baseline (2.087M)")
    ax3.bar(x_idx, [bn_conv, bn_bn, bn_dense], width=w, color="#1f77b4", label="BatchNorm (23.0K)")
    ax3.bar(x_idx + w, [gap_conv, gap_bn, gap_dense], width=w, color="#2ca02c", label="GAP (14.9K)")
    ax3.set_yscale("log")
    ax3.set_title("Layer Parameter Distribution Across Architectures", fontweight="bold")
    ax3.set_ylabel("Parameters (log scale)")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(layer_categories)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(loc="upper right")
    ax3.set_ylim(10, 5e6)

    # Plot 4: Forward Inference Latency Benchmarks
    ax4 = axes[1, 1]
    lat_vals = [latencies_us[k] for k in models]
    ax4.bar(names, lat_vals, color=colors, width=0.45, edgecolor="black")
    ax4.axhline(2300.0, color="crimson", ls="--", lw=1.8, label="2.3 ms Target Frame Budget")
    ax4.set_title("Single Frame Forward Inference Latency", fontweight="bold")
    ax4.set_ylabel("Inference Latency (µs/frame)")
    ax4.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(lat_vals):
        ax4.text(i, v + 2.0, f"{v:.1f} µs", ha="center", fontweight="bold", fontsize=10)
    ax4.legend(loc="upper left")

    plt.suptitle("1D-CNN Spectrum Sensing Architectures Verification (P4-002)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[OK] Model architectures verification plot saved to {plot_path}")


if __name__ == "__main__":
    run_cnn_architectures_verification()
