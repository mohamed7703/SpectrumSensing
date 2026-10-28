r"""Model Export, Quantization & Edge Latency Verification (P4-006).

Exports all 4 spectrum sensing models into deployment-ready formats (.npz, .tflite, .onnx),
applies FP16 and INT8 post-training quantization, profiles latency against the 2.3 ms SDR budget,
and generates publication-ready visualization.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    BaselineCNN,
    BatchNormCNN,
    GAPCNN,
    BiGRUModel,
    ModelExporter,
    quantize_affine_int8,
    quantize_fp16,
)


def run_export_verification():
    print("=" * 75)
    print("Cognitive Radio Spectrum Sensing — Model Export & Edge Quantization Suite")
    print("=" * 75)

    export_dir = os.path.join("saved_models", "exported")
    exporter = ModelExporter(output_dir=export_dir)

    models = {
        "Baseline 1D-CNN": BaselineCNN(),
        "BatchNorm 1D-CNN": BatchNormCNN(),
        "GAP 1D-CNN": GAPCNN(),
        "Bi-GRU Temporal": BiGRUModel(),
    }

    # 1. Export Models into Target Edge Formats
    print("\n--- Exporting Models to Edge Formats (.npz, .tflite, .onnx) ---")
    exported_files = []
    for name, model in models.items():
        npz_file = exporter.export_weights_npz(model)
        tflite_fp32 = exporter.export_tflite(model, quantization="none")
        tflite_fp16 = exporter.export_tflite(model, quantization="fp16")
        tflite_int8 = exporter.export_tflite(model, quantization="int8")
        onnx_file = exporter.export_onnx(model)

        print(f"Exported {name:<18} -> NPZ, TFLite (FP32/FP16/INT8), ONNX in '{export_dir}'")
        exported_files.extend([npz_file, tflite_fp32, tflite_fp16, tflite_int8, onnx_file])

    # 2. Quantization Compression Analysis
    print("\n--- Post-Training Quantization Precision & Compression ---")
    synthetic_weights = np.random.randn(23009).astype(np.float32)  # Size of BatchNorm CNN
    _, q_int8_summary = quantize_affine_int8(synthetic_weights)
    _, q_fp16_summary = quantize_fp16(synthetic_weights)

    print(f"FP32 Baseline : {q_int8_summary.original_bytes / 1024.0:.2f} KB (1.0x)")
    print(f"FP16 Half     : {q_fp16_summary.quantized_bytes / 1024.0:.2f} KB ({q_fp16_summary.compression_ratio:.1f}x compression, MSE: {q_fp16_summary.mean_squared_error:.2e})")
    print(f"INT8 Integer  : {q_int8_summary.quantized_bytes / 1024.0:.2f} KB ({q_int8_summary.compression_ratio:.1f}x compression, MSE: {q_int8_summary.mean_squared_error:.2e})")

    # 3. Edge Inference Profiling
    print("\n--- Edge Hardware Inference Latency Benchmark (1024 I/Q Samples) ---")
    profiles = {}
    for name, model in models.items():
        rep = exporter.profile_edge_inference(model, num_iterations=120)
        profiles[name] = rep
        status = "COMPLIANT (< 2.3 ms)" if rep.realtime_compliant else "NON-COMPLIANT"
        print(f"{name:<20}: Mean {rep.mean_latency_us:>6.2f} µs | P95 {rep.p95_latency_us:>6.2f} µs | {rep.throughput_fps:>8.1f} FPS | {status}")

    # 4. Generate Publication Plot
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    plot_path = os.path.join("results", "plots", "p4_006_model_edge_export.png")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Model Memory Footprint across Quantization Modes (FP32 vs FP16 vs INT8)
    model_names = list(models.keys())
    fp32_sizes = [models[m].count_parameters()["total_params"] * 4 / 1024.0 for m in model_names]
    fp16_sizes = [s / 2.0 for s in fp32_sizes]
    int8_sizes = [s / 4.0 for s in fp32_sizes]

    x = np.arange(len(model_names))
    width = 0.25

    axs[0, 0].bar(x - width, fp32_sizes, width, label="FP32 (Original)", color="#4C72B0", edgecolor="black")
    axs[0, 0].bar(x, fp16_sizes, width, label="FP16 (2x Quantized)", color="#55A868", edgecolor="black")
    axs[0, 0].bar(x + width, int8_sizes, width, label="INT8 (4x Quantized)", color="#C44E52", edgecolor="black")

    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel("Model Footprint (KB, Log Scale)", fontsize=11)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels([m.split()[0] for m in model_names], fontsize=10)
    axs[0, 0].set_title("Edge Storage Footprint by Quantization Mode", fontsize=12, fontweight="bold")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5, which="both")
    axs[0, 0].legend()

    # Subplot 2: Inference Latency vs Real-Time SDR Budget (2.3 ms)
    mean_lats = [profiles[m].mean_latency_us for m in model_names]
    p95_lats = [profiles[m].p95_latency_us for m in model_names]

    axs[0, 1].bar(x - width/2, mean_lats, width, label="Mean Latency", color="#8172B3", edgecolor="black")
    axs[0, 1].bar(x + width/2, p95_lats, width, label="P95 Latency", color="#CCB974", edgecolor="black")
    axs[0, 1].axhline(y=2300.0, color="crimson", linestyle="--", lw=2, label="2.3 ms SDR Real-time Budget")

    axs[0, 1].set_ylabel(r"Latency per Frame ($\mu$s)", fontsize=11)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels([m.split()[0] for m in model_names], fontsize=10)
    axs[0, 1].set_title("Edge Inference Latency & P95 Jitter", fontsize=12, fontweight="bold")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend()

    # Subplot 3: Quantization Reconstruction Error Distribution (INT8)
    reconstructed_int8 = q_int8_summary.scale * (synthetic_weights.astype(np.float32) - q_int8_summary.zero_point)
    error_int8 = synthetic_weights - reconstructed_int8
    axs[1, 0].hist(error_int8, bins=50, color="#C44E52", alpha=0.75, edgecolor="black")
    axs[1, 0].axvline(x=0.0, color="black", linestyle="--")
    axs[1, 0].set_xlabel("Reconstruction Error (Original - INT8 Dequantized)", fontsize=11)
    axs[1, 0].set_ylabel("Parameter Count", fontsize=11)
    axs[1, 0].set_title(r"INT8 Quantization Error ($\text{MSE} = " + f"{q_int8_summary.mean_squared_error:.2e}" + r"$)", fontsize=12, fontweight="bold")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Throughput Comparison (Frames per Second)
    fps_vals = [profiles[m].throughput_fps for m in model_names]
    bars_fps = axs[1, 1].bar([m.split()[0] for m in model_names], fps_vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"], edgecolor="black")
    axs[1, 1].set_ylabel("Inference Throughput (FPS)", fontsize=11)
    axs[1, 1].set_title("Sensing Throughput across Architectures", fontsize=12, fontweight="bold")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    for bar in bars_fps:
        h = bar.get_height()
        axs[1, 1].annotate(f"{int(h):,} FPS",
                           xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 4), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nModel export and edge profiling plot saved to: {plot_path}")

    return exported_files, profiles


if __name__ == "__main__":
    run_export_verification()
