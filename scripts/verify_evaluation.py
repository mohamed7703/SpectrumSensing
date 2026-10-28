r"""Benchmark Evaluation and Comparison Suite (P4-005).

Reproduces Figure 6 (ROC curves) and Figure 7 (Pd vs SNR) from Abdelbaset et al.
(Sensors 2024), demonstrating deep learning superiority below the SNR wall (-8.2 dB).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import BaselineCNN, BatchNormCNN, GAPCNN, BiGRUModel
from src.evaluation.evaluator import BenchmarkEvaluator
from src.training.dataset_loader import SyntheticRadioMLGenerator


def run_benchmark_suite():
    print("=" * 75)
    print("Cognitive Radio Spectrum Sensing — Full Benchmark Evaluation Suite")
    print("=" * 75)

    evaluator = BenchmarkEvaluator(random_seed=42)
    gen = SyntheticRadioMLGenerator(random_seed=42)

    # 1. Instantiate Models
    models = {
        "Baseline 1D-CNN": BaselineCNN(),
        "BatchNorm 1D-CNN": BatchNormCNN(),
        "GAP 1D-CNN": GAPCNN(),
        "Bi-GRU Temporal": BiGRUModel(),
    }

    # 2. Evaluate ROC Curves on Unseen Test Dataset
    print("\n--- Generating Receiver Operating Characteristic (ROC) Curves ---")
    test_ds = gen.generate_dataset(frames_per_class=12, h0_ratio=0.5)
    print(f"Test frames: {len(test_ds.x)} (H0: {np.sum(test_ds.y_binary == 0)}, H1: {np.sum(test_ds.y_binary == 1)})")

    roc_results = {}
    for name, model in models.items():
        scores = model.predict_proba(test_ds.x)
        roc = evaluator.compute_roc(test_ds.y_binary, scores)
        roc_results[name] = roc
        print(f"{name:<20}: AUC = {roc.auc:.4f}, EER = {roc.eer * 100:.2f}%")

    # Classical Energy Detector ROC on same frames
    ed_scores = np.mean(test_ds.x[..., 0]**2 + test_ds.x[..., 1]**2, axis=1)
    ed_roc = evaluator.compute_roc(test_ds.y_binary, ed_scores)
    roc_results["Energy Detector"] = ed_roc
    print(f"{'Energy Detector':<20}: AUC = {ed_roc.auc:.4f}, EER = {ed_roc.eer * 100:.2f}%")

    # 3. Sweep Detection Probability (Pd) vs SNR across [-20 dB, +30 dB]
    print("\n--- Sweeping Detection Probability (Pd) vs SNR (-20 dB to +30 dB) ---")
    snrs = [-20, -16, -12, -10, -8, -6, -4, -2, 0, 2, 4, 8, 12, 16, 20, 25, 30]

    snr_results = {}
    for name, model in models.items():
        print(f"Sweeping {name}...")
        snr_res = evaluator.evaluate_model_snr_sweep(model, snrs=snrs, frames_per_snr=25, target_pfa=0.05)
        snr_results[name] = snr_res

    print("Evaluating Classical Energy Detector under 1.0 dB noise uncertainty...")
    ed_snr = evaluator.evaluate_energy_detector_snr_sweep(
        snrs=snrs,
        num_samples=1024,
        noise_uncertainty_db=1.0,
        target_pfa=0.05,
        trials_per_snr=60,
    )
    snr_results["Energy Detector"] = ed_snr

    # Print Comparative Pd Table across Key SNR Regimes
    print("\n--- Detection Probability (Pd) at Target Pfa = 0.05 ---")
    header = f"{'Model':<20} | {'-16 dB':<8} | {'-10 dB':<8} | {'-8 dB (Wall)':<12} | {'0 dB':<8} | {'+12 dB':<8}"
    print(header)
    print("-" * len(header))
    target_snrs_idx = [1, 3, 4, 8, 12]
    for name, res in snr_results.items():
        vals = [f"{res.pd[idx]*100:.1f}%" for idx in target_snrs_idx]
        print(f"{name:<20} | {vals[0]:<8} | {vals[1]:<8} | {vals[2]:<12} | {vals[3]:<8} | {vals[4]:<8}")

    # 4. Confusion Matrices (Low SNR vs High SNR)
    cm_low = evaluator.compute_confusion_matrix(
        test_ds.y_binary,
        models["BatchNorm 1D-CNN"].predict_proba(test_ds.x),
        threshold=0.5,
    )

    # 5. Generate Figures (Reproducing Paper Figures 6 and 7)
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    plot_path = os.path.join("results", "plots", "p4_005_benchmark_evaluation.png")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Panel A: Paper Figure 6 Reproduction — ROC Curves (Log-scale Pfa)
    colors = {
        "Baseline 1D-CNN": "#4C72B0",
        "BatchNorm 1D-CNN": "#55A868",
        "GAP 1D-CNN": "#C44E52",
        "Bi-GRU Temporal": "#8172B3",
        "Energy Detector": "#333333",
    }
    linestyles = {
        "Baseline 1D-CNN": "-",
        "BatchNorm 1D-CNN": "-",
        "GAP 1D-CNN": "--",
        "Bi-GRU Temporal": "-.",
        "Energy Detector": ":",
    }

    for name, roc in roc_results.items():
        axs[0, 0].plot(
            roc.pfa,
            roc.pd,
            label=f"{name} (AUC={roc.auc:.3f})",
            color=colors[name],
            linestyle=linestyles[name],
            lw=2.0 if "Energy" not in name else 2.5,
        )
    axs[0, 0].plot([0, 1], [0, 1], "k:", alpha=0.3, label="Random Guess")
    axs[0, 0].set_xlim([0.0, 1.0])
    axs[0, 0].set_ylim([0.0, 1.05])
    axs[0, 0].set_xlabel(r"False Alarm Probability ($P_{fa}$)", fontsize=11)
    axs[0, 0].set_ylabel(r"Detection Probability ($P_d$)", fontsize=11)
    axs[0, 0].set_title("Figure 6: ROC Curves Comparison", fontsize=12, fontweight="bold")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="lower right", fontsize=9)

    # Panel B: Paper Figure 7 Reproduction — Pd vs SNR with SNR Wall Annotation
    for name, res in snr_results.items():
        axs[0, 1].plot(
            res.snrs,
            res.pd * 100,
            marker="o",
            label=name,
            color=colors[name],
            linestyle=linestyles[name],
            lw=2.0,
            ms=4,
        )

    # Annotate SNR Wall at -8.2 dB
    axs[0, 1].axvline(x=-8.2, color="crimson", linestyle="--", lw=2, label=r"SNR Wall ($-8.2$ dB)")
    axs[0, 1].axhspan(0, 50, xmin=0.0, xmax=(-8.2 - (-20)) / 50.0, color="pink", alpha=0.25, label="Sub-Wall Collapse Zone")
    axs[0, 1].axhline(y=90.0, color="green", linestyle=":", lw=1.5, label=r"IEEE 802.22 Target $P_d \geq 90\%$")

    axs[0, 1].set_xlim([-20, 30])
    axs[0, 1].set_ylim([0, 105])
    axs[0, 1].set_xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=11)
    axs[0, 1].set_ylabel(r"Detection Probability $P_d$ (%)", fontsize=11)
    axs[0, 1].set_title(r"Figure 7: Detection Probability ($P_d$) vs SNR ($P_{fa}=0.05$)", fontsize=12, fontweight="bold")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend(loc="lower right", fontsize=9)

    # Panel C: Confusion Matrix for BatchNorm 1D-CNN
    im = axs[1, 0].imshow(cm_low, cmap="Blues", interpolation="nearest")
    axs[1, 0].set_title(f"Confusion Matrix: BatchNorm 1D-CNN (Accuracy = {np.trace(cm_low)/np.sum(cm_low)*100:.1f}%)", fontsize=12, fontweight="bold")
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xticklabels([r"$\mathcal{H}_0$ (Noise)", r"$\mathcal{H}_1$ (Signal)"], fontsize=10)
    axs[1, 0].set_yticklabels([r"$\mathcal{H}_0$ (Noise)", r"$\mathcal{H}_1$ (Signal)"], fontsize=10)
    axs[1, 0].set_xlabel("Predicted Class", fontsize=11)
    axs[1, 0].set_ylabel("True Class", fontsize=11)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_low[i, j] > np.max(cm_low) / 2 else "black"
            axs[1, 0].text(j, i, f"{cm_low[i, j]:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # Panel D: Architectural Radar / Bar Trade-off Summary (Params vs AUC vs Latency)
    arch_labels = ["Baseline", "BatchNorm", "GAP", "Bi-GRU"]
    auc_scores = [roc_results[f"{k} 1D-CNN" if "Bi-GRU" not in k else "Bi-GRU Temporal"].auc for k in arch_labels]
    param_counts_k = [2087.5, 23.0, 14.9, 117.9]  # in thousands

    x = np.arange(len(arch_labels))
    width = 0.35

    ax_d1 = axs[1, 1]
    ax_d2 = ax_d1.twinx()

    b1 = ax_d1.bar(x - width/2, param_counts_k, width, label="Parameters (k)", color="#4C72B0", alpha=0.85)
    b2 = ax_d2.bar(x + width/2, auc_scores, width, label="ROC AUC", color="#55A868", alpha=0.85)

    ax_d1.set_yscale("log")
    ax_d1.set_ylabel("Parameters (x 1,000, Log Scale)", color="#4C72B0", fontsize=11)
    ax_d2.set_ylabel("ROC AUC Score", color="#55A868", fontsize=11)
    ax_d2.set_ylim([0.8, 1.02])
    ax_d1.set_xticks(x)
    ax_d1.set_xticklabels(arch_labels, fontsize=10)
    ax_d1.set_title("Parameter Efficiency vs Sensing Discrimination (AUC)", fontsize=12, fontweight="bold")
    ax_d1.grid(True, linestyle="--", alpha=0.5)

    # Combined legend
    lines1, labels1 = ax_d1.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nBenchmark evaluation plot saved to: {plot_path}")

    return roc_results, snr_results


if __name__ == "__main__":
    run_benchmark_suite()
