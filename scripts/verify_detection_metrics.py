"""Verification script for P1-003: Detection Metrics and ROC Analysis.

Generates a family of Receiver Operating Characteristic (ROC) curves across
the key SNR transition regime (-16 dB to -6 dB), comparing theoretical non-central
Chi-square predictions with empirical Monte Carlo evaluations, and plots AUC vs SNR.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.math_models.detection_metrics import (
    compute_detection_metrics,
    compute_empirical_roc,
    compute_theoretical_roc,
)
from src.math_models.energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    compute_energy_statistic,
)
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


def run_verification():
    print("=" * 60)
    print("Verifying P1-003: ROC Curves & Detection Metrics")
    print("=" * 60)

    l_samples = 1024
    noise_var = 1.0
    snr_levels = [-16, -13, -10, -7]
    colors = ["#4575b4", "#74add1", "#f46d43", "#d73027"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Family of ROC Curves (Pd vs Pfa)
    for snr, col in zip(snr_levels, colors):
        # 1. Theoretical curve
        th_roc = compute_theoretical_roc(snr_db=snr, num_samples=l_samples, noise_variance=noise_var)
        axes[0].plot(
            th_roc.pfa,
            th_roc.pd,
            color=col,
            linewidth=2,
            label=f"Theory SNR = {snr} dB (AUC = {th_roc.auc:.3f})",
        )

        # 2. Empirical Monte Carlo simulation (4000 frames)
        sim = HypothesisTestingSimulator(
            HypothesisConfig(num_samples=l_samples, noise_variance=noise_var, snr_db=snr, modulation="qpsk"),
            seed=42 + abs(snr),
        )
        x_batch, y_batch = sim.generate_batch(num_frames=4000, p_h1=0.5, as_iq_channels=True)
        scores = compute_energy_statistic(x_batch, normalize=False, noise_variance=noise_var)
        emp_roc = compute_empirical_roc(scores, y_batch, num_thresholds=150)

        # Scatter a subset of empirical points to show alignment
        sub_idx = np.linspace(0, len(emp_roc.pfa) - 1, 20, dtype=int)
        axes[0].plot(
            emp_roc.pfa[sub_idx],
            emp_roc.pd[sub_idx],
            "o",
            color=col,
            markersize=5,
            alpha=0.8,
        )

    # Reference diagonal (random guess)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_xlabel(r"Probability of False Alarm ($P_{\text{FA}}$)")
    axes[0].set_ylabel(r"Probability of Detection ($P_d$)")
    axes[0].set_title(rf"ROC Curves across SNR Regimes ($L={l_samples}$)")
    axes[0].legend(loc="lower right", framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Area Under Curve (AUC) vs SNR
    snr_sweep = np.linspace(-20, 0, 25)
    auc_curve = [compute_theoretical_roc(s, l_samples, noise_var).auc for s in snr_sweep]

    axes[1].plot(snr_sweep, auc_curve, "b-", linewidth=2.5, label="Theoretical AUC")
    axes[1].axhline(0.5, color="gray", linestyle=":", label="Lower Bound (0.50)")
    axes[1].axhline(1.0, color="green", linestyle=":", label="Upper Bound (1.00)")
    axes[1].axvline(-8.0, color="crimson", linestyle="--", alpha=0.7, label=r"Paper Filter Cutoff ($-8$ dB)")

    axes[1].set_xlabel("Signal-to-Noise Ratio (dB)")
    axes[1].set_ylabel("Area Under ROC Curve (AUC)")
    axes[1].set_title(r"Sensing Separability: AUC vs SNR ($L=1024$)")
    axes[1].legend(loc="lower right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PROJECT_ROOT / "results" / "plots" / "p1_003_roc_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"ROC figure successfully saved to: {out_path}")
    print("P1-003 Verification completed successfully!")


if __name__ == "__main__":
    run_verification()
