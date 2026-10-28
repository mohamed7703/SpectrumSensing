"""Verification script for P1-002: Energy Detection Decision Statistic.

Generates validation plots comparing the central and non-central Chi-Square
distributions, threshold placement, and analytical vs empirical Pd curves.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.math_models.energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    analytical_pd,
    compute_energy_statistic,
    compute_theoretical_threshold,
)
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


def run_verification():
    print("=" * 60)
    print("Verifying P1-002: Energy Detection Decision Statistic")
    print("=" * 60)

    l_samples = 1024
    noise_var = 1.0
    pfa_target = 0.05
    snr_eval_db = -10.0

    cfg = EnergyDetectorConfig(
        num_samples=l_samples,
        noise_variance=noise_var,
        pfa_target=pfa_target,
        method="exact",
    )
    detector = EnergyDetector(cfg)
    thresh = detector.threshold

    print(f"L={l_samples}, PFA_target={pfa_target}, Threshold gamma={thresh:.4f}")

    # Generate 10,000 frames under H0 and H1
    sim_h0 = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples, noise_variance=noise_var), seed=42)
    h0_frames, _ = sim_h0.generate_batch(num_frames=10_000, p_h1=0.0, as_iq_channels=True)

    sim_h1 = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples, noise_variance=noise_var, snr_db=snr_eval_db), seed=43)
    h1_frames, _ = sim_h1.generate_batch(num_frames=10_000, p_h1=1.0, as_iq_channels=True)

    stat_h0 = compute_energy_statistic(h0_frames, normalize=False, noise_variance=noise_var)
    stat_h1 = compute_energy_statistic(h1_frames, normalize=False, noise_variance=noise_var)

    emp_pfa = np.mean(stat_h0 >= thresh)
    emp_pd = np.mean(stat_h1 >= thresh)
    theory_pd = analytical_pd(pfa=pfa_target, snr_db=snr_eval_db, num_samples=l_samples, noise_variance=noise_var)

    print(f"Empirical PFA = {emp_pfa:.4f} (Target = {pfa_target:.4f})")
    print(f"At SNR={snr_eval_db} dB: Empirical Pd = {emp_pd:.4f}, Analytical Pd = {theory_pd:.4f}")

    # Sweep SNR from -20 to 0 dB to compare Analytical vs Empirical curves
    snr_range = np.linspace(-18, -4, 15)
    analytical_curve = [analytical_pd(pfa_target, s, l_samples, noise_var) for s in snr_range]
    empirical_curve = []

    for s in snr_range:
        sim_s = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples, noise_variance=noise_var, snr_db=s), seed=123)
        h1_sweep, _ = sim_s.generate_batch(num_frames=2000, p_h1=1.0, as_iq_channels=True)
        stats_s = compute_energy_statistic(h1_sweep, normalize=False, noise_variance=noise_var)
        empirical_curve.append(np.mean(stats_s >= thresh))

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Probability Density of Test Statistic D(Y)
    h1_label = f"H_1 (PU @ {snr_eval_db} dB)"
    cfar_label = f"CFAR Threshold $\\gamma={thresh:.4f}$"
    pfa_label = f"Target $P_{{\\text{{FA}}}}={pfa_target}$"

    axes[0].hist(stat_h0, bins=60, density=True, alpha=0.6, label="H_0 (Noise only)", color="steelblue")
    axes[0].hist(stat_h1, bins=60, density=True, alpha=0.6, label=h1_label, color="crimson")
    axes[0].axvline(thresh, color="black", linestyle="--", linewidth=2, label=cfar_label)
    axes[0].set_title("Energy Decision Statistic Distribution $D(Y)$")
    axes[0].set_xlabel("Test Statistic $D(Y) = \\frac{1}{L}\\sum |r(n)|^2$")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Pd vs SNR
    axes[1].plot(snr_range, analytical_curve, "b-", linewidth=2, label="Analytical $P_d$ (Non-central $\\chi^2$)")
    axes[1].plot(snr_range, empirical_curve, "ro", markersize=6, label="Monte Carlo Empirical $P_d$")
    axes[1].axhline(pfa_target, color="gray", linestyle=":", label=pfa_label)
    axes[1].set_title(r"Probability of Detection $P_d$ vs SNR ($L=1024$)")
    axes[1].set_xlabel("Signal-to-Noise Ratio (dB)")
    axes[1].set_ylabel(r"Detection Probability $P_d$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = PROJECT_ROOT / "results" / "plots" / "p1_002_energy_detection.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"Plot saved successfully to: {out_plot}")
    print("P1-002 Verification completed successfully!")


if __name__ == "__main__":
    run_verification()
