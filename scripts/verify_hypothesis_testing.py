"""Standalone verification script for P1-001: Binary Hypothesis Testing.

Generates demonstration plots comparing H0 (idle noise) and H1 (PU active)
under various SNRs and modulations, and logs empirical verification metrics.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


def run_verification():
    print("=" * 60)
    print("Verifying P1-001: Binary Hypothesis Testing Model")
    print("=" * 60)

    # 1. Instantiate simulator with SNR = 3 dB, 1024 samples
    cfg = HypothesisConfig(num_samples=1024, noise_variance=1.0, snr_db=3.0, modulation="qpsk")
    sim = HypothesisTestingSimulator(cfg, seed=42)

    stats = sim.theoretical_statistics()
    print(f"Theoretical stats: {stats}")

    # 2. Generate 2000 frames
    x_batch, y_batch = sim.generate_batch(num_frames=2000, p_h1=0.5, as_iq_channels=True)
    print(f"Generated batch: X={x_batch.shape}, y={y_batch.shape}")

    # Compute empirical power under H0 and H1
    h0_idx = np.where(y_batch == 0)[0]
    h1_idx = np.where(y_batch == 1)[0]

    h0_frames = x_batch[h0_idx]  # (N_0, L, 2)
    h1_frames = x_batch[h1_idx]  # (N_1, L, 2)

    emp_var_h0 = np.mean(h0_frames**2) * 2.0  # Sum of I^2 + Q^2 variance
    emp_var_h1 = np.mean(h1_frames**2) * 2.0

    print(f"H0 Variance -> Theoretical: {stats['var_h0']:.4f}, Empirical: {emp_var_h0:.4f}")
    print(f"H1 Variance -> Theoretical: {stats['var_h1']:.4f}, Empirical: {emp_var_h1:.4f}")

    # 3. Create visual figure for verification
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series (In-Phase component)
    time_axis = np.arange(100)
    axes[0, 0].plot(time_axis, h0_frames[0, :100, 0], label=r"$\mathcal{H}_0$ (Noise)", color="steelblue")
    axes[0, 0].plot(time_axis, h1_frames[0, :100, 0], label=r"$\mathcal{H}_1$ (PU + Noise)", color="crimson", alpha=0.7)
    axes[0, 0].set_title("Time-Domain Waveform (In-Phase, first 100 samples)")
    axes[0, 0].set_xlabel("Sample index $n$")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # I/Q Constellation scatter
    axes[0, 1].scatter(h0_frames[0, :, 0], h0_frames[0, :, 1], alpha=0.4, label=r"$\mathcal{H}_0$", color="steelblue", s=15)
    axes[0, 1].scatter(h1_frames[0, :, 0], h1_frames[0, :, 1], alpha=0.4, label=r"$\mathcal{H}_1$ (QPSK @ 3dB)", color="crimson", s=15)
    axes[0, 1].set_title("I/Q Plane Constellation")
    axes[0, 1].set_xlabel("In-Phase (I)")
    axes[0, 1].set_ylabel("Quadrature (Q)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Amplitude histograms
    amp_h0 = np.sqrt(h0_frames[:, :, 0] ** 2 + h0_frames[:, :, 1] ** 2).flatten()
    amp_h1 = np.sqrt(h1_frames[:, :, 0] ** 2 + h1_frames[:, :, 1] ** 2).flatten()
    axes[1, 0].hist(amp_h0, bins=50, density=True, alpha=0.5, label=r"$\mathcal{H}_0$ (Rayleigh envelope)", color="steelblue")
    axes[1, 0].hist(amp_h1, bins=50, density=True, alpha=0.5, label=r"$\mathcal{H}_1$ Envelope", color="crimson")
    axes[1, 0].set_title("Signal Envelope Distribution $|r(n)|$")
    axes[1, 0].set_xlabel("Envelope $|r(n)|$")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Instantaneous Power distribution
    power_h0 = (h0_frames[:, :, 0] ** 2 + h0_frames[:, :, 1] ** 2).flatten()
    power_h1 = (h1_frames[:, :, 0] ** 2 + h1_frames[:, :, 1] ** 2).flatten()
    axes[1, 1].hist(power_h0, bins=50, density=True, alpha=0.5, label=r"$\mathcal{H}_0$ Power (exp)", color="steelblue")
    axes[1, 1].hist(power_h1, bins=50, density=True, alpha=0.5, label=r"$\mathcal{H}_1$ Power", color="crimson")
    axes[1, 1].set_title("Instantaneous Sample Power $|r(n)|^2$")
    axes[1, 1].set_xlabel("Power")
    axes[1, 1].set_ylabel("Probability Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot = PROJECT_ROOT / "results" / "plots" / "p1_001_hypothesis_testing.png"
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"Plot saved successfully to: {output_plot}")
    print("Verification completed cleanly!")


if __name__ == "__main__":
    run_verification()
