"""Verification script for P1-004: Wireless Channel Fading Models.

Plots analytical envelope distributions for Rayleigh, Rician, and Nakagami-m fading,
and compares empirical detection performance under AWGN vs Rayleigh vs Rician channels.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.math_models.channel_fading import (
    FadingChannel,
    FadingChannelConfig,
    generate_fading_coefficients,
    nakagami_pdf,
    rayleigh_pdf,
    rician_pdf,
)
from src.math_models.energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    analytical_pd,
    compute_energy_statistic,
)
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


def run_verification():
    print("=" * 60)
    print("Verifying P1-004: Channel Fading Models & Impact on Sensing")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Analytical and Empirical Fading Envelope PDFs
    a_axis = np.linspace(0.0, 3.0, 500)
    omega = 1.0

    # 1. Rayleigh
    axes[0].plot(a_axis, rayleigh_pdf(a_axis, sigma=np.sqrt(omega / 2.0)), "b-", linewidth=2, label="Rayleigh (Theory)")
    h_ray = generate_fading_coefficients(30_000, FadingChannelConfig(fading_type="rayleigh"))
    axes[0].hist(np.abs(h_ray), bins=60, density=True, alpha=0.3, color="blue", label="Rayleigh (Empirical)")

    # 2. Rician (K = 4)
    axes[0].plot(a_axis, rician_pdf(a_axis, k_factor=4.0, average_power=omega), "r-", linewidth=2, label="Rician $K=4$ (Theory)")
    h_ric = generate_fading_coefficients(30_000, FadingChannelConfig(fading_type="rician", rician_k=4.0))
    axes[0].hist(np.abs(h_ric), bins=60, density=True, alpha=0.3, color="red", label="Rician $K=4$ (Empirical)")

    # 3. Nakagami-m (m = 2.0)
    axes[0].plot(a_axis, nakagami_pdf(a_axis, m_param=2.0, average_power=omega), "g-", linewidth=2, label="Nakagami $m=2$ (Theory)")
    h_nak = generate_fading_coefficients(30_000, FadingChannelConfig(fading_type="nakagami", nakagami_m=2.0))
    axes[0].hist(np.abs(h_nak), bins=60, density=True, alpha=0.3, color="green", label="Nakagami $m=2$ (Empirical)")

    axes[0].set_title(r"Fading Envelope Distributions $|h|$ ($\Omega=1$)")
    axes[0].set_xlabel("Envelope Amplitude $a = |h|$")
    axes[0].set_ylabel("Probability Density $f(a)$")
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Detection Probability Pd vs SNR: AWGN vs Rayleigh vs Rician
    l_samples = 1024
    detector = EnergyDetector(EnergyDetectorConfig(num_samples=l_samples, pfa_target=0.05))

    snr_sweep = np.linspace(-16, 8, 13)
    pd_awgn_theory = [analytical_pd(0.05, s, l_samples) for s in snr_sweep]
    pd_rayleigh_emp = []
    pd_rician_emp = []

    chan_ray = FadingChannel(FadingChannelConfig(fading_type="rayleigh", block_fading=True), seed=42)
    chan_ric = FadingChannel(FadingChannelConfig(fading_type="rician", rician_k=4.0, block_fading=True), seed=43)

    for s in snr_sweep:
        sim = HypothesisTestingSimulator(HypothesisConfig(num_samples=l_samples, snr_db=s), seed=100 + int(s))

        # Rayleigh batch
        frames_ray = [sim.generate_batch(1, 1.0, channel_gain=generate_fading_coefficients(1, chan_ray.config, rng=sim.rng)[0], as_iq_channels=True)[0][0] for _ in range(1500)]
        res_ray = detector.evaluate_performance(np.array(frames_ray), np.ones(1500))
        pd_rayleigh_emp.append(res_ray["pd"])

        # Rician batch
        frames_ric = [sim.generate_batch(1, 1.0, channel_gain=generate_fading_coefficients(1, chan_ric.config, rng=sim.rng)[0], as_iq_channels=True)[0][0] for _ in range(1500)]
        res_ric = detector.evaluate_performance(np.array(frames_ric), np.ones(1500))
        pd_rician_emp.append(res_ric["pd"])

    axes[1].plot(snr_sweep, pd_awgn_theory, "k--", linewidth=2, label="AWGN (No Fading, Theory)")
    axes[1].plot(snr_sweep, pd_rician_emp, "r-s", markersize=5, linewidth=1.8, label="Rician ($K=4$, Monte Carlo)")
    axes[1].plot(snr_sweep, pd_rayleigh_emp, "b-o", markersize=5, linewidth=1.8, label="Rayleigh (NLOS, Monte Carlo)")
    axes[1].axhline(0.05, color="gray", linestyle=":", label=r"Target $P_{\text{FA}}=0.05$")
    axes[1].axhline(0.95, color="green", linestyle=":", label=r"Reliable Sensing Threshold ($95\%$)")

    axes[1].set_title(r"Impact of Multipath Fading on Sensing $P_d$ ($L=1024$)")
    axes[1].set_xlabel("Average SNR (dB)")
    axes[1].set_ylabel(r"Detection Probability $P_d$")
    axes[1].legend(loc="lower right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PROJECT_ROOT / "results" / "plots" / "p1_004_channel_fading.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Fading verification figure saved to: {out_path}")
    print("P1-004 Verification completed successfully!")


if __name__ == "__main__":
    run_verification()
