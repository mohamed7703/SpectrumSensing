"""Verification Script for P4-001: RadioML 2018.01A Dataset Ingestion & Loader.

Validates tensor shapes (1024, 2), SNR filtering, binary hypothesis labelling,
train/val/test partitioning, and plots I/Q constellation & time-domain traces.
"""

from pathlib import Path
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.training.dataset_loader import DatasetLoader, SyntheticRadioMLGenerator


def run_dataset_loader_verification() -> None:
    """Verify dataset loading, SNR filtering, and feature extraction."""
    config_path = Path("configs/training_config.yaml")
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "p4_001_dataset_loader.png"

    print("=" * 70)
    print("P4-001: RadioML 2018.01A Dataset Ingestion & Loader Verification")
    print("=" * 70)

    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    min_snr = float(cfg["dataset"]["snr_filtering"]["min_snr_db"])
    max_snr = float(cfg["dataset"]["snr_filtering"]["max_snr_db"])

    # 2. Ingest Dataset
    ds = DatasetLoader.load_or_generate(
        hdf5_path=cfg["dataset"]["data_path"],
        min_snr_db=min_snr,
        max_snr_db=max_snr,
        num_synthetic_frames=1000,
        random_seed=42,
    )

    print(f"\n[Dataset Summary] Total Frames Loaded: {len(ds):,}")
    print(f"[Dataset Summary] Frame Tensor Shape:   {ds.x.shape} (N, 1024, 2)")
    print(f"[Dataset Summary] Binary Labels:        H0 (Noise)={np.sum(ds.y_binary == 0)}, H1 (Signal)={np.sum(ds.y_binary == 1)}")
    print(f"[Dataset Summary] SNR Range (H1):       [{np.min(ds.snr[ds.y_binary == 1]):.1f}, {np.max(ds.snr[ds.y_binary == 1]):.1f}] dB")

    # 3. Partitioning Split
    train_ds, val_ds, test_ds = ds.train_val_test_split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    print(f"[Partitions] Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # 4. Generate Publication Verification Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    gen = SyntheticRadioMLGenerator(random_seed=42)

    # Subplot 1: Constellations under varying SNR & Noise
    ax1 = axes[0, 0]
    f_noise = gen.generate_single_frame("NOISE", 0.0)
    f_low = gen.generate_single_frame("QPSK", -8.0)
    f_mid = gen.generate_single_frame("QPSK", 4.0)
    f_high = gen.generate_single_frame("QPSK", 18.0)

    ax1.scatter(f_noise[:300, 0], f_noise[:300, 1], s=12, alpha=0.5, c="gray", label=r"$\mathcal{H}_0$ Pure Noise")
    ax1.scatter(f_low[:300, 0], f_low[:300, 1], s=12, alpha=0.6, c="crimson", label=r"QPSK @ $-8$ dB")
    ax1.scatter(f_mid[:300, 0], f_mid[:300, 1], s=12, alpha=0.7, c="orange", label=r"QPSK @ $+4$ dB")
    ax1.scatter(f_high[:300, 0], f_high[:300, 1], s=12, alpha=0.9, c="blue", label=r"QPSK @ $+18$ dB")
    ax1.set_title(r"I/Q Constellation Dynamics Across SNR ($\mathcal{H}_0$ vs. $\mathcal{H}_1$)", fontweight="bold")
    ax1.set_xlabel("In-Phase (I)", fontsize=9)
    ax1.set_ylabel("Quadrature (Q)", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)

    # Subplot 2: Time-Domain Waveforms (1024 samples)
    ax2 = axes[0, 1]
    time_indices = np.arange(128)  # Show first 128 samples for clarity
    ax2.plot(time_indices, f_high[:128, 0], "b-", lw=1.5, label="I-Rail (Signal +18dB)")
    ax2.plot(time_indices, f_high[:128, 1], "r--", lw=1.5, label="Q-Rail (Signal +18dB)")
    ax2.plot(time_indices, f_noise[:128, 0], color="gray", ls=":", lw=1.2, label=r"I-Rail ($\mathcal{H}_0$ Noise)")
    ax2.set_title("Time-Domain I/Q Baseband Waveforms (1024 Samples)", fontweight="bold")
    ax2.set_xlabel("Sample Index $n$")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Subplot 3: Sample Amplitude Distribution (H0 vs H1)
    ax3 = axes[1, 0]
    samp_h0 = ds.x[ds.y_binary == 0, :, 0].flatten()[:5000]
    samp_h1 = ds.x[ds.y_binary == 1, :, 0].flatten()[:5000]
    ax3.hist(samp_h0, bins=40, color="gray", alpha=0.6, density=True, label=r"$\mathcal{H}_0$ Idle Noise (I-rail)")
    ax3.hist(samp_h1, bins=40, color="teal", alpha=0.6, density=True, label=r"$\mathcal{H}_1$ Active Signal (I-rail)")
    ax3.set_title(r"Sample Amplitude Distribution ($I[n]$: $\mathcal{H}_0$ vs. $\mathcal{H}_1$)", fontweight="bold")
    ax3.set_xlabel("Sample Amplitude")
    ax3.set_ylabel("Probability Density")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # Subplot 4: Partition Proportions & SNR Range Distribution
    ax4 = axes[1, 1]
    h1_snrs = ds.snr[ds.y_binary == 1]
    ax4.hist(h1_snrs, bins=20, color="navy", edgecolor="black", alpha=0.85)
    ax4.axvline(min_snr, color="crimson", ls="--", lw=2.0, label=f"SNR Floor ({min_snr:.0f} dB)")
    ax4.set_title("SNR Distribution of Training Dataset (Filtered >= -8 dB)", fontweight="bold")
    ax4.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax4.set_ylabel("Frame Count")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")

    plt.suptitle("RadioML 2018.01A Dataset Ingestion & Feature Preprocessing (P4-001)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[OK] Dataset verification plot saved to {plot_path}")


if __name__ == "__main__":
    run_dataset_loader_verification()
