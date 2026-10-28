"""Verification script for P2-002: I/Q Signal Preprocessing and Augmentations.

Visualizes DC offset removal, Gram-Schmidt I/Q imbalance correction,
power normalization, and data augmentations (CFO, phase, timing).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.iq_normalizer import (
    correct_iq_imbalance,
    normalize_power,
    remove_dc_offset,
)
from src.preprocessing.augmentation import apply_cfo
from src.dsp.spectrogram import compute_spectrogram


def run_verification():
    print("=" * 60)
    print("Verifying P2-002: I/Q Preprocessing & Normalization Pipeline")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 2048

    # 1. Synthesize 16-QAM or QPSK signal
    bits_i = 2 * rng.integers(0, 2, n) - 1
    bits_q = 2 * rng.integers(0, 2, n) - 1
    ideal_qpsk = (bits_i + 1j * bits_q) / np.sqrt(2.0)

    # 2. Inject direct-conversion analog impairments
    # DC bias (LO leakage)
    dc_bias = 0.40 + 0.30j
    # Gain imbalance (25% higher Q gain) and phase skew (20 degrees)
    gain_imbalance = 1.25
    phase_skew_rad = np.radians(20.0)

    i_skew = np.real(ideal_qpsk)
    q_skew = gain_imbalance * (np.sin(phase_skew_rad) * i_skew + np.cos(phase_skew_rad) * np.imag(ideal_qpsk))
    distorted_sig = (i_skew + 1j * q_skew) + dc_bias

    # Add channel noise
    distorted_sig += rng.normal(0, 0.08, n) + 1j * rng.normal(0, 0.08, n)

    # 3. Apply preprocessing pipeline
    # Step A: Remove DC offset
    dc_cleaned = remove_dc_offset(distorted_sig)
    # Step B: Correct I/Q amplitude & phase imbalance
    orth_corrected = correct_iq_imbalance(dc_cleaned)
    # Step C: Normalize to unit power
    fully_normalized = normalize_power(orth_corrected, target_power=1.0)

    # Verification calculations
    cov_before = float(np.mean((np.real(distorted_sig) - np.mean(np.real(distorted_sig))) * (np.imag(distorted_sig) - np.mean(np.imag(distorted_sig)))))
    cov_after = float(np.mean(np.real(fully_normalized) * np.imag(fully_normalized)))
    power_after = float(np.mean(np.abs(fully_normalized) ** 2))

    print(f"I/Q Covariance -> Distorted: {cov_before:.4f}, Corrected: {cov_after:.4f}")
    print(f"Final Normalized Power: {power_after:.4f} (Target = 1.0000)")

    # 4. Generate multi-panel demonstration figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Constellation BEFORE Preprocessing
    axes[0].scatter(np.real(distorted_sig), np.imag(distorted_sig), alpha=0.5, color="crimson", s=20)
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_title(f"Impaired Frontend (DC bias + Skew)\nCovariance = {cov_before:.3f}")
    axes[0].set_xlabel("In-Phase (I)")
    axes[0].set_ylabel("Quadrature (Q)")
    axes[0].set_xlim([-2.5, 2.5])
    axes[0].set_ylim([-2.5, 2.5])
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Constellation AFTER Preprocessing
    axes[1].scatter(np.real(fully_normalized), np.imag(fully_normalized), alpha=0.5, color="teal", s=20)
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_title(f"Cleaned & Normalized Constellation\nPower = {power_after:.3f}, Cov = {cov_after:.4f}")
    axes[1].set_xlabel("In-Phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlim([-2.5, 2.5])
    axes[1].set_ylim([-2.5, 2.5])
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Spectrogram of CFO-Augmented Signal
    cfo_sig = apply_cfo(fully_normalized, freq_offset_hz=2.0e5, sample_rate=1.0e6)
    times, freqs, spec_db = compute_spectrogram(cfo_sig, sample_rate=1.0e6, nperseg=64, noverlap=48, return_db=True)

    im = axes[2].pcolormesh(times * 1e3, freqs / 1e3, spec_db.T, shading="gouraud", cmap="plasma")
    fig.colorbar(im, ax=axes[2], label="Power (dB)")
    axes[2].set_title("CFO Data Augmentation (+200 kHz Shift)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Frequency (kHz)")

    plt.tight_layout()
    out_path = PROJECT_ROOT / "results" / "plots" / "p2_002_iq_preprocessing.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Preprocessing plot saved successfully to: {out_path}")
    print("P2-002 Verification completed successfully!")


if __name__ == "__main__":
    run_verification()
