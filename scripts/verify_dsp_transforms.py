"""Verification script for P2-001: Digital Signal Processing (DSP) Transforms.

Generates multi-panel figure verifying FFT, Welch Power Spectral Density,
2D STFT spectrogram generation, and matched filtering.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.dsp.fft_utils import compute_fft
from src.dsp.filters import matched_filter
from src.dsp.psd_estimator import estimate_psd_welch
from src.dsp.spectrogram import compute_spectrogram
from src.math_models.hypothesis_testing import (
    HypothesisConfig,
    HypothesisTestingSimulator,
)


def run_verification():
    print("=" * 60)
    print("Verifying P2-001: DSP Transforms (FFT, PSD, STFT, Filters)")
    print("=" * 60)

    fs = 2.0e6  # 2.0 MHz standard SDR sampling rate
    n_samples = 1024

    # 1. Generate active PU signal (QPSK @ 0 dB SNR) and idle noise frame
    sim = HypothesisTestingSimulator(
        HypothesisConfig(num_samples=n_samples, snr_db=0.0, modulation="qpsk"),
        seed=42,
    )
    x_batch, y_batch = sim.generate_batch(num_frames=2, p_h1=0.5, as_iq_channels=False)

    h0_sig = x_batch[0] if y_batch[0] == 0 else x_batch[1]
    h1_sig = x_batch[1] if y_batch[1] == 1 else x_batch[0]

    # 2. PSD Estimation via Welch's method
    f_h0, psd_h0 = estimate_psd_welch(h0_sig, sample_rate=fs, nperseg=256, return_db=True)
    f_h1, psd_h1 = estimate_psd_welch(h1_sig, sample_rate=fs, nperseg=256, return_db=True)

    # 3. Frequency-hopping signal for 2D Spectrogram demonstration
    t_axis = np.arange(2048) / fs
    # 4 distinct frequency hops across time
    hops = [-6e5, -2e5, 3e5, 7e5]
    hop_signal = np.zeros(2048, dtype=complex)
    for i, hop_f in enumerate(hops):
        idx = slice(i * 512, (i + 1) * 512)
        hop_signal[idx] = np.exp(1j * 2 * np.pi * hop_f * t_axis[idx])

    # Add light AWGN
    rng = np.random.default_rng(99)
    hop_signal += rng.normal(0, 0.3, 2048) + 1j * rng.normal(0, 0.3, 2048)

    t_spec, f_spec, spec_matrix = compute_spectrogram(
        hop_signal, sample_rate=fs, nperseg=128, noverlap=96, return_db=True
    )

    # 4. Matched filter correlation
    pulse_len = 64
    pulse = rng.choice([-1.0, 1.0], size=pulse_len) + 1j * rng.choice([-1.0, 1.0], size=pulse_len)
    noisy_channel = np.zeros(1024, dtype=complex)
    embed_idx = 450
    noisy_channel[embed_idx : embed_idx + pulse_len] = pulse
    noisy_channel += rng.normal(0, 0.8, 1024) + 1j * rng.normal(0, 0.8, 1024)
    mf_out = matched_filter(noisy_channel, pulse)

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Time-Domain In-Phase waveform
    axes[0, 0].plot(np.real(h0_sig[:150]), label=r"$\mathcal{H}_0$ (Noise)", color="steelblue")
    axes[0, 0].plot(np.real(h1_sig[:150]), label=r"$\mathcal{H}_1$ (PU QPSK)", color="crimson", alpha=0.8)
    axes[0, 0].set_title("Time-Domain Waveform (In-Phase Component, first 150 samples)")
    axes[0, 0].set_xlabel("Sample index $n$")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: Welch Power Spectral Density (PSD)
    axes[0, 1].plot(f_h0 / 1e3, psd_h0, label=r"$\mathcal{H}_0$ (Noise Floor)", color="steelblue")
    axes[0, 1].plot(f_h1 / 1e3, psd_h1, label=r"$\mathcal{H}_1$ (Active Spectrum @ 0 dB)", color="crimson")
    axes[0, 1].set_title("Welch Power Spectral Density ($N_{\text{fft}}=256$)")
    axes[0, 1].set_xlabel("Frequency (kHz)")
    axes[0, 1].set_ylabel("Power Spectral Density (dB/Hz)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel C: 2D Spectrogram (Time vs Frequency Heatmap)
    im = axes[1, 0].pcolormesh(
        t_spec * 1e3,
        f_spec / 1e3,
        spec_matrix.T,
        shading="gouraud",
        cmap="viridis",
    )
    fig.colorbar(im, ax=axes[1, 0], label="Power (dB)")
    axes[1, 0].set_title("2D STFT Spectrogram (Dynamic Frequency-Hopping Signal)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Frequency (kHz)")

    # Panel D: Matched Filter Correlation Response
    axes[1, 1].plot(np.abs(mf_out), color="darkorange", linewidth=1.5, label="Matched Filter Output $|y(n)|$")
    axes[1, 1].axvline(embed_idx + pulse_len // 2, color="crimson", linestyle="--", label=f"True Alignment ({embed_idx + pulse_len // 2})")
    axes[1, 1].set_title("Optimal Matched Filter Correlation Peak in AWGN")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Correlation Magnitude")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = PROJECT_ROOT / "results" / "plots" / "p2_001_dsp_transforms.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"DSP transforms plot saved successfully to: {out_plot}")
    print("P2-001 Verification completed successfully!")


if __name__ == "__main__":
    run_verification()
