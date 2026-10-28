r"""Verification and Training Dynamics Demonstration for Spectrum Sensing (P4-004).

Simulates the training pipeline across multiple epochs, tracking BCE loss convergence,
detection probability (Pd), false alarm rate (Pfa), and AUC progression matching the paper.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.dataset_loader import DatasetLoader
from src.training.trainer import (
    SpectrumSensingTrainer,
    EarlyStopping,
    ModelCheckpoint,
    compute_binary_metrics,
)
from src.models import BaselineCNN, BatchNormCNN, GAPCNN, BiGRUModel


def run_training_verification():
    print("=" * 70)
    print("Cognitive Radio Spectrum Sensing — Training Pipeline Verification")
    print("=" * 70)

    # 1. Load Dataset (Synthetic RadioML Generator Fallback with SNR >= -8 dB)
    print("\n--- Ingesting RadioML Frames ---")
    dataset = DatasetLoader.load_or_generate(
        hdf5_path="data/GOLD_XYZ_OSC.0001_1024.hdf5",
        min_snr_db=-8.0,
        max_snr_db=30.0,
        num_synthetic_frames=600,
    )
    train_split, val_split, test_split = dataset.train_val_test_split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    print(f"Train split: {len(train_split.x)} frames (SNR range: {train_split.snr.min():.0f} to {train_split.snr.max():.0f} dB)")
    print(f"Val split:   {len(val_split.x)} frames")
    print(f"Test split:  {len(test_split.x)} frames")

    # 2. Setup Trainer and Callbacks
    trainer = SpectrumSensingTrainer(config_path="configs/training_config.yaml")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4)

    ckpt_dir = os.path.join("saved_models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(filepath=os.path.join(ckpt_dir, "batchnorm_cnn"), monitor="val_loss")

    # 3. Train BatchNorm 1D-CNN Model
    model = BatchNormCNN()
    epochs = 12
    batch_size = 64

    print(f"\n--- Training {model.name} for {epochs} Epochs ---")
    history = trainer.fit(
        model=model,
        train_data=(train_split.x, train_split.y_binary),
        val_data=(val_split.x, val_split.y_binary),
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        verbose=True,
    )

    # 4. Evaluate on Held-Out Test Set
    print("\n--- Evaluating on Test Set ---")
    test_metrics = trainer.evaluate(model, (test_split.x, test_split.y_binary))
    print(f"Test Loss:     {test_metrics.loss:.4f}")
    print(f"Test Accuracy: {test_metrics.accuracy * 100:.2f}%")
    print(f"Detection Pd:  {test_metrics.detection_probability * 100:.2f}%")
    print(f"False Alarm:   {test_metrics.false_alarm_rate * 100:.2f}%")
    print(f"ROC AUC:       {test_metrics.auc:.4f}")

    # 5. Generate Verification Plots
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    plot_path = os.path.join("results", "plots", "p4_004_training_pipeline.png")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = np.arange(1, len(history["loss"]) + 1)

    # Subplot 1: Loss Convergence (Train vs Val)
    axs[0, 0].plot(epochs_range, history["loss"], "o-", label="Train BCE Loss", color="#4C72B0", lw=2)
    axs[0, 0].plot(epochs_range, history["val_loss"], "s--", label="Val BCE Loss", color="#C44E52", lw=2)
    axs[0, 0].set_xlabel("Epoch", fontsize=11)
    axs[0, 0].set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    axs[0, 0].set_title("Training & Validation Loss Convergence", fontsize=12, fontweight="bold")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend()

    # Subplot 2: Accuracy Trajectory
    axs[0, 1].plot(epochs_range, np.array(history["accuracy"]) * 100, "o-", label="Train Accuracy", color="#55A868", lw=2)
    axs[0, 1].plot(epochs_range, np.array(history["val_accuracy"]) * 100, "s--", label="Val Accuracy", color="#8172B3", lw=2)
    axs[0, 1].set_xlabel("Epoch", fontsize=11)
    axs[0, 1].set_ylabel("Accuracy (%)", fontsize=11)
    axs[0, 1].set_title("Accuracy Trajectory Across Epochs", fontsize=12, fontweight="bold")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend()

    # Subplot 3: Detection Probability Pd and False Alarm Rate Pfa
    axs[1, 0].plot(epochs_range, np.array(history["val_pd"]) * 100, "o-", label=r"Validation $P_d$ (Recall)", color="#2ca02c", lw=2)
    axs[1, 0].plot(epochs_range, np.array(history["val_pfa"]) * 100, "v--", label=r"Validation $P_{fa}$ (False Alarm)", color="#d62728", lw=2)
    axs[1, 0].axhline(y=90.0, color="gray", linestyle=":", label=r"IEEE 802.22 Target $P_d \geq 90\%$")
    axs[1, 0].axhline(y=10.0, color="black", linestyle=":", label=r"IEEE 802.22 Limit $P_{fa} \leq 10\%$")
    axs[1, 0].set_xlabel("Epoch", fontsize=11)
    axs[1, 0].set_ylabel("Rate (%)", fontsize=11)
    axs[1, 0].set_title(r"Cognitive Radio Sensing Metrics ($P_d$ vs $P_{fa}$)", fontsize=12, fontweight="bold")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend()

    # Subplot 4: Validation ROC-AUC Trajectory
    axs[1, 1].plot(epochs_range, history["val_auc"], "D-", label="Validation ROC-AUC", color="#9467bd", lw=2)
    axs[1, 1].set_xlabel("Epoch", fontsize=11)
    axs[1, 1].set_ylabel("Area Under ROC Curve (AUC)", fontsize=11)
    axs[1, 1].set_title("Discriminative Capability (ROC-AUC)", fontsize=12, fontweight="bold")
    axs[1, 1].set_ylim([0.8, 1.02])
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nTraining pipeline verification plot saved to: {plot_path}")

    return history, test_metrics


if __name__ == "__main__":
    run_training_verification()
