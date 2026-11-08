r"""Benchmark Evaluation Engine for Cognitive Radio Spectrum Sensing (P4-005).

Provides rigorous benchmark comparisons between deep learning architectures
(Baseline 1D-CNN, BatchNorm 1D-CNN, GAP 1D-CNN, Bi-GRU) and classical Energy Detection
across the full SNR spectrum ($-20\text{ dB}$ to $+30\text{ dB}$).

Mathematical Formulation:
    1. Receiver Operating Characteristic (ROC):
       For decision threshold $\eta \in [0, 1]$:
       $$P_{fa}(\eta) = P(\hat{p} \ge \eta \mid \mathcal{H}_0) = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
       $$P_d(\eta) = P(\hat{p} \ge \eta \mid \mathcal{H}_1) = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
       $$\text{AUC} = \int_{0}^1 P_d(P_{fa}) \, dP_{fa}$$

    2. Fixed-signal-power uncertainty model:
       For noise variance in [1/rho, rho], robust mean separation requires
       S > rho - 1/rho (about -3.33 dB for a 1 dB bound). This is not a
       universal impossibility result or evidence of neural superiority.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from src.models.base_model import BaseSpectrumClassifier
from src.math_models.energy_detection import compute_energy_statistic, compute_theoretical_threshold
from src.training.dataset_loader import SyntheticRadioMLGenerator, RadioMLDataset
from src.math_models.detection_metrics import compute_empirical_roc, validate_binary_inputs
from src.math_models.threshold_calibration import calibrate_score_threshold


@dataclass
class ROCResult:
    """Container for Receiver Operating Characteristic curve data."""
    pfa: np.ndarray
    pd: np.ndarray
    thresholds: np.ndarray
    auc: float
    eer: float  # Equal Error Rate where Pfa = 1 - Pd


@dataclass
class SNRSweepResult:
    """Detection performance as a function of SNR."""
    snrs: np.ndarray
    pd: np.ndarray
    pfa: np.ndarray


@dataclass
class ModelBenchmarkResult:
    """Comprehensive benchmark results for a single architecture."""
    name: str
    total_params: int
    trainable_params: int
    inference_latency_us: float
    roc: ROCResult
    snr_sweep: SNRSweepResult
    confusion_matrix: np.ndarray  # 2x2 [[TN, FP], [FN, TP]]


class BenchmarkEvaluator:
    """Cognitive Radio Spectrum Sensing Benchmark and Comparison Suite."""

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        """Initialize benchmark evaluator."""
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def compute_roc(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        num_thresholds: int = 250,
    ) -> ROCResult:
        """Compute empirical ROC curve, Area Under Curve (AUC), and Equal Error Rate (EER).

        Args:
            y_true: Binary ground truth labels (0 for H0, 1 for H1).
            scores: Continuous prediction scores or probabilities.
            num_thresholds: Compatibility argument; all distinct score thresholds
                are used, independent of plotting resolution.

        Returns:
            ROCResult dataclass.
        """
        roc = compute_empirical_roc(scores, y_true, num_thresholds=num_thresholds)
        # Linear interpolation on the empirical ROC (randomized threshold
        # operating points). EER need not be attainable by one fixed threshold.
        difference = roc.pfa + roc.pd - 1.0
        upper = int(np.flatnonzero(difference >= 0)[0])
        if difference[upper] == 0:
            eer = float(roc.pfa[upper])
        else:
            lower = upper - 1
            fraction = -difference[lower] / (difference[upper] - difference[lower])
            eer = float(roc.pfa[lower] + fraction * (roc.pfa[upper] - roc.pfa[lower]))
        return ROCResult(
            pfa=roc.pfa,
            pd=roc.pd,
            thresholds=roc.thresholds,
            auc=roc.auc,
            eer=eer,
        )

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Generate 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        scores, y_true = validate_binary_inputs(scores, y_true)
        if not np.isscalar(threshold) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite real scalar")
        decisions = (scores >= threshold).astype(np.int32)

        tn = int(np.sum((decisions == 0) & (y_true == 0)))
        fp = int(np.sum((decisions == 1) & (y_true == 0)))
        fn = int(np.sum((decisions == 0) & (y_true == 1)))
        tp = int(np.sum((decisions == 1) & (y_true == 1)))

        return np.array([[tn, fp], [fn, tp]], dtype=np.int32)

    def evaluate_model_snr_sweep(
        self,
        model: BaseSpectrumClassifier,
        snrs: List[float],
        frames_per_snr: int = 40,
        target_pfa: float = 0.05,
    ) -> SNRSweepResult:
        """Measure model detection probability across SNR levels from -20 dB to +30 dB.

        Args:
            model: Spectrum sensing classifier.
            snrs: List of SNR levels in dB.
            frames_per_snr: Evaluation frames per SNR step.
            target_pfa: Nominal false alarm target constraint.

        Returns:
            SNRSweepResult containing snrs, Pd array, and Pfa array.
        """
        gen = SyntheticRadioMLGenerator(random_seed=self.random_seed)
        pd_vals = []
        pfa_vals = []

        # Find calibrated threshold for target Pfa under H0 noise
        noise_frames = np.array([gen.generate_single_frame("NOISE", 0.0) for _ in range(200)])
        noise_scores = model.predict_proba(noise_frames)
        # Threshold corresponding to upper (1 - target_pfa) quantile
        threshold = calibrate_score_threshold(noise_scores, target_pfa).threshold

        for snr in snrs:
            # Generate H1 frames across modulations at this SNR
            h1_frames = np.array([gen.generate_single_frame("QPSK", snr) for _ in range(frames_per_snr)])
            h1_scores = np.asarray(model.predict_proba(h1_frames), dtype=np.float64)
            pd_vals.append(float(np.mean(h1_scores >= threshold)))

            # H0 noise evaluation
            h0_frames = np.array([gen.generate_single_frame("NOISE", 0.) for _ in range(frames_per_snr)])
            h0_scores = np.asarray(model.predict_proba(h0_frames), dtype=np.float64)
            pfa_vals.append(float(np.mean(h0_scores >= threshold)))

        return SNRSweepResult(
            snrs=np.array(snrs, dtype=np.float32),
            pd=np.array(pd_vals, dtype=np.float32),
            pfa=np.array(pfa_vals, dtype=np.float32),
        )

    def evaluate_energy_detector_snr_sweep(
        self,
        snrs: List[float],
        num_samples: int = 1024,
        noise_uncertainty_db: float = 1.0,
        target_pfa: float = 0.05,
        trials_per_snr: int = 100,
    ) -> SNRSweepResult:
        r"""Evaluate classical Energy Detector with noise power uncertainty.

        Estimates average performance under log-uniform bounded noise variance.
        Fixed nominal signal power is independent of noise uncertainty. This
        average-case Monte Carlo is not proof of a worst-case SNR wall.

        Args:
            snrs: List of SNR levels in dB.
            num_samples: Frame sample count N (1024).
            noise_uncertainty_db: Noise uncertainty bound $\Delta_{\text{dB}}$ (default 1.0 dB).
            target_pfa: Target false alarm rate.
            trials_per_snr: Monte-Carlo trials per SNR level.

        Returns:
            SNRSweepResult for Energy Detector.
        """
        rng = np.random.default_rng(self.random_seed)
        rho = 10.0 ** (noise_uncertainty_db / 10.0)

        threshold = compute_theoretical_threshold(target_pfa, num_samples, noise_variance=rho)

        pd_list = []
        pfa_list = []
        log_rho = np.log(rho)

        for snr in snrs:
            snr_lin = 10.0 ** (snr / 10.0)
            det_count = 0
            fa_count = 0

            for _ in range(trials_per_snr):
                # Noise variance fluctuates log-uniformly in [1/rho, rho]
                sigma2_n = np.exp(rng.uniform(-log_rho, log_rho))
                noise = rng.normal(0.0, np.sqrt(sigma2_n / 2.0), (num_samples, 2))

                # H0 trial (pure noise)
                test_stat_h0 = np.mean(noise[:, 0]**2 + noise[:, 1]**2)
                if test_stat_h0 >= threshold:
                    fa_count += 1

                # H1 trial (signal + noise)
                sig_amp = np.sqrt(snr_lin / 2.0)
                signal = sig_amp * np.ones((num_samples, 2))
                rx = signal + noise
                test_stat_h1 = np.mean(rx[:, 0]**2 + rx[:, 1]**2)
                if test_stat_h1 >= threshold:
                    det_count += 1

            pd_list.append(det_count / trials_per_snr)
            pfa_list.append(fa_count / trials_per_snr)

        return SNRSweepResult(
            snrs=np.array(snrs, dtype=np.float32),
            pd=np.array(pd_list, dtype=np.float32),
            pfa=np.array(pfa_list, dtype=np.float32),
        )
