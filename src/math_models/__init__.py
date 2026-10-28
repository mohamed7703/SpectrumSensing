"""Mathematical models package for cognitive radio spectrum sensing.

This package contains theoretical and statistical foundations:
- Binary hypothesis testing (H0 vs H1)
- Energy detection decision statistics
- Detection metrics (Pd, Pfa) and ROC curve analysis
- Channel fading models (Rayleigh, Rician, Nakagami-m)
- Noise models and the SNR wall phenomenon
"""

from .hypothesis_testing import (
    HypothesisTestingSimulator,
    HypothesisConfig,
    generate_pu_signal,
    generate_awgn,
    simulate_received_signal,
)
from .energy_detection import (
    EnergyDetector,
    EnergyDetectorConfig,
    compute_energy_statistic,
    compute_theoretical_threshold,
    analytical_pd,
)
from .detection_metrics import (
    ConfusionMatrix,
    DetectionMetrics,
    ROCCurveResult,
    compute_confusion_matrix,
    compute_detection_metrics,
    compute_empirical_roc,
    compute_theoretical_roc,
)
from .channel_fading import (
    FadingChannel,
    FadingChannelConfig,
    rayleigh_pdf,
    rician_pdf,
    nakagami_pdf,
    generate_fading_coefficients,
)
from .noise_models import (
    CalibratedAWGNGenerator,
    NoiseModelConfig,
    compute_snr_wall,
    compute_sample_complexity,
)

__all__ = [
    "HypothesisTestingSimulator",
    "HypothesisConfig",
    "generate_pu_signal",
    "generate_awgn",
    "simulate_received_signal",
    "EnergyDetector",
    "EnergyDetectorConfig",
    "compute_energy_statistic",
    "compute_theoretical_threshold",
    "analytical_pd",
    "ConfusionMatrix",
    "DetectionMetrics",
    "ROCCurveResult",
    "compute_confusion_matrix",
    "compute_detection_metrics",
    "compute_empirical_roc",
    "compute_theoretical_roc",
    "FadingChannel",
    "FadingChannelConfig",
    "rayleigh_pdf",
    "rician_pdf",
    "nakagami_pdf",
    "generate_fading_coefficients",
    "CalibratedAWGNGenerator",
    "NoiseModelConfig",
    "compute_snr_wall",
    "compute_sample_complexity",
]
