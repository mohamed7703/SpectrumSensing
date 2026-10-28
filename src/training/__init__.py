"""Training Pipeline and Optimization Module for Spectrum Sensing Models.

Includes RadioML dataset ingestion, Binary Cross-Entropy loss with L2 decay,
SGD optimization, early stopping, and training coordination.
"""

from src.training.dataset_loader import (
    RadioMLDataset,
    SyntheticRadioMLGenerator,
    DatasetLoader,
)
from src.training.loss_functions import BinaryCrossEntropyLoss
from src.training.trainer import (
    EvaluationMetrics,
    compute_binary_metrics,
    SGDOptimizer,
    EarlyStopping,
    ModelCheckpoint,
    TrainingHistory,
    SpectrumSensingTrainer,
)

__all__ = [
    "RadioMLDataset",
    "SyntheticRadioMLGenerator",
    "DatasetLoader",
    "BinaryCrossEntropyLoss",
    "EvaluationMetrics",
    "compute_binary_metrics",
    "SGDOptimizer",
    "EarlyStopping",
    "ModelCheckpoint",
    "TrainingHistory",
    "SpectrumSensingTrainer",
]
