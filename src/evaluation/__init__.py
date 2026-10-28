"""Cognitive Radio Spectrum Sensing Evaluation and Benchmarking Module.

Provides comprehensive benchmark evaluation between deep learning architectures
and classical detection methods across ROC curves, SNR sweeps, and confusion matrices.
"""

from src.evaluation.evaluator import (
    ROCResult,
    SNRSweepResult,
    ModelBenchmarkResult,
    BenchmarkEvaluator,
)

__all__ = [
    "ROCResult",
    "SNRSweepResult",
    "ModelBenchmarkResult",
    "BenchmarkEvaluator",
]
