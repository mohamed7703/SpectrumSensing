"""Recalculate saved audit scores without rewriting historical results or figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.evaluator import BenchmarkEvaluator


def main() -> None:
    """Read explicit frozen scores and write a new, exclusive measurement record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new run record")
    with np.load(args.scores, allow_pickle=False) as data:
        labels = data["labels"]
        results = {}
        for name, key in (("energy_detector", "energy"), ("historical_feature_classifier", "model_scores")):
            scores = data[key]
            positive = scores[labels == 1]
            negative = scores[labels == 0]
            # Small frozen audit fixture only: intentionally independent slow oracle.
            expected = float(np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative)))
            roc = BenchmarkEvaluator().compute_roc(labels, scores)
            if not np.isclose(roc.auc, expected, atol=1e-12, rtol=0):
                raise AssertionError(f"{name}: ROC AUC disagrees with independent ranking")
            results[name] = {"auc": roc.auc, "independent_rank_auc": expected, "eer_interpolated": roc.eer}
    record = {
        "scope": "Recalculation of frozen synthetic audit scores; no neural training or deployment validation",
        "input_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT/"src/math_models/detection_metrics.py", ROOT/"src/evaluation/evaluator.py")},
        "observations": int(len(labels)),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
