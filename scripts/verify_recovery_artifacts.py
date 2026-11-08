"""Verify preserved evidence, native gradient connectivity and recovery artifact identity."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.training.dataset_loader import SyntheticRadioMLGenerator


def sha256(path: Path) -> str:
    """Compute a streamed SHA-256 identifier for the specified local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Check explicit input records and write an exclusive verification artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--native-record", type=Path, required=True)
    parser.add_argument("--assets-record", type=Path, required=True)
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing verification record")
    baseline = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    preserved = []
    for filename, expected in baseline.items():
        path = Path(filename)
        protected = ("results" in path.parts or "saved_models" in path.parts or path.parent.name == "scripts"
                     or path.name in {"COMPREHENSIVE_VERIFICATION_REPORT.md", "sensors-24-07907.pdf"})
        if protected:
            if sha256(path) != expected:
                raise AssertionError(f"historical evidence changed: {path}")
            preserved.append(str(path))
    assets = json.loads(args.assets_record.read_text(encoding="utf-8"))
    local_assets = []
    for category in ["notebooks", "models", "captures"]:
        for item in assets[category]:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise AssertionError(f"original local asset changed: {item['path']}")
            local_assets.append(item["path"])
    test_tree = ET.parse(args.test_results)
    cases = test_tree.findall(".//testcase")
    if not cases or any(case.find("failure") is not None or case.find("error") is not None for case in cases):
        raise AssertionError("test results contain failures/errors or no test cases")
    record = json.loads(args.native_record.read_text(encoding="utf-8"))
    data = record["config"]["data"]
    validation = SyntheticRadioMLGenerator(data["validation_seed"]).generate_dataset(
        data["validation_frames_per_condition"], data["modulations"], data["snrs_db"], data["h0_fraction"])
    import tensorflow as tf
    native = {}
    exported = []
    for name, evidence in record["models"].items():
        path = Path(evidence["checkpoint"])
        training = json.loads((path.parent/"training.json").read_text(encoding="utf-8"))
        if sha256(path) != training["checkpoint_sha256"]:
            raise AssertionError("checkpoint identity changed")
        model = tf.keras.models.load_model(path, safe_mode=True)
        with tf.GradientTape() as tape:
            predictions = model(validation.x, training=False)
            # No weight decay or regularizer term: gradients must come from BCE.
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                validation.y_binary[:, None].astype(np.float32), predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        observations = []
        for variable, gradient in zip(model.trainable_variables, gradients):
            if gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient))):
                raise AssertionError(f"disconnected/nonfinite BCE gradient: {variable.path}")
            observations.append({"variable": variable.path, "shape": list(variable.shape),
                                 "bce_gradient_l2": float(tf.linalg.norm(gradient).numpy())})
        for layer in ["conv1d_1", "conv1d_2"]:
            kernel = next(item for item in observations if item["variable"].endswith(f"{layer}/kernel"))
            if kernel["bce_gradient_l2"] <= 0:
                raise AssertionError("convolution kernel has zero BCE gradient")
        native[name] = {"checkpoint_sha256": sha256(path), "parameters": model.count_params(),
                        "optimizer_iterations": int(model.optimizer.iterations.numpy()), "variables": observations,
                        "scope": "BCE graph connectivity with dropout disabled; no generalization or RNG-resume claim"}
        for mode, item in evidence["exports"].items():
            export_path = Path(item["path"])
            if sha256(export_path) != item["sha256"]:
                raise AssertionError("export identity changed")
            exported.append({"model": name, "mode": mode, "sha256": item["sha256"], "bytes": export_path.stat().st_size})
    packages = {distribution.metadata["Name"]: importlib.metadata.version(distribution.metadata["Name"])
                for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
              "preserved_historical_files": preserved, "preserved_local_assets": local_assets,
              "tests": {"file": str(args.test_results), "sha256": sha256(args.test_results), "cases": len(cases),
                        "skipped": sum(case.find("skipped") is not None for case in cases), "failures": 0, "errors": 0},
              "native_bce_gradients": native, "verified_exports": exported, "packages": dict(sorted(packages.items())),
              "limits": ["Does not validate original scientific conclusions", "No RadioML dataset download or full evaluation",
                         "Historical reports remain evidence of the baseline, not corrected-run certificates"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"preserved_historical_files": len(preserved), "preserved_local_assets": len(local_assets),
                      "passing_test_cases": len(cases), "verified_exports": len(exported),
                      "convolution_bce_gradient_norms": {name: [item["bce_gradient_l2"] for item in details["variables"]
                                                               if "conv1d" in item["variable"] and item["variable"].endswith("kernel")]
                                                         for name, details in native.items()}}, indent=2))


if __name__ == "__main__":
    main()
