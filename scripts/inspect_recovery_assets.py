"""Read local research assets without executing notebooks or modifying originals."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    """Identify the supplied local artifact using a streamed SHA-256 hash."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8*1024*1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Inspect specified local directory and write an exclusive provenance record."""
    import tensorflow as tf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapt-null-quantization-metadata", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing inspection")
    output = {"scope": "local artifact/code inspection only; no notebook execution or dataset download",
              "source_directory": str(args.directory), "tensorflow": tf.__version__,
              "notebooks": [], "models": [], "captures": []}
    for path in sorted(args.directory.glob("*.ipynb")):
        item = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        try:
            notebook = json.loads(path.read_text(encoding="utf-8-sig"))
            item["code_cells"] = [{"cell_index_zero_based": i, "source": "".join(cell["source"]),
                                   "execution_count": cell.get("execution_count"), "stored_outputs": len(cell.get("outputs", []))}
                                  for i, cell in enumerate(notebook.get("cells", [])) if cell.get("cell_type") == "code"]
        except (ValueError, UnicodeError) as error:
            item["parse_error"] = f"{type(error).__name__}: {error}"
        output["notebooks"].append(item)
    allowed = {"InputLayer", "Conv1D", "BatchNormalization", "MaxPooling1D", "GlobalAveragePooling1D",
               "Flatten", "Dense", "Dropout", "Activation", "ReLU"}
    rng = np.random.default_rng(1305)
    probe = rng.normal(0, 1/np.sqrt(2), (8, 1024, 2)).astype(np.float32)
    for path in sorted(args.directory.glob("*.h5")):
        item = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        with h5py.File(path, "r") as stream:
            architecture = json.loads(stream.attrs["model_config"])
            item["saved_keras_version"] = str(stream.attrs.get("keras_version"))
            item["layers"] = [layer["class_name"] for layer in architecture["config"]["layers"]]
            training = stream.attrs.get("training_config")
            item["saved_training_config"] = json.loads(training) if training else None
        if not set(item["layers"]).issubset(allowed):
            item["native_probe"] = {"status": "not_run", "reason": "architecture includes types outside the reviewed built-in allowlist"}
        else:
            try:
                model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
                predictions = np.asarray(model(probe, training=False)).reshape(-1)
                item["native_probe"] = {"status": "executed", "name": model.name,
                                        "input_shape": model.input_shape, "output_shape": model.output_shape,
                                        "parameters": model.count_params(), "all_weights_finite": all(np.all(np.isfinite(w)) for w in model.get_weights()),
                                        "gaussian_input_scores": predictions.tolist(), "seed": 1305,
                                        "scope": "load/execution only; synthetic input scores are not a performance estimate"}
            except Exception as error:
                item["native_probe"] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
                if args.adapt_null_quantization_metadata and "quantization_config" in str(error):
                    # Explicit in-memory compatibility mapping, never edit the HDF5.
                    adjusted = json.loads(json.dumps(architecture))
                    removed = []
                    for layer in adjusted["config"]["layers"]:
                        settings = layer["config"]
                        if "quantization_config" in settings:
                            if settings["quantization_config"] is not None:
                                raise ValueError("refusing to remove non-null quantization state")
                            settings.pop("quantization_config")
                            removed.append(settings["name"])
                    # Legacy HDF5 configs omit modern JSON module registrations;
                    # Model.from_config accepts their functional graph config.
                    adapted = tf.keras.Model.from_config(adjusted["config"])
                    adapted.load_weights(path)
                    verified_values = 0
                    with h5py.File(path, "r") as stream:
                        for layer in adapted.layers:
                            group = stream["model_weights"][layer.name]
                            original_weights = [np.asarray(group[name]) for name in group.attrs["weight_names"]]
                            loaded_weights = layer.get_weights()
                            if len(original_weights) != len(loaded_weights):
                                raise ValueError("weight count changed under metadata adaptation")
                            for original, loaded in zip(original_weights, loaded_weights):
                                np.testing.assert_array_equal(original, loaded)
                                verified_values += original.size
                    if verified_values != adapted.count_params():
                        raise ValueError("not all native parameter/state values matched the original HDF5")
                    item["adapted_probe"] = {"status": "executed_with_explicit_metadata_adaptation",
                                             "removed_only_null_quantization_config_in_layers": removed,
                                             "original_file_unchanged": digest(path) == item["sha256"],
                                             "bitwise_verified_weight_and_state_values": verified_values,
                                             "parameters": adapted.count_params(),
                                             "gaussian_input_scores": np.asarray(adapted(probe, training=False)).reshape(-1).tolist(),
                                             "scope": "compatibility and execution diagnostic only; no historical performance validation"}
        output["models"].append(item)
    for path in sorted(args.directory.glob("*noise*.dat")):
        size = path.stat().st_size
        item = {"path": str(path), "bytes": size, "sha256": digest(path),
                "format_assumption": "little-endian complex64, inferred from notebook; no capture metadata establishes format/RF ground truth"}
        if size % 8 == 0:
            samples = np.memmap(path, mode="r", dtype="<c8")
            prefix = np.asarray(samples[:min(len(samples), 400000)])
            finite = np.isfinite(prefix)
            item["prefix_statistics"] = {"inspected_samples": len(prefix), "finite_samples": int(finite.sum()),
                                         "mean_real": float(prefix.real[finite].mean()), "mean_imaginary": float(prefix.imag[finite].mean()),
                                         "mean_power": float(np.mean(np.abs(prefix[finite].astype(np.complex128))**2)),
                                         "complete_1024_sample_frames": size//8//1024}
            del samples
        output["captures"].append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"notebooks": len(output["notebooks"]), "models": [{"path": item["path"], "probe": item["native_probe"]} for item in output["models"]],
                      "captures": output["captures"]}, indent=2))


if __name__ == "__main__":
    main()
