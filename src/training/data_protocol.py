"""Local preparation primitives; source groups must come from real provenance."""

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .dataset_loader import MODULATION_CLASSES, RadioMLDataset


def grouped_partition_indices(groups: np.ndarray, proportions: dict[str, float],
                              random_seed: int) -> dict[str, np.ndarray]:
    """Assign whole caller-supplied groups to four immutable-role partitions.

    Args:
        groups: One nonempty string capture/session/realization ID per frame.
            Frame indices are not evidence of independent source groups.
        proportions: Positive group fractions for train, validation, calibration, test.
        random_seed: Seed controlling group allocation.

    Fractions apply to groups, not frame counts; unequal group sizes can yield
    unequal class/SNR mixtures. Inspect and publish actual resulting counts.
    This helper does not invent unavailable RadioML capture-group provenance.
    """
    names = ["train", "validation", "calibration", "test"]
    ids = np.asarray(groups)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "US" or np.any(ids == ""):
        raise ValueError("groups must be a nonempty vector of nonempty source IDs")
    if set(proportions) != set(names):
        raise ValueError("four explicit train/validation/calibration/test proportions required")
    ratios = np.asarray([proportions[name] for name in names], dtype=float)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("positive finite proportions summing to one required")
    unique = np.unique(ids)
    if len(unique) < len(names):
        raise ValueError("at least four independent groups required")
    unique = np.random.default_rng(random_seed).permutation(unique)
    # Reserve one group per role, then distribute remaining groups by largest remainder.
    desired = ratios * (len(unique)-len(names))
    counts = np.floor(desired).astype(int) + 1
    leftovers = len(unique)-counts.sum()
    counts[np.argsort(-(desired-np.floor(desired)), kind="stable")[:leftovers]] += 1
    offset = 0
    result = {}
    for name, count in zip(names, counts):
        result[name] = np.flatnonzero(np.isin(ids, unique[offset:offset+count]))
        offset += count
    return result


def iter_hdf5_batches(path: Path, indices: np.ndarray, batch_size: int) -> Iterator[RadioMLDataset]:
    """Read explicitly selected H1 rows in bounded batches, preserving caller order.

    Args:
        path: Local X/Y/Z HDF5; no download or synthetic fallback occurs.
        indices: Unique, in-range integer source rows from an independent manifest.
        batch_size: Maximum number of I/Q frames read in a single operation.

    The caller owns SNR selection and grouped manifests. RadioML supplies H1;
    independently sourced and consistently processed H0 is still required.
    """
    rows = np.asarray(indices)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows):
        raise ValueError("indices must be a unique integer vector")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    with h5py.File(path, "r") as stream:
        x, y, z = [stream[name] for name in ["X", "Y", "Z"]]
        if x.ndim != 3 or x.shape[-1] != 2 or y.shape != (len(x), 24) or z.size != len(x):
            raise ValueError("inconsistent X/Y/Z schema")
        if np.any(rows < 0) or np.any(rows >= len(x)):
            raise ValueError("source indices out of range")
        for offset in range(0, len(rows), batch_size):
            selected = rows[offset:offset+batch_size]
            sort = np.argsort(selected)
            inverse = np.argsort(sort)
            sorted_rows = selected[sort]
            frames = np.asarray(x[sorted_rows], dtype=np.float32)[inverse]
            labels = np.asarray(y[sorted_rows])[inverse]
            snrs = np.asarray(z[sorted_rows], dtype=np.float32).reshape(-1)[inverse]
            if (not np.all(np.isfinite(frames)) or not np.all(np.isfinite(snrs))
                    or not np.all((labels == 0) | (labels == 1)) or not np.all(labels.sum(axis=1) == 1)):
                raise ValueError("nonfinite samples/metadata or invalid one-hot modulation labels")
            names = np.asarray(MODULATION_CLASSES)[np.argmax(labels, axis=1)]
            yield RadioMLDataset(frames, np.ones(len(frames), dtype=np.int32), names, snrs)
