"""Check whole-group isolation and bounded HDF5 reads using a local tiny fixture."""

import h5py
import numpy as np
import pytest
from src.training.data_protocol import grouped_partition_indices, iter_hdf5_batches


def test_whole_groups_are_complete_disjoint_and_repeatable() -> None:
    """No source group crosses roles, and every source frame is assigned once."""
    groups = np.repeat(np.array([f"capture{i}" for i in range(20)]), np.arange(1, 21))
    proportions = {"train": .55, "validation": .15, "calibration": .15, "test": .15}
    first = grouped_partition_indices(groups, proportions, 44)
    second = grouped_partition_indices(groups, proportions, 44)
    seen = set()
    for name, indices in first.items():
        np.testing.assert_array_equal(indices, second[name])
        sources = set(groups[indices])
        assert not (seen & sources)
        seen.update(sources)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(len(groups)))


def test_streaming_reads_bounded_rows_and_preserves_unsorted_order(tmp_path, monkeypatch) -> None:
    """HDF5 I/Q indexing is limited to each declared batch, never the complete file."""
    path = tmp_path/"fixture.hdf5"
    frames = np.arange(12*16*2, dtype=np.float32).reshape(12, 16, 2)
    with h5py.File(path, "w") as stream:
        stream["X"] = frames
        stream["Y"] = np.eye(24)[np.arange(12)]
        stream["Z"] = np.arange(12).reshape(-1, 1)
    original = h5py.Dataset.__getitem__
    reads = []
    def observed(dataset, key):
        """Record I/Q selection lengths while delegating fixture reads unchanged."""
        if dataset.name == "/X":
            reads.append(len(key))
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", observed)
    selected = np.array([10, 2, 8, 1, 6])
    batches = list(iter_hdf5_batches(path, selected, 2))
    np.testing.assert_array_equal(np.concatenate([batch.x for batch in batches]), frames[selected])
    np.testing.assert_array_equal(np.concatenate([batch.snr for batch in batches]), selected)
    assert reads == [2, 2, 1]


def test_streaming_rejects_duplicate_source_rows(tmp_path) -> None:
    """Repeated row IDs must not silently alter effective test counts."""
    with pytest.raises(ValueError, match="unique"):
        list(iter_hdf5_batches(tmp_path/"unused.hdf5", np.array([1, 1]), 2))
