r"""RadioML 2018.01A Dataset Ingestion and Preprocessing Pipeline.

Provides memory-efficient ingestion for the RadioML 2018.01A dataset
(2,555,904 frames of shape (1024, 2) across 24 modulation classes and 26 SNR levels),
binary hypothesis mapping ($\mathcal{H}_0$ Noise vs $\mathcal{H}_1$ Signal),
SNR filtering and an explicitly limited synthetic pulse-signal fixture generator.

Mathematical Notation:
    The binary hypothesis for cognitive radio sensing is:
    $$y = \begin{cases}
        0 \ (\mathcal{H}_0), & \text{if PU is absent (pure thermal noise } w[n]\text{)} \\
        1 \ (\mathcal{H}_1), & \text{if PU is active (modulated RF carrier } s[n] + w[n]\text{)}
    \end{cases}$$
    Input feature tensors are normalized complex baseband signals:
    $$\mathbf{X}_i \in \mathbb{R}^{1024 \times 2}, \quad \mathbf{X}_i[n] = [I[n], Q[n]]$$
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from pathlib import Path
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# 24 standard RadioML 2018.01A modulation types
MODULATION_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

# 26 standard SNR levels from -20 dB to +30 dB (step = 2 dB)
ALL_SNR_LEVELS = [
    -20, -18, -16, -14, -12, -10, -8, -6, -4, -2,
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
]


@dataclass
class RadioMLDataset:
    """In-memory or sliced partition of the spectrum sensing dataset.

    Attributes:
        x: Array of shape (N, 1024, 2) representing I and Q channels.
        y_binary: Binary hypothesis labels (0 for H0 noise, 1 for H1 signal).
        y_mod: Modulation class labels or names.
        snr: Signal-to-Noise Ratio in dB for each frame.
    """
    x: np.ndarray
    y_binary: np.ndarray
    y_mod: np.ndarray
    snr: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def filter_by_snr(self, min_snr_db: float = -8.0, max_snr_db: float = 30.0) -> RadioMLDataset:
        """Filter dataset to include only frames within specified SNR window.

        This is an experimental selection policy, not a universal detectability
        limit. A robust energy-detector wall depends on the uncertainty model.

        Args:
            min_snr_db: Minimum SNR in dB (inclusive).
            max_snr_db: Maximum SNR in dB (inclusive).

        Returns:
            Filtered RadioMLDataset.
        """
        # Note: Include H0 pure noise samples regardless of SNR tag
        mask = (self.snr >= min_snr_db) & (self.snr <= max_snr_db) | (self.y_binary == 0)
        return RadioMLDataset(
            x=self.x[mask],
            y_binary=self.y_binary[mask],
            y_mod=self.y_mod[mask],
            snr=self.snr[mask],
        )

    def train_val_test_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = 42,
    ) -> Tuple[RadioMLDataset, RadioMLDataset, RadioMLDataset]:
        """Randomly split frames for local smoke tests; this is not grouped or stratified.

        Use capture/session groups and four independent train/validation/
        calibration/test partitions for scientific evaluation. This helper
        cannot establish independence between adjacent or common-source frames.

        Args:
            train_ratio: Fraction for training (e.g. 0.70).
            val_ratio: Fraction for validation (e.g. 0.15).
            test_ratio: Fraction for testing (e.g. 0.15).
            random_seed: RNG seed for reproducible shuffling.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if not np.all(np.isfinite(ratios)) or np.any(ratios < 0) or not np.isclose(ratios.sum(), 1):
            raise ValueError("split ratios must be finite, nonnegative and sum to one")
        total = len(self)
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(total)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        idx_train = indices[:train_end]
        idx_val = indices[train_end:val_end]
        idx_test = indices[val_end:]

        train_ds = RadioMLDataset(self.x[idx_train], self.y_binary[idx_train], self.y_mod[idx_train], self.snr[idx_train])
        val_ds = RadioMLDataset(self.x[idx_val], self.y_binary[idx_val], self.y_mod[idx_val], self.snr[idx_val])
        test_ds = RadioMLDataset(self.x[idx_test], self.y_binary[idx_test], self.y_mod[idx_test], self.snr[idx_test])

        return train_ds, val_ds, test_ds

    def batch_generator(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield batches of (X, y_binary) for model training or evaluation.

        Args:
            batch_size: Number of frames per mini-batch.
            shuffle: Whether to permute indices before each epoch.
            random_seed: RNG seed for shuffling.

        Yields:
            Tuple of (batch_X of shape (B, 1024, 2), batch_y of shape (B,)).
        """
        n = len(self)
        rng = np.random.default_rng(random_seed)
        indices = np.arange(n)
        if shuffle:
            indices = rng.permutation(n)

        for start_idx in range(0, n, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield self.x[batch_idx], self.y_binary[batch_idx]


class SyntheticRadioMLGenerator:
    """Synthetic rectangular-pulse fixtures using a RadioML-like tensor schema.

    Allows complete unit testing, training pipeline validation, and benchmarking
    without the external HDF5. These are not a reproduction of RadioML waveforms,
    channel impairments, or modulation coverage. Noise power is fixed at one;
    SNR scales signal power. Both hypotheses retain raw received frame energy.
    """

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

    def generate_single_frame(
        self,
        modulation: str,
        snr_db: float,
        num_samples: int = 1024,
    ) -> np.ndarray:
        """Synthesize a single (1024, 2) I/Q frame with calibrated AWGN at given SNR.

        Args:
            modulation: Modulation class name or 'NOISE'.
            snr_db: Target Signal-to-Noise Ratio in dB.
            num_samples: Length of frame (standard 1024).

        Returns:
            np.ndarray of shape (num_samples, 2) float32.
        """
        if not np.isfinite(snr_db) or num_samples < 1 or int(num_samples) != num_samples:
            raise ValueError("finite SNR and positive integer sample count required")
        modulation = modulation.upper()
        if modulation in ("NOISE", "H0"):
            # H0: Pure complex circular AWGN
            noise = (self.rng.normal(0.0, 1.0 / np.sqrt(2.0), num_samples) +
                     1j * self.rng.normal(0.0, 1.0 / np.sqrt(2.0), num_samples))
            return np.stack([noise.real, noise.imag], axis=-1).astype(np.float32)

        # Synthesize digital symbol carrier
        sps = 8
        n_symbols = int(np.ceil(num_samples / sps)) + 2

        if modulation == "OOK":
            syms = self.rng.choice(np.array([0., np.sqrt(2.)]), size=n_symbols)
        elif modulation in ("4ASK", "8ASK"):
            order = int(modulation[:-3])
            const = np.arange(1-order, order, 2, dtype=float)
            syms = self.rng.choice(const / np.sqrt(np.mean(const**2)), size=n_symbols)
        elif modulation == "BPSK":
            const = np.array([-1.0, 1.0])
            syms = self.rng.choice(const, size=n_symbols)
        elif modulation == "QPSK":
            const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2.0)
            syms = self.rng.choice(const, size=n_symbols)
        elif modulation in ("8PSK", "16PSK", "32PSK"):
            angles = np.linspace(0, 2 * np.pi, int(modulation[:-3]), endpoint=False)
            const = np.exp(1j * angles)
            syms = self.rng.choice(const, size=n_symbols)
        elif modulation in ("16QAM", "64QAM", "256QAM"):
            width = int(np.sqrt(int(modulation[:-3])))
            grid = np.arange(1-width, width, 2, dtype=float)
            grid /= np.sqrt(2 * np.mean(grid**2))
            i_syms = self.rng.choice(grid, size=n_symbols)
            q_syms = self.rng.choice(grid, size=n_symbols)
            syms = i_syms + 1j * q_syms
        else:
            raise ValueError(f"modulation {modulation!r} is not supported by this limited synthetic generator")

        if syms is not None:
            pulses = np.repeat(syms, sps)[:num_samples]
            # Small random carrier frequency offset
            cfo = self.rng.uniform(-0.02, 0.02)
            t = np.arange(num_samples)
            sig = pulses * np.exp(1j * 2 * np.pi * cfo * t)

        # Power normalize clean signal
        sig_pow = np.mean(np.abs(sig) ** 2)
        if sig_pow > 0:
            sig = sig / np.sqrt(sig_pow)

        # Add calibrated AWGN
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0
        noise = (self.rng.normal(0.0, np.sqrt(noise_var / 2.0), num_samples) +
                 1j * self.rng.normal(0.0, np.sqrt(noise_var / 2.0), num_samples))
        rx = np.sqrt(snr_linear) * sig + noise

        return np.stack([rx.real, rx.imag], axis=-1).astype(np.float32)

    def generate_dataset(
        self,
        frames_per_class: int = 20,
        modulations: Optional[List[str]] = None,
        snrs: Optional[List[int]] = None,
        h0_ratio: float = 0.5,
    ) -> RadioMLDataset:
        """Generate balanced synthetic RadioML dataset for spectrum sensing.

        Args:
            frames_per_class: Number of frames per (modulation, SNR) combination.
            modulations: Supported subset; defaults to first eight digital classes.
            snrs: SNR levels; defaults to [-8,-4,0,4,8,12,16,20].
            h0_ratio: Fraction of H0 (idle noise) frames to include for balanced sensing.

        Returns:
            RadioMLDataset with shape (N, 1024, 2).
        """
        if not 0 <= h0_ratio < 1 or frames_per_class < 1 or int(frames_per_class) != frames_per_class:
            raise ValueError("require 0 <= h0_ratio < 1 and positive integer frames_per_class")
        mods = modulations or MODULATION_CLASSES[:8]
        snr_list = snrs or [-8, -4, 0, 4, 8, 12, 16, 20]

        x_list = []
        y_bin_list = []
        y_mod_list = []
        snr_vals = []

        # 1. Generate H1 active signal frames
        for m in mods:
            for s in snr_list:
                for _ in range(frames_per_class):
                    frame = self.generate_single_frame(modulation=m, snr_db=float(s))
                    x_list.append(frame)
                    y_bin_list.append(1)  # H1: PU present
                    y_mod_list.append(m)
                    snr_vals.append(float(s))

        num_h1 = len(x_list)
        # 2. Generate H0 idle noise frames to achieve target h0_ratio
        num_h0 = int(num_h1 * (h0_ratio / (1.0 - h0_ratio))) if h0_ratio < 1.0 else num_h1

        for _ in range(num_h0):
            frame = self.generate_single_frame(modulation="NOISE", snr_db=0.0)
            x_list.append(frame)
            y_bin_list.append(0)  # H0: PU absent
            y_mod_list.append("NOISE")
            snr_vals.append(-99.0)  # Tag noise with -99 dB

        return RadioMLDataset(
            x=np.array(x_list, dtype=np.float32),
            y_binary=np.array(y_bin_list, dtype=np.int32),
            y_mod=np.array(y_mod_list),
            snr=np.array(snr_vals, dtype=np.float32),
        )


class DatasetLoader:
    """Unified loader supporting real RadioML HDF5 files and synthetic fallback."""

    @classmethod
    def load_or_generate(
        cls,
        hdf5_path: Optional[Union[str, Path]] = None,
        min_snr_db: float = -8.0,
        max_snr_db: float = 30.0,
        num_synthetic_frames: int = 500,
        random_seed: int = 42,
        allow_synthetic_fallback: bool = False,
    ) -> RadioMLDataset:
        """Load an explicitly supplied HDF5, or explicitly request synthetic fixtures.

        Args:
            hdf5_path: Path to RadioML HDF5 file (e.g. data/GOLD_XYZ_OSC.0001_1024.hdf5).
            min_snr_db: Filter threshold for SNR.
            max_snr_db: Max SNR threshold.
            num_synthetic_frames: Number of synthetic frames if generating.
            random_seed: RNG seed for synthetic generator.
            allow_synthetic_fallback: Opt in to replacing a missing explicit path
                with synthetic fixtures. Omitted hdf5_path requests fixtures.

        Returns:
            RadioMLDataset ready for training or evaluation.
        """
        if hdf5_path is not None:
            p = Path(hdf5_path)
            if p.exists():
                return cls.load_from_hdf5(p, min_snr_db, max_snr_db)
            if not allow_synthetic_fallback:
                raise FileNotFoundError(f"Requested HDF5 file does not exist: {p}")

        # Explicitly requested synthetic fixtures, never substitute invisibly.
        gen = SyntheticRadioMLGenerator(random_seed=random_seed)
        frames_per = max(2, num_synthetic_frames // (8 * 8))
        ds = gen.generate_dataset(frames_per_class=frames_per)
        return ds.filter_by_snr(min_snr_db, max_snr_db)

    @classmethod
    def load_from_hdf5(
        cls,
        hdf5_path: Union[str, Path],
        min_snr_db: float = -8.0,
        max_snr_db: float = 30.0,
        max_samples: Optional[int] = None,
    ) -> RadioMLDataset:
        """Load a bounded selected subset, filtering metadata before reading I/Q.

        This convenience function returns an in-memory array. It is not memory
        mapping or an out-of-core training loader. A 512 MiB I/Q guard requires
        callers to cap large selections. All RadioML rows remain H1.

        Args:
            hdf5_path: Path to .hdf5 file.
            min_snr_db: SNR filter floor.
            max_snr_db: SNR filter ceiling.
            max_samples: Optional cap on frames to read into RAM.

        Returns:
            RadioMLDataset.
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required to read RadioML HDF5 files.")

        with h5py.File(hdf5_path, "r") as f:
            # Standard RadioML 2018.01A keys: 'X', 'Y', 'Z'
            x_ds = f["X"]
            y_ds = f["Y"]
            z_ds = f["Z"]

            if not np.isfinite(min_snr_db) or not np.isfinite(max_snr_db) or min_snr_db > max_snr_db:
                raise ValueError("SNR bounds must be finite and ordered")
            if max_samples is not None and (max_samples < 1 or int(max_samples) != max_samples):
                raise ValueError("max_samples must be a positive integer")
            if (x_ds.ndim != 3 or x_ds.shape[-1] != 2 or y_ds.shape != (len(x_ds), 24)
                    or z_ds.size != len(x_ds)):
                raise ValueError("invalid RadioML X/Y/Z schema or inconsistent lengths")
            # Z is small metadata; avoid reading X before selecting its indices.
            snr_all = np.asarray(z_ds[:], dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(snr_all)):
                raise ValueError("nonfinite SNR metadata")
            indices = np.flatnonzero((snr_all >= min_snr_db) & (snr_all <= max_snr_db))
            if max_samples is not None:
                indices = indices[:int(max_samples)]
            n = len(indices)
            if n * np.prod(x_ds.shape[1:]) * np.dtype(np.float32).itemsize > 512 * 1024**2:
                raise MemoryError("selection exceeds 512 MiB; specify max_samples or use bounded batch ingestion")
            x = np.asarray(x_ds[indices], dtype=np.float32)
            y_onehot = np.asarray(y_ds[indices])
            snr = snr_all[indices]
            if not np.all(np.isfinite(x)) or not np.all((y_onehot == 0) | (y_onehot == 1)) or not np.all(y_onehot.sum(axis=-1) == 1):
                raise ValueError("invalid I/Q samples or one-hot modulation labels")

            # Map one-hot to class names and binary labels
            # In RadioML, all 24 classes are active signals (H1). Pure noise frames can be added or filtered.
            mod_indices = np.argmax(y_onehot, axis=-1)
            mod_names = np.array([MODULATION_CLASSES[idx] if idx < len(MODULATION_CLASSES) else "UNKNOWN" for idx in mod_indices])
            y_binary = np.ones(n, dtype=np.int32)  # All RadioML frames are H1 signals

        ds = RadioMLDataset(x=x, y_binary=y_binary, y_mod=mod_names, snr=snr)
        return ds.filter_by_snr(min_snr_db, max_snr_db)
