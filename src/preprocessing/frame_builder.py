"""Frame Builder and Stream Chunking for Cognitive Radio Sensing.

Segments continuous streaming SDR baseband samples into discrete fixed-length
observation frames matching the RadioML 2018.01A tensor specification: :math:`(N, 1024, 2)`.
"""

from __future__ import annotations

from typing import Literal, Tuple
import numpy as np


def chunk_stream_to_frames(
    continuous_samples: np.ndarray,
    frame_length: int = 1024,
    stride: int | None = None,
    remainder_mode: Literal["drop", "pad"] = "drop",
    as_iq_channels: bool = True,
) -> np.ndarray:
    r"""Slice a continuous time-series stream into fixed-size observation frames.

    Args:
        continuous_samples: 1D complex array of streamed samples, or 2D real array.
        frame_length: Observation window size :math:`L` (default 1024).
        stride: Hop size between consecutive frames. If None, equals `frame_length` (non-overlapping).
        remainder_mode: 'drop' discards leftover samples; 'pad' zero-pads the final frame.
        as_iq_channels: If True, returns tensor shape :math:`(N, L, 2)`. If False, returns :math:`(N, L)`.

    Returns:
        3D or 2D array of batched frames.
    """
    if continuous_samples.ndim == 2 and continuous_samples.shape[-1] == 2 and not np.iscomplexobj(continuous_samples):
        sig = continuous_samples[:, 0] + 1j * continuous_samples[:, 1]
    else:
        sig = np.asarray(continuous_samples, dtype=complex)

    total_len = len(sig)
    step = stride or frame_length

    if total_len < frame_length:
        if remainder_mode == "drop":
            empty_shape = (0, frame_length, 2) if as_iq_channels else (0, frame_length)
            return np.empty(empty_shape, dtype=float if as_iq_channels else complex)
        else:
            padded = np.zeros(frame_length, dtype=complex)
            padded[:total_len] = sig
            sig = padded
            total_len = frame_length

    # Calculate number of frames
    num_frames = (total_len - frame_length) // step + 1
    frames = []

    for i in range(num_frames):
        start = i * step
        end = start + frame_length
        frames.append(sig[start:end])

    # Handle remainder if padding requested
    remainder_start = num_frames * step
    if remainder_mode == "pad" and remainder_start < total_len:
        leftover = sig[remainder_start:]
        if len(leftover) > 0:
            padded = np.zeros(frame_length, dtype=complex)
            padded[: len(leftover)] = leftover
            frames.append(padded)

    frames_arr = np.array(frames, dtype=complex)

    if as_iq_channels:
        # Format into (N, L, 2)
        i_comp = np.real(frames_arr)
        q_comp = np.imag(frames_arr)
        return np.stack([i_comp, q_comp], axis=-1)

    return frames_arr


def complex_to_iq(complex_frames: np.ndarray) -> np.ndarray:
    """Convert complex array of shape (..., L) to real I/Q array of shape (..., L, 2)."""
    return np.stack([np.real(complex_frames), np.imag(complex_frames)], axis=-1)


def iq_to_complex(iq_frames: np.ndarray) -> np.ndarray:
    """Convert real I/Q array of shape (..., L, 2) to complex array of shape (..., L)."""
    if iq_frames.shape[-1] != 2:
        raise ValueError(f"Last axis must have size 2, got {iq_frames.shape[-1]}")
    return iq_frames[..., 0] + 1j * iq_frames[..., 1]
