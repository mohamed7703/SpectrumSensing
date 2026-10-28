"""Physical-layer I/Q signal preprocessing package.

Provides tools for cleaning, framing, augmenting, and normalizing RF baseband signals:
- DC offset and carrier leakage removal
- Gram-Schmidt I/Q imbalance correction
- Exact average power normalization
- Continuous stream to (1024, 2) frame building
- Calibrated AWGN noise injection across target SNR levels
- Stochastic RF data augmentations (CFO, phase jitter, timing shift)
"""

from .iq_normalizer import (
    correct_iq_imbalance,
    normalize_power,
    remove_dc_offset,
)
from .frame_builder import (
    chunk_stream_to_frames,
    complex_to_iq,
    iq_to_complex,
)
from .noise_injector import (
    inject_batch_snr_sweep,
    inject_noise_at_snr,
)
from .augmentation import (
    apply_cfo,
    apply_random_phase_rotation,
    apply_time_shift,
    augment_frame,
)

__all__ = [
    "remove_dc_offset",
    "correct_iq_imbalance",
    "normalize_power",
    "chunk_stream_to_frames",
    "complex_to_iq",
    "iq_to_complex",
    "inject_noise_at_snr",
    "inject_batch_snr_sweep",
    "apply_cfo",
    "apply_random_phase_rotation",
    "apply_time_shift",
    "augment_frame",
]
