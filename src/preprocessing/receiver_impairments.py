r"""Session-level receiver front-end impairments for I/Q frames.

A receiver session fixes its analog imperfections for as long as the radio stays
tuned, so every frame recorded in that session shares one draw. This module
applies four impairments in the order a real chain meets them:

1. Anti-alias filtering colours the noise (Butterworth magnitude response,
   zero phase, applied circularly over each frame):

   .. math::
       |H(f)|^2 = \frac{1}{1 + (|f| / f_c)^{2N}}, \qquad f_c = \alpha / 2

   with :math:`\alpha` the cutoff as a fraction of Nyquist and :math:`N` the order.
   For white noise of power :math:`\sigma^2` the output power is
   :math:`\sigma^2 G_f` with :math:`G_f = \frac{1}{L}\sum_k |H(f_k)|^2`.
2. IQ imbalance with amplitude mismatch :math:`g = 10^{\epsilon/20}` and phase
   mismatch :math:`\varphi`:

   .. math::
       I' = I, \qquad Q' = g\,(Q\cos\varphi - I\sin\varphi)

   Equivalently :math:`y = \mu x + \nu x^*` with
   :math:`\mu = (1 + g e^{-j\varphi})/2` and :math:`\nu = (1 - g e^{j\varphi})/2`.
   For circular noise :math:`E|y|^2 = (|\mu|^2 + |\nu|^2)\sigma^2 = \frac{1+g^2}{2}\sigma^2`.
3. DC offset :math:`d \in \mathbb{C}`, added to every sample.
4. A :math:`B`-bit ADC with full scale :math:`F` per component, step
   :math:`\Delta = 2F / 2^B`, rounding to the nearest code and clipping to
   :math:`[-2^{B-1}, 2^{B-1}-1]\,\Delta`.

Noise power is renormalised to one after steps 1 and 2 (divide by
:math:`\sqrt{G_f}` and :math:`\sqrt{(1+g^2)/2}`), so the mean energy cue keeps its
meaning: noise-only frames carry mean power :math:`1 + |d|^2`. Coloured noise has
fewer effective degrees of freedom, so the per-frame energy varies more (by about
:math:`\overline{|H|^4}/\overline{|H|^2}^2` in variance) and an energy detector
gets weaker. The signal is filtered
with the noise, so its in-band SNR rises by roughly :math:`1/G_f` relative to
the nominal SNR set at the antenna port. No random gain is applied; receiver
gain drift stays an experimental variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImpairmentRanges:
    """Bounds from which sessions are drawn, plus fixed filter and ADC settings.

    Attributes:
        dc_power_db: (low, high) DC offset power in dB relative to unit noise power.
        iq_gain_db_max: Largest absolute IQ amplitude mismatch in dB.
        iq_phase_deg_max: Largest absolute IQ phase mismatch in degrees.
        cutoff_fraction: (low, high) filter cutoff as a fraction of Nyquist.
        filter_order: Butterworth order :math:`N`.
        adc_bits: ADC resolution :math:`B`.
        adc_full_scale: ADC full scale :math:`F` per component, in noise-power units.
    """

    dc_power_db: tuple[float, float]
    iq_gain_db_max: float
    iq_phase_deg_max: float
    cutoff_fraction: tuple[float, float]
    filter_order: int
    adc_bits: int
    adc_full_scale: float


@dataclass(frozen=True)
class ImpairmentSession:
    """One receiver session's fixed impairments."""

    dc: complex
    iq_gain_db: float
    iq_phase_deg: float
    cutoff_fraction: float


def _check_frames(x: np.ndarray) -> np.ndarray:
    """Return frames as a finite float64 array of shape (N, L, 2) or raise ValueError."""
    frames = np.asarray(x)
    if frames.ndim != 3 or frames.shape[-1] != 2:
        raise ValueError("frames must have shape (N, L, 2)")
    if not np.all(np.isfinite(frames)):
        raise ValueError("frames must be finite")
    return frames.astype(np.float64)


def _lowpass_response(cutoff_fraction: float, order: int, length: int) -> np.ndarray:
    """Return the magnitude response |H(f_k)| on the FFT grid."""
    freq = np.abs(np.fft.fftfreq(length))
    return 1.0 / np.sqrt(1.0 + (freq / (0.5 * cutoff_fraction)) ** (2 * order))


def lowpass_noise_power_gain(cutoff_fraction: float, order: int, length: int) -> float:
    r"""Return :math:`G_f = \frac{1}{L}\sum_k |H(f_k)|^2` for circular filtering of white noise.

    Args:
        cutoff_fraction: Cutoff :math:`\alpha` as a fraction of Nyquist, in (0, 1].
        order: Butterworth order :math:`N`, at least 1.
        length: Frame length :math:`L`.
    """
    if not 0.0 < cutoff_fraction <= 1.0 or order < 1 or length < 1:
        raise ValueError("require 0 < cutoff_fraction <= 1, order >= 1 and length >= 1")
    return float(np.mean(_lowpass_response(cutoff_fraction, order, length) ** 2))


def apply_lowpass(x: np.ndarray, cutoff_fraction: float, order: int, renormalise: bool = True) -> np.ndarray:
    r"""Filter each frame circularly with the Butterworth magnitude response.

    Args:
        x: Frames of shape (N, L, 2).
        cutoff_fraction: Cutoff as a fraction of Nyquist.
        order: Butterworth order.
        renormalise: Divide by :math:`\sqrt{G_f}` so white noise keeps its power.

    Returns:
        New float32 array of the same shape.
    """
    frames = _check_frames(x)
    length = frames.shape[1]
    z = np.fft.fft(frames[..., 0] + 1j * frames[..., 1], axis=1)
    z = np.fft.ifft(z * _lowpass_response(cutoff_fraction, order, length), axis=1)
    if renormalise:
        z = z / np.sqrt(lowpass_noise_power_gain(cutoff_fraction, order, length))
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def apply_iq_imbalance(x: np.ndarray, gain_db: float, phase_deg: float, renormalise: bool = True) -> np.ndarray:
    r"""Apply :math:`I' = I,\ Q' = g(Q\cos\varphi - I\sin\varphi)`.

    Args:
        x: Frames of shape (N, L, 2).
        gain_db: Amplitude mismatch :math:`\epsilon` in dB, :math:`g = 10^{\epsilon/20}`.
        phase_deg: Phase mismatch :math:`\varphi` in degrees.
        renormalise: Divide by :math:`\sqrt{(1+g^2)/2}` so circular noise keeps its power.

    Returns:
        New float32 array of the same shape.
    """
    frames = _check_frames(x)
    g, phi = 10.0 ** (gain_db / 20.0), np.deg2rad(phase_deg)
    i, q = frames[..., 0], frames[..., 1]
    out = np.stack([i, g * (q * np.cos(phi) - i * np.sin(phi))], axis=-1)
    if renormalise:
        out = out / np.sqrt((1.0 + g * g) / 2.0)
    return out.astype(np.float32)


def apply_dc_offset(x: np.ndarray, dc: complex) -> np.ndarray:
    """Add a complex DC offset: real part to I, imaginary part to Q. Returns a new float32 array."""
    frames = _check_frames(x)
    return (frames + np.array([dc.real, dc.imag])).astype(np.float32)


def quantize_adc(x: np.ndarray, bits: int, full_scale: float) -> np.ndarray:
    r"""Round to :math:`B`-bit codes of step :math:`\Delta = 2F/2^B` and clip.

    Args:
        x: Frames of shape (N, L, 2).
        bits: ADC resolution :math:`B`, at least 2.
        full_scale: Full scale :math:`F` per component, positive.

    Returns:
        New float32 array on the code grid, within :math:`[-F,\ F-\Delta]`.
    """
    if bits < 2 or full_scale <= 0.0:
        raise ValueError("require bits >= 2 and full_scale > 0")
    frames = _check_frames(x)
    step = 2.0 * full_scale / 2 ** bits
    codes = np.clip(np.round(frames / step), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (codes * step).astype(np.float32)


def draw_session(rng: np.random.Generator, ranges: ImpairmentRanges) -> ImpairmentSession:
    """Draw one session: DC power uniform in dB with uniform phase, uniform IQ mismatch, uniform cutoff."""
    low, high = ranges.dc_power_db
    dc_amplitude = 10.0 ** (rng.uniform(low, high) / 20.0)
    dc = dc_amplitude * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    gain_db = rng.uniform(-ranges.iq_gain_db_max, ranges.iq_gain_db_max)
    phase_deg = rng.uniform(-ranges.iq_phase_deg_max, ranges.iq_phase_deg_max)
    cutoff = rng.uniform(*ranges.cutoff_fraction)
    return ImpairmentSession(complex(dc), float(gain_db), float(phase_deg), float(cutoff))


def apply_impairments(x: np.ndarray, session: ImpairmentSession, ranges: ImpairmentRanges) -> np.ndarray:
    """Run filter, IQ imbalance, DC offset and ADC for one session; returns a new float32 array."""
    y = apply_lowpass(x, session.cutoff_fraction, ranges.filter_order)
    y = apply_iq_imbalance(y, session.iq_gain_db, session.iq_phase_deg)
    y = apply_dc_offset(y, session.dc)
    return quantize_adc(y, ranges.adc_bits, ranges.adc_full_scale)


def impair_in_sessions(
    x: np.ndarray, rng: np.random.Generator, ranges: ImpairmentRanges, frames_per_session: int
) -> tuple[np.ndarray, list[ImpairmentSession]]:
    """Split frames into consecutive sessions, drawing and applying one session per chunk.

    Args:
        x: Frames of shape (N, L, 2).
        rng: Source of the session draws.
        ranges: Draw bounds and fixed settings.
        frames_per_session: Frames sharing one draw, at least 1.

    Returns:
        The impaired float32 frames and the session drawn for each chunk.
    """
    if int(frames_per_session) != frames_per_session or frames_per_session < 1:
        raise ValueError("frames_per_session must be a positive integer")
    frames = _check_frames(x)
    chunks, sessions = [], []
    for start in range(0, len(frames), frames_per_session):
        session = draw_session(rng, ranges)
        sessions.append(session)
        chunks.append(apply_impairments(frames[start:start + frames_per_session], session, ranges))
    return np.concatenate(chunks) if chunks else frames.astype(np.float32), sessions
