"""Tests for session-level receiver impairments applied to I/Q frames."""

import numpy as np
import pytest

from src.preprocessing.receiver_impairments import (
    ImpairmentRanges,
    ImpairmentSession,
    apply_dc_offset,
    apply_impairments,
    apply_iq_imbalance,
    apply_lowpass,
    draw_session,
    impair_in_sessions,
    lowpass_noise_power_gain,
    quantize_adc,
)

LENGTH = 1024


def noise(count: int, seed: int = 0) -> np.ndarray:
    """Return unit-power circular complex Gaussian noise as float32 (count, 1024, 2) frames."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(2.0)
    return rng.normal(0.0, scale, (count, LENGTH, 2)).astype(np.float32)


def power(x: np.ndarray) -> float:
    """Mean complex sample power of (N, L, 2) frames."""
    return float(np.mean(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1)))


@pytest.fixture
def ranges() -> ImpairmentRanges:
    """Return the proposed impairment ranges."""
    return ImpairmentRanges(dc_power_db=(-40.0, -25.0), iq_gain_db_max=0.5, iq_phase_deg_max=3.0,
                            cutoff_fraction=(0.35, 0.45), filter_order=4, adc_bits=12,
                            adc_full_scale=8.0)


def neutral(ranges: ImpairmentRanges) -> ImpairmentSession:
    """Return a session whose only active impairment is the (near-transparent) filter."""
    return ImpairmentSession(dc=0j, iq_gain_db=0.0, iq_phase_deg=0.0, cutoff_fraction=0.45)


def test_iq_imbalance_matches_closed_form() -> None:
    """I is unchanged and Q' = g (Q cos(phi) - I sin(phi))."""
    x = noise(2, 1)
    gain_db, phase_deg = 0.4, 2.5
    y = apply_iq_imbalance(x, gain_db, phase_deg, renormalise=False)
    g, phi = 10 ** (gain_db / 20), np.deg2rad(phase_deg)
    np.testing.assert_allclose(y[..., 0], x[..., 0], atol=1e-6)
    np.testing.assert_allclose(y[..., 1], g * (x[..., 1] * np.cos(phi) - x[..., 0] * np.sin(phi)), atol=1e-5)


def test_iq_imbalance_renormalisation_keeps_noise_power() -> None:
    """Dividing by sqrt((1+g^2)/2) preserves circular-noise power."""
    x = noise(40, 2)
    y = apply_iq_imbalance(x, 0.5, 3.0, renormalise=True)
    assert power(y) == pytest.approx(power(x), rel=0.01)
    assert power(apply_iq_imbalance(x, 0.5, 3.0, renormalise=False)) != pytest.approx(power(x), rel=1e-4)


def test_iq_imbalance_zero_is_identity() -> None:
    """Zero gain and phase mismatch leaves frames unchanged."""
    x = noise(2, 3)
    np.testing.assert_allclose(apply_iq_imbalance(x, 0.0, 0.0), x, atol=1e-6)


def test_dc_offset_shifts_the_mean() -> None:
    """A complex offset adds its real part to I and imaginary part to Q."""
    x = noise(2, 4)
    y = apply_dc_offset(x, 0.03 - 0.02j)
    np.testing.assert_allclose(y[..., 0] - x[..., 0], 0.03, atol=1e-6)
    np.testing.assert_allclose(y[..., 1] - x[..., 1], -0.02, atol=1e-6)


def test_lowpass_colours_noise_and_matches_analytic_power_gain() -> None:
    """Measured noise power ratio equals the analytic mean |H|^2 and neighbours correlate."""
    x = noise(60, 5)
    cutoff, order = 0.4, 4
    y = apply_lowpass(x, cutoff, order, renormalise=False)
    gain = lowpass_noise_power_gain(cutoff, order, LENGTH)
    assert 0.3 < gain < 0.7
    assert power(y) / power(x) == pytest.approx(gain, rel=0.03)
    z = y[..., 0] + 1j * y[..., 1]
    lag1 = np.mean(np.real(z[:, 1:] * np.conj(z[:, :-1]))) / np.mean(np.abs(z) ** 2)
    assert lag1 > 0.3


def test_lowpass_renormalisation_restores_unit_noise_power() -> None:
    """The energy cue keeps its meaning after colouring."""
    x = noise(60, 6)
    y = apply_lowpass(x, 0.4, 4, renormalise=True)
    assert power(y) == pytest.approx(1.0, rel=0.03)


def test_quantize_adc_lies_on_the_grid_and_clips() -> None:
    """Outputs are integer multiples of the LSB inside the two's-complement range."""
    bits, full_scale = 12, 8.0
    step = 2 * full_scale / 2 ** bits
    x = np.array([[[0.0, 0.00123], [7.9999, -8.5], [20.0, -20.0]]], dtype=np.float32)
    y = quantize_adc(x, bits, full_scale)
    codes = y / step
    np.testing.assert_allclose(codes, np.round(codes), atol=1e-4)
    assert y.max() <= full_scale - step + 1e-6 and y.min() >= -full_scale - 1e-6
    assert y[0, 2, 0] == pytest.approx(full_scale - step, abs=1e-6)
    assert y[0, 2, 1] == pytest.approx(-full_scale, abs=1e-6)


def test_quantize_adc_error_is_below_half_lsb_inside_range() -> None:
    """Rounding error stays within half an LSB when nothing clips."""
    x = noise(4, 7)
    y = quantize_adc(x, 12, 8.0)
    assert np.max(np.abs(y - x)) <= 0.5 * (16.0 / 4096) + 1e-6


def test_draw_session_is_seeded_and_inside_ranges(ranges: ImpairmentRanges) -> None:
    """Draws are reproducible, vary across calls, and respect every bound."""
    a, b = draw_session(np.random.default_rng(11), ranges), draw_session(np.random.default_rng(11), ranges)
    assert a == b
    rng = np.random.default_rng(12)
    draws = [draw_session(rng, ranges) for _ in range(200)]
    assert len({d.cutoff_fraction for d in draws}) > 100
    for d in draws:
        assert -40.0 - 1e-9 <= 10 * np.log10(abs(d.dc) ** 2) <= -25.0 + 1e-9
        assert abs(d.iq_gain_db) <= 0.5 and abs(d.iq_phase_deg) <= 3.0
        assert 0.35 <= d.cutoff_fraction <= 0.45


def test_apply_impairments_keeps_unit_noise_power_apart_from_dc(ranges: ImpairmentRanges) -> None:
    """Noise-only frames carry power 1 + |dc|^2 after the full chain."""
    session = ImpairmentSession(dc=0.04 + 0.03j, iq_gain_db=0.4, iq_phase_deg=2.0, cutoff_fraction=0.4)
    y = apply_impairments(noise(80, 8), session, ranges)
    assert y.dtype == np.float32 and y.shape == (80, LENGTH, 2)
    assert power(y) == pytest.approx(1.0 + abs(session.dc) ** 2, rel=0.02)


def test_apply_impairments_does_not_mutate_input(ranges: ImpairmentRanges) -> None:
    """The input array is left untouched."""
    x = noise(3, 9)
    before = x.copy()
    apply_impairments(x, draw_session(np.random.default_rng(1), ranges), ranges)
    np.testing.assert_array_equal(x, before)


def test_apply_impairments_adds_no_random_gain(ranges: ImpairmentRanges) -> None:
    """A tone far below the cutoff keeps its amplitude, so no gain drift is injected."""
    t = np.arange(LENGTH)
    tone = 0.5 * np.exp(2j * np.pi * 8 / LENGTH * t)
    x = np.stack([tone.real, tone.imag], axis=-1)[None].astype(np.float32)
    session = ImpairmentSession(dc=0j, iq_gain_db=0.0, iq_phase_deg=0.0, cutoff_fraction=0.45)
    y = apply_impairments(x, session, ranges)
    gain = lowpass_noise_power_gain(0.45, ranges.filter_order, LENGTH)
    assert power(y) / power(x) == pytest.approx(1.0 / gain, rel=0.01)


def test_impair_in_sessions_groups_frames_by_session(ranges: ImpairmentRanges) -> None:
    """Frames inside one chunk share a draw; chunks get independent draws."""
    x = noise(25, 10)
    y, sessions = impair_in_sessions(x, np.random.default_rng(3), ranges, frames_per_session=10)
    assert len(sessions) == 3 and y.shape == x.shape
    again, same = impair_in_sessions(x, np.random.default_rng(3), ranges, frames_per_session=10)
    assert sessions == same
    np.testing.assert_array_equal(y, again)
    assert len({s.cutoff_fraction for s in sessions}) == 3
    first = apply_impairments(x[:10], sessions[0], ranges)
    np.testing.assert_allclose(y[:10], first, atol=1e-6)


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.zeros((2, 8, 3)), np.full((1, 8, 2), np.nan)])
def test_rejects_malformed_frames(ranges: ImpairmentRanges, bad: np.ndarray) -> None:
    """Wrong shapes and non-finite values raise ValueError."""
    with pytest.raises(ValueError):
        apply_impairments(bad, neutral(ranges), ranges)


def test_impair_in_sessions_rejects_nonpositive_session_size(ranges: ImpairmentRanges) -> None:
    """A session must contain at least one frame."""
    with pytest.raises(ValueError):
        impair_in_sessions(noise(2, 11), np.random.default_rng(0), ranges, frames_per_session=0)
