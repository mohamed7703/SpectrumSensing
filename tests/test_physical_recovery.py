"""Physical-layer regressions independent of latency and positive-power smoke tests."""

import numpy as np
import pytest

from src.sdr_interface.mock_sdr import MockSDR
from src.sdr_interface.stream_handler import CircularRingBuffer
from src.math_models.energy_detection import compute_energy_statistic


@pytest.mark.parametrize('prefix,size', [(0, 10), (0, 25), (3, 11), (8, 21)])
def test_oversize_buffer_write_retains_latest_samples_and_consistent_counters(prefix, size):
    """Writes at/above capacity keep the newest capacity samples, including unread overflow."""
    buffer = CircularRingBuffer(capacity=10)
    buffer.write(np.arange(prefix, dtype=np.complex64))
    incoming = np.arange(100, 100+size, dtype=np.complex64)
    buffer.write(incoming)
    expected = np.r_[np.arange(prefix), incoming][-10:]
    assert buffer.available_samples == 10
    assert buffer.free_space == 0
    assert buffer.dropped_samples == prefix + size - 10
    assert buffer.total_written == prefix + size
    np.testing.assert_array_equal(buffer.read(10, timeout_s=.01), expected)


def test_invalid_buffer_inputs_do_not_corrupt_state():
    """Reject impossible capacities/read lengths and multidimensional writes."""
    with pytest.raises(ValueError):
        CircularRingBuffer(capacity=0)
    buffer = CircularRingBuffer(capacity=10)
    with pytest.raises(ValueError):
        buffer.write(np.ones((2, 2)))
    with pytest.raises(ValueError):
        buffer.read(-1, timeout_s=.01)
    assert buffer.available_samples == 0


def test_mock_gain_does_not_normalize_away_signal_energy():
    """At fixed gain, 10 dB signal adds ten noise powers before clipping."""
    powers = []
    for signal_type in ['noise', 'qpsk']:
        sdr = MockSDR(signal_type=signal_type, snr_db=10., gain=0., adc_bits=16, random_seed=42)
        with sdr:
            powers.append(float(compute_energy_statistic(sdr.read_samples(200000))))
    assert powers[1]/powers[0] == pytest.approx(11., rel=.03)


def test_calibration_uses_confidence_order_statistic_and_strict_tie_boundary():
    """For 100 frames, the maximum has >99% confidence coverage of a 5% upper tail."""
    from src.math_models.energy_detection import calibrate_energy_threshold
    frames = np.ones((100, 16), dtype=complex)
    calibration = calibrate_energy_threshold(frames, target_pfa=.05, confidence_level=.99)
    assert calibration.order_rank == 100
    assert np.all(compute_energy_statistic(frames) < calibration.threshold)
    with pytest.raises(ValueError, match='insufficient'):
        calibrate_energy_threshold(frames[:2], target_pfa=.05, confidence_level=.99)


def test_calibrated_active_mock_detection_with_independent_noise_false_alarm_check():
    """Fresh H0 and H1 records check correctness; timing success is not the criterion."""
    from src.math_models.energy_detection import calibrate_energy_threshold
    from src.math_models.detection_metrics import binomial_rate_interval
    def capture(kind, seed, count):
        sdr = MockSDR(adc_bits=12, sample_rate=2.4e6, signal_type=kind, snr_db=10., random_seed=seed)
        with sdr:
            return np.concatenate([sdr.read_samples(4096).reshape(4, 1024) for _ in range(count//4)])
    calibration = calibrate_energy_threshold(capture('noise', 1729, 400))
    h0 = compute_energy_statistic(capture('noise', 1730, 3000))
    h1 = compute_energy_statistic(capture('qpsk', 42, 300))
    false_alarms = int(np.sum(h0 >= calibration.threshold))
    detections = int(np.sum(h1 >= calibration.threshold))
    assert binomial_rate_interval(false_alarms, len(h0), alternative='less')[1] <= .05
    assert binomial_rate_interval(detections, len(h1), alternative='greater')[0] >= .9
