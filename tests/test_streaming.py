"""Unit Tests for Real-Time Streaming Ring Buffer, Latency Profiler, and Duty-Cycle Scheduler.

Tests thread-safe circular ring buffer, overrun/underrun accounting,
asynchronous StreamHandler polling, latency profiling percentiles,
and predictive duty cycle modulation.
"""

import time
import pytest
import numpy as np

from src.sdr_interface.mock_sdr import MockSDR
from src.sdr_interface.stream_handler import (
    CircularRingBuffer,
    StreamHandler,
    LatencyProfiler,
    DutyCycleController,
)


class TestCircularRingBuffer:
    """Test ring buffer functionality, wraparound, and overflow/underflow handling."""

    def test_basic_write_read_no_wrap(self) -> None:
        """Sequential write and read within single buffer pass."""
        buf = CircularRingBuffer(capacity=100)
        assert buf.capacity == 100
        assert buf.available_samples == 0
        assert buf.free_space == 100

        data = np.arange(40, dtype=np.complex64)
        buf.write(data)
        assert buf.available_samples == 40
        assert buf.free_space == 60

        read_data = buf.read(40)
        assert len(read_data) == 40
        assert np.array_equal(read_data, data)
        assert buf.available_samples == 0

    def test_buffer_wraparound(self) -> None:
        """Ensure circular wrap across end of buffer works seamlessly."""
        buf = CircularRingBuffer(capacity=100)
        # Advance write pointer to 80
        buf.write(np.zeros(80, dtype=np.complex64))
        buf.read(80)
        assert buf.available_samples == 0

        # Now write 40 samples (will wrap around from index 80 to 20)
        data = np.arange(40, dtype=np.complex64) + 1j * 5.0
        buf.write(data)
        assert buf.available_samples == 40

        read_data = buf.read(40)
        assert np.array_equal(read_data, data)
        assert buf.available_samples == 0

    def test_buffer_overrun_telemetry(self) -> None:
        """Writing more than available space should drop oldest samples and increment overrun."""
        buf = CircularRingBuffer(capacity=50)
        buf.write(np.ones(40, dtype=np.complex64))
        assert buf.available_samples == 40

        # Write 20 more samples into 10 remaining spaces -> 10 samples overflow
        new_data = np.full(20, 2.0 + 2.0j, dtype=np.complex64)
        buf.write(new_data)

        telemetry = buf.get_telemetry()
        assert telemetry["available"] == 50
        assert telemetry["overrun_count"] == 1
        assert telemetry["dropped_samples"] == 10
        assert telemetry["fill_percentage"] == 100.0

        # Read all 50: first 30 should be 1.0, last 20 should be 2.0
        read_data = buf.read(50)
        assert np.all(read_data[:30] == 1.0)
        assert np.all(read_data[30:] == 2.0 + 2.0j)

    def test_buffer_underrun_timeout(self) -> None:
        """Reading more than available with timeout should raise TimeoutError."""
        buf = CircularRingBuffer(capacity=100)
        buf.write(np.zeros(20, dtype=np.complex64))

        with pytest.raises(TimeoutError):
            buf.read(50, timeout_s=0.05)

        telemetry = buf.get_telemetry()
        assert telemetry["underrun_count"] == 1


class TestStreamHandler:
    """Test asynchronous StreamHandler background worker."""

    def test_async_streaming_from_mock_sdr(self) -> None:
        """StreamHandler should continuously ingest samples from MockSDR."""
        sdr = MockSDR(random_seed=42)
        handler = StreamHandler(sdr_device=sdr, buffer_capacity=100000, chunk_size=2048)

        handler.start()
        # Wait briefly for worker thread to populate buffer
        time.sleep(0.05)

        # Pull 3 consecutive 1024-sample frames
        for _ in range(3):
            frame = handler.get_frame(frame_size=1024, timeout_s=1.0)
            assert len(frame) == 1024
            assert frame.dtype == np.complex64

        handler.stop()
        telemetry = handler.get_telemetry()
        assert telemetry["total_written"] >= 3072
        assert telemetry["total_read"] == 3072


class TestLatencyProfiler:
    """Test latency metric computations and threshold validations."""

    def test_latency_statistics_calculation(self) -> None:
        profiler = LatencyProfiler(target_budget_ms=2.3, embedded_ceiling_ms=5.0)

        # Record 100 samples around 1.0 ms
        for val in np.linspace(0.8e-3, 1.4e-3, 100):
            profiler.record("capture", val)

        stats = profiler.get_statistics("capture")
        assert stats["count"] == 100
        assert 0.8 <= stats["mean_ms"] <= 1.4
        assert stats["meets_target_budget"] is True
        assert stats["meets_embedded_ceiling"] is True

    def test_total_pipeline_latency(self) -> None:
        profiler = LatencyProfiler(target_budget_ms=2.3, embedded_ceiling_ms=5.0)
        for _ in range(50):
            profiler.record("capture", 0.5e-3)
            profiler.record("dsp", 0.8e-3)
            profiler.record("inference", 0.6e-3)

        total_stats = profiler.get_statistics()
        assert total_stats["count"] == 50
        # Total should be ~1.9 ms
        assert np.isclose(total_stats["mean_ms"], 1.9, atol=0.05)
        assert total_stats["meets_target_budget"] is True


class TestDutyCycleController:
    """Test adaptive duty cycle scheduling."""

    def test_duty_cycle_adaptation_on_idle(self) -> None:
        ctrl = DutyCycleController(base_duty_cycle=1.0, min_duty_cycle=0.3)
        assert ctrl.duty_cycle == 1.0

        # Repeated idle channels should reduce duty cycle
        for _ in range(10):
            ctrl.update(is_channel_busy=False)

        assert ctrl.duty_cycle < 1.0
        assert ctrl.duty_cycle >= 0.3

    def test_sleep_time_computation(self) -> None:
        ctrl = DutyCycleController(base_duty_cycle=0.5, min_duty_cycle=0.2)
        # At 50% duty cycle, T_sleep equals T_sense
        sleep_t = ctrl.compute_sleep_time(sense_duration_s=0.01)
        assert np.isclose(sleep_t, 0.01)
