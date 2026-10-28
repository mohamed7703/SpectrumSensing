r"""Real-Time Streaming Ring Buffer, Latency Profiler, and Duty-Cycle Scheduler.

Provides thread-safe circular buffering for high-throughput SDR I/Q ingestion,
hardware overrun/underrun accounting, high-precision latency profiling,
and predictive duty-cycle optimization.

Mathematical Notation:
    The continuous frame processing budget $T_{\text{budget}}$ must satisfy:
    $$T_{\text{capture}} + T_{\text{DSP}} + T_{\text{infer}} \le T_{\text{budget}} = 2.3\text{ ms}$$
    For sample rate $f_s$ and frame length $L$, the acquisition time is:
    $$T_{\text{acq}} = \frac{L}{f_s}$$
    The duty cycle $\eta \in (0, 1]$ defines active sensing fraction:
    $$\eta = \frac{T_{\text{sense}}}{T_{\text{sense}} + T_{\text{sleep}}}$$
"""

from typing import Optional, Dict, Any, Tuple, List
import time
import threading
import numpy as np


class CircularRingBuffer:
    """Thread-safe circular ring buffer for complex baseband I/Q samples.

    Supports continuous high-speed streaming between asynchronous SDR hardware
    threads (producer) and DSP/AI inference processing pipelines (consumer).
    """

    def __init__(self, capacity: int = 1048576, dtype: np.dtype = np.complex64) -> None:
        """Initialize ring buffer.

        Args:
            capacity: Total sample capacity (e.g., 1M complex samples).
            dtype: Data type (default complex64).
        """
        self.capacity = int(capacity)
        self.dtype = np.dtype(dtype)
        self.buffer = np.zeros(self.capacity, dtype=self.dtype)

        self._head: int = 0  # Write pointer
        self._tail: int = 0  # Read pointer
        self._available: int = 0  # Unread samples count
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        # Telemetry counters
        self.total_written: int = 0
        self.total_read: int = 0
        self.overrun_count: int = 0
        self.dropped_samples: int = 0
        self.underrun_count: int = 0

    @property
    def available_samples(self) -> int:
        """Number of readable samples currently buffered."""
        with self._lock:
            return self._available

    @property
    def free_space(self) -> int:
        """Available space remaining before buffer overrun."""
        with self._lock:
            return self.capacity - self._available

    def write(self, data: np.ndarray) -> int:
        """Write contiguous samples into the ring buffer.

        If write size exceeds available space, existing unread samples are
        overwritten and overrun counters are incremented.

        Args:
            data: 1D numpy array of samples.

        Returns:
            Number of samples accepted.
        """
        data = np.asarray(data, dtype=self.dtype)
        n = len(data)
        if n == 0:
            return 0

        with self._lock:
            space = self.capacity - self._available
            if n > space:
                # Buffer overrun: drop oldest unread samples
                overflow = n - space
                self.overrun_count += 1
                self.dropped_samples += overflow
                self._tail = (self._tail + overflow) % self.capacity
                self._available -= overflow

            # Write data in 1 or 2 chunks (if wrapping around)
            first_chunk = min(n, self.capacity - self._head)
            self.buffer[self._head : self._head + first_chunk] = data[:first_chunk]

            second_chunk = n - first_chunk
            if second_chunk > 0:
                self.buffer[0:second_chunk] = data[first_chunk:n]

            self._head = (self._head + n) % self.capacity
            self._available += n
            self.total_written += n

            self._not_empty.notify_all()
            return n

    def read(self, num_samples: int, timeout_s: Optional[float] = None) -> np.ndarray:
        """Read requested number of samples from the ring buffer.

        Blocks until `num_samples` are available or timeout expires.

        Args:
            num_samples: Number of samples to read.
            timeout_s: Maximum wait time in seconds (None blocks indefinitely).

        Returns:
            1D numpy array of samples of length `num_samples`.

        Raises:
            TimeoutError: If timeout expires before enough samples arrive.
        """
        if num_samples > self.capacity:
            raise ValueError(f"Requested {num_samples} exceeds buffer capacity {self.capacity}")

        deadline = (time.perf_counter() + timeout_s) if timeout_s is not None else None

        with self._not_empty:
            while self._available < num_samples:
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0 or not self._not_empty.wait(timeout=remaining):
                        self.underrun_count += 1
                        raise TimeoutError(
                            f"Ring buffer read timed out: {self._available}/{num_samples} samples available."
                        )
                else:
                    self._not_empty.wait()

            # Read out data in 1 or 2 segments
            out = np.empty(num_samples, dtype=self.dtype)
            first_chunk = min(num_samples, self.capacity - self._tail)
            out[:first_chunk] = self.buffer[self._tail : self._tail + first_chunk]

            second_chunk = num_samples - first_chunk
            if second_chunk > 0:
                out[first_chunk:] = self.buffer[0:second_chunk]

            self._tail = (self._tail + num_samples) % self.capacity
            self._available -= num_samples
            self.total_read += num_samples
            return out

    def clear(self) -> None:
        """Flush buffer contents and reset pointers."""
        with self._lock:
            self._head = 0
            self._tail = 0
            self._available = 0

    def get_telemetry(self) -> Dict[str, Any]:
        """Return diagnostic metrics on buffer health."""
        with self._lock:
            fill_pct = (self._available / self.capacity) * 100.0
            return {
                "capacity": self.capacity,
                "available": self._available,
                "fill_percentage": fill_pct,
                "total_written": self.total_written,
                "total_read": self.total_read,
                "overrun_count": self.overrun_count,
                "dropped_samples": self.dropped_samples,
                "underrun_count": self.underrun_count,
            }


class StreamHandler:
    """Asynchronous background streaming manager for SDR hardware frontends."""

    def __init__(
        self,
        sdr_device: Any,
        buffer_capacity: int = 1048576,
        chunk_size: int = 4096,
    ) -> None:
        """Initialize stream handler.

        Args:
            sdr_device: Configured BaseSDR instance.
            buffer_capacity: Ring buffer capacity in samples.
            chunk_size: Samples fetched per hardware USB read cycle.
        """
        self.sdr = sdr_device
        self.chunk_size = chunk_size
        self.buffer = CircularRingBuffer(capacity=buffer_capacity, dtype=np.complex64)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Launch background acquisition thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        if not self.sdr.is_streaming:
            self.sdr.start_rx()

        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()

    def _acquisition_loop(self) -> None:
        """Internal background loop polling SDR hardware."""
        while not self._stop_event.is_set():
            try:
                samples = self.sdr.read_samples(self.chunk_size)
                if samples.ndim == 2:
                    # Multi-channel MIMO: push channel 0 into primary buffer
                    samples = samples[0]
                self.buffer.write(samples)
            except Exception:
                if self._stop_event.is_set():
                    break
                time.sleep(0.001)

    def get_frame(self, frame_size: int = 1024, timeout_s: float = 1.0) -> np.ndarray:
        """Pull a synchronized frame of samples for DSP or AI processing."""
        return self.buffer.read(num_samples=frame_size, timeout_s=timeout_s)

    def stop(self) -> None:
        """Halt streaming thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.sdr.is_streaming:
            self.sdr.stop_rx()

    def get_telemetry(self) -> Dict[str, Any]:
        """Retrieve ring buffer and hardware telemetry."""
        return self.buffer.get_telemetry()


class LatencyProfiler:
    """High-resolution latency benchmarking profiler for real-time sensing pipeline."""

    def __init__(self, target_budget_ms: float = 2.3, embedded_ceiling_ms: float = 5.0) -> None:
        """Initialize latency profiler.

        Args:
            target_budget_ms: Paper/Deployment target budget (2.3 ms).
            embedded_ceiling_ms: Raspberry Pi / embedded CPU ceiling (5.0 ms).
        """
        self.target_budget_ms = target_budget_ms
        self.embedded_ceiling_ms = embedded_ceiling_ms
        self._stage_timings: Dict[str, List[float]] = {}

    def record(self, stage: str, duration_sec: float) -> None:
        """Record execution duration in seconds for a specific pipeline stage."""
        if stage not in self._stage_timings:
            self._stage_timings[stage] = []
        self._stage_timings[stage].append(duration_sec * 1000.0)  # Store in ms

    def get_statistics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Compute summary statistics for a given stage or total pipeline."""
        if stage is not None:
            vals = np.array(self._stage_timings.get(stage, []))
        else:
            # Total end-to-end latency across all stages per run
            lengths = [len(v) for v in self._stage_timings.values()]
            if not lengths or min(lengths) == 0:
                return {}
            n = min(lengths)
            vals = np.zeros(n)
            for series in self._stage_timings.values():
                vals += np.array(series[:n])

        if len(vals) == 0:
            return {}

        mean_ms = float(np.mean(vals))
        p50_ms = float(np.median(vals))
        p95_ms = float(np.percentile(vals, 95))
        p99_ms = float(np.percentile(vals, 99))
        max_ms = float(np.max(vals))
        meets_target = bool(p95_ms <= self.target_budget_ms)
        meets_embedded = bool(p95_ms <= self.embedded_ceiling_ms)

        return {
            "count": len(vals),
            "mean_ms": mean_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "max_ms": max_ms,
            "target_budget_ms": self.target_budget_ms,
            "meets_target_budget": meets_target,
            "meets_embedded_ceiling": meets_embedded,
        }

    def reset(self) -> None:
        """Clear recorded timings."""
        self._stage_timings.clear()


class DutyCycleController:
    """Predictive scheduling controller to minimize cognitive radio energy consumption.

    Dynamically modulates sensing duty cycle based on primary user activity
    predictions, achieving up to 50% energy reduction while preserving sensing QoS.
    """

    def __init__(self, base_duty_cycle: float = 1.0, min_duty_cycle: float = 0.2) -> None:
        """Initialize duty cycle controller.

        Args:
            base_duty_cycle: Initial duty cycle fraction in (0.0, 1.0].
            min_duty_cycle: Lower clamp on duty cycle fraction to ensure PU safety.
        """
        self.duty_cycle = float(np.clip(base_duty_cycle, min_duty_cycle, 1.0))
        self.min_duty_cycle = min_duty_cycle
        self._consecutive_idle_count: int = 0
        self._consecutive_busy_count: int = 0

    def update(self, is_channel_busy: bool) -> float:
        """Adapt duty cycle based on sensing decision.

        When channel is consistently idle or busy for prolonged spans,
        duty cycle can be reduced to save energy; upon state changes,
        duty cycle returns to 100% to avoid collision.

        Args:
            is_channel_busy: True if PU is detected active (H1).

        Returns:
            Updated duty cycle fraction in [min_duty_cycle, 1.0].
        """
        if is_channel_busy:
            self._consecutive_busy_count += 1
            self._consecutive_idle_count = 0
            if self._consecutive_busy_count > 5:
                # Channel reliably occupied, reduce duty cycle
                self.duty_cycle = max(self.min_duty_cycle, self.duty_cycle * 0.9)
            else:
                self.duty_cycle = 1.0
        else:
            self._consecutive_idle_count += 1
            self._consecutive_busy_count = 0
            if self._consecutive_idle_count > 5:
                # Channel stably vacant, reduce duty cycle
                self.duty_cycle = max(self.min_duty_cycle, self.duty_cycle * 0.85)
            else:
                self.duty_cycle = 1.0

        return self.duty_cycle

    def compute_sleep_time(self, sense_duration_s: float) -> float:
        """Compute sleep duration in seconds for current duty cycle."""
        if self.duty_cycle >= 0.999:
            return 0.0
        # eta = T_sense / (T_sense + T_sleep) => T_sleep = T_sense * (1 - eta) / eta
        return sense_duration_s * (1.0 - self.duty_cycle) / self.duty_cycle
