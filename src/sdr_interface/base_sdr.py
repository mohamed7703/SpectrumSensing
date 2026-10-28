r"""Base SDR Hardware Abstraction Layer.

Defines the abstract base class and common contracts for all Software-Defined
Radio (SDR) hardware frontends and mock drivers in the cognitive radio system.

Mathematical Notation:
    The sampled complex baseband signal $r[n]$ is represented as:
    $$r[n] = I[n] + j Q[n]$$
    Quantization error for an $b$-bit ADC with peak-to-peak voltage $V_{pp}$
    has step size:
    $$\Delta = \frac{V_{pp}}{2^b - 1}$$
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np


class SDRError(Exception):
    """Base exception for all SDR-related operations."""
    pass


class DeviceNotFoundError(SDRError):
    """Raised when the specified physical SDR device is not detected."""
    pass


class DriverNotInstalledError(SDRError):
    """Raised when third-party SDR driver or Python bindings are missing."""
    pass


class DeviceConfigurationError(SDRError):
    """Raised when invalid frequency, rate, or gain parameters are requested."""
    pass


class BaseSDR(ABC):
    """Abstract Base Class for Software-Defined Radio hardware interfaces.

    Provides a unified polymorphic interface across diverse hardware architectures:
    RTL-SDR (8-bit), HackRF One (8-bit), PlutoSDR (12-bit), LimeSDR (12-bit 2x2 MIMO),
    USRP (12-14 bit), and synthetic HITL mock generators.
    """

    def __init__(
        self,
        device_name: str,
        adc_bits: int,
        freq_range: Tuple[float, float],
        max_bandwidth: float,
        gain_range: Tuple[float, float],
        channels: int = 1,
        simulated: bool = False,
    ) -> None:
        """Initialize base SDR attributes.

        Args:
            device_name: Descriptive name of the SDR frontend.
            adc_bits: ADC resolution in bits (e.g., 8, 12, 14).
            freq_range: Tuple of (min_freq_hz, max_freq_hz).
            max_bandwidth: Maximum instantaneous RF bandwidth in Hz.
            gain_range: Tuple of (min_gain_db, max_gain_db).
            channels: Number of coherent RX channels (1 for SISO, 2 for MIMO).
            simulated: If True, operates in simulation mode without hardware.
        """
        self.device_name = device_name
        self.adc_bits = adc_bits
        self.freq_range = freq_range
        self.max_bandwidth = max_bandwidth
        self.gain_range = gain_range
        self.channels = channels
        self.simulated = simulated

        self._center_freq: float = (freq_range[0] + freq_range[1]) / 2.0
        self._sample_rate: float = min(2.4e6, max_bandwidth)
        self._gain: float = (gain_range[0] + gain_range[1]) / 2.0
        self._is_streaming: bool = False

    @property
    def center_freq(self) -> float:
        """Current tuned center frequency in Hz."""
        return self._center_freq

    @property
    def sample_rate(self) -> float:
        """Current baseband sampling rate in samples/second."""
        return self._sample_rate

    @property
    def gain(self) -> float:
        """Current RF receiver gain in dB."""
        return self._gain

    @property
    def is_streaming(self) -> bool:
        """True if the device receiver is currently active and streaming."""
        return self._is_streaming

    def validate_frequency(self, freq: float) -> bool:
        """Check if target frequency is supported by hardware tuner."""
        return self.freq_range[0] <= freq <= self.freq_range[1]

    def validate_sample_rate(self, rate: float) -> bool:
        """Check if target sample rate is within ADC bandwidth limits."""
        return 0 < rate <= self.max_bandwidth

    def validate_gain(self, gain: float) -> bool:
        """Check if gain is within hardware programmable range."""
        return self.gain_range[0] <= gain <= self.gain_range[1]

    def quantize_samples(self, samples: np.ndarray) -> np.ndarray:
        """Simulate ADC bit depth quantization and rail saturation.

        Models finite ADC precision by mapping normalized continuous I/Q values
        in [-1.0, 1.0] to integer levels $[-2^{b-1}, 2^{b-1}-1]$ and scaling back.

        Args:
            samples: Continuous complex baseband samples.

        Returns:
            Quantized complex samples as np.complex64.
        """
        if self.adc_bits >= 16:
            return samples.astype(np.complex64)

        max_int = (1 << (self.adc_bits - 1)) - 1
        min_int = -(1 << (self.adc_bits - 1))

        # Quantize Real (I) rail
        i_int = np.clip(np.round(samples.real * max_int), min_int, max_int)
        # Quantize Imag (Q) rail
        q_int = np.clip(np.round(samples.imag * max_int), min_int, max_int)

        quantized = (i_int + 1j * q_int) / max_int
        return quantized.astype(np.complex64)

    @abstractmethod
    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Configure frontend RF and baseband parameters.

        Args:
            center_freq: Center frequency in Hz.
            sample_rate: Sampling rate in Hz.
            gain: Total or stage gain in dB.
            **kwargs: Frontend-specific parameters (e.g. AGC, antenna port).

        Raises:
            DeviceConfigurationError: If any parameter is out of bounds.
        """
        pass

    @abstractmethod
    def start_rx(self) -> None:
        """Start asynchronous or buffered streaming reception."""
        pass

    @abstractmethod
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read a contiguous chunk of complex I/Q baseband samples.

        Args:
            num_samples: Number of samples to read per channel.

        Returns:
            Complex baseband numpy array of shape (num_samples,) or (channels, num_samples).
        """
        pass

    @abstractmethod
    def stop_rx(self) -> None:
        """Halt streaming reception and flush hardware buffers."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release USB device handle and teardown driver resources."""
        pass

    def __enter__(self) -> "BaseSDR":
        """Enter context manager, starting reception."""
        self.start_rx()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, stopping reception and releasing device."""
        try:
            self.stop_rx()
        finally:
            self.close()

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} device='{self.device_name}' "
            f"freq={self.center_freq/1e6:.3f}MHz rate={self.sample_rate/1e6:.2f}MSPS "
            f"gain={self.gain:.1f}dB adc={self.adc_bits}b sim={self.simulated}>"
        )
