"""LimeSDR-USB Hardware Driver.

Wraps the Lime Microsystems LimeSDR-USB (LMS7002M MIMO transceiver, 12-bit ADC,
61.44 MHz IBW, 100 kHz to 3800 MHz, 2x2 MIMO) with BaseSDR interface.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from src.sdr_interface.base_sdr import BaseSDR, DeviceConfigurationError, DeviceNotFoundError, DriverNotInstalledError
from src.sdr_interface.mock_sdr import MockSDR


class LimeSDRDriver(BaseSDR):
    """Driver for LimeSDR-USB MIMO transceiver (LMS7002M).

    Specifications:
        - ADC Resolution: 12-bit
        - Tuning Range: 100 kHz to 3800 MHz
        - Max Bandwidth: 61.44 MSPS instantaneous
        - Programmable Gain: 0.0 dB to 73.0 dB
        - Channels: 2 Coherent RX channels (2x2 MIMO)
    """

    def __init__(
        self,
        device_index: int = 0,
        channels: int = 2,
        simulated: bool = False,
        center_freq: float = 2412.0e6,
        sample_rate: float = 10.0e6,
        gain: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize LimeSDR interface.

        Args:
            device_index: USB device index.
            channels: Number of active RX channels (1 or 2).
            simulated: If True, uses synthetic HITL backend.
            center_freq: Initial center frequency in Hz.
            sample_rate: Initial sampling rate in Hz.
            gain: Initial RF gain in dB.
        """
        super().__init__(
            device_name="LimeSDR-USB",
            adc_bits=12,
            freq_range=(100.0e3, 3800.0e6),
            max_bandwidth=61.44e6,
            gain_range=(0.0, 73.0),
            channels=channels,
            simulated=simulated,
        )
        self.device_index = device_index
        self._sdr_handle: Optional[Any] = None
        self._mock_backend: Optional[MockSDR] = None

        if self.simulated:
            self._mock_backend = MockSDR(
                device_name=f"{self.device_name} (Simulated)",
                adc_bits=self.adc_bits,
                freq_range=self.freq_range,
                max_bandwidth=self.max_bandwidth,
                gain_range=self.gain_range,
                channels=self.channels,
            )
        else:
            self._init_hardware()

        self.configure(center_freq=center_freq, sample_rate=sample_rate, gain=gain)

    def _init_hardware(self) -> None:
        """Attempt to bind via SoapySDR."""
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX
            args = dict(driver="lime")
            self._sdr_handle = SoapySDR.Device(args)
        except ImportError:
            raise DriverNotInstalledError(
                "SoapySDR with LimeSuite bindings is not installed. "
                "Run in simulation mode with simulated=True."
            )
        except Exception as e:
            raise DeviceNotFoundError(
                f"Failed to find LimeSDR device: {e}. Use simulated=True for HITL emulation."
            )

    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Configure LimeSDR frequency, sample rate, and RF gain."""
        if center_freq is not None:
            if not self.validate_frequency(center_freq):
                raise DeviceConfigurationError(
                    f"LimeSDR frequency {center_freq/1e6:.2f} MHz outside [{self.freq_range[0]/1e6:.2f}, {self.freq_range[1]/1e6:.2f}] MHz"
                )
            self._center_freq = center_freq

        if sample_rate is not None:
            if not self.validate_sample_rate(sample_rate):
                raise DeviceConfigurationError(
                    f"LimeSDR sample rate {sample_rate/1e6:.2f} MSPS exceeds max 61.44 MSPS"
                )
            self._sample_rate = sample_rate

        if gain is not None:
            if not self.validate_gain(gain):
                raise DeviceConfigurationError(
                    f"LimeSDR gain {gain:.1f} dB outside [{self.gain_range[0]}, {self.gain_range[1]}] dB"
                )
            self._gain = float(gain)

        if self._mock_backend is not None:
            self._mock_backend.configure(
                center_freq=self._center_freq,
                sample_rate=self._sample_rate,
                gain=self._gain,
                **kwargs,
            )

    def start_rx(self) -> None:
        """Begin streaming samples."""
        self._is_streaming = True
        if self._mock_backend is not None:
            self._mock_backend.start_rx()

    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read complex I/Q baseband samples."""
        if not self._is_streaming:
            raise RuntimeError("LimeSDR receiver is not streaming. Call start_rx() first.")

        if self._mock_backend is not None:
            return self._mock_backend.read_samples(num_samples)

        raise DeviceNotFoundError("Physical LimeSDR hardware stream not active.")

    def stop_rx(self) -> None:
        """Halt streaming."""
        self._is_streaming = False
        if self._mock_backend is not None:
            self._mock_backend.stop_rx()

    def close(self) -> None:
        """Release LimeSDR handle."""
        self.stop_rx()
        if self._mock_backend is not None:
            self._mock_backend.close()
