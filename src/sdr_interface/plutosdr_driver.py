"""ADALM-PLUTO (PlutoSDR) Hardware Driver.

Wraps the Analog Devices ADALM-PLUTO (AD9363/AD9364 transceiver, 12-bit ADC,
20 MHz IBW, 325 MHz to 3800 MHz) with BaseSDR interface and simulated fallback.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from src.sdr_interface.base_sdr import BaseSDR, DeviceConfigurationError, DeviceNotFoundError, DriverNotInstalledError
from src.sdr_interface.mock_sdr import MockSDR


class PlutoSDRDriver(BaseSDR):
    """Driver for ADALM-PLUTO SDR (AD9363 RF Agile Transceiver).

    Specifications:
        - ADC Resolution: 12-bit
        - Tuning Range: 325 MHz to 3800 MHz (extendable to 70 MHz - 6 GHz)
        - Max Bandwidth: 20.0 MSPS instantaneous
        - Programmable Gain: -4.0 dB to 71.0 dB (manual or AGC)
        - Interface: USB 2.0 / Ethernet over USB (default URI: ip:192.168.2.1)
    """

    def __init__(
        self,
        uri: str = "ip:192.168.2.1",
        channels: int = 1,
        simulated: bool = False,
        center_freq: float = 2412.0e6,
        sample_rate: float = 5.0e6,
        gain: float = 30.0,
        gain_control_mode: str = "manual",
        **kwargs: Any,
    ) -> None:
        """Initialize PlutoSDR interface.

        Args:
            uri: Device URI connection string (e.g. 'ip:192.168.2.1' or 'usb:1.2.5').
            simulated: If True, uses synthetic HITL backend.
            center_freq: Initial center frequency in Hz.
            sample_rate: Initial sampling rate in Hz.
            gain: Initial RX gain in dB.
            gain_control_mode: 'manual', 'slow_attack', 'fast_attack', or 'hybrid'.
        """
        super().__init__(
            device_name="ADALM-PLUTO",
            adc_bits=12,
            freq_range=(325.0e6, 3800.0e6),
            max_bandwidth=20.0e6,
            gain_range=(-4.0, 71.0),
            channels=1,
            simulated=simulated,
        )
        self.uri = uri
        self.gain_control_mode = gain_control_mode
        self._sdr_handle: Optional[Any] = None
        self._mock_backend: Optional[MockSDR] = None

        if self.simulated:
            self._mock_backend = MockSDR(
                device_name=f"{self.device_name} (Simulated)",
                adc_bits=self.adc_bits,
                freq_range=self.freq_range,
                max_bandwidth=self.max_bandwidth,
                gain_range=self.gain_range,
                channels=1,
            )
        else:
            self._init_hardware()

        self.configure(
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
            gain_control_mode=gain_control_mode,
        )

    def _init_hardware(self) -> None:
        """Attempt to bind via pyadi-iio (adi.Pluto)."""
        try:
            import adi
            self._sdr_handle = adi.Pluto(self.uri)
        except ImportError:
            raise DriverNotInstalledError(
                "pyadi-iio package is not installed. Install via 'pip install pyadi-iio' "
                "or run in simulation mode with simulated=True."
            )
        except Exception as e:
            raise DeviceNotFoundError(
                f"Failed to connect to PlutoSDR at {self.uri}: {e}. "
                "Verify network/USB connection or use simulated=True."
            )

    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        gain_control_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Configure PlutoSDR center frequency, decimation rate, and gain."""
        if center_freq is not None:
            if not self.validate_frequency(center_freq):
                raise DeviceConfigurationError(
                    f"PlutoSDR frequency {center_freq/1e6:.2f} MHz outside [{self.freq_range[0]/1e6:.1f}, {self.freq_range[1]/1e6:.1f}] MHz"
                )
            self._center_freq = center_freq
            if self._sdr_handle is not None:
                self._sdr_handle.rx_lo = int(center_freq)

        if sample_rate is not None:
            if not self.validate_sample_rate(sample_rate):
                raise DeviceConfigurationError(
                    f"PlutoSDR sample rate {sample_rate/1e6:.2f} MSPS exceeds max 20 MSPS"
                )
            self._sample_rate = sample_rate
            if self._sdr_handle is not None:
                self._sdr_handle.sample_rate = int(sample_rate)

        if gain is not None:
            if not self.validate_gain(gain):
                raise DeviceConfigurationError(
                    f"PlutoSDR gain {gain:.1f} dB outside [{self.gain_range[0]}, {self.gain_range[1]}] dB"
                )
            self._gain = float(gain)
            if self._sdr_handle is not None:
                self._sdr_handle.rx_hardwaregain_chan0 = float(gain)

        if gain_control_mode is not None:
            self.gain_control_mode = gain_control_mode
            if self._sdr_handle is not None:
                self._sdr_handle.gain_control_mode_chan0 = gain_control_mode

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
            raise RuntimeError("PlutoSDR receiver is not streaming. Call start_rx() first.")

        if self._mock_backend is not None:
            return self._mock_backend.read_samples(num_samples)

        if self._sdr_handle is not None:
            self._sdr_handle.rx_buffer_size = num_samples
            raw = self._sdr_handle.rx()
            # Pluto returns complex raw values normalized to 12-bit full scale
            return (raw / 2048.0).astype(np.complex64)

        raise DeviceNotFoundError("Physical PlutoSDR hardware stream not active.")

    def stop_rx(self) -> None:
        """Halt streaming."""
        self._is_streaming = False
        if self._mock_backend is not None:
            self._mock_backend.stop_rx()

    def close(self) -> None:
        """Release PlutoSDR connection."""
        self.stop_rx()
        if self._sdr_handle is not None:
            self._sdr_handle = None
        if self._mock_backend is not None:
            self._mock_backend.close()
