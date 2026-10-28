"""RTL-SDR Blog V3 Hardware Driver.

Wraps the Realtek RTL2832U DVB-T / R820T2 frontend (8-bit ADC, 2.4 MHz BW)
with polymorphic BaseSDR lifecycle and simulated fallback.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from src.sdr_interface.base_sdr import BaseSDR, DeviceConfigurationError, DeviceNotFoundError, DriverNotInstalledError
from src.sdr_interface.mock_sdr import MockSDR


class RTLSDRDriver(BaseSDR):
    """Driver for RTL-SDR Blog V3 dongles (RTL2832U + R820T2).

    Specifications:
        - ADC Resolution: 8-bit (I and Q rails in uint8 [0, 255])
        - Tuning Range: 24 MHz to 1766 MHz
        - Max Bandwidth: 2.4 MSPS continuous (3.2 MSPS burst)
        - Programmable Gain: 0.0 to 49.6 dB in 29 discrete steps
    """

    SUPPORTED_GAINS = [
        0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
        19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
        38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6,
    ]

    def __init__(
        self,
        device_index: int = 0,
        channels: int = 1,
        simulated: bool = False,
        center_freq: float = 100.0e6,
        sample_rate: float = 2.4e6,
        gain: float = 20.0,
        **kwargs: Any,
    ) -> None:
        """Initialize RTL-SDR interface.

        Args:
            device_index: USB enumeration index (0 for first dongle).
            simulated: If True, uses synthetic HITL backend.
            center_freq: Initial center frequency in Hz.
            sample_rate: Initial sampling rate in Hz.
            gain: Initial gain in dB.
        """
        super().__init__(
            device_name="RTL-SDR Blog V3",
            adc_bits=8,
            freq_range=(24.0e6, 1766.0e6),
            max_bandwidth=2.4e6,
            gain_range=(0.0, 49.6),
            channels=1,
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
                channels=1,
            )
        else:
            self._init_hardware()

        self.configure(center_freq=center_freq, sample_rate=sample_rate, gain=gain)

    def _init_hardware(self) -> None:
        """Attempt to bind to librtlsdr via pyrtlsdr."""
        try:
            from rtlsdr import RtlSdr
            self._sdr_handle = RtlSdr(device_index=self.device_index)
        except ImportError:
            raise DriverNotInstalledError(
                "pyrtlsdr package is not installed. Install via 'pip install pyrtlsdr' "
                "or run in simulation mode with simulated=True."
            )
        except Exception as e:
            raise DeviceNotFoundError(
                f"Failed to open RTL-SDR dongle index {self.device_index}: {e}. "
                "Ensure device is plugged in or use simulated=True."
            )

    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[Union[float, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Configure RTL-SDR tuner and baseband decimation."""
        if center_freq is not None:
            if not self.validate_frequency(center_freq):
                raise DeviceConfigurationError(
                    f"RTL-SDR frequency {center_freq/1e6:.2f} MHz outside [{self.freq_range[0]/1e6:.1f}, {self.freq_range[1]/1e6:.1f}] MHz"
                )
            self._center_freq = center_freq
            if self._sdr_handle is not None:
                self._sdr_handle.center_freq = center_freq

        if sample_rate is not None:
            if not self.validate_sample_rate(sample_rate):
                raise DeviceConfigurationError(
                    f"RTL-SDR sample rate {sample_rate/1e6:.2f} MSPS exceeds max 2.4 MSPS"
                )
            self._sample_rate = sample_rate
            if self._sdr_handle is not None:
                self._sdr_handle.sample_rate = sample_rate

        if gain is not None:
            if isinstance(gain, (int, float)):
                if not self.validate_gain(float(gain)):
                    raise DeviceConfigurationError(
                        f"RTL-SDR gain {gain} dB outside [{self.gain_range[0]}, {self.gain_range[1]}] dB"
                    )
                # Find closest supported gain
                closest_gain = min(self.SUPPORTED_GAINS, key=lambda g: abs(g - float(gain)))
                self._gain = closest_gain
                if self._sdr_handle is not None:
                    self._sdr_handle.gain = closest_gain
            elif str(gain).lower() == "auto":
                self._gain = 0.0
                if self._sdr_handle is not None:
                    self._sdr_handle.gain = "auto"

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
            raise RuntimeError("RTL-SDR receiver is not streaming. Call start_rx() first.")

        if self._mock_backend is not None:
            return self._mock_backend.read_samples(num_samples)

        if self._sdr_handle is not None:
            raw_samples = self._sdr_handle.read_samples(num_samples)
            return np.asarray(raw_samples, dtype=np.complex64)

        raise DeviceNotFoundError("No RTL-SDR hardware or mock backend available.")

    def stop_rx(self) -> None:
        """Stop streaming."""
        self._is_streaming = False
        if self._mock_backend is not None:
            self._mock_backend.stop_rx()

    def close(self) -> None:
        """Close device connection."""
        self.stop_rx()
        if self._sdr_handle is not None:
            try:
                self._sdr_handle.close()
            except Exception:
                pass
            self._sdr_handle = None
        if self._mock_backend is not None:
            self._mock_backend.close()
