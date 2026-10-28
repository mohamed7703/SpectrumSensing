"""Ettus Research USRP Hardware Driver (B210 / X310).

Wraps the Ettus Research USRP B210 (12-bit, 56 MHz BW) and USRP X310
(14-bit, 160 MHz BW) via UHD (USRP Hardware Driver) with BaseSDR interface.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from src.sdr_interface.base_sdr import BaseSDR, DeviceConfigurationError, DeviceNotFoundError, DriverNotInstalledError
from src.sdr_interface.mock_sdr import MockSDR


class USRPDriver(BaseSDR):
    """Driver for Ettus Research USRP series (B210 / X310).

    Specifications:
        - B210: 12-bit ADC, 70 MHz to 6000 MHz, 56 MSPS BW, 0-76 dB gain, 2x2 MIMO
        - X310: 14-bit ADC, 10 MHz to 6000 MHz, 160 MSPS BW, 0-38 dB gain, 2 channels
    """

    MODEL_SPECS = {
        "b210": {
            "name": "USRP B210",
            "adc_bits": 12,
            "freq_range": (70.0e6, 6000.0e6),
            "max_bandwidth": 56.0e6,
            "gain_range": (0.0, 76.0),
            "channels": 2,
        },
        "x310": {
            "name": "USRP X310",
            "adc_bits": 14,
            "freq_range": (10.0e6, 6000.0e6),
            "max_bandwidth": 160.0e6,
            "gain_range": (0.0, 38.0),
            "channels": 2,
        },
    }

    def __init__(
        self,
        model: str = "b210",
        device_args: str = "",
        channels: int = 1,
        simulated: bool = False,
        center_freq: float = 2412.0e6,
        sample_rate: float = 16.0e6,
        gain: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize USRP interface.

        Args:
            model: "b210" or "x310".
            device_args: UHD device identification arguments (e.g., "type=b200" or "addr=192.168.10.2").
            channels: Number of active coherent channels (1 or 2).
            simulated: If True, uses synthetic HITL backend.
            center_freq: Initial center frequency in Hz.
            sample_rate: Initial sampling rate in Hz.
            gain: Initial RF gain in dB.
        """
        model_key = model.lower()
        if model_key not in self.MODEL_SPECS:
            raise ValueError(f"Unknown USRP model '{model}'. Supported: {list(self.MODEL_SPECS.keys())}")

        specs = self.MODEL_SPECS[model_key]
        super().__init__(
            device_name=specs["name"],
            adc_bits=specs["adc_bits"],
            freq_range=specs["freq_range"],
            max_bandwidth=specs["max_bandwidth"],
            gain_range=specs["gain_range"],
            channels=channels,
            simulated=simulated,
        )
        self.model = model_key
        self.device_args = device_args
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
        """Attempt to bind via uhd."""
        try:
            import uhd
            self._sdr_handle = uhd.usrp.MultiUSRP(self.device_args)
        except ImportError:
            raise DriverNotInstalledError(
                "UHD Python bindings (uhd) are not installed. "
                "Run in simulation mode with simulated=True."
            )
        except Exception as e:
            raise DeviceNotFoundError(
                f"Failed to find USRP device with args '{self.device_args}': {e}. "
                "Ensure device is connected or use simulated=True."
            )

    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Configure USRP center frequency, sampling rate, and gain."""
        if center_freq is not None:
            if not self.validate_frequency(center_freq):
                raise DeviceConfigurationError(
                    f"USRP frequency {center_freq/1e6:.2f} MHz outside [{self.freq_range[0]/1e6:.1f}, {self.freq_range[1]/1e6:.1f}] MHz"
                )
            self._center_freq = center_freq
            if self._sdr_handle is not None:
                for ch in range(self.channels):
                    self._sdr_handle.set_rx_freq(center_freq, ch)

        if sample_rate is not None:
            if not self.validate_sample_rate(sample_rate):
                raise DeviceConfigurationError(
                    f"USRP sample rate {sample_rate/1e6:.2f} MSPS exceeds max {self.max_bandwidth/1e6:.2f} MSPS"
                )
            self._sample_rate = sample_rate
            if self._sdr_handle is not None:
                for ch in range(self.channels):
                    self._sdr_handle.set_rx_rate(sample_rate, ch)

        if gain is not None:
            if not self.validate_gain(gain):
                raise DeviceConfigurationError(
                    f"USRP gain {gain:.1f} dB outside [{self.gain_range[0]}, {self.gain_range[1]}] dB"
                )
            self._gain = float(gain)
            if self._sdr_handle is not None:
                for ch in range(self.channels):
                    self._sdr_handle.set_rx_gain(float(gain), ch)

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
            raise RuntimeError("USRP receiver is not streaming. Call start_rx() first.")

        if self._mock_backend is not None:
            return self._mock_backend.read_samples(num_samples)

        raise DeviceNotFoundError("Physical USRP hardware stream not active.")

    def stop_rx(self) -> None:
        """Halt streaming."""
        self._is_streaming = False
        if self._mock_backend is not None:
            self._mock_backend.stop_rx()

    def close(self) -> None:
        """Release USRP handle."""
        self.stop_rx()
        if self._mock_backend is not None:
            self._mock_backend.close()
