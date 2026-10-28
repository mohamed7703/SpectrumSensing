"""HackRF One Hardware Driver.

Wraps the Great Scott Gadgets HackRF One frontend (8-bit ADC/DAC, 20 MHz BW,
1 MHz to 6 GHz tuning) with BaseSDR interface and simulated fallback.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from src.sdr_interface.base_sdr import BaseSDR, DeviceConfigurationError, DeviceNotFoundError, DriverNotInstalledError
from src.sdr_interface.mock_sdr import MockSDR


class HackRFDriver(BaseSDR):
    """Driver for HackRF One SDR (MAX2837 + MAX5864 + RFFC5072).

    Specifications:
        - ADC Resolution: 8-bit
        - Tuning Range: 1.0 MHz to 6000.0 MHz
        - Max Bandwidth: 20.0 MSPS
        - Gain Stages: LNA (0-40 dB, 8 dB steps), VGA (0-62 dB, 2 dB steps)
        - Operating Mode: Half-duplex transceiver
    """

    def __init__(
        self,
        device_index: int = 0,
        channels: int = 1,
        simulated: bool = False,
        center_freq: float = 915.0e6,
        sample_rate: float = 10.0e6,
        lna_gain: float = 16.0,
        vga_gain: float = 20.0,
        **kwargs: Any,
    ) -> None:
        """Initialize HackRF interface.

        Args:
            device_index: USB device index.
            simulated: If True, uses synthetic HITL backend.
            center_freq: Initial frequency in Hz.
            sample_rate: Initial sampling rate in Hz.
            lna_gain: LNA amplifier gain in dB (0 to 40 dB).
            vga_gain: VGA baseband gain in dB (0 to 62 dB).
        """
        super().__init__(
            device_name="HackRF One",
            adc_bits=8,
            freq_range=(1.0e6, 6000.0e6),
            max_bandwidth=20.0e6,
            gain_range=(0.0, 102.0),  # LNA (40) + VGA (62)
            channels=1,
            simulated=simulated,
        )
        self.device_index = device_index
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain
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
            gain=lna_gain + vga_gain,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
        )

    def _init_hardware(self) -> None:
        """Attempt to bind to libhackrf or pyhackrf/SoapySDR."""
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX
            args = dict(driver="hackrf")
            self._sdr_handle = SoapySDR.Device(args)
        except ImportError:
            try:
                import hackrf
                self._sdr_handle = hackrf.HackRF()
            except ImportError:
                raise DriverNotInstalledError(
                    "Neither SoapySDR nor pyhackrf is installed. "
                    "Run with simulated=True for HITL emulation."
                )
            except Exception as e:
                raise DeviceNotFoundError(f"Failed to open HackRF device: {e}")
        except Exception as e:
            raise DeviceNotFoundError(f"HackRF device not found via SoapySDR: {e}")

    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        lna_gain: Optional[float] = None,
        vga_gain: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Configure HackRF center frequency, sample rate, and amplifier stages."""
        if center_freq is not None:
            if not self.validate_frequency(center_freq):
                raise DeviceConfigurationError(
                    f"HackRF frequency {center_freq/1e6:.2f} MHz outside [{self.freq_range[0]/1e6:.1f}, {self.freq_range[1]/1e6:.1f}] MHz"
                )
            self._center_freq = center_freq

        if sample_rate is not None:
            if not self.validate_sample_rate(sample_rate):
                raise DeviceConfigurationError(
                    f"HackRF sample rate {sample_rate/1e6:.2f} MSPS exceeds max 20 MSPS"
                )
            self._sample_rate = sample_rate

        if lna_gain is not None:
            lna_val = float(lna_gain)
            if not (0.0 <= lna_val <= 40.0):
                raise DeviceConfigurationError(f"HackRF LNA gain {lna_val} dB outside [0, 40] dB")
            self.lna_gain = round(lna_val / 8.0) * 8.0  # 8 dB steps

        if vga_gain is not None:
            vga_val = float(vga_gain)
            if not (0.0 <= vga_val <= 62.0):
                raise DeviceConfigurationError(f"HackRF VGA gain {vga_val} dB outside [0, 62] dB")
            self.vga_gain = round(vga_val / 2.0) * 2.0  # 2 dB steps

        if gain is not None and lna_gain is None and vga_gain is None:
            # Distribute total gain across LNA and VGA stages
            clamped = max(0.0, min(102.0, float(gain)))
            self.lna_gain = min(40.0, round(clamped * 0.4 / 8.0) * 8.0)
            self.vga_gain = min(62.0, round((clamped - self.lna_gain) / 2.0) * 2.0)

        self._gain = self.lna_gain + self.vga_gain

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
            raise RuntimeError("HackRF receiver is not streaming. Call start_rx() first.")

        if self._mock_backend is not None:
            return self._mock_backend.read_samples(num_samples)

        raise DeviceNotFoundError("Physical HackRF hardware stream not active.")

    def stop_rx(self) -> None:
        """Halt streaming."""
        self._is_streaming = False
        if self._mock_backend is not None:
            self._mock_backend.stop_rx()

    def close(self) -> None:
        """Release HackRF handle."""
        self.stop_rx()
        if self._mock_backend is not None:
            self._mock_backend.close()
