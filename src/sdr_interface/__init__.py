"""SDR Hardware Interface and Abstraction Module.

Provides unified polymorphic drivers across RTL-SDR, HackRF One, PlutoSDR,
LimeSDR, USRP, and high-fidelity Mock SDR for HITL simulation.
"""

from src.sdr_interface.base_sdr import (
    BaseSDR,
    SDRError,
    DeviceNotFoundError,
    DriverNotInstalledError,
    DeviceConfigurationError,
)
from src.sdr_interface.mock_sdr import MockSDR
from src.sdr_interface.rtlsdr_driver import RTLSDRDriver
from src.sdr_interface.hackrf_driver import HackRFDriver
from src.sdr_interface.plutosdr_driver import PlutoSDRDriver
from src.sdr_interface.limesdr_driver import LimeSDRDriver
from src.sdr_interface.usrp_driver import USRPDriver
from src.sdr_interface.factory import SDRFactory
from src.sdr_interface.stream_handler import (
    CircularRingBuffer,
    StreamHandler,
    LatencyProfiler,
    DutyCycleController,
)

__all__ = [
    "BaseSDR",
    "SDRError",
    "DeviceNotFoundError",
    "DriverNotInstalledError",
    "DeviceConfigurationError",
    "MockSDR",
    "RTLSDRDriver",
    "HackRFDriver",
    "PlutoSDRDriver",
    "LimeSDRDriver",
    "USRPDriver",
    "SDRFactory",
    "CircularRingBuffer",
    "StreamHandler",
    "LatencyProfiler",
    "DutyCycleController",
]
