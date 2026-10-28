"""SDR Device Factory.

Provides a unified factory to instantiate appropriate SDR drivers from
configuration files, device aliases, or hardware detection.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import yaml

from src.sdr_interface.base_sdr import BaseSDR
from src.sdr_interface.mock_sdr import MockSDR
from src.sdr_interface.rtlsdr_driver import RTLSDRDriver
from src.sdr_interface.hackrf_driver import HackRFDriver
from src.sdr_interface.plutosdr_driver import PlutoSDRDriver
from src.sdr_interface.limesdr_driver import LimeSDRDriver
from src.sdr_interface.usrp_driver import USRPDriver


class SDRFactory:
    """Factory for SDR hardware and mock frontends."""

    DRIVER_MAP = {
        "rtlsdr": RTLSDRDriver,
        "rtl-sdr": RTLSDRDriver,
        "hackrf": HackRFDriver,
        "hackrf-one": HackRFDriver,
        "plutosdr": PlutoSDRDriver,
        "pluto": PlutoSDRDriver,
        "adalm-pluto": PlutoSDRDriver,
        "limesdr": LimeSDRDriver,
        "lime": LimeSDRDriver,
        "usrp": USRPDriver,
        "usrp_b210": USRPDriver,
        "usrp_x310": USRPDriver,
        "mock": MockSDR,
        "mock_sdr": MockSDR,
    }

    @classmethod
    def create(
        cls,
        device_type: str,
        simulated: bool = False,
        **kwargs: Any,
    ) -> BaseSDR:
        """Create an SDR driver instance by device name.

        Args:
            device_type: Device key ('rtlsdr', 'hackrf', 'plutosdr', 'limesdr', 'usrp_b210', 'mock').
            simulated: If True, operates in simulation mode without hardware.
            **kwargs: Extra parameters passed to driver constructor.

        Returns:
            Configured BaseSDR driver instance.
        """
        key = device_type.lower().replace(" ", "_")
        if key not in cls.DRIVER_MAP:
            raise ValueError(
                f"Unknown SDR device '{device_type}'. Available devices: {list(cls.DRIVER_MAP.keys())}"
            )

        driver_cls = cls.DRIVER_MAP[key]

        if key in ("usrp_b210", "usrp_x310"):
            model = "b210" if "b210" in key else "x310"
            return driver_cls(model=model, simulated=simulated, **kwargs)

        if driver_cls is MockSDR:
            return MockSDR(**kwargs)

        return driver_cls(simulated=simulated, **kwargs)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        device_type: str = "mock_sdr",
        simulated: bool = True,
    ) -> BaseSDR:
        """Instantiate an SDR driver using parameters from a YAML config file.

        Args:
            config_path: Path to YAML configuration file (e.g. configs/sdr_config.yaml).
            device_type: Key of the device profile within the YAML.
            simulated: Whether to run in simulated HITL mode.

        Returns:
            Configured BaseSDR instance.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"SDR config file not found: {config_path}")

        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)

        defaults = cfg.get("defaults", {})
        device_cfgs = cfg.get("devices", {})
        dev_cfg = device_cfgs.get(device_type, {})

        freq = dev_cfg.get("center_freq", defaults.get("center_freq", 2.4e9))
        rate = dev_cfg.get("default_sample_rate", defaults.get("sample_rate", 2.4e6))
        gain = dev_cfg.get("gain", defaults.get("gain", 20.0))
        channels = dev_cfg.get("channels", 1)

        # Ensure frequency is clamped within hardware supported range
        freq_range = dev_cfg.get("freq_range")
        if freq_range:
            min_f, max_f = float(freq_range[0]), float(freq_range[1])
            if not (min_f <= float(freq) <= max_f):
                freq = (min_f + max_f) / 2.0
            else:
                freq = float(freq)

        # Ensure sample rate is within max bandwidth
        max_bw = dev_cfg.get("max_bandwidth")
        if max_bw and float(rate) > float(max_bw):
            rate = float(max_bw)
        else:
            rate = float(rate)

        kwargs: Dict[str, Any] = {
            "center_freq": float(freq),
            "sample_rate": float(rate),
            "gain": float(gain),
        }
        if "channels" in dev_cfg:
            kwargs["channels"] = channels

        return cls.create(device_type=device_type, simulated=simulated, **kwargs)
