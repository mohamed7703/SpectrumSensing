"""Unit Tests for Unified SDR Hardware Abstraction Layer.

Tests polymorphic interface, ADC quantization, MockSDR signal synthesis,
simulated hardware drivers, SDR factory, and exception handling.
"""

import pytest
import numpy as np
from pathlib import Path

from src.sdr_interface.base_sdr import (
    BaseSDR,
    DeviceConfigurationError,
    DeviceNotFoundError,
    DriverNotInstalledError,
)
from src.sdr_interface.mock_sdr import MockSDR
from src.sdr_interface.rtlsdr_driver import RTLSDRDriver
from src.sdr_interface.hackrf_driver import HackRFDriver
from src.sdr_interface.plutosdr_driver import PlutoSDRDriver
from src.sdr_interface.limesdr_driver import LimeSDRDriver
from src.sdr_interface.usrp_driver import USRPDriver
from src.sdr_interface.factory import SDRFactory


class TestBaseSDRQuantization:
    """Test ADC bit depth quantization and validation logic."""

    def test_quantization_levels_8bit(self) -> None:
        """8-bit ADC should quantize continuous samples into 256 discrete levels."""
        sdr = MockSDR(adc_bits=8)
        # Uniform sweep from -1 to 1
        samples = np.linspace(-1.0, 1.0, 1000) + 1j * np.linspace(-1.0, 1.0, 1000)
        q = sdr.quantize_samples(samples)

        # Discrete steps should equal 1 / 127
        diffs = np.unique(np.diff(np.sort(np.unique(q.real))))
        assert len(diffs) > 0
        min_step = np.min(diffs[diffs > 1e-6])
        expected_step = 1.0 / 127.0
        assert np.isclose(min_step, expected_step, atol=1e-4)

    def test_quantization_16bit_preserves_float(self) -> None:
        """16-bit or float MockSDR should preserve full resolution."""
        sdr = MockSDR(adc_bits=16)
        samples = np.array([0.123456 + 0.654321j], dtype=np.complex64)
        q = sdr.quantize_samples(samples)
        assert np.isclose(q[0], samples[0], atol=1e-6)

    def test_validation_bounds(self) -> None:
        """Parameter validation bounds checking."""
        sdr = MockSDR(
            freq_range=(100e6, 200e6),
            max_bandwidth=10e6,
            gain_range=(0.0, 50.0),
        )
        assert sdr.validate_frequency(150e6)
        assert not sdr.validate_frequency(50e6)
        assert not sdr.validate_frequency(250e6)

        assert sdr.validate_sample_rate(5e6)
        assert not sdr.validate_sample_rate(15e6)

        assert sdr.validate_gain(25.0)
        assert not sdr.validate_gain(-5.0)
        assert not sdr.validate_gain(60.0)


class TestMockSDR:
    """Test Mock SDR generator for HITL simulation."""

    def test_mock_streaming_lifecycle(self) -> None:
        """Verify start_rx, read_samples, and stop_rx cycle."""
        sdr = MockSDR(random_seed=42)
        assert not sdr.is_streaming

        with pytest.raises(RuntimeError):
            sdr.read_samples(1024)

        sdr.start_rx()
        assert sdr.is_streaming

        samples = sdr.read_samples(1024)
        assert len(samples) == 1024
        assert samples.dtype == np.complex64
        assert np.all(np.isfinite(samples))

        sdr.stop_rx()
        assert not sdr.is_streaming

    def test_mock_signal_types(self) -> None:
        """Test tone, qpsk, ofdm, and noise generation."""
        for sig_type in ["noise", "tone", "qpsk", "ofdm"]:
            sdr = MockSDR(signal_type=sig_type, snr_db=15.0, random_seed=42)
            sdr.start_rx()
            samples = sdr.read_samples(2048)
            sdr.stop_rx()
            assert len(samples) == 2048
            power = np.mean(np.abs(samples) ** 2)
            assert power > 0.0

    def test_mock_fading_models(self) -> None:
        """Test Rayleigh and Rician channel fading emulation."""
        for fading in ["rayleigh", "rician"]:
            sdr = MockSDR(signal_type="qpsk", fading_model=fading, random_seed=123)
            sdr.start_rx()
            samples = sdr.read_samples(1024)
            sdr.stop_rx()
            assert len(samples) == 1024
            assert np.all(np.isfinite(samples))

    def test_context_manager(self) -> None:
        """Verify BaseSDR context manager automatically handles lifecycle."""
        with MockSDR(random_seed=99) as sdr:
            assert sdr.is_streaming
            samples = sdr.read_samples(512)
            assert len(samples) == 512
        assert not sdr.is_streaming


class TestSimulatedDrivers:
    """Test concrete drivers operating under simulated HITL mode."""

    def test_rtlsdr_simulated(self) -> None:
        """RTL-SDR driver in simulation mode."""
        sdr = RTLSDRDriver(simulated=True, center_freq=100e6, sample_rate=2.4e6, gain=20.0)
        assert sdr.device_name == "RTL-SDR Blog V3"
        assert sdr.adc_bits == 8

        # Test supported gain snapping
        sdr.configure(gain=21.0)
        assert sdr.gain == 20.7  # Closest supported step

        # Out of bounds freq
        with pytest.raises(DeviceConfigurationError):
            sdr.configure(center_freq=2000e6)

        with sdr:
            samples = sdr.read_samples(1024)
            assert len(samples) == 1024

    def test_hackrf_simulated(self) -> None:
        """HackRF One driver in simulation mode."""
        sdr = HackRFDriver(simulated=True, center_freq=915e6, sample_rate=10e6, lna_gain=16.0, vga_gain=20.0)
        assert sdr.device_name == "HackRF One"
        assert sdr.adc_bits == 8

        with sdr:
            samples = sdr.read_samples(1024)
            assert len(samples) == 1024

        # Gain splitting
        sdr.configure(gain=50.0)
        assert sdr.gain > 0

    def test_plutosdr_simulated(self) -> None:
        """PlutoSDR driver in simulation mode."""
        sdr = PlutoSDRDriver(simulated=True, center_freq=2412e6, sample_rate=5e6, gain=30.0)
        assert sdr.device_name == "ADALM-PLUTO"
        assert sdr.adc_bits == 12

        with sdr:
            samples = sdr.read_samples(1024)
            assert len(samples) == 1024

    def test_limesdr_mimo_simulated(self) -> None:
        """LimeSDR MIMO driver in simulation mode."""
        sdr = LimeSDRDriver(simulated=True, channels=2, center_freq=2412e6, sample_rate=10e6)
        assert sdr.channels == 2
        assert sdr.adc_bits == 12

        with sdr:
            samples = sdr.read_samples(1024)
            assert samples.shape == (2, 1024)

    def test_usrp_b210_and_x310_simulated(self) -> None:
        """USRP B210 and X310 in simulation mode."""
        b210 = USRPDriver(model="b210", simulated=True, center_freq=2412e6)
        assert b210.adc_bits == 12
        with b210:
            samples = b210.read_samples(1024)
            assert len(samples) == 1024

        x310 = USRPDriver(model="x310", simulated=True, center_freq=2412e6)
        assert x310.adc_bits == 14
        with x310:
            samples = x310.read_samples(1024)
            assert len(samples) == 1024


class TestDriverHardwareDetectionErrors:
    """Test that missing native drivers raise clean DriverNotInstalledError or DeviceNotFoundError."""

    def test_rtlsdr_raises_without_hardware(self) -> None:
        with pytest.raises((DriverNotInstalledError, DeviceNotFoundError)):
            RTLSDRDriver(simulated=False)

    def test_hackrf_raises_without_hardware(self) -> None:
        with pytest.raises((DriverNotInstalledError, DeviceNotFoundError)):
            HackRFDriver(simulated=False)

    def test_pluto_raises_without_hardware(self) -> None:
        with pytest.raises((DriverNotInstalledError, DeviceNotFoundError)):
            PlutoSDRDriver(simulated=False)

    def test_lime_raises_without_hardware(self) -> None:
        with pytest.raises((DriverNotInstalledError, DeviceNotFoundError)):
            LimeSDRDriver(simulated=False)

    def test_usrp_raises_without_hardware(self) -> None:
        with pytest.raises((DriverNotInstalledError, DeviceNotFoundError)):
            USRPDriver(model="b210", simulated=False)


class TestSDRFactory:
    """Test SDRFactory creation and YAML configuration loader."""

    def test_factory_create_by_name(self) -> None:
        sdr = SDRFactory.create("rtlsdr", simulated=True)
        assert isinstance(sdr, RTLSDRDriver)

        sdr_mimo = SDRFactory.create("limesdr", simulated=True, channels=2)
        assert isinstance(sdr_mimo, LimeSDRDriver)

        sdr_mock = SDRFactory.create("mock")
        assert isinstance(sdr_mock, MockSDR)

    def test_factory_from_yaml(self) -> None:
        config_path = Path("configs/sdr_config.yaml")
        assert config_path.exists()

        sdr_rtl = SDRFactory.from_config(config_path, "rtlsdr", simulated=True)
        assert isinstance(sdr_rtl, RTLSDRDriver)
        assert sdr_rtl.sample_rate == 2.4e6

        sdr_b210 = SDRFactory.from_config(config_path, "usrp_b210", simulated=True)
        assert isinstance(sdr_b210, USRPDriver)
        assert sdr_b210.adc_bits == 12
