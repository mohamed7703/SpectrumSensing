"""High-Fidelity Mock SDR Generator for Hardware-in-the-Loop (HITL) Simulation.

Provides synthetic RF generation matching real physical SDR characteristics
(noise floor, primary user modulation, fading, and ADC quantization) without
requiring physical USB RF hardware to be connected.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

from src.sdr_interface.base_sdr import BaseSDR, DeviceConfigurationError


class MockSDR(BaseSDR):
    """High-fidelity Hardware-in-the-Loop Mock SDR receiver.

    Simulates an arbitrary SDR frontend emitting complex baseband I/Q samples
    with configurable Primary User (PU) signals, fading channels, and noise floor.
    """

    def __init__(
        self,
        device_name: str = "MockSDR",
        adc_bits: int = 16,
        freq_range: Tuple[float, float] = (1.0e6, 10000.0e6),
        max_bandwidth: float = 200.0e6,
        gain_range: Tuple[float, float] = (0.0, 100.0),
        channels: int = 1,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        noise_floor_dbm: float = -100.0,
        signal_type: str = "qpsk",
        snr_db: Optional[float] = 10.0,
        fading_model: Optional[str] = None,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Mock SDR frontend.

        Args:
            device_name: Descriptive name.
            adc_bits: ADC bit resolution for quantization simulation.
            freq_range: Programmable frequency bounds (min_hz, max_hz).
            max_bandwidth: Maximum instantaneous bandwidth in Hz.
            gain_range: Programmable gain bounds (min_db, max_db).
            channels: Number of coherent RX channels.
            noise_floor_dbm: Synthesized thermal noise floor in dBm.
            signal_type: Emulated PU transmission: "noise", "tone", "qpsk", "ofdm".
            snr_db: Signal-to-Noise Ratio in dB. None implies pure noise (H0).
            fading_model: Optional channel fading ("rayleigh", "rician", or None).
            random_seed: Optional deterministic PRNG seed.
        """
        super().__init__(
            device_name=device_name,
            adc_bits=adc_bits,
            freq_range=freq_range,
            max_bandwidth=max_bandwidth,
            gain_range=gain_range,
            channels=channels,
            simulated=True,
        )
        self.noise_floor_dbm = noise_floor_dbm
        self.signal_type = signal_type
        self.snr_db = snr_db
        self.fading_model = fading_model
        self.rng = np.random.default_rng(random_seed)
        self._sample_counter: int = 0

        if center_freq is not None or sample_rate is not None or gain is not None:
            self.configure(center_freq=center_freq, sample_rate=sample_rate, gain=gain)

    def configure(
        self,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None,
        gain: Optional[float] = None,
        signal_type: Optional[str] = None,
        snr_db: Optional[float] = None,
        fading_model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Configure mock receiver parameters."""
        if center_freq is not None:
            if not self.validate_frequency(center_freq):
                raise DeviceConfigurationError(
                    f"Frequency {center_freq/1e6:.2f} MHz outside [{self.freq_range[0]/1e6:.2f}, {self.freq_range[1]/1e6:.2f}] MHz"
                )
            self._center_freq = center_freq

        if sample_rate is not None:
            if not self.validate_sample_rate(sample_rate):
                raise DeviceConfigurationError(
                    f"Sample rate {sample_rate/1e6:.2f} MSPS exceeds max bandwidth {self.max_bandwidth/1e6:.2f} MSPS"
                )
            self._sample_rate = sample_rate

        if gain is not None:
            if not self.validate_gain(gain):
                raise DeviceConfigurationError(
                    f"Gain {gain:.1f} dB outside [{self.gain_range[0]:.1f}, {self.gain_range[1]:.1f}] dB"
                )
            self._gain = gain

        if signal_type is not None:
            self.signal_type = signal_type
        if snr_db is not None:
            self.snr_db = snr_db
        if fading_model is not None:
            self.fading_model = fading_model

    def start_rx(self) -> None:
        """Start mock streaming reception."""
        self._is_streaming = True
        self._sample_counter = 0

    def _generate_synthetic_signal(self, n: int) -> np.ndarray:
        """Synthesize baseband Primary User signal before channel and noise."""
        if self.signal_type == "noise" or self.snr_db is None:
            return np.zeros(n, dtype=np.complex64)

        elif self.signal_type == "tone":
            # CW carrier at 10% of Nyquist bandwidth
            fc = 0.1 * (self.sample_rate / 2.0)
            t = (np.arange(n) + self._sample_counter) / self.sample_rate
            sig = np.exp(1j * 2 * np.pi * fc * t)

        elif self.signal_type == "qpsk":
            # Over-sampled QPSK constellation with random symbol transitions
            sps = 8  # Samples per symbol
            n_symbols = int(np.ceil(n / sps)) + 1
            constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            syms = self.rng.choice(constellation, size=n_symbols)
            # Repeat to form square pulses
            pulses = np.repeat(syms, sps)[:n]
            # Small frequency offset
            cfo = 0.02 * (self.sample_rate / 2.0)
            t = (np.arange(n) + self._sample_counter) / self.sample_rate
            sig = pulses * np.exp(1j * 2 * np.pi * cfo * t)

        elif self.signal_type == "ofdm":
            # Simplified multi-carrier superposition
            num_subcarriers = 16
            sig = np.zeros(n, dtype=np.complex64)
            for k in range(-num_subcarriers // 2, num_subcarriers // 2):
                if k == 0:
                    continue  # Null DC subcarrier
                fk = k * (self.sample_rate / (num_subcarriers * 4))
                phase = self.rng.uniform(0, 2 * np.pi)
                t = (np.arange(n) + self._sample_counter) / self.sample_rate
                sig += np.exp(1j * (2 * np.pi * fk * t + phase))
            sig /= np.sqrt(num_subcarriers)
        else:
            raise ValueError(f"Unsupported synthetic signal type: {self.signal_type}")

        # Normalize signal unit power
        sig_pow = np.mean(np.abs(sig) ** 2)
        if sig_pow > 0:
            sig = sig / np.sqrt(sig_pow)
        return sig.astype(np.complex64)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """Generate a chunk of complex baseband samples.

        Args:
            num_samples: Number of samples per channel.

        Returns:
            np.ndarray of shape (num_samples,) for SISO or (channels, num_samples) for MIMO.
        """
        if not self._is_streaming:
            raise RuntimeError("Mock SDR receiver is not streaming. Call start_rx() first.")

        channel_data = []
        for _ in range(self.channels):
            # Synthesize PU signal
            s = self._generate_synthetic_signal(num_samples)

            # Apply fading if configured
            if self.fading_model == "rayleigh" and self.signal_type != "noise":
                h_i = self.rng.normal(0, 1.0 / np.sqrt(2))
                h_q = self.rng.normal(0, 1.0 / np.sqrt(2))
                h = h_i + 1j * h_q
                s = s * h
            elif self.fading_model == "rician" and self.signal_type != "noise":
                k_factor = 4.0  # Rician K-factor ~6 dB
                mu = np.sqrt(k_factor / (k_factor + 1.0))
                sigma = np.sqrt(1.0 / (2.0 * (k_factor + 1.0)))
                h = (mu + self.rng.normal(0, sigma)) + 1j * self.rng.normal(0, sigma)
                s = s * h

            # Generate AWGN noise with unit variance
            w = (self.rng.normal(0, 1.0 / np.sqrt(2), num_samples) +
                 1j * self.rng.normal(0, 1.0 / np.sqrt(2), num_samples))

            if self.signal_type == "noise" or self.snr_db is None:
                # Idle channel H0
                r = w
            else:
                # Active channel H1: scale signal by linear SNR
                snr_linear = 10.0 ** (self.snr_db / 10.0)
                r = np.sqrt(snr_linear) * s + w

            # Normalize to nominal receiver amplitude range [-1.0, 1.0]
            # Scaled by gain
            gain_factor = 10.0 ** ((self.gain - 30.0) / 40.0)
            rms = np.sqrt(np.mean(np.abs(r) ** 2))
            if rms > 0:
                r = (r / rms) * 0.25 * gain_factor

            # Apply ADC quantization
            r_quant = self.quantize_samples(r)
            channel_data.append(r_quant)

        self._sample_counter += num_samples

        if self.channels == 1:
            return channel_data[0]
        else:
            return np.vstack(channel_data)

    def stop_rx(self) -> None:
        """Halt mock streaming."""
        self._is_streaming = False

    def close(self) -> None:
        """Release mock resources."""
        self._is_streaming = False
