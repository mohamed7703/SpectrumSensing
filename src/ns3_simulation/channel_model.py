r"""Wireless Channel Propagation Model for CRN Simulation.

Implements path loss, shadow fading, and small-scale fading models for
computing received signal strength between any pair of network nodes.

Mathematical Notation:
    **Log-Distance Path Loss Model:**
    $$PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right) + X_{\sigma}$$
    where $n$ is the path-loss exponent, $d_0$ is the reference distance,
    and $X_{\sigma} \sim \mathcal{N}(0, \sigma^2)$ is log-normal shadow fading.

    **Received Power:**
    $$P_r(d) = P_t - PL(d) + G_t + G_r \quad [\text{dBm}]$$

    **Thermal Noise Floor:**
    $$N_0 = -174 + 10\log_{10}(B) + NF \quad [\text{dBm}]$$
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np

from src.ns3_simulation.topology_builder import Node


class ChannelModel:
    """Wireless propagation channel with path loss, shadowing, and fading.

    Computes the received signal strength indicator (RSSI) and instantaneous
    SNR between any two nodes in the CRN topology.
    """

    def __init__(
        self,
        propagation_model: Literal["free_space", "log_distance", "two_ray_ground"] = "log_distance",
        path_loss_exponent: float = 3.5,
        reference_distance_m: float = 1.0,
        reference_loss_db: float = 46.67,
        shadowing_std_db: float = 8.0,
        fading_model: Literal["none", "rayleigh", "rician"] = "rayleigh",
        rician_k_factor: float = 4.0,
        noise_figure_db: float = 7.0,
        thermal_noise_dbm_per_hz: float = -174.0,
        bandwidth_hz: float = 2.4e6,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize channel model.

        Args:
            propagation_model: Large-scale path loss model type.
            path_loss_exponent: Exponent $n$ for log-distance model.
            reference_distance_m: Reference distance $d_0$ in meters.
            reference_loss_db: Path loss at reference distance $PL(d_0)$ in dB.
            shadowing_std_db: Log-normal shadow fading std $\\sigma$ in dB.
            fading_model: Small-scale fading type.
            rician_k_factor: Rician K-factor (only for rician model).
            noise_figure_db: Receiver noise figure in dB.
            thermal_noise_dbm_per_hz: Thermal noise density in dBm/Hz.
            bandwidth_hz: Channel bandwidth in Hz.
            random_seed: RNG seed for reproducibility.
        """
        self.propagation_model = propagation_model
        self.path_loss_exponent = path_loss_exponent
        self.reference_distance_m = reference_distance_m
        self.reference_loss_db = reference_loss_db
        self.shadowing_std_db = shadowing_std_db
        self.fading_model = fading_model
        self.rician_k_factor = rician_k_factor
        self.noise_figure_db = noise_figure_db
        self.thermal_noise_dbm_per_hz = thermal_noise_dbm_per_hz
        self.bandwidth_hz = bandwidth_hz
        self.rng = np.random.default_rng(random_seed)

        # Precompute noise floor
        self.noise_floor_dbm = (
            thermal_noise_dbm_per_hz
            + 10.0 * np.log10(bandwidth_hz)
            + noise_figure_db
        )

    def compute_path_loss(self, distance_m: float) -> float:
        r"""Compute large-scale path loss in dB.

        Args:
            distance_m: Distance between TX and RX in meters.

        Returns:
            Path loss in dB (positive value).
        """
        d = max(distance_m, self.reference_distance_m)

        if self.propagation_model == "free_space":
            # Friis free space: PL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
            pl = self.reference_loss_db + 20.0 * np.log10(d / self.reference_distance_m)

        elif self.propagation_model == "log_distance":
            pl = self.reference_loss_db + 10.0 * self.path_loss_exponent * np.log10(
                d / self.reference_distance_m
            )

        elif self.propagation_model == "two_ray_ground":
            # Simplified two-ray: exponent transitions from 2 to 4
            crossover = 50.0  # meters
            if d < crossover:
                pl = self.reference_loss_db + 20.0 * np.log10(d / self.reference_distance_m)
            else:
                pl_cross = self.reference_loss_db + 20.0 * np.log10(crossover / self.reference_distance_m)
                pl = pl_cross + 40.0 * np.log10(d / crossover)
        else:
            raise ValueError(f"Unknown propagation model: {self.propagation_model}")

        return float(pl)

    def compute_shadow_fading(self) -> float:
        """Sample log-normal shadow fading in dB."""
        if self.shadowing_std_db <= 0:
            return 0.0
        return float(self.rng.normal(0.0, self.shadowing_std_db))

    def compute_small_scale_fading_db(self) -> float:
        """Sample small-scale fading gain in dB.

        Returns:
            Fading gain in dB (typically negative, representing power loss).
        """
        if self.fading_model == "none":
            return 0.0

        elif self.fading_model == "rayleigh":
            # |h|^2 ~ Exponential(1) for Rayleigh
            h_sq = self.rng.exponential(1.0)
            return float(10.0 * np.log10(max(h_sq, 1e-20)))

        elif self.fading_model == "rician":
            k = self.rician_k_factor
            sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
            mu = np.sqrt(k / (k + 1.0))
            h_re = mu + self.rng.normal(0.0, sigma)
            h_im = self.rng.normal(0.0, sigma)
            h_sq = h_re ** 2 + h_im ** 2
            return float(10.0 * np.log10(max(h_sq, 1e-20)))

        else:
            raise ValueError(f"Unknown fading model: {self.fading_model}")

    def compute_rssi(
        self,
        tx_node: Node,
        rx_node: Node,
        include_shadowing: bool = True,
        include_fading: bool = True,
    ) -> float:
        """Compute received signal strength indicator (RSSI) in dBm.

        Args:
            tx_node: Transmitting node.
            rx_node: Receiving node.
            include_shadowing: Include log-normal shadow fading.
            include_fading: Include small-scale multipath fading.

        Returns:
            Received power in dBm.
        """
        from src.ns3_simulation.topology_builder import TopologyBuilder
        d = TopologyBuilder.distance(tx_node, rx_node)

        pl = self.compute_path_loss(d)
        shadow = self.compute_shadow_fading() if include_shadowing else 0.0
        fading = self.compute_small_scale_fading_db() if include_fading else 0.0

        rssi = tx_node.tx_power_dbm - pl - shadow + fading
        return float(rssi)

    def compute_snr(
        self,
        tx_node: Node,
        rx_node: Node,
        include_shadowing: bool = True,
        include_fading: bool = True,
    ) -> float:
        """Compute received Signal-to-Noise Ratio in dB.

        Args:
            tx_node: Transmitting node.
            rx_node: Receiving node.
            include_shadowing: Include log-normal shadow fading.
            include_fading: Include small-scale multipath fading.

        Returns:
            SNR in dB.
        """
        rssi = self.compute_rssi(tx_node, rx_node, include_shadowing, include_fading)
        return float(rssi - self.noise_floor_dbm)

    def is_signal_detectable(
        self,
        tx_node: Node,
        rx_node: Node,
        detection_threshold_dbm: float = -90.0,
    ) -> bool:
        """Check whether signal from tx_node can be detected at rx_node.

        Args:
            tx_node: Transmitting node.
            rx_node: Receiving node.
            detection_threshold_dbm: Minimum RSSI for detection.

        Returns:
            True if RSSI exceeds detection threshold.
        """
        rssi = self.compute_rssi(tx_node, rx_node)
        return rssi >= detection_threshold_dbm
