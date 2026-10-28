r"""Pure ALOHA MAC Protocol for Cognitive Radio Networks.

Baseline unslotted random access protocol. Secondary Users transmit immediately
upon packet arrival without carrier sensing. Collisions occur if any other transmission
starts within the vulnerable interval of duration $2 T_p$.

Mathematical Notation:
    The vulnerable period is:
    $$V = 2 T_p$$
    For Poisson arrival traffic with offered load $G = \lambda T_p$:
    $$P_{\text{success}} = e^{-2G}$$
    Normalized throughput $S$ is:
    $$S(G) = G e^{-2G}$$
    Maximum theoretical throughput is:
    $$S_{\max} = \frac{1}{2e} \approx 0.1839 \quad \text{at } G = 0.5$$
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import SecondaryUser, PrimaryUser
from src.ns3_simulation.traffic_generator import Packet


class PureAlohaMAC:
    """Pure ALOHA unslotted contention MAC protocol."""

    def __init__(
        self,
        engine: SimulationEngine,
        data_rate_bps: float = 1.0e6,
        max_retransmissions: int = 5,
        backoff_min_s: float = 0.001,
        backoff_max_s: float = 0.032,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize Pure ALOHA MAC.

        Args:
            engine: DES simulation engine.
            data_rate_bps: Channel transmission bit rate.
            max_retransmissions: Maximum retry attempts before packet drop.
            backoff_min_s: Minimum random backoff delay.
            backoff_max_s: Maximum random backoff delay.
            random_seed: RNG seed for reproducible backoffs.
        """
        self.engine = engine
        self.data_rate_bps = data_rate_bps
        self.max_retransmissions = max_retransmissions
        self.backoff_min_s = backoff_min_s
        self.backoff_max_s = backoff_max_s
        self.rng = np.random.default_rng(random_seed)

        # Active transmissions tracking: {packet_id: (su_node_id, start_time, end_time, packet)}
        self.active_transmissions: Dict[int, Tuple[int, float, float, Packet]] = {}
        # Collision flags for active packets: {packet_id: bool}
        self.packet_collided: Dict[int, bool] = {}

        # Global statistics
        self.packets_transmitted: int = 0
        self.packets_successful: int = 0
        self.packets_collided_count: int = 0
        self.packets_dropped_count: int = 0
        self.pu_collisions: int = 0

    def compute_transmission_duration(self, packet_size_bytes: int) -> float:
        """Calculate packet air-time duration in seconds."""
        return (packet_size_bytes * 8.0) / self.data_rate_bps

    def send_packet(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Attempt immediate packet transmission without carrier sensing.

        Args:
            su: Transmitting Secondary User.
            packet: Packet to send.
            primary_users: Optional list of PUs for primary collision tracking.
        """
        self.packets_transmitted += 1
        t_duration = self.compute_transmission_duration(packet.size_bytes)
        t_start = self.engine.now
        t_end = t_start + t_duration
        packet.send_time = t_start

        self.engine.log_trace(
            action="-",
            node_id=su.node_id,
            details=f"pure_aloha_tx pkt_id={packet.packet_id} dur={t_duration*1e3:.2f}ms",
        )

        has_collision = False

        # 1. Check collision with active Secondary Users
        for active_pid, (other_node, other_start, other_end, other_pkt) in list(self.active_transmissions.items()):
            # Any overlap in transmission windows corrupts both packets
            has_collision = True
            self.packet_collided[active_pid] = True

        # 2. Check collision with Primary Users (if active)
        if primary_users:
            for pu in primary_users:
                if pu.is_active:
                    has_collision = True
                    self.pu_collisions += 1

        self.active_transmissions[packet.packet_id] = (su.node_id, t_start, t_end, packet)
        self.packet_collided[packet.packet_id] = has_collision

        # Schedule end of packet transmission
        self.engine.schedule(
            delay=t_duration,
            callback=self._on_transmission_complete,
            su=su,
            packet=packet,
            primary_users=primary_users,
        )

    def _on_transmission_complete(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Handle end of transmission, ACK or collision detection."""
        if packet.packet_id in self.active_transmissions:
            del self.active_transmissions[packet.packet_id]

        collided = self.packet_collided.pop(packet.packet_id, False)

        if not collided:
            # Successful transmission
            packet.receive_time = self.engine.now
            su.packets_sent += 1
            su.total_delay_s += (packet.receive_time - packet.creation_time)
            su.has_packet = False
            self.packets_successful += 1

            self.engine.log_trace(
                action="r",
                node_id=su.node_id,
                details=f"pure_aloha_rx_ack pkt_id={packet.packet_id} delay={(packet.receive_time - packet.creation_time)*1e3:.2f}ms",
            )
        else:
            # Collision occurred
            self.packets_collided_count += 1
            su.packets_collided += 1

            if packet.retransmissions < self.max_retransmissions:
                # Exponential backoff
                packet.retransmissions += 1
                backoff_window = min(self.backoff_max_s, self.backoff_min_s * (2 ** packet.retransmissions))
                backoff_delay = float(self.rng.uniform(self.backoff_min_s, backoff_window))

                self.engine.schedule(
                    delay=backoff_delay,
                    callback=self.send_packet,
                    su=su,
                    packet=packet,
                    primary_users=primary_users,
                )
            else:
                # Max retries exceeded -> drop packet
                packet.dropped = True
                su.packets_dropped += 1
                su.has_packet = False
                self.packets_dropped_count += 1

                self.engine.log_trace(
                    action="d",
                    node_id=su.node_id,
                    details=f"pure_aloha_drop pkt_id={packet.packet_id} max_retries_reached",
                )

    @staticmethod
    def theoretical_throughput(g: float) -> float:
        r"""Compute theoretical Pure ALOHA throughput $S = G e^{-2G}$."""
        return float(g * np.exp(-2.0 * g))
