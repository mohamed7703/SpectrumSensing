r"""Interference-Limited Adaptive CSMA (ILA-CSMA) MAC Protocol.

Implements two-stage hybrid spectrum sensing for Cognitive Radio Networks:
- Stage 1: Channel Activity Detection (CAD) during DIFS preamble.
- Stage 2: Deep RSSI / Energy Detection against adaptive threshold $\gamma_{\text{th}}$.
- CSMA-HS (Interference Screening): Immediate deferral upon PU activity detection.
- CSMA-AB (Adaptive Backoff): Binary exponential backoff with density adaptation.

Mathematical Notation:
    The two-stage decision rule is:
    $$\text{Sense}(t) = \begin{cases}
        \text{Busy}, & \text{if } \text{CAD}(t) \ge \tau_{\text{CAD}} \text{ or } \text{RSSI}(t) \ge \gamma_{\text{th}} \\
        \text{Idle}, & \text{otherwise}
    \end{cases}$$
    Contention window adapts via binary exponential backoff:
    $$W_k = \min\left(W_{\max}, W_{\min} \cdot 2^k\right), \quad B_k \sim \mathcal{U}[0, W_k]$$
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import SecondaryUser, PrimaryUser
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.traffic_generator import Packet


class ILaCsmaMAC:
    """Interference-Limited Adaptive CSMA protocol with two-stage hybrid sensing."""

    def __init__(
        self,
        engine: SimulationEngine,
        channel: ChannelModel,
        data_rate_bps: float = 1.0e6,
        cad_threshold_dbm: float = -90.0,
        rssi_threshold_dbm: float = -85.0,
        difs_s: float = 0.000050,  # 50 µs DCF Inter-Frame Space
        sifs_s: float = 0.000010,  # 10 µs Short Inter-Frame Space
        slot_duration_s: float = 0.000020,  # 20 µs backoff slot
        cw_min_slots: int = 15,
        cw_max_slots: int = 1023,
        max_retransmissions: int = 7,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize ILA-CSMA MAC protocol.

        Args:
            engine: DES simulation engine.
            channel: Wireless propagation channel model for RSSI calculation.
            data_rate_bps: Transmission bit rate.
            cad_threshold_dbm: Stage 1 Channel Activity Detection threshold.
            rssi_threshold_dbm: Stage 2 Deep RSSI detection threshold.
            difs_s: DIFS duration for carrier sensing.
            sifs_s: SIFS inter-frame space.
            slot_duration_s: Duration of one backoff slot.
            cw_min_slots: Minimum contention window size.
            cw_max_slots: Maximum contention window size.
            max_retransmissions: Retry limit before packet drop.
            random_seed: RNG seed for reproducible backoffs.
        """
        self.engine = engine
        self.channel = channel
        self.data_rate_bps = data_rate_bps
        self.cad_threshold_dbm = cad_threshold_dbm
        self.rssi_threshold_dbm = rssi_threshold_dbm
        self.difs_s = difs_s
        self.sifs_s = sifs_s
        self.slot_duration_s = slot_duration_s
        self.cw_min_slots = cw_min_slots
        self.cw_max_slots = cw_max_slots
        self.max_retransmissions = max_retransmissions
        self.rng = np.random.default_rng(random_seed)

        # Active transmissions on the channel: {su_id: (packet, end_time)}
        self.active_transmissions: Dict[int, Tuple[Packet, float]] = {}

        # Global statistics
        self.packets_transmitted: int = 0
        self.packets_successful: int = 0
        self.packets_collided_count: int = 0
        self.packets_dropped_count: int = 0
        self.pu_collisions: int = 0
        self.carrier_senses_performed: int = 0
        self.carrier_sense_busies: int = 0

    def compute_transmission_duration(self, packet_size_bytes: int) -> float:
        """Air-time transmission duration."""
        return (packet_size_bytes * 8.0) / self.data_rate_bps

    def sense_channel(
        self,
        su: SecondaryUser,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> bool:
        """Perform two-stage hybrid spectrum sensing (CAD + RSSI).

        Returns:
            True if channel is BUSY (occupied by PU or other SUs), False if IDLE.
        """
        self.carrier_senses_performed += 1

        # Check ongoing SU transmissions
        if len(self.active_transmissions) > 0:
            self.carrier_sense_busies += 1
            return True  # Busy due to other SU transmission

        # Check Primary User activity via Channel Model
        if primary_users:
            for pu in primary_users:
                if pu.is_active:
                    # Stage 1: CAD Fast Preamble Check
                    rssi = self.channel.compute_rssi(pu, su, include_shadowing=False, include_fading=False)
                    if rssi >= self.cad_threshold_dbm:
                        self.carrier_sense_busies += 1
                        return True

                    # Stage 2: Deep RSSI against threshold
                    if rssi >= self.rssi_threshold_dbm:
                        self.carrier_sense_busies += 1
                        return True

        return False  # Channel is clear

    def queue_packet(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Initiate ILA-CSMA carrier sensing and transmission procedure."""
        # Wait for DIFS sensing duration
        self.engine.schedule(
            delay=self.difs_s,
            callback=self._on_difs_complete,
            su=su,
            packet=packet,
            primary_users=primary_users,
        )

    def _on_difs_complete(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Evaluate channel state at conclusion of DIFS observation window."""
        is_busy = self.sense_channel(su, primary_users)

        if not is_busy:
            # Channel is idle -> transmit packet
            self._start_transmission(su, packet, primary_users)
        else:
            # Channel busy -> CSMA-HS Deferral and CSMA-AB Adaptive Backoff
            self._backoff_and_retry(su, packet, primary_users)

    def _start_transmission(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Start packet transmission on the channel."""
        self.packets_transmitted += 1
        t_duration = self.compute_transmission_duration(packet.size_bytes)
        t_start = self.engine.now
        t_end = t_start + t_duration
        packet.send_time = t_start

        self.engine.log_trace(
            action="-",
            node_id=su.node_id,
            details=f"ila_csma_tx pkt_id={packet.packet_id} dur={t_duration*1e3:.2f}ms",
        )

        # Detect collision if another transmission is already in the air
        is_collided = len(self.active_transmissions) > 0

        # Check collision with PU
        if primary_users:
            for pu in primary_users:
                if pu.is_active:
                    is_collided = True
                    self.pu_collisions += 1
                    break

        self.active_transmissions[su.node_id] = (packet, t_end)

        self.engine.schedule(
            delay=t_duration,
            callback=self._on_transmission_finish,
            su=su,
            packet=packet,
            is_collided=is_collided,
            primary_users=primary_users,
        )

    def _on_transmission_finish(
        self,
        su: SecondaryUser,
        packet: Packet,
        is_collided: bool,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Handle end of transmission and ACK timeout."""
        if su.node_id in self.active_transmissions:
            del self.active_transmissions[su.node_id]

        if not is_collided:
            # Successful reception
            packet.receive_time = self.engine.now
            su.packets_sent += 1
            su.total_delay_s += (packet.receive_time - packet.creation_time)
            su.has_packet = False
            self.packets_successful += 1

            self.engine.log_trace(
                action="r",
                node_id=su.node_id,
                details=f"ila_csma_ack pkt_id={packet.packet_id}",
            )
        else:
            # Collision occurred -> backoff and retry
            self.packets_collided_count += 1
            su.packets_collided += 1
            self._backoff_and_retry(su, packet, primary_users)

    def _backoff_and_retry(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Execute binary exponential backoff with slot adjustment."""
        if packet.retransmissions < self.max_retransmissions:
            packet.retransmissions += 1
            cw = min(self.cw_max_slots, self.cw_min_slots * (2 ** packet.retransmissions))
            backoff_slots = int(self.rng.integers(0, cw + 1))
            backoff_time = backoff_slots * self.slot_duration_s

            self.engine.schedule(
                delay=backoff_time,
                callback=self.queue_packet,
                su=su,
                packet=packet,
                primary_users=primary_users,
            )
        else:
            # Retries exhausted -> drop packet
            packet.dropped = True
            su.packets_dropped += 1
            su.has_packet = False
            self.packets_dropped_count += 1

            self.engine.log_trace(
                action="d",
                node_id=su.node_id,
                details=f"ila_csma_drop pkt_id={packet.packet_id} max_retries_reached",
            )
