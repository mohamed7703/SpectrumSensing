r"""Slotted ALOHA MAC Protocol for Cognitive Radio Networks.

Synchronized contention protocol where time is divided into discrete slots
of duration $T_{\text{slot}}$. Packets arriving in slot $k$ defer transmission
until the boundary of slot $k+1$. Vulnerable interval is halved to $T_{\text{slot}}$.

Mathematical Notation:
    The vulnerable period is:
    $$V = T_{\text{slot}}$$
    For Poisson arrival traffic with offered load $G = \lambda T_{\text{slot}}$:
    $$P_{\text{success}} = e^{-G}$$
    Normalized throughput $S$ is:
    $$S(G) = G e^{-G}$$
    Maximum theoretical throughput is:
    $$S_{\max} = \frac{1}{e} \approx 0.3679 \quad \text{at } G = 1.0$$
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import SecondaryUser, PrimaryUser
from src.ns3_simulation.traffic_generator import Packet
from src.ns3_simulation.transmission_tracker import TransmissionTracker


class SlottedAlohaMAC:
    """Slotted ALOHA synchronized contention MAC protocol."""

    def __init__(
        self,
        engine: SimulationEngine,
        slot_duration_s: float = 0.008192,  # Matches 1024 bytes @ 1 Mbps
        data_rate_bps: float = 1.0e6,
        max_retransmissions: int = 5,
        backoff_min_slots: int = 1,
        backoff_max_slots: int = 16,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize Slotted ALOHA MAC.

        Args:
            engine: DES simulation engine.
            slot_duration_s: Synchronized time slot duration in seconds.
            data_rate_bps: Channel transmission bit rate.
            max_retransmissions: Maximum retry attempts before packet drop.
            backoff_min_slots: Minimum backoff duration in integer slots.
            backoff_max_slots: Maximum backoff duration in integer slots.
            random_seed: RNG seed for reproducible backoffs.
        """
        self.engine = engine
        self.slot_duration_s = slot_duration_s
        self.data_rate_bps = data_rate_bps
        self.max_retransmissions = max_retransmissions
        self.backoff_min_slots = backoff_min_slots
        self.backoff_max_slots = backoff_max_slots
        self.rng = np.random.default_rng(random_seed)

        # Slot occupancy map: {slot_index: [(su_node, packet)]}
        self.slot_reservations: Dict[int, List[Tuple[SecondaryUser, Packet]]] = {}
        self._primary_users_by_packet: Dict[int, List[PrimaryUser]] = {}
        self.transmission_tracker = TransmissionTracker(engine)

        # Global statistics
        self.packets_transmitted: int = 0
        self.packets_successful: int = 0
        self.packets_collided_count: int = 0
        self.packets_dropped_count: int = 0
        self.pu_collisions: int = 0

    def get_next_slot_boundary(self, current_time: float) -> Tuple[int, float]:
        """Compute the next synchronized slot index and absolute boundary time.

        Args:
            current_time: Current simulation time.

        Returns:
            Tuple of (slot_index, slot_start_time).
        """
        slot_idx = int(np.floor(current_time / self.slot_duration_s)) + 1
        slot_start_time = slot_idx * self.slot_duration_s
        return slot_idx, slot_start_time

    def queue_packet(
        self,
        su: SecondaryUser,
        packet: Packet,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Schedule packet for transmission at the beginning of the next slot.

        Args:
            su: Transmitting Secondary User.
            packet: Packet to send.
            primary_users: Optional list of PUs for primary collision tracking.
        """
        if packet.size_bytes * 8 / self.data_rate_bps > self.slot_duration_s:
            raise ValueError("packet airtime exceeds the configured slot duration")
        self._primary_users_by_packet[packet.packet_id] = list(primary_users or [])
        slot_idx, slot_time = self.get_next_slot_boundary(self.engine.now)

        if slot_idx not in self.slot_reservations:
            self.slot_reservations[slot_idx] = []
            # Schedule slot execution event
            delay_to_slot = slot_time - self.engine.now
            self.engine.schedule(
                delay=delay_to_slot,
                callback=self._execute_slot,
                slot_index=slot_idx,
                primary_users=primary_users,
            )

        self.slot_reservations[slot_idx].append((su, packet))

    def _execute_slot(
        self,
        slot_index: int,
        primary_users: Optional[List[PrimaryUser]] = None,
    ) -> None:
        """Process all packets synchronized to transmit in this time slot."""
        transmissions = self.slot_reservations.pop(slot_index, [])
        num_transmissions = len(transmissions)
        if num_transmissions == 0:
            return

        self.packets_transmitted += num_transmissions

        # Each attempt occupies the full slot, including padding/guard time.
        # Success or retry is decided only at completion, using interval overlap.
        for su, packet in transmissions:
            packet.send_time = self.engine.now
            pus = self._primary_users_by_packet[packet.packet_id]
            self.transmission_tracker.start(packet.packet_id, self.slot_duration_s, pus)
            self.engine.log_trace("-", su.node_id, f"slotted_aloha_tx pkt_id={packet.packet_id} size={packet.size_bytes}B slot={slot_index}")
            self.engine.schedule(self.slot_duration_s, self._finish_packet,
                                 su=su, packet=packet, slot_index=slot_index)

    def _finish_packet(self, su: SecondaryUser, packet: Packet, slot_index: int) -> None:
        """Resolve one specified packet at the end of its reserved slot."""
        attempt = self.transmission_tracker.finish(packet.packet_id)
        self.pu_collisions += int(attempt.pu_overlap_s > 0)
        if not attempt.collided:
            packet.receive_time = self.engine.now
            su.packets_sent += 1
            su.total_delay_s += (packet.receive_time - packet.creation_time)
            su.has_packet = False
            self.packets_successful += 1
            self._primary_users_by_packet.pop(packet.packet_id, None)

            self.engine.log_trace(
                action="r",
                node_id=su.node_id,
                details=f"slotted_aloha_ack pkt_id={packet.packet_id} slot={slot_index}",
            )
        else:
            self.packets_collided_count += 1
            su.packets_collided += 1
            if packet.retransmissions < self.max_retransmissions:
                packet.retransmissions += 1
                max_win = min(self.backoff_max_slots, self.backoff_min_slots * (2 ** packet.retransmissions))
                backoff_slots = int(self.rng.integers(self.backoff_min_slots, max_win + 1))
                target_slot = slot_index + 1 + backoff_slots
                target_time = target_slot * self.slot_duration_s
                if target_slot not in self.slot_reservations:
                    self.slot_reservations[target_slot] = []
                    self.engine.schedule_at(target_time, self._execute_slot, slot_index=target_slot)
                self.slot_reservations[target_slot].append((su, packet))
            else:
                packet.dropped = True
                su.packets_dropped += 1
                su.has_packet = False
                self.packets_dropped_count += 1
                self._primary_users_by_packet.pop(packet.packet_id, None)
                self.engine.log_trace("d", su.node_id, f"slotted_aloha_drop pkt_id={packet.packet_id} max_retries_reached")

    @staticmethod
    def theoretical_throughput(g: float) -> float:
        r"""Compute theoretical Slotted ALOHA throughput $S = G e^{-G}$."""
        return float(g * np.exp(-g))
