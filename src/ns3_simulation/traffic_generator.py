r"""Traffic Generator for CRN Discrete-Event Simulation.

Generates deterministic, Poisson, and bursty packet arrivals for Secondary User
nodes. Each packet carries metadata for end-to-end delay tracking.

Mathematical Notation:
    **Poisson Arrivals:**
    Inter-arrival times are i.i.d. exponential:
    $$\tau_i \sim \text{Exp}(\lambda), \quad \lambda = \text{rate (packets/s)}$$

    **Offered Load:**
    For $N$ SUs each generating at rate $\lambda$ with packet duration $T_p$:
    $$G = N \lambda T_p$$
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, List
import numpy as np

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import SecondaryUser


@dataclass
class Packet:
    """Network packet with metadata for delay and delivery tracking.

    Attributes:
        packet_id: Unique packet identifier.
        source_id: Originating SU node_id.
        destination_id: Destination node_id (or -1 for broadcast).
        size_bytes: Packet payload size in bytes.
        creation_time: Simulation time when packet was created.
        send_time: Simulation time when transmission started (or None).
        receive_time: Simulation time when packet was received (or None).
        retransmissions: Number of retransmission attempts.
        dropped: Whether the packet was permanently dropped.
    """
    packet_id: int
    source_id: int
    destination_id: int = -1
    size_bytes: int = 1024
    creation_time: float = 0.0
    send_time: Optional[float] = None
    receive_time: Optional[float] = None
    retransmissions: int = 0
    dropped: bool = False

    @property
    def delay(self) -> Optional[float]:
        """End-to-end delay in seconds, or None if not yet received."""
        if self.receive_time is not None and self.send_time is not None:
            return self.receive_time - self.creation_time
        return None


class TrafficGenerator:
    """Packet traffic source for Secondary User nodes.

    Schedules packet arrivals according to deterministic, Poisson,
    or bursty traffic models, and enqueues them onto the SU's transmit queue.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        traffic_model: Literal["deterministic", "poisson", "bursty"] = "deterministic",
        packet_size_bytes: int = 1024,
        inter_packet_interval_s: float = 0.01,
        poisson_rate_pps: float = 100.0,
        num_packets: int = 1000,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize traffic generator.

        Args:
            engine: Reference to the DES simulation engine.
            traffic_model: Arrival process type.
            packet_size_bytes: Payload size per packet.
            inter_packet_interval_s: Fixed interval for deterministic traffic.
            poisson_rate_pps: Mean arrival rate for Poisson traffic (packets/s).
            num_packets: Total packets to generate per SU.
            random_seed: RNG seed.
        """
        self.engine = engine
        self.traffic_model = traffic_model
        self.packet_size_bytes = packet_size_bytes
        self.inter_packet_interval_s = inter_packet_interval_s
        self.poisson_rate_pps = poisson_rate_pps
        self.num_packets = num_packets
        self.rng = np.random.default_rng(random_seed)
        self._packet_counter: int = 0
        self.generated_packets: List[Packet] = []

    def _next_packet_id(self) -> int:
        pid = self._packet_counter
        self._packet_counter += 1
        return pid

    def start(self, su: SecondaryUser) -> None:
        """Begin generating traffic for a given SU node.

        Args:
            su: The SecondaryUser node to generate traffic for.
        """
        self._schedule_next(su, packets_remaining=self.num_packets)

    def _schedule_next(self, su: SecondaryUser, packets_remaining: int) -> None:
        """Schedule the next packet arrival event."""
        if packets_remaining <= 0:
            return

        if self.traffic_model == "deterministic":
            delay = self.inter_packet_interval_s
        elif self.traffic_model == "poisson":
            delay = float(self.rng.exponential(1.0 / self.poisson_rate_pps))
        elif self.traffic_model == "bursty":
            # Two-state: burst (fast) or idle (slow)
            if self.rng.random() < 0.3:
                delay = float(self.rng.exponential(self.inter_packet_interval_s * 0.1))
            else:
                delay = float(self.rng.exponential(self.inter_packet_interval_s * 3.0))
        else:
            raise ValueError(f"Unknown traffic model: {self.traffic_model}")

        self.engine.schedule(
            delay=delay,
            callback=self._on_packet_arrival,
            su=su,
            packets_remaining=packets_remaining,
        )

    def _on_packet_arrival(self, su: SecondaryUser, packets_remaining: int) -> None:
        """Event callback: new packet arrives at SU transmit queue."""
        pkt = Packet(
            packet_id=self._next_packet_id(),
            source_id=su.node_id,
            size_bytes=self.packet_size_bytes,
            creation_time=self.engine.now,
        )
        self.generated_packets.append(pkt)
        su.has_packet = True

        # Log enqueue event in NS-3 format
        self.engine.log_trace(
            action="+",
            node_id=su.node_id,
            details=f"pkt_id={pkt.packet_id} size={pkt.size_bytes}B",
        )

        # Schedule next arrival
        self._schedule_next(su, packets_remaining - 1)

    def compute_offered_load(
        self, num_sus: int, packet_duration_s: float
    ) -> float:
        r"""Compute theoretical offered load $G = N \lambda T_p$.

        Args:
            num_sus: Number of active Secondary Users.
            packet_duration_s: Duration of one packet transmission.

        Returns:
            Offered load (dimensionless).
        """
        if self.traffic_model == "deterministic":
            lam = 1.0 / self.inter_packet_interval_s
        elif self.traffic_model == "poisson":
            lam = self.poisson_rate_pps
        else:
            lam = 1.0 / self.inter_packet_interval_s

        return num_sus * lam * packet_duration_s
