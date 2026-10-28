r"""CRN Topology Builder for Cognitive Radio Network Simulation.

Defines Primary User (PU) base stations and Secondary User (SU) nodes,
positions them in a 2D service area, and models PU ON/OFF channel occupancy.

Mathematical Notation:
    PU activity follows a two-state continuous-time Markov chain:
    $$\text{ON} \xrightarrow{\mu_{\text{off}}} \text{OFF} \xrightarrow{\mu_{\text{on}}} \text{ON}$$
    with steady-state PU utilization:
    $$\rho_{\text{PU}} = \frac{\mathbb{E}[T_{\text{on}}]}{\mathbb{E}[T_{\text{on}}] + \mathbb{E}[T_{\text{off}}]}$$
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
import numpy as np


@dataclass
class Node:
    """Generic network node with position and identity.

    Attributes:
        node_id: Unique integer identifier.
        position: (x, y) coordinates in meters.
        node_type: 'PU' for Primary User, 'SU' for Secondary User.
        tx_power_dbm: Transmit power in dBm.
    """
    node_id: int
    position: Tuple[float, float]
    node_type: str = "SU"
    tx_power_dbm: float = 20.0


@dataclass
class PrimaryUser(Node):
    """Primary User base station with ON/OFF spectrum occupancy model.

    Attributes:
        mean_on_s: Mean ON-period duration in seconds (exponential).
        mean_off_s: Mean OFF-period duration in seconds (exponential).
        is_active: Current transmission state.
    """
    mean_on_s: float = 2.0
    mean_off_s: float = 3.0
    is_active: bool = False

    def __post_init__(self) -> None:
        self.node_type = "PU"

    @property
    def utilization(self) -> float:
        r"""Steady-state channel utilization $\rho = E[T_{on}] / (E[T_{on}] + E[T_{off}])$."""
        return self.mean_on_s / (self.mean_on_s + self.mean_off_s)


@dataclass
class SecondaryUser(Node):
    """Secondary User cognitive radio node.

    Attributes:
        sensing_period_ms: How often this SU senses the channel (ms).
        has_packet: Whether SU has a packet queued for transmission.
        backoff_counter: Current MAC backoff slot counter.
        packets_sent: Total successfully transmitted packets.
        packets_dropped: Total dropped (collided or max-retries) packets.
        total_delay_s: Cumulative packet delay for averaging.
    """
    sensing_period_ms: float = 10.0
    has_packet: bool = False
    backoff_counter: int = 0
    packets_sent: int = 0
    packets_dropped: int = 0
    packets_collided: int = 0
    total_delay_s: float = 0.0

    def __post_init__(self) -> None:
        self.node_type = "SU"

    @property
    def average_delay_s(self) -> float:
        """Mean per-packet delay in seconds."""
        if self.packets_sent == 0:
            return 0.0
        return self.total_delay_s / self.packets_sent


class TopologyBuilder:
    """Constructs CRN topologies with PU base stations and SU nodes.

    Places PU and SU nodes in a bounded 2D service area and provides
    distance calculations for propagation loss modeling.
    """

    def __init__(
        self,
        area_width: float = 500.0,
        area_height: float = 500.0,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize topology builder.

        Args:
            area_width: Service area width in meters.
            area_height: Service area height in meters.
            random_seed: RNG seed for reproducible node placement.
        """
        self.area_width = area_width
        self.area_height = area_height
        self.rng = np.random.default_rng(random_seed)
        self.primary_users: List[PrimaryUser] = []
        self.secondary_users: List[SecondaryUser] = []
        self._next_id: int = 0

    def _get_next_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_primary_user(
        self,
        position: Tuple[float, float],
        tx_power_dbm: float = 30.0,
        mean_on_s: float = 2.0,
        mean_off_s: float = 3.0,
    ) -> PrimaryUser:
        """Add a Primary User base station at a fixed position.

        Args:
            position: (x, y) in meters.
            tx_power_dbm: PU transmit power in dBm.
            mean_on_s: Mean ON duration for occupancy model.
            mean_off_s: Mean OFF duration for occupancy model.

        Returns:
            The created PrimaryUser node.
        """
        pu = PrimaryUser(
            node_id=self._get_next_id(),
            position=position,
            tx_power_dbm=tx_power_dbm,
            mean_on_s=mean_on_s,
            mean_off_s=mean_off_s,
        )
        self.primary_users.append(pu)
        return pu

    def add_secondary_users(
        self,
        count: int,
        placement: Literal["random_uniform", "grid", "cluster"] = "random_uniform",
        tx_power_dbm: float = 20.0,
        sensing_period_ms: float = 10.0,
        cluster_center: Optional[Tuple[float, float]] = None,
        cluster_radius: float = 100.0,
    ) -> List[SecondaryUser]:
        """Add multiple Secondary User nodes.

        Args:
            count: Number of SU nodes to place.
            placement: Placement strategy.
            tx_power_dbm: SU transmit power.
            sensing_period_ms: Channel sensing period.
            cluster_center: Center for cluster placement.
            cluster_radius: Radius for cluster placement.

        Returns:
            List of created SecondaryUser nodes.
        """
        sus: List[SecondaryUser] = []

        if placement == "random_uniform":
            xs = self.rng.uniform(0, self.area_width, count)
            ys = self.rng.uniform(0, self.area_height, count)
            for x, y in zip(xs, ys):
                su = SecondaryUser(
                    node_id=self._get_next_id(),
                    position=(float(x), float(y)),
                    tx_power_dbm=tx_power_dbm,
                    sensing_period_ms=sensing_period_ms,
                )
                sus.append(su)

        elif placement == "grid":
            side = int(np.ceil(np.sqrt(count)))
            x_spacing = self.area_width / (side + 1)
            y_spacing = self.area_height / (side + 1)
            placed = 0
            for i in range(1, side + 1):
                for j in range(1, side + 1):
                    if placed >= count:
                        break
                    su = SecondaryUser(
                        node_id=self._get_next_id(),
                        position=(i * x_spacing, j * y_spacing),
                        tx_power_dbm=tx_power_dbm,
                        sensing_period_ms=sensing_period_ms,
                    )
                    sus.append(su)
                    placed += 1

        elif placement == "cluster":
            cx, cy = cluster_center or (self.area_width / 2, self.area_height / 2)
            for _ in range(count):
                angle = self.rng.uniform(0, 2 * np.pi)
                r = cluster_radius * np.sqrt(self.rng.uniform(0, 1))
                x = np.clip(cx + r * np.cos(angle), 0, self.area_width)
                y = np.clip(cy + r * np.sin(angle), 0, self.area_height)
                su = SecondaryUser(
                    node_id=self._get_next_id(),
                    position=(float(x), float(y)),
                    tx_power_dbm=tx_power_dbm,
                    sensing_period_ms=sensing_period_ms,
                )
                sus.append(su)

        self.secondary_users.extend(sus)
        return sus

    @property
    def all_nodes(self) -> List[Node]:
        """Return all PU and SU nodes."""
        return list(self.primary_users) + list(self.secondary_users)

    @staticmethod
    def distance(a: Node, b: Node) -> float:
        r"""Euclidean distance between two nodes in meters.

        $$d = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}$$
        """
        return float(np.sqrt(
            (a.position[0] - b.position[0]) ** 2 +
            (a.position[1] - b.position[1]) ** 2
        ))

    def get_topology_summary(self) -> str:
        """Return a human-readable summary of the topology."""
        lines = [
            f"CRN Topology: {self.area_width:.0f}m x {self.area_height:.0f}m",
            f"  Primary Users:   {len(self.primary_users)}",
        ]
        for pu in self.primary_users:
            lines.append(
                f"    PU-{pu.node_id}: pos=({pu.position[0]:.0f}, {pu.position[1]:.0f}), "
                f"power={pu.tx_power_dbm:.0f}dBm, util={pu.utilization:.1%}"
            )
        lines.append(f"  Secondary Users: {len(self.secondary_users)}")
        return "\n".join(lines)
