"""Temporal overlap ledger for the simplified single-channel collision model.

Any positive overlap corrupts a packet. This is an explicit conservative temporal
model, not a SINR/capture model or evidence of interference at a PU receiver.
PU state changes must be scheduled through the simulation event engine.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import PrimaryUser


@dataclass
class TransmissionAttempt:
    """One half-open on-air interval and its accumulated interference durations."""

    packet_id: int
    start: float
    end: float
    primary_users: List[PrimaryUser]
    pu_overlap_s: float = 0.0
    su_overlap_s: float = 0.0

    @property
    def collided(self) -> bool:
        """Return whether positive-duration primary or secondary overlap occurred."""
        return self.pu_overlap_s > 0 or self.su_overlap_s > 0


class TransmissionTracker:
    """Integrate active attempts before each scheduled state mutation."""

    def __init__(self, engine: SimulationEngine) -> None:
        """Attach a tracker to the supplied engine for one protocol's medium."""
        self.engine = engine
        self.active: Dict[int, TransmissionAttempt] = {}
        self.completed: List[TransmissionAttempt] = []
        self.last_time = engine.now
        engine.add_time_observer(self.advance)

    def advance(self, time: float) -> None:
        """Accumulate overlap between the preceding and supplied event times."""
        attempts = list(self.active.values())
        for attempt in attempts:
            left = max(self.last_time, attempt.start)
            right = min(time, attempt.end)
            if right <= left:
                continue
            if any(pu.is_active for pu in attempt.primary_users):
                attempt.pu_overlap_s += right - left
            # All active attempts began no later than last_time. Thus any
            # overlapping peers cover a common prefix of this interval.
            peer_end = max((other.end for other in attempts if other is not attempt), default=left)
            attempt.su_overlap_s += max(0.0, min(right, peer_end) - left)
        self.last_time = time

    def start(self, packet_id: int, duration: float, primary_users: List[PrimaryUser]) -> None:
        """Record a new attempt with specified packet ID, duration and PU objects."""
        if duration <= 0:
            raise ValueError("transmission duration must be positive")
        if packet_id in self.active:
            raise ValueError("packet already has an active transmission")
        self.active[packet_id] = TransmissionAttempt(
            packet_id, self.engine.now, self.engine.now + duration, list(primary_users)
        )

    def finish(self, packet_id: int) -> TransmissionAttempt:
        """Remove and return the completed attempt for the specified packet ID."""
        attempt = self.active.pop(packet_id)
        self.completed.append(attempt)
        return attempt
