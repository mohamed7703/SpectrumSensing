r"""Discrete-Event Simulation Engine for Cognitive Radio Networks.

Provides a lightweight pure-Python discrete-event simulation (DES) core with
a priority-queue event scheduler, simulation clock, and event callback dispatch.
Produces NS-3-compatible ASCII trace output for downstream analysis.

Mathematical Notation:
    Simulation time $t \in [0, T_{\max}]$ advances discretely via the
    next-event time advance algorithm:
    $$t_{\text{next}} = \min_{e \in \mathcal{Q}} \{ e.time \}$$
    where $\mathcal{Q}$ is the pending event priority queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
import heapq


@dataclass(order=True)
class Event:
    """A scheduled simulation event.

    Attributes:
        time: Absolute simulation time in seconds when this event fires.
        priority: Tie-breaker for simultaneous events (lower = higher priority).
        event_id: Monotonically increasing unique identifier.
        callback: Function to invoke when event fires.
        kwargs: Arguments passed to the callback.
    """
    time: float
    priority: int = field(compare=True, default=0)
    event_id: int = field(compare=True, default=0)
    callback: Callable[..., None] = field(compare=False, default=None)  # type: ignore
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    cancelled: bool = field(compare=False, default=False, repr=False)


class SimulationEngine:
    """Priority-queue driven discrete-event simulation core.

    Maintains a global simulation clock and dispatches events in
    chronological order. Supports event scheduling, cancellation,
    and trace logging.
    """

    def __init__(self, max_time: float = 10.0) -> None:
        """Initialize the DES engine.

        Args:
            max_time: Maximum simulation duration in seconds.
        """
        self.max_time = max_time
        self.now: float = 0.0
        self._event_queue: List[Event] = []
        self._event_counter: int = 0
        self._trace_log: List[str] = []
        self.stats: Dict[str, Any] = {}

    def schedule(
        self,
        delay: float,
        callback: Callable[..., None],
        priority: int = 0,
        **kwargs: Any,
    ) -> Event:
        """Schedule an event to fire after a relative delay.

        Args:
            delay: Seconds from current simulation time until event fires.
            callback: Function to invoke.
            priority: Tie-breaking priority (lower fires first at same time).
            **kwargs: Additional arguments for the callback.

        Returns:
            The scheduled Event object (can be cancelled).
        """
        evt = Event(
            time=self.now + delay,
            priority=priority,
            event_id=self._event_counter,
            callback=callback,
            kwargs=kwargs,
        )
        self._event_counter += 1
        heapq.heappush(self._event_queue, evt)
        return evt

    def schedule_at(
        self,
        abs_time: float,
        callback: Callable[..., None],
        priority: int = 0,
        **kwargs: Any,
    ) -> Event:
        """Schedule an event at an absolute simulation time.

        Args:
            abs_time: Absolute simulation time in seconds.
            callback: Function to invoke.
            priority: Tie-breaking priority.
            **kwargs: Additional arguments.

        Returns:
            The scheduled Event object.
        """
        evt = Event(
            time=abs_time,
            priority=priority,
            event_id=self._event_counter,
            callback=callback,
            kwargs=kwargs,
        )
        self._event_counter += 1
        heapq.heappush(self._event_queue, evt)
        return evt

    def cancel(self, event: Event) -> None:
        """Cancel a previously scheduled event (lazy deletion)."""
        event.cancelled = True

    def run(self) -> None:
        """Execute the simulation until event queue is exhausted or max_time reached."""
        while self._event_queue:
            evt = heapq.heappop(self._event_queue)
            if evt.cancelled:
                continue
            if evt.time > self.max_time:
                break
            self.now = evt.time
            evt.callback(**evt.kwargs)

    def log_trace(self, action: str, node_id: int, details: str) -> None:
        """Append an NS-3-style ASCII trace entry.

        Format: ``<action> <time> <node_id> <details>``

        Args:
            action: Trace action character ('+' enqueue, '-' dequeue, 'r' receive, 'd' drop).
            node_id: Source or destination node identifier.
            details: Packet or event description string.
        """
        entry = f"{action} {self.now:.9f} /NodeList/{node_id} {details}"
        self._trace_log.append(entry)

    def get_trace(self) -> List[str]:
        """Return the full NS-3-compatible ASCII trace log."""
        return self._trace_log

    def reset(self) -> None:
        """Reset simulation state for a new run."""
        self.now = 0.0
        self._event_queue.clear()
        self._event_counter = 0
        self._trace_log.clear()
        self.stats.clear()
