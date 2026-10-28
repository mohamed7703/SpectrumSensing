r"""NS-3 ASCII Event Trace Parser and End-to-End Delay Analysis Pipeline.

Parses discrete-event network simulation traces, extracts per-packet timestamps
(enqueue, transmit, receive, drop), computes Packet Delivery Ratio (PDR),
calculates end-to-end latency distributions, and benchmarks MAC protocols.

Mathematical Notation:
    The End-to-End (E2E) delay for packet $i$ is:
    $$D_i = T_{\text{rx}, i} - T_{\text{enq}, i}$$
    Packet Delivery Ratio (PDR) is:
    $$\text{PDR} = \frac{N_{\text{received}}}{N_{\text{generated}}} \times 100\%$$
    Delay jitter (mean absolute packet delay variation) is:
    $$J = \frac{1}{N-1} \sum_{i=1}^{N-1} |D_{i+1} - D_i|$$
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import re
import numpy as np


@dataclass
class PacketRecord:
    """Detailed record for a tracked packet across its transmission lifecycle."""
    packet_id: int
    source_node: int
    size_bytes: int = 1024
    enqueue_time: Optional[float] = None
    tx_time: Optional[float] = None
    rx_time: Optional[float] = None
    dropped: bool = False
    drop_reason: Optional[str] = None
    drop_time: Optional[float] = None

    @property
    def is_delivered(self) -> bool:
        """True if packet successfully reached receiver and was ACKed."""
        return self.rx_time is not None and not self.dropped

    @property
    def e2e_delay_s(self) -> Optional[float]:
        """Total end-to-end delay from enqueue to reception."""
        if self.rx_time is not None and self.enqueue_time is not None:
            return self.rx_time - self.enqueue_time
        return None

    @property
    def channel_access_delay_s(self) -> Optional[float]:
        """MAC queuing and contention delay before physical transmission."""
        if self.tx_time is not None and self.enqueue_time is not None:
            return self.tx_time - self.enqueue_time
        return None


class TraceParser:
    """Parser for NS-3 and DES ASCII simulation trace logs."""

    # Regex patterns for NS-3 trace format: <action> <time> /NodeList/<id> <details>
    _LINE_REGEX = re.compile(
        r"^([+\-rd])\s+([\d\.]+)\s+/NodeList/(\d+)\s+(.*)$"
    )
    _PKT_ID_REGEX = re.compile(r"pkt_id=(\d+)")
    _SIZE_REGEX = re.compile(r"size=(\d+)B?")

    def __init__(self) -> None:
        """Initialize trace parser."""
        self.packets: Dict[int, PacketRecord] = {}
        self.raw_events: List[Dict[str, Any]] = []

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single NS-3 ASCII trace line."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        m = self._LINE_REGEX.match(line)
        if not m:
            return None

        action, timestamp_str, node_id_str, details = m.groups()
        timestamp = float(timestamp_str)
        node_id = int(node_id_str)

        pkt_m = self._PKT_ID_REGEX.search(details)
        if not pkt_m:
            return None
        packet_id = int(pkt_m.group(1))

        size = 1024
        size_m = self._SIZE_REGEX.search(details)
        if size_m:
            size = int(size_m.group(1))

        if packet_id not in self.packets:
            self.packets[packet_id] = PacketRecord(
                packet_id=packet_id,
                source_node=node_id,
                size_bytes=size,
            )

        rec = self.packets[packet_id]

        if action == "+":
            # Packet enqueued at source
            rec.enqueue_time = timestamp
        elif action == "-":
            # Packet dequeued / transmission started
            rec.tx_time = timestamp
        elif action == "r":
            # Packet received / ACKed
            rec.rx_time = timestamp
        elif action == "d":
            # Packet dropped
            rec.dropped = True
            rec.drop_time = timestamp
            rec.drop_reason = details

        event_data = {
            "action": action,
            "time": timestamp,
            "node_id": node_id,
            "packet_id": packet_id,
            "details": details,
        }
        self.raw_events.append(event_data)
        return event_data

    def parse_lines(self, lines: List[str]) -> None:
        """Parse multiple trace log lines."""
        for line in lines:
            self.parse_line(line)

    def parse_file(self, filepath: Union[str, Path]) -> None:
        """Read and parse an entire NS-3 trace file (.tr)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.parse_line(line)

    def compute_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive end-to-end network performance metrics.

        Returns:
            Dictionary with PDR, delay statistics, jitter, and throughput.
        """
        total_packets = len(self.packets)
        if total_packets == 0:
            return {
                "total_packets": 0,
                "delivered_packets": 0,
                "dropped_packets": 0,
                "pdr_percent": 0.0,
                "drop_rate_percent": 0.0,
            }

        delivered = [p for p in self.packets.values() if p.is_delivered]
        dropped = [p for p in self.packets.values() if p.dropped]

        pdr = (len(delivered) / total_packets) * 100.0
        drop_rate = (len(dropped) / total_packets) * 100.0

        # End-to-end delays in milliseconds
        e2e_delays_ms = np.array([p.e2e_delay_s * 1000.0 for p in delivered if p.e2e_delay_s is not None])

        if len(e2e_delays_ms) > 0:
            mean_delay = float(np.mean(e2e_delays_ms))
            median_delay = float(np.median(e2e_delays_ms))
            p95_delay = float(np.percentile(e2e_delays_ms, 95))
            p99_delay = float(np.percentile(e2e_delays_ms, 99))
            min_delay = float(np.min(e2e_delays_ms))
            max_delay = float(np.max(e2e_delays_ms))
            std_delay = float(np.std(e2e_delays_ms))

            # Jitter: mean |D_{i+1} - D_i|
            if len(e2e_delays_ms) > 1:
                jitter_ms = float(np.mean(np.abs(np.diff(e2e_delays_ms))))
            else:
                jitter_ms = 0.0
        else:
            mean_delay = median_delay = p95_delay = p99_delay = min_delay = max_delay = std_delay = jitter_ms = 0.0

        # Simulation duration
        if self.raw_events:
            sim_duration_s = max(e["time"] for e in self.raw_events) - min(e["time"] for e in self.raw_events)
            sim_duration_s = max(sim_duration_s, 0.001)
        else:
            sim_duration_s = 1.0

        # Throughput in kbps
        total_delivered_bytes = sum(p.size_bytes for p in delivered)
        throughput_kbps = (total_delivered_bytes * 8.0) / (sim_duration_s * 1000.0)

        return {
            "total_packets": total_packets,
            "delivered_packets": len(delivered),
            "dropped_packets": len(dropped),
            "pdr_percent": float(pdr),
            "drop_rate_percent": float(drop_rate),
            "mean_delay_ms": mean_delay,
            "median_delay_ms": median_delay,
            "p95_delay_ms": p95_delay,
            "p99_delay_ms": p99_delay,
            "min_delay_ms": min_delay,
            "max_delay_ms": max_delay,
            "std_delay_ms": std_delay,
            "jitter_ms": jitter_ms,
            "throughput_kbps": float(throughput_kbps),
            "simulation_duration_s": float(sim_duration_s),
        }

    def reset(self) -> None:
        """Clear parser state."""
        self.packets.clear()
        self.raw_events.clear()
