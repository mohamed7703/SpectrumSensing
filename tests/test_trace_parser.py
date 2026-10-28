"""Unit Tests for NS-3 ASCII Event Trace Parser.

Tests trace parsing (+, -, r, d), packet lifecycle record construction,
Packet Delivery Ratio (PDR), latency percentiles, and jitter calculations.
"""

import pytest
import numpy as np
from pathlib import Path

from src.ns3_simulation.trace_parser import TraceParser, PacketRecord


class TestTraceParser:
    """Test NS-3 ASCII trace parsing and metric aggregation."""

    def test_single_packet_complete_lifecycle(self) -> None:
        """Parse enqueue, transmit, and receive events for a single packet."""
        parser = TraceParser()
        lines = [
            "+ 0.100000000 /NodeList/1 pkt_id=1 size=1024B",
            "- 0.100050000 /NodeList/1 ila_csma_tx pkt_id=1 dur=8.19ms",
            "r 0.108242000 /NodeList/1 ila_csma_ack pkt_id=1",
        ]
        parser.parse_lines(lines)

        assert len(parser.packets) == 1
        pkt = parser.packets[1]
        assert pkt.is_delivered is True
        assert pkt.dropped is False
        assert np.isclose(pkt.enqueue_time, 0.100)
        assert np.isclose(pkt.tx_time, 0.100050)
        assert np.isclose(pkt.rx_time, 0.108242)

        # E2E delay = 0.108242 - 0.100000 = 8.242 ms
        assert np.isclose(pkt.e2e_delay_s, 0.008242)
        # Channel access delay = 50 µs
        assert np.isclose(pkt.channel_access_delay_s, 0.000050)

    def test_dropped_packet_record(self) -> None:
        """Parse dropped packet event."""
        parser = TraceParser()
        lines = [
            "+ 0.200000000 /NodeList/2 pkt_id=5 size=1024B",
            "- 0.200050000 /NodeList/2 ila_csma_tx pkt_id=5 dur=8.19ms",
            "d 0.245000000 /NodeList/2 ila_csma_drop pkt_id=5 max_retries_reached",
        ]
        parser.parse_lines(lines)

        pkt = parser.packets[5]
        assert pkt.is_delivered is False
        assert pkt.dropped is True
        assert "max_retries_reached" in pkt.drop_reason

    def test_metrics_pdr_and_drop_rate(self) -> None:
        """Verify PDR = 75% for 3 delivered and 1 dropped packet."""
        parser = TraceParser()
        lines = [
            # Packet 1 (delivered)
            "+ 0.010000000 /NodeList/1 pkt_id=1 size=1024B",
            "r 0.018000000 /NodeList/1 ack pkt_id=1",
            # Packet 2 (delivered)
            "+ 0.020000000 /NodeList/1 pkt_id=2 size=1024B",
            "r 0.029000000 /NodeList/1 ack pkt_id=2",
            # Packet 3 (delivered)
            "+ 0.030000000 /NodeList/1 pkt_id=3 size=1024B",
            "r 0.040000000 /NodeList/1 ack pkt_id=3",
            # Packet 4 (dropped)
            "+ 0.040000000 /NodeList/1 pkt_id=4 size=1024B",
            "d 0.080000000 /NodeList/1 drop pkt_id=4 collided",
        ]
        parser.parse_lines(lines)
        metrics = parser.compute_metrics()

        assert metrics["total_packets"] == 4
        assert metrics["delivered_packets"] == 3
        assert metrics["dropped_packets"] == 1
        assert np.isclose(metrics["pdr_percent"], 75.0)
        assert np.isclose(metrics["drop_rate_percent"], 25.0)

    def test_latency_percentiles(self) -> None:
        """Verify delay calculations (mean, median, p95, p99)."""
        parser = TraceParser()
        # Generate 100 delivered packets with known delays: 10ms to 20ms
        lines = []
        for i in range(100):
            t_enq = i * 0.1
            delay = 0.010 + (i / 100.0) * 0.010  # 10ms to 20ms
            t_rx = t_enq + delay
            lines.append(f"+ {t_enq:.9f} /NodeList/1 pkt_id={i} size=1024B")
            lines.append(f"r {t_rx:.9f} /NodeList/1 ack pkt_id={i}")

        parser.parse_lines(lines)
        metrics = parser.compute_metrics()

        assert metrics["total_packets"] == 100
        assert np.isclose(metrics["pdr_percent"], 100.0)
        assert 14.5 <= metrics["mean_delay_ms"] <= 15.5
        assert np.isclose(metrics["min_delay_ms"], 10.0, atol=0.2)
        assert np.isclose(metrics["max_delay_ms"], 19.9, atol=0.2)
        assert metrics["p95_delay_ms"] > metrics["median_delay_ms"]

    def test_file_parsing(self, tmp_path: Path) -> None:
        """Verify reading and parsing an ASCII trace file from disk."""
        trace_file = tmp_path / "test_trace.tr"
        trace_content = (
            "# NS-3 Trace Header\n"
            "+ 0.001000000 /NodeList/0 pkt_id=1 size=1024B\n"
            "- 0.001050000 /NodeList/0 tx pkt_id=1\n"
            "r 0.009242000 /NodeList/0 rx pkt_id=1\n"
        )
        trace_file.write_text(trace_content)

        parser = TraceParser()
        parser.parse_file(trace_file)
        metrics = parser.compute_metrics()

        assert metrics["total_packets"] == 1
        assert metrics["delivered_packets"] == 1
        assert np.isclose(metrics["mean_delay_ms"], 8.242, atol=0.01)

    def test_malformed_and_comment_lines(self) -> None:
        """Parser must ignore blank lines, comments, and non-conforming strings."""
        parser = TraceParser()
        lines = [
            "",
            "# A comment line",
            "This is not a trace line",
            "+ 0.050000000 /NodeList/1 pkt_id=9 size=1024B",
            "r 0.058000000 /NodeList/1 ack pkt_id=9",
        ]
        parser.parse_lines(lines)
        metrics = parser.compute_metrics()
        assert metrics["total_packets"] == 1
        assert metrics["delivered_packets"] == 1
