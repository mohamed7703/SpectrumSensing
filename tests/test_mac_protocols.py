"""Unit Tests for CRN Cognitive MAC Protocols.

Tests Pure ALOHA random access, Slotted ALOHA synchronized access, and
ILA-CSMA two-stage hybrid sensing with PU protection and adaptive backoff.
"""

import pytest
import numpy as np

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import TopologyBuilder, PrimaryUser, SecondaryUser
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.traffic_generator import Packet
from src.ns3_simulation.mac_pure_aloha import PureAlohaMAC
from src.ns3_simulation.mac_slotted_aloha import SlottedAlohaMAC
from src.ns3_simulation.mac_ila_csma import ILaCsmaMAC


class TestPureAlohaMAC:
    """Test unslotted Pure ALOHA contention MAC."""

    def test_single_packet_success(self) -> None:
        """Isolated packet transmission should succeed without collision."""
        engine = SimulationEngine(max_time=1.0)
        mac = PureAlohaMAC(engine=engine, data_rate_bps=1e6)
        su = SecondaryUser(node_id=1, position=(0.0, 0.0))
        pkt = Packet(packet_id=1, source_id=1, size_bytes=1024)

        mac.send_packet(su, pkt)
        engine.run()

        assert mac.packets_transmitted == 1
        assert mac.packets_successful == 1
        assert mac.packets_collided_count == 0
        assert pkt.receive_time is not None

    def test_overlapping_packets_collide(self) -> None:
        """Two overlapping SU transmissions must collide."""
        engine = SimulationEngine(max_time=1.0)
        mac = PureAlohaMAC(engine=engine, max_retransmissions=0)
        su1 = SecondaryUser(node_id=1, position=(0.0, 0.0))
        su2 = SecondaryUser(node_id=2, position=(10.0, 0.0))

        pkt1 = Packet(packet_id=1, source_id=1, size_bytes=1024)
        pkt2 = Packet(packet_id=2, source_id=2, size_bytes=1024)

        # Transmit at t=0.0 and t=0.002 (packet duration is ~8.19 ms, so they overlap)
        engine.schedule(0.0, lambda: mac.send_packet(su1, pkt1))
        engine.schedule(0.002, lambda: mac.send_packet(su2, pkt2))
        engine.run()

        assert mac.packets_collided_count >= 1
        assert mac.packets_successful == 0

    def test_pu_collision_detection(self) -> None:
        """Transmission while PU is active must flag collision."""
        engine = SimulationEngine(max_time=1.0)
        mac = PureAlohaMAC(engine=engine, max_retransmissions=0)
        pu = PrimaryUser(node_id=0, position=(100.0, 100.0), is_active=True)
        su = SecondaryUser(node_id=1, position=(0.0, 0.0))
        pkt = Packet(packet_id=1, source_id=1)

        mac.send_packet(su, pkt, primary_users=[pu])
        engine.run()

        assert mac.pu_collisions == 1
        assert mac.packets_successful == 0

    def test_pure_aloha_theoretical_throughput(self) -> None:
        """Throughput S = G * exp(-2G) peaks at G = 0.5 with S = 1/(2e)."""
        g_vals = np.linspace(0.01, 3.0, 500)
        s_vals = [PureAlohaMAC.theoretical_throughput(g) for g in g_vals]

        max_idx = np.argmax(s_vals)
        opt_g = g_vals[max_idx]
        max_s = s_vals[max_idx]

        assert np.isclose(opt_g, 0.5, atol=0.05)
        assert np.isclose(max_s, 1.0 / (2.0 * np.e), atol=0.01)


class TestSlottedAlohaMAC:
    """Test synchronized Slotted ALOHA MAC."""

    def test_slot_boundary_alignment(self) -> None:
        """Packets must synchronize to future slot boundaries."""
        engine = SimulationEngine(max_time=1.0)
        mac = SlottedAlohaMAC(engine=engine, slot_duration_s=0.01)

        slot_idx, slot_time = mac.get_next_slot_boundary(current_time=0.003)
        assert slot_idx == 1
        assert np.isclose(slot_time, 0.01)

        slot_idx2, slot_time2 = mac.get_next_slot_boundary(current_time=0.015)
        assert slot_idx2 == 2
        assert np.isclose(slot_time2, 0.02)

    def test_single_packet_slot_success(self) -> None:
        """Single transmission in a slot succeeds."""
        engine = SimulationEngine(max_time=1.0)
        mac = SlottedAlohaMAC(engine=engine, slot_duration_s=0.01)
        su = SecondaryUser(node_id=1, position=(0.0, 0.0))
        pkt = Packet(packet_id=1, source_id=1)

        mac.queue_packet(su, pkt)
        engine.run()

        assert mac.packets_successful == 1
        assert mac.packets_collided_count == 0

    def test_same_slot_contention_collision(self) -> None:
        """Two SUs queueing into the same slot must collide."""
        engine = SimulationEngine(max_time=1.0)
        mac = SlottedAlohaMAC(engine=engine, slot_duration_s=0.01, max_retransmissions=0)
        su1 = SecondaryUser(node_id=1, position=(0.0, 0.0))
        su2 = SecondaryUser(node_id=2, position=(10.0, 0.0))

        pkt1 = Packet(packet_id=1, source_id=1)
        pkt2 = Packet(packet_id=2, source_id=2)

        mac.queue_packet(su1, pkt1)
        mac.queue_packet(su2, pkt2)
        engine.run()

        assert mac.packets_collided_count == 2
        assert mac.packets_successful == 0

    def test_different_slots_succeed(self) -> None:
        """Transmissions in separate slots both succeed without interference."""
        engine = SimulationEngine(max_time=1.0)
        mac = SlottedAlohaMAC(engine=engine, slot_duration_s=0.01)
        su1 = SecondaryUser(node_id=1, position=(0.0, 0.0))
        su2 = SecondaryUser(node_id=2, position=(10.0, 0.0))

        pkt1 = Packet(packet_id=1, source_id=1)
        pkt2 = Packet(packet_id=2, source_id=2)

        # Queue pkt1 at t=0.0 (slot 1)
        mac.queue_packet(su1, pkt1)
        # Queue pkt2 at t=0.015 (slot 2)
        engine.schedule(0.015, lambda: mac.queue_packet(su2, pkt2))
        engine.run()

        assert mac.packets_successful == 2
        assert mac.packets_collided_count == 0

    def test_slotted_aloha_theoretical_throughput(self) -> None:
        """Slotted ALOHA S = G * exp(-G) peaks at G = 1.0 with S = 1/e (double Pure ALOHA)."""
        g_vals = np.linspace(0.01, 3.0, 500)
        s_vals = [SlottedAlohaMAC.theoretical_throughput(g) for g in g_vals]

        max_idx = np.argmax(s_vals)
        opt_g = g_vals[max_idx]
        max_s = s_vals[max_idx]

        assert np.isclose(opt_g, 1.0, atol=0.05)
        assert np.isclose(max_s, 1.0 / np.e, atol=0.01)


class TestILaCsmaMAC:
    """Test ILA-CSMA two-stage sensing and adaptive backoff protocol."""

    def test_clear_channel_transmission(self) -> None:
        """When channel is clear, SU transmits immediately after DIFS."""
        engine = SimulationEngine(max_time=1.0)
        channel = ChannelModel(random_seed=42)
        mac = ILaCsmaMAC(engine=engine, channel=channel, data_rate_bps=1e6)

        su = SecondaryUser(node_id=1, position=(0.0, 0.0))
        pkt = Packet(packet_id=1, source_id=1, size_bytes=1024)

        mac.queue_packet(su, pkt)
        engine.run()

        assert mac.packets_successful == 1
        assert mac.packets_collided_count == 0
        assert pkt.receive_time is not None

    def test_pu_detection_and_deferral(self) -> None:
        """When PU is active, ILA-CSMA must detect activity and defer/backoff."""
        engine = SimulationEngine(max_time=1.0)
        channel = ChannelModel(random_seed=42)
        mac = ILaCsmaMAC(
            engine=engine,
            channel=channel,
            cad_threshold_dbm=-95.0,
            rssi_threshold_dbm=-90.0,
        )

        pu = PrimaryUser(node_id=0, position=(50.0, 50.0), is_active=True, tx_power_dbm=30.0)
        su = SecondaryUser(node_id=1, position=(0.0, 0.0))

        # Channel sensing must report busy
        is_busy = mac.sense_channel(su, primary_users=[pu])
        assert is_busy is True

    def test_pu_turns_off_su_transmits(self) -> None:
        """SU defers while PU is active, then transmits once PU turns OFF."""
        engine = SimulationEngine(max_time=1.0)
        channel = ChannelModel(random_seed=42)
        mac = ILaCsmaMAC(engine=engine, channel=channel)

        pu = PrimaryUser(node_id=0, position=(50.0, 50.0), is_active=True, tx_power_dbm=30.0)
        su = SecondaryUser(node_id=1, position=(0.0, 0.0))
        pkt = Packet(packet_id=1, source_id=1)

        # Deactivate PU at t=0.05
        def deactivate_pu() -> None:
            pu.is_active = False

        engine.schedule(0.005, deactivate_pu)
        mac.queue_packet(su, pkt, primary_users=[pu])
        engine.run()

        # Packet should eventually succeed once PU vacates band
        assert mac.packets_successful == 1
        assert mac.pu_collisions == 0
