"""Adversarial packet timing checks; no inference about real RF protection."""

import pytest

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.topology_builder import PrimaryUser, SecondaryUser
from src.ns3_simulation.traffic_generator import Packet
from src.ns3_simulation.mac_ila_csma import ILaCsmaMAC
from src.ns3_simulation.mac_pure_aloha import PureAlohaMAC
from src.ns3_simulation.mac_slotted_aloha import SlottedAlohaMAC
from src.ns3_simulation.trace_parser import TraceParser


def make_mac(kind: str, engine: SimulationEngine):
    """Create a no-retry protocol for the specified deterministic test engine."""
    if kind == "csma":
        return ILaCsmaMAC(engine, ChannelModel(random_seed=1), max_retransmissions=0)
    if kind == "pure":
        return PureAlohaMAC(engine, max_retransmissions=0)
    return SlottedAlohaMAC(engine, slot_duration_s=0.01, max_retransmissions=0)


def start(mac, su: SecondaryUser, packet: Packet, pus: list) -> None:
    """Submit a packet through the protocol's public entry point."""
    if isinstance(mac, PureAlohaMAC):
        mac.send_packet(su, packet, pus)
    else:
        mac.queue_packet(su, packet, pus)


@pytest.mark.parametrize("kind", ["csma", "pure", "slotted"])
def test_primary_pulse_entirely_inside_packet(kind: str) -> None:
    """A PU pulse starting and ending during a packet must corrupt that attempt."""
    engine = SimulationEngine(0.1)
    mac = make_mac(kind, engine)
    su = SecondaryUser(node_id=1, position=(0, 0))
    pu = PrimaryUser(node_id=0, position=(1, 0), is_active=False)
    packet = Packet(packet_id=1, source_id=1)
    offset = 0.01 if kind == "slotted" else 0.0
    start(mac, su, packet, [pu])
    engine.schedule_at(offset + 0.002, lambda: setattr(pu, "is_active", True))
    engine.schedule_at(offset + 0.004, lambda: setattr(pu, "is_active", False))
    engine.run()
    assert mac.pu_collisions == 1
    assert mac.packets_collided_count == 1
    assert mac.packets_successful == 0
    assert packet.dropped and packet.receive_time is None


def test_slotted_packet_still_pending_at_horizon() -> None:
    """A slot starting at 10ms cannot deliver a packet by a 15ms horizon."""
    engine = SimulationEngine(0.015)
    mac = make_mac("slotted", engine)
    packet = Packet(packet_id=1, source_id=1)
    start(mac, SecondaryUser(node_id=1, position=(0, 0)), packet, [])
    engine.run()
    assert mac.packets_transmitted == 1
    assert mac.packets_successful == 0
    assert packet.receive_time is None
    assert not any(line.startswith("r ") for line in engine.get_trace())


def test_slotted_receive_trace_matches_packet_time() -> None:
    """The receive trace and packet metadata must both refer to slot completion."""
    engine = SimulationEngine(0.03)
    mac = make_mac("slotted", engine)
    packet = Packet(packet_id=1, source_id=1)
    start(mac, SecondaryUser(node_id=1, position=(0, 0)), packet, [])
    engine.run()
    parser = TraceParser()
    parser.parse_lines(engine.get_trace())
    assert packet.receive_time == pytest.approx(0.02)
    assert parser.packets[1].rx_time == pytest.approx(packet.receive_time)
    assert parser.packets[1].tx_time == pytest.approx(0.01)


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_scheduler_rejects_invalid_delay(delay: float) -> None:
    """Invalid delays must not silently move time backwards or corrupt the heap."""
    with pytest.raises(ValueError):
        SimulationEngine().schedule(delay, lambda: None)


def test_run_preserves_events_beyond_horizon() -> None:
    """An unfinished event remains available if the observation horizon extends."""
    engine = SimulationEngine(1)
    fired = []
    engine.schedule_at(2, lambda: fired.append(engine.now))
    engine.run()
    engine.max_time = 3
    engine.run()
    assert fired == [2]


def test_trace_accounts_pending_and_full_observation_window() -> None:
    """Throughput uses a declared horizon, and pending packets complete the ledger."""
    parser = TraceParser()
    parser.parse_lines([
        "+ 0.100000000 /NodeList/1 pkt_id=1 size=100B",
        "r 0.200000000 /NodeList/1 pkt_id=1",
        "+ 0.300000000 /NodeList/1 pkt_id=2 size=100B",
    ])
    metrics = parser.compute_metrics(observation_duration_s=1.0)
    assert metrics["throughput_kbps"] == pytest.approx(0.8)
    assert metrics["pending_packets"] == 1
    assert metrics["delivered_packets"] + metrics["dropped_packets"] + metrics["pending_packets"] == 2


@pytest.mark.parametrize("kind", ["csma", "pure", "slotted"])
def test_primary_onset_exactly_at_packet_end_is_not_overlap(kind: str) -> None:
    """Half-open airtime intervals exclude a PU starting at the completion instant."""
    engine = SimulationEngine(0.1)
    mac = make_mac(kind, engine)
    su = SecondaryUser(node_id=1, position=(0, 0))
    pu = PrimaryUser(node_id=0, position=(1, 0), is_active=False)
    packet = Packet(packet_id=1, source_id=1)
    end = {"pure": 0.008192, "csma": 0.008242, "slotted": 0.02}[kind]
    # Insert onset first to challenge event tie ordering at the boundary.
    engine.schedule_at(end, lambda: setattr(pu, "is_active", True))
    start(mac, su, packet, [pu])
    engine.run()
    assert mac.packets_successful == 1
    assert mac.pu_collisions == 0


def test_tracker_records_symmetric_secondary_overlap_and_duration() -> None:
    """Two overlapping attempts must both fail with the same known intersection."""
    engine = SimulationEngine(0.1)
    mac = make_mac("pure", engine)
    for packet_id, time in [(1, 0.0), (2, 0.002)]:
        su = SecondaryUser(node_id=packet_id, position=(packet_id, 0))
        packet = Packet(packet_id=packet_id, source_id=packet_id)
        engine.schedule_at(time, mac.send_packet, su=su, packet=packet)
    engine.run()
    attempts = mac.transmission_tracker.completed
    assert len(attempts) == 2
    assert [a.su_overlap_s for a in attempts] == pytest.approx([0.006192, 0.006192])
    assert mac.packets_collided_count == 2
    assert mac.packets_dropped_count == 2


def test_back_to_back_secondary_packets_do_not_collide() -> None:
    """A packet beginning exactly at its predecessor's end has zero overlap."""
    engine = SimulationEngine(0.1)
    mac = make_mac("pure", engine)
    su = SecondaryUser(node_id=1, position=(0, 0))
    # Start event inserted before the first packet's completion event.
    engine.schedule_at(0.008192, mac.send_packet, su=su, packet=Packet(packet_id=2, source_id=1))
    mac.send_packet(su, Packet(packet_id=1, source_id=1))
    engine.run()
    assert mac.packets_successful == 2
    assert mac.packets_collided_count == 0


def test_slotted_rejects_payload_larger_than_slot() -> None:
    """Slot occupancy must cover the payload's stated bit rate and size."""
    engine = SimulationEngine(0.1)
    mac = SlottedAlohaMAC(engine, slot_duration_s=0.001)
    with pytest.raises(ValueError, match="airtime"):
        mac.queue_packet(SecondaryUser(node_id=1, position=(0, 0)), Packet(packet_id=1, source_id=1))
