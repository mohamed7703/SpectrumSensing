"""Unit Tests for CRN Simulation Architecture, Topology & Traffic Generator.

Tests discrete-event simulation engine, CRN topology construction, wireless
channel propagation loss, and multi-model traffic generation.
"""

import pytest
import numpy as np

from src.ns3_simulation.sim_engine import SimulationEngine, Event
from src.ns3_simulation.topology_builder import (
    PrimaryUser,
    SecondaryUser,
    TopologyBuilder,
)
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.traffic_generator import TrafficGenerator, Packet


class TestSimulationEngine:
    """Test discrete-event simulation core."""

    def test_events_fire_in_order(self) -> None:
        """Events must fire in strict chronological order."""
        engine = SimulationEngine(max_time=10.0)
        order: list = []

        engine.schedule(0.3, lambda: order.append("C"))
        engine.schedule(0.1, lambda: order.append("A"))
        engine.schedule(0.2, lambda: order.append("B"))

        engine.run()
        assert order == ["A", "B", "C"]

    def test_simulation_clock_advances(self) -> None:
        """Simulation clock must equal the last event's scheduled time."""
        engine = SimulationEngine(max_time=10.0)
        times: list = []

        def record_time() -> None:
            times.append(engine.now)

        engine.schedule(1.0, record_time)
        engine.schedule(3.0, record_time)
        engine.schedule(5.0, record_time)
        engine.run()

        assert np.isclose(times[0], 1.0)
        assert np.isclose(times[1], 3.0)
        assert np.isclose(times[2], 5.0)

    def test_max_time_respected(self) -> None:
        """Events scheduled beyond max_time should not fire."""
        engine = SimulationEngine(max_time=2.0)
        fired = []

        engine.schedule(1.0, lambda: fired.append("within"))
        engine.schedule(3.0, lambda: fired.append("beyond"))

        engine.run()
        assert fired == ["within"]

    def test_event_cancellation(self) -> None:
        """Cancelled events must not fire."""
        engine = SimulationEngine(max_time=10.0)
        fired = []

        engine.schedule(0.1, lambda: fired.append("A"))
        evt_b = engine.schedule(0.2, lambda: fired.append("B"))
        engine.schedule(0.3, lambda: fired.append("C"))
        engine.cancel(evt_b)

        engine.run()
        assert fired == ["A", "C"]

    def test_trace_logging(self) -> None:
        """Trace entries must record correct action, time, and node."""
        engine = SimulationEngine(max_time=10.0)

        def log_something() -> None:
            engine.log_trace("+", 3, "pkt_id=42 size=1024B")

        engine.schedule(0.5, log_something)
        engine.run()

        trace = engine.get_trace()
        assert len(trace) == 1
        assert "+ 0.500000000" in trace[0]
        assert "/NodeList/3" in trace[0]

    def test_self_scheduling_chain(self) -> None:
        """Events can schedule further events (chain reaction)."""
        engine = SimulationEngine(max_time=1.0)
        count = [0]

        def tick() -> None:
            count[0] += 1
            if engine.now + 0.1 <= engine.max_time:
                engine.schedule(0.1, tick)

        engine.schedule(0.0, tick)
        engine.run()
        assert count[0] == 11  # 0.0, 0.1, ..., 1.0


class TestTopologyBuilder:
    """Test CRN topology construction."""

    def test_primary_user_properties(self) -> None:
        """PU nodes should have correct steady-state utilization."""
        topo = TopologyBuilder(random_seed=42)
        pu = topo.add_primary_user(
            position=(250.0, 450.0),
            tx_power_dbm=30.0,
            mean_on_s=2.0,
            mean_off_s=3.0,
        )
        assert pu.node_type == "PU"
        assert np.isclose(pu.utilization, 0.4)
        assert pu.tx_power_dbm == 30.0

    def test_secondary_user_random_placement(self) -> None:
        """Random uniform placement should generate N SUs within area bounds."""
        topo = TopologyBuilder(area_width=500.0, area_height=500.0, random_seed=42)
        sus = topo.add_secondary_users(count=20, placement="random_uniform")

        assert len(sus) == 20
        for su in sus:
            assert su.node_type == "SU"
            assert 0 <= su.position[0] <= 500.0
            assert 0 <= su.position[1] <= 500.0

    def test_grid_placement(self) -> None:
        """Grid placement should be reproducible and evenly spaced."""
        topo = TopologyBuilder(area_width=400.0, area_height=400.0, random_seed=42)
        sus = topo.add_secondary_users(count=9, placement="grid")
        assert len(sus) == 9

    def test_cluster_placement(self) -> None:
        """Cluster placement should center nodes near the given point."""
        topo = TopologyBuilder(random_seed=42)
        center = (250.0, 250.0)
        sus = topo.add_secondary_users(count=50, placement="cluster", cluster_center=center, cluster_radius=80.0)
        assert len(sus) == 50
        # Mean position should be near center
        mean_x = np.mean([su.position[0] for su in sus])
        mean_y = np.mean([su.position[1] for su in sus])
        assert abs(mean_x - center[0]) < 40.0
        assert abs(mean_y - center[1]) < 40.0

    def test_distance_calculation(self) -> None:
        """Euclidean distance between two nodes."""
        a = PrimaryUser(node_id=0, position=(0.0, 0.0))
        b = SecondaryUser(node_id=1, position=(300.0, 400.0))
        d = TopologyBuilder.distance(a, b)
        assert np.isclose(d, 500.0)

    def test_unique_node_ids(self) -> None:
        """Every node in the topology must have a unique ID."""
        topo = TopologyBuilder(random_seed=42)
        topo.add_primary_user(position=(100.0, 100.0))
        topo.add_primary_user(position=(200.0, 200.0))
        topo.add_secondary_users(count=5)

        ids = [n.node_id for n in topo.all_nodes]
        assert len(ids) == len(set(ids))


class TestChannelModel:
    """Test wireless propagation channel."""

    def test_path_loss_increases_with_distance(self) -> None:
        """Path loss must increase monotonically with distance."""
        ch = ChannelModel(propagation_model="log_distance", random_seed=42)
        pl_10 = ch.compute_path_loss(10.0)
        pl_100 = ch.compute_path_loss(100.0)
        pl_500 = ch.compute_path_loss(500.0)
        assert pl_10 < pl_100 < pl_500

    def test_noise_floor_calculation(self) -> None:
        """Noise floor = -174 + 10*log10(BW) + NF."""
        ch = ChannelModel(
            thermal_noise_dbm_per_hz=-174.0,
            bandwidth_hz=2.4e6,
            noise_figure_db=7.0,
            random_seed=42,
        )
        # N0 = -174 + 10*log10(2.4e6) + 7 = -174 + 63.8 + 7 = -103.2 dBm
        expected = -174.0 + 10.0 * np.log10(2.4e6) + 7.0
        assert np.isclose(ch.noise_floor_dbm, expected, atol=0.1)

    def test_rssi_decreases_with_distance(self) -> None:
        """RSSI should decrease as receiver moves farther from transmitter."""
        ch = ChannelModel(
            propagation_model="log_distance",
            shadowing_std_db=0.0,
            fading_model="none",
            random_seed=42,
        )
        pu = PrimaryUser(node_id=0, position=(0.0, 0.0), tx_power_dbm=30.0)
        su_near = SecondaryUser(node_id=1, position=(50.0, 0.0))
        su_far = SecondaryUser(node_id=2, position=(400.0, 0.0))

        rssi_near = ch.compute_rssi(pu, su_near, include_shadowing=False, include_fading=False)
        rssi_far = ch.compute_rssi(pu, su_far, include_shadowing=False, include_fading=False)
        assert rssi_near > rssi_far

    def test_snr_positive_near_transmitter(self) -> None:
        """Near a strong transmitter, SNR should be positive."""
        ch = ChannelModel(fading_model="none", shadowing_std_db=0.0, random_seed=42)
        pu = PrimaryUser(node_id=0, position=(0.0, 0.0), tx_power_dbm=30.0)
        su = SecondaryUser(node_id=1, position=(10.0, 0.0))
        snr = ch.compute_snr(pu, su, include_shadowing=False, include_fading=False)
        assert snr > 0.0

    def test_rayleigh_fading_produces_variance(self) -> None:
        """Rayleigh fading should create varying RSSI across many realizations."""
        ch = ChannelModel(fading_model="rayleigh", shadowing_std_db=0.0, random_seed=42)
        pu = PrimaryUser(node_id=0, position=(0.0, 0.0), tx_power_dbm=30.0)
        su = SecondaryUser(node_id=1, position=(100.0, 0.0))

        rssi_values = [ch.compute_rssi(pu, su, include_shadowing=False) for _ in range(200)]
        assert np.std(rssi_values) > 1.0  # Rayleigh creates spread

    def test_detectability_boundary(self) -> None:
        """Signal detectability should depend on RSSI vs threshold."""
        ch = ChannelModel(fading_model="none", shadowing_std_db=0.0, random_seed=42)
        pu = PrimaryUser(node_id=0, position=(0.0, 0.0), tx_power_dbm=30.0)
        su_close = SecondaryUser(node_id=1, position=(5.0, 0.0))
        su_far = SecondaryUser(node_id=2, position=(5000.0, 0.0))

        assert ch.is_signal_detectable(pu, su_close, detection_threshold_dbm=-90.0) is True
        assert ch.is_signal_detectable(pu, su_far, detection_threshold_dbm=-90.0) is False


class TestTrafficGenerator:
    """Test packet generation and arrival scheduling."""

    def test_deterministic_packet_count(self) -> None:
        """Deterministic traffic should generate exactly num_packets packets."""
        engine = SimulationEngine(max_time=100.0)
        topo = TopologyBuilder(random_seed=42)
        sus = topo.add_secondary_users(count=1)

        gen = TrafficGenerator(
            engine=engine,
            traffic_model="deterministic",
            num_packets=50,
            inter_packet_interval_s=0.01,
        )
        gen.start(sus[0])
        engine.run()

        assert len(gen.generated_packets) == 50

    def test_poisson_packet_generation(self) -> None:
        """Poisson traffic should generate approximately expected count."""
        engine = SimulationEngine(max_time=100.0)
        topo = TopologyBuilder(random_seed=42)
        sus = topo.add_secondary_users(count=1)

        gen = TrafficGenerator(
            engine=engine,
            traffic_model="poisson",
            poisson_rate_pps=100.0,
            num_packets=200,
            random_seed=42,
        )
        gen.start(sus[0])
        engine.run()

        # Should generate exactly 200 packets (capped by num_packets)
        assert len(gen.generated_packets) == 200

    def test_packet_metadata_correctness(self) -> None:
        """Each packet should have valid source_id and creation_time."""
        engine = SimulationEngine(max_time=100.0)
        topo = TopologyBuilder(random_seed=42)
        sus = topo.add_secondary_users(count=1)

        gen = TrafficGenerator(
            engine=engine,
            traffic_model="deterministic",
            num_packets=10,
            inter_packet_interval_s=0.01,
        )
        gen.start(sus[0])
        engine.run()

        for pkt in gen.generated_packets:
            assert pkt.source_id == sus[0].node_id
            assert pkt.creation_time >= 0.0
            assert pkt.size_bytes == 1024

    def test_trace_log_contains_enqueue_events(self) -> None:
        """Each packet arrival should produce a '+' enqueue trace entry."""
        engine = SimulationEngine(max_time=100.0)
        topo = TopologyBuilder(random_seed=42)
        sus = topo.add_secondary_users(count=1)

        gen = TrafficGenerator(
            engine=engine,
            traffic_model="deterministic",
            num_packets=5,
        )
        gen.start(sus[0])
        engine.run()

        trace = engine.get_trace()
        assert len(trace) == 5
        for entry in trace:
            assert entry.startswith("+")

    def test_offered_load_computation(self) -> None:
        """Offered load G = N * lambda * T_p."""
        gen = TrafficGenerator(
            engine=SimulationEngine(),
            traffic_model="deterministic",
            inter_packet_interval_s=0.01,  # 100 pps
        )
        # 10 SUs, 100 pps each, 1 ms packet => G = 10 * 100 * 0.001 = 1.0
        g = gen.compute_offered_load(num_sus=10, packet_duration_s=0.001)
        assert np.isclose(g, 1.0)

    def test_multi_su_traffic_isolation(self) -> None:
        """Multiple SU traffic streams should have independent packet IDs."""
        engine = SimulationEngine(max_time=100.0)
        topo = TopologyBuilder(random_seed=42)
        sus = topo.add_secondary_users(count=3)

        gen = TrafficGenerator(
            engine=engine,
            traffic_model="deterministic",
            num_packets=10,
        )
        for su in sus:
            gen.start(su)

        engine.run()

        # 3 SUs * 10 packets = 30 total, all unique IDs
        assert len(gen.generated_packets) == 30
        ids = [p.packet_id for p in gen.generated_packets]
        assert len(ids) == len(set(ids))
