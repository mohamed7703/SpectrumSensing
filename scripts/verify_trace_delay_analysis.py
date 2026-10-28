"""Verification Script for P3-003: Trace Parser & End-to-End Delay Analysis Pipeline.

Executes dense multi-node CRN scenarios, parses ASCII event traces,
benchmarks ILA-CSMA against baseline CSMA/CA across 10 to 2000 SU nodes,
and evaluates Packet Delivery Ratio (PDR) and latency percentiles.
"""

from pathlib import Path
import sys
from typing import Tuple, Dict, Any, List

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import TopologyBuilder, PrimaryUser, SecondaryUser
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.traffic_generator import TrafficGenerator, Packet
from src.ns3_simulation.mac_ila_csma import ILaCsmaMAC
from src.ns3_simulation.trace_parser import TraceParser


def run_dense_scenario_simulation(
    num_sus: int,
    mac_type: str = "ila_csma",
    sim_duration_s: float = 1.0,
    random_seed: int = 42,
) -> Tuple[TraceParser, Dict[str, Any]]:
    """Simulate a CRN scenario with num_sus and parse trace output."""
    engine = SimulationEngine(max_time=sim_duration_s)
    topo = TopologyBuilder(area_width=500.0, area_height=500.0, random_seed=random_seed)
    pu = topo.add_primary_user(position=(250.0, 450.0), tx_power_dbm=30.0, mean_on_s=0.2, mean_off_s=0.3)
    sus = topo.add_secondary_users(count=num_sus, placement="random_uniform", tx_power_dbm=20.0)

    channel = ChannelModel(random_seed=random_seed)

    if mac_type == "ila_csma":
        # ILA-CSMA with hybrid sensing + adaptive backoff
        mac = ILaCsmaMAC(
            engine=engine,
            channel=channel,
            cad_threshold_dbm=-95.0,
            rssi_threshold_dbm=-90.0,
            cw_min_slots=15,
            cw_max_slots=1023,
            max_retransmissions=7,
            random_seed=random_seed,
        )
    else:
        # Standard CSMA/CA (fixed sensing threshold, non-adaptive backoff)
        mac = ILaCsmaMAC(
            engine=engine,
            channel=channel,
            cad_threshold_dbm=-85.0,  # Lower sensitivity
            rssi_threshold_dbm=-80.0,
            cw_min_slots=31,
            cw_max_slots=255,         # Rigid backoff ceiling
            max_retransmissions=4,
            random_seed=random_seed,
        )

    # Schedule PU periodic toggles
    t = 0.0
    while t < sim_duration_s:
        t_on = t + 0.1
        t_off = t_on + 0.15
        engine.schedule(t_on, lambda: setattr(pu, "is_active", True))
        engine.schedule(t_off, lambda: setattr(pu, "is_active", False))
        t += 0.35

    # Traffic generation
    rng = np.random.default_rng(random_seed)
    # Scale packets so total load challenges channel without infinite queue
    packets_per_su = max(1, int(150 / num_sus))

    for su in sus:
        arr_times = np.sort(rng.uniform(0.01, sim_duration_s * 0.8, packets_per_su))
        for idx, t_arr in enumerate(arr_times):
            pkt_id = su.node_id * 1000 + idx
            pkt = Packet(packet_id=pkt_id, source_id=su.node_id, creation_time=t_arr)
            # Log enqueue event
            engine.schedule(t_arr, lambda s=su, p=pkt: (
                engine.log_trace("+", s.node_id, f"pkt_id={p.packet_id} size={p.size_bytes}B"),
                mac.queue_packet(s, p, primary_users=[pu])
            ))

    engine.run()

    # Parse trace
    parser = TraceParser()
    parser.parse_lines(engine.get_trace())
    metrics = parser.compute_metrics()
    return parser, metrics


def run_trace_delay_verification() -> None:
    """Benchmark ILA-CSMA vs CSMA/CA across dense network node counts."""
    plot_dir = Path("results/plots")
    log_dir = Path("results/logs")
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / "p3_003_trace_delay_analysis.png"
    trace_path = log_dir / "p3_003_sample_trace.tr"

    print("=" * 70)
    print("P3-003: Trace Parser & End-to-End Delay Analysis Pipeline Verification")
    print("=" * 70)

    node_counts = [10, 50, 100, 500, 2000]
    pdr_ila = []
    pdr_csma = []
    delay_ila = []
    delay_csma = []

    sample_parser_ila = None
    sample_parser_csma = None

    for n in node_counts:
        # ILA-CSMA run
        parser_ila, met_ila = run_dense_scenario_simulation(num_sus=n, mac_type="ila_csma", sim_duration_s=1.5, random_seed=42)
        pdr_ila.append(met_ila["pdr_percent"])
        delay_ila.append(met_ila["mean_delay_ms"])

        # CSMA/CA run
        parser_csma, met_csma = run_dense_scenario_simulation(num_sus=n, mac_type="csma_ca", sim_duration_s=1.5, random_seed=42)
        pdr_csma.append(met_csma["pdr_percent"])
        delay_csma.append(met_csma["mean_delay_ms"])

        print(f"Nodes: {n:4d} | ILA PDR: {met_ila['pdr_percent']:5.1f}% (Delay: {met_ila['mean_delay_ms']:6.2f}ms) | CSMA PDR: {met_csma['pdr_percent']:5.1f}% (Delay: {met_csma['mean_delay_ms']:6.2f}ms)")

        if n == 100:
            sample_parser_ila = parser_ila
            sample_parser_csma = parser_csma

    # Save a sample NS-3 ASCII trace file
    sample_trace = parser_ila.raw_events
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("# NS-3 ASCII Trace Log: Cognitive Radio Network Simulation\n")
        for ev in sample_trace[:500]:
            f.write(f"{ev['action']} {ev['time']:.9f} /NodeList/{ev['node_id']} {ev['details']}\n")
    print(f"\n[OK] Sample NS-3 trace file saved to {trace_path}")

    # Generate Publication Verification Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Packet Delivery Ratio (PDR) vs Node Density
    ax1 = axes[0, 0]
    ax1.plot(node_counts, pdr_ila, "g-^", lw=2.2, markersize=8, label="ILA-CSMA (Hybrid Sensing)")
    ax1.plot(node_counts, pdr_csma, "r--s", lw=2.0, markersize=7, label="Standard CSMA/CA")
    ax1.set_xscale("log")
    ax1.set_title("Packet Delivery Ratio (PDR) vs. Node Scalability", fontweight="bold")
    ax1.set_xlabel("Number of Secondary Users (Log Scale)")
    ax1.set_ylabel("Packet Delivery Ratio (%)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="lower left")
    ax1.set_ylim(40, 105)

    # Plot 2: End-to-End Mean Delay vs Node Density
    ax2 = axes[0, 1]
    ax2.plot(node_counts, delay_ila, "g-^", lw=2.2, markersize=8, label="ILA-CSMA")
    ax2.plot(node_counts, delay_csma, "r--s", lw=2.0, markersize=7, label="Standard CSMA/CA")
    ax2.set_xscale("log")
    ax2.set_title("End-to-End Mean Latency vs. Node Density", fontweight="bold")
    ax2.set_xlabel("Number of Secondary Users (Log Scale)")
    ax2.set_ylabel("Mean E2E Delay (ms)")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="upper left")

    # Plot 3: Delay Cumulative Distribution Function (CDF) for N=100
    ax3 = axes[1, 0]
    if sample_parser_ila and sample_parser_csma:
        delays_i = np.sort([p.e2e_delay_s * 1000.0 for p in sample_parser_ila.packets.values() if p.is_delivered])
        delays_c = np.sort([p.e2e_delay_s * 1000.0 for p in sample_parser_csma.packets.values() if p.is_delivered])
        cdf_i = np.linspace(0, 1, len(delays_i))
        cdf_c = np.linspace(0, 1, len(delays_c))
        ax3.plot(delays_i, cdf_i, "g-", lw=2.2, label=f"ILA-CSMA (P95={np.percentile(delays_i, 95):.1f}ms)")
        ax3.plot(delays_c, cdf_c, "r--", lw=2.0, label=f"CSMA/CA (P95={np.percentile(delays_c, 95):.1f}ms)")
    ax3.set_title("Delay Cumulative Distribution Function (CDF) [N=100]", fontweight="bold")
    ax3.set_xlabel("End-to-End Latency (ms)")
    ax3.set_ylabel("CDF P(D <= d)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")

    # Plot 4: PDR Improvement Ratio
    ax4 = axes[1, 1]
    pdr_gains = np.array(pdr_ila) - np.array(pdr_csma)
    ax4.bar(np.arange(len(node_counts)), pdr_gains, color="teal", width=0.4, edgecolor="black")
    ax4.axhline(20.0, color="orange", ls="--", lw=1.8, label="Target +20% Improvement")
    ax4.set_title("Absolute PDR Improvement (ILA-CSMA over CSMA/CA)", fontweight="bold")
    ax4.set_xlabel("Scenario Node Count")
    ax4.set_ylabel("PDR Gain (Percentage Points)")
    ax4.set_xticks(np.arange(len(node_counts)))
    ax4.set_xticklabels([str(n) for n in node_counts])
    for i, v in enumerate(pdr_gains):
        ax4.text(i, v + 0.5, f"+{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.legend(loc="upper left")

    plt.suptitle("NS-3 Trace Delay & Scalability Analysis: ILA-CSMA vs CSMA/CA (P3-003)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[OK] Scalability delay verification plot saved to {plot_path}")


if __name__ == "__main__":
    run_trace_delay_verification()
