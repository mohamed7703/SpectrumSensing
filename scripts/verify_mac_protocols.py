"""Verification Script for P3-002: Cognitive MAC Protocols.

Evaluates throughput vs offered load curves S(G) across Pure ALOHA,
Slotted ALOHA, and ILA-CSMA, benchmarking against theoretical limits
and measuring Primary User collision protection.
"""

from pathlib import Path
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.topology_builder import TopologyBuilder, PrimaryUser, SecondaryUser
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.traffic_generator import Packet
from src.ns3_simulation.mac_pure_aloha import PureAlohaMAC
from src.ns3_simulation.mac_slotted_aloha import SlottedAlohaMAC
from src.ns3_simulation.mac_ila_csma import ILaCsmaMAC


def run_mac_verification() -> None:
    """Run empirical MAC protocol throughput and collision benchmark."""
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "p3_002_mac_protocols.png"

    print("=" * 70)
    print("P3-002: Cognitive MAC Protocols (Pure ALOHA, Slotted ALOHA, ILA-CSMA)")
    print("=" * 70)

    # 1. Theoretical Throughput Curves
    g_theoretical = np.linspace(0.01, 3.5, 300)
    s_pure_theory = [PureAlohaMAC.theoretical_throughput(g) for g in g_theoretical]
    s_slotted_theory = [SlottedAlohaMAC.theoretical_throughput(g) for g in g_theoretical]

    # 2. Empirical Monte Carlo Simulation Across Offered Loads
    g_sweep = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    s_pure_empirical = []
    s_slotted_empirical = []
    s_ila_empirical = []
    pu_coll_pure = []
    pu_coll_ila = []

    packet_duration = 0.008192  # 1024 bytes at 1 Mbps

    channel = ChannelModel(random_seed=42)
    rng = np.random.default_rng(42)

    for g in g_sweep:
        sim_time = 2.0
        num_packets = int(np.round((g / packet_duration) * sim_time))
        num_packets = max(10, min(500, num_packets))

        # Simulation with PU intermittently active (50% duty cycle)
        pu = PrimaryUser(node_id=0, position=(250.0, 250.0), is_active=False, tx_power_dbm=30.0)

        # -- A. Pure ALOHA --
        eng_pure = SimulationEngine(max_time=sim_time)
        mac_pure = PureAlohaMAC(engine=eng_pure, max_retransmissions=0, random_seed=42)
        su_pure = SecondaryUser(node_id=1, position=(200.0, 200.0))

        # Schedule PU toggles
        eng_pure.schedule(0.5, lambda: setattr(pu, "is_active", True))
        eng_pure.schedule(1.0, lambda: setattr(pu, "is_active", False))
        eng_pure.schedule(1.5, lambda: setattr(pu, "is_active", True))

        arrival_times = np.sort(rng.uniform(0.0, sim_time - packet_duration, num_packets))
        for idx, t_arr in enumerate(arrival_times):
            pkt = Packet(packet_id=idx, source_id=1, creation_time=t_arr)
            eng_pure.schedule(t_arr, lambda p=pkt: mac_pure.send_packet(su_pure, p, primary_users=[pu]))

        eng_pure.run()
        s_pure = (mac_pure.packets_successful * packet_duration) / sim_time
        s_pure_empirical.append(s_pure)
        pu_coll_pure.append(mac_pure.pu_collisions)

        # -- B. Slotted ALOHA --
        pu.is_active = False
        eng_slotted = SimulationEngine(max_time=sim_time)
        mac_slotted = SlottedAlohaMAC(engine=eng_slotted, slot_duration_s=packet_duration, max_retransmissions=0, random_seed=42)
        su_slotted = SecondaryUser(node_id=1, position=(200.0, 200.0))

        for idx, t_arr in enumerate(arrival_times):
            pkt = Packet(packet_id=idx, source_id=1, creation_time=t_arr)
            eng_slotted.schedule(t_arr, lambda p=pkt: mac_slotted.queue_packet(su_slotted, p, primary_users=None))

        eng_slotted.run()
        s_slotted = (mac_slotted.packets_successful * packet_duration) / sim_time
        s_slotted_empirical.append(s_slotted)

        # -- C. ILA-CSMA --
        pu.is_active = False
        eng_ila = SimulationEngine(max_time=sim_time)
        mac_ila = ILaCsmaMAC(
            engine=eng_ila,
            channel=channel,
            cad_threshold_dbm=-95.0,
            rssi_threshold_dbm=-90.0,
            max_retransmissions=5,
            random_seed=42,
        )
        su_ila = SecondaryUser(node_id=1, position=(200.0, 200.0))

        # Same PU schedule
        eng_ila.schedule(0.5, lambda: setattr(pu, "is_active", True))
        eng_ila.schedule(1.0, lambda: setattr(pu, "is_active", False))
        eng_ila.schedule(1.5, lambda: setattr(pu, "is_active", True))

        for idx, t_arr in enumerate(arrival_times):
            pkt = Packet(packet_id=idx, source_id=1, creation_time=t_arr)
            eng_ila.schedule(t_arr, lambda p=pkt: mac_ila.queue_packet(su_ila, p, primary_users=[pu]))

        eng_ila.run()
        s_ila = (mac_ila.packets_successful * packet_duration) / sim_time
        s_ila_empirical.append(s_ila)
        pu_coll_ila.append(mac_ila.pu_collisions)

        print(f"G = {g:4.2f} | Pure S: {s_pure:5.3f} (PU Coll: {mac_pure.pu_collisions:2d}) | Slotted S: {s_slotted:5.3f} | ILA-CSMA S: {s_ila:5.3f} (PU Coll: {mac_ila.pu_collisions:2d})")

    # 3. Generate Verification Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Subplot 1: Throughput vs Offered Load S(G)
    ax1 = axes[0, 0]
    ax1.plot(g_theoretical, s_pure_theory, "r--", lw=2.0, label="Pure ALOHA (Theory: $G e^{-2G}$)")
    ax1.plot(g_theoretical, s_slotted_theory, "b--", lw=2.0, label="Slotted ALOHA (Theory: $G e^{-G}$)")
    ax1.plot(g_sweep, s_pure_empirical, "ro", markersize=6, label="Pure ALOHA (Empirical)")
    ax1.plot(g_sweep, s_slotted_empirical, "bs", markersize=6, label="Slotted ALOHA (Empirical)")
    ax1.plot(g_sweep, s_ila_empirical, "g^-", lw=2.2, markersize=7, label="ILA-CSMA Cognitive MAC")
    ax1.axhline(1.0 / (2.0 * np.e), color="red", ls=":", alpha=0.6, label=r"Pure Max ($1/2e \approx 0.184$)")
    ax1.axhline(1.0 / np.e, color="blue", ls=":", alpha=0.6, label=r"Slotted Max ($1/e \approx 0.368$)")
    ax1.set_title("Throughput vs. Offered Load $S(G)$", fontweight="bold")
    ax1.set_xlabel("Offered Load $G$ (dimensionless)")
    ax1.set_ylabel("Normalized Channel Throughput $S$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 0.85)

    # Subplot 2: Packet Delivery Ratio (PDR) vs Offered Load
    ax2 = axes[0, 1]
    pdr_pure = np.array(s_pure_empirical) / g_sweep
    pdr_slotted = np.array(s_slotted_empirical) / g_sweep
    pdr_ila = np.array(s_ila_empirical) / g_sweep
    ax2.plot(g_sweep, np.clip(pdr_pure * 100.0, 0, 100), "r--o", lw=1.8, label="Pure ALOHA")
    ax2.plot(g_sweep, np.clip(pdr_slotted * 100.0, 0, 100), "b--s", lw=1.8, label="Slotted ALOHA")
    ax2.plot(g_sweep, np.clip(pdr_ila * 100.0, 0, 100), "g-^", lw=2.2, label="ILA-CSMA Cognitive MAC")
    ax2.axhline(90.0, color="gray", ls=":", label="90% Reliability QoS Target")
    ax2.set_title("Packet Delivery Ratio (PDR) vs. Traffic Load", fontweight="bold")
    ax2.set_xlabel("Offered Load $G$")
    ax2.set_ylabel("PDR (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 105)

    # Subplot 3: Primary User Protection / Collision Comparison
    ax3 = axes[1, 0]
    bar_width = 0.35
    x_idx = np.arange(len(g_sweep))
    ax3.bar(x_idx - bar_width / 2, pu_coll_pure, width=bar_width, color="crimson", label="Pure ALOHA (No Sensing)", alpha=0.85)
    ax3.bar(x_idx + bar_width / 2, pu_coll_ila, width=bar_width, color="forestgreen", label="ILA-CSMA (Hybrid Sensing)", alpha=0.85)
    ax3.set_title("Primary User Collision Count (Interference Ingress)", fontweight="bold")
    ax3.set_xlabel("Offered Load $G$")
    ax3.set_ylabel("PU Collision Events")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels([f"{g:.2f}" for g in g_sweep])
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(loc="upper left")

    # Subplot 4: Relative Performance Gain of ILA-CSMA
    ax4 = axes[1, 1]
    gains = []
    for s_i, s_p in zip(s_ila_empirical, s_pure_empirical):
        gain = ((s_i - s_p) / max(1e-4, s_p)) * 100.0
        gains.append(min(gain, 400.0))
    ax4.plot(g_sweep, gains, "purple", marker="D", lw=2.0, label="ILA-CSMA Throughput Gain vs Pure ALOHA (%)")
    ax4.axhline(100.0, color="orange", ls="--", label="+100% (2x Improvement)")
    ax4.set_title("Throughput Gain from Intelligent Spectrum Sensing", fontweight="bold")
    ax4.set_xlabel("Offered Load $G$")
    ax4.set_ylabel("Throughput Gain (%)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")

    plt.suptitle("CRN MAC Protocol Comparison: Pure ALOHA, Slotted ALOHA & ILA-CSMA (P3-002)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[OK] MAC protocols verification plot saved to {plot_path}")


if __name__ == "__main__":
    run_mac_verification()
