"""Verification Script for P3-004: Game-Theoretic Nash Equilibrium Analysis.

Analyzes non-cooperative spectrum contention, computes symmetric Nash Equilibrium
strategies, compares against Social Optimum, evaluates Price of Anarchy (PoA),
and verifies Primary User interference constraints.
"""

from pathlib import Path
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.ns3_simulation.nash_equilibrium import SpectrumContentionGame


def run_nash_verification() -> None:
    """Run empirical and analytical Nash Equilibrium verification."""
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "p3_004_nash_equilibrium.png"

    print("=" * 70)
    print("P3-004: Game-Theoretic Nash Equilibrium Spectrum Sharing Verification")
    print("=" * 70)

    n_range = np.array([2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100])

    p_star_low_cost = []
    p_star_high_cost = []
    p_soc_list = []
    poa_list = []
    pu_interf_list = []

    for n in n_range:
        game_low = SpectrumContentionGame(num_sus=n, reward=10.0, collision_cost=1.0, tx_energy_cost=0.5, max_pu_interference_prob=0.1)
        game_high = SpectrumContentionGame(num_sus=n, reward=10.0, collision_cost=5.0, tx_energy_cost=1.0, max_pu_interference_prob=0.1)

        p_l = game_low.compute_symmetric_nash_equilibrium()
        p_h = game_high.compute_symmetric_nash_equilibrium()
        p_soc = game_high.compute_social_optimum()
        poa = game_high.compute_price_of_anarchy()

        # Interference probability into PU
        p_interf = 1.0 - (1.0 - p_h) ** n

        p_star_low_cost.append(p_l)
        p_star_high_cost.append(p_h)
        p_soc_list.append(p_soc)
        poa_list.append(poa)
        pu_interf_list.append(p_interf)

        print(f"N = {n:3d} SUs | NE p* (Low Cost): {p_l:5.3f} | NE p* (High Cost): {p_h:5.3f} | Social Opt: {p_soc:5.3f} | PoA: {poa:4.2f} | PU Interf: {p_interf*100:4.1f}%")

    # Generate Publication Verification Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Nash Equilibrium Strategy vs Network Scale
    ax1 = axes[0, 0]
    ax1.plot(n_range, p_star_low_cost, "b-o", lw=2.0, label="NE $p^*$ (Low Collision Cost $C=1$)")
    ax1.plot(n_range, p_star_high_cost, "r-s", lw=2.0, label="NE $p^*$ (High Collision Cost $C=5$)")
    ax1.plot(n_range, p_soc_list, "g--^", lw=2.0, label="Social Optimum $p_{\\text{soc}}$")
    ax1.set_xscale("log")
    ax1.set_title("Equilibrium Access Probability $p^*$ vs. Network Size $N$", fontweight="bold")
    ax1.set_xlabel("Number of Competing Secondary Users $N$ (Log Scale)")
    ax1.set_ylabel("Transmission Access Probability $p^*$")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 0.7)

    # Plot 2: Utility Surface Heatmap for N = 5
    ax2 = axes[0, 1]
    game_n5 = SpectrumContentionGame(num_sus=5, reward=10.0, collision_cost=3.0, tx_energy_cost=1.0)
    p_grid = np.linspace(0.0, 0.8, 100)
    P_i, P_opp = np.meshgrid(p_grid, p_grid)
    U_grid = np.zeros_like(P_i)
    for i in range(100):
        for j in range(100):
            U_grid[i, j] = game_n5.compute_utility(P_i[i, j], P_opp[i, j])

    cp = ax2.contourf(P_i, P_opp, U_grid, levels=25, cmap="magma")
    cbar = plt.colorbar(cp, ax=ax2)
    cbar.set_label("Individual Expected Utility $U_i$", fontsize=9)
    # Plot symmetric diagonal and NE point
    p_star_n5 = game_n5.compute_symmetric_nash_equilibrium()
    ax2.plot([0, 0.8], [0, 0.8], "w--", lw=1.5, label="Symmetric Line ($p_i = p_{\\text{opp}}$)")
    ax2.plot(p_star_n5, p_star_n5, "c*", markersize=14, markeredgecolor="black", label=f"Nash Equilibrium ($p^*={p_star_n5:.2f}$)")
    ax2.set_title("Individual Utility Surface & Fixed Point (N = 5)", fontweight="bold")
    ax2.set_xlabel("Own Transmission Probability $p_i$")
    ax2.set_ylabel("Opponent Transmission Probability $p_{\\text{opp}}$")
    ax2.legend(loc="upper right", framealpha=0.9)

    # Plot 3: Price of Anarchy (PoA) vs Network Scale
    ax3 = axes[1, 0]
    ax3.plot(n_range, poa_list, "purple", marker="D", lw=2.2, label="Price of Anarchy (PoA)")
    ax3.axhline(1.0, color="gray", ls="--", label="Ideal Efficiency Bound (PoA = 1.0)")
    ax3.set_xscale("log")
    ax3.set_title("Price of Anarchy (PoA) vs. Network Contention", fontweight="bold")
    ax3.set_xlabel("Number of Secondary Users $N$ (Log Scale)")
    ax3.set_ylabel("PoA = Social Welfare / NE Welfare")
    ax3.grid(True, alpha=0.3, which="both")
    ax3.legend(loc="upper right")
    ax3.set_ylim(0.9, 2.5)

    # Plot 4: Primary User Interference Protection
    ax4 = axes[1, 1]
    ax4.plot(n_range, np.array(pu_interf_list) * 100.0, "teal", marker="o", lw=2.2, label="Measured PU Interference Prob (%)")
    ax4.axhline(10.0, color="crimson", ls="--", lw=2.0, label="QoS Constraint Threshold ($\epsilon_{\\text{PU}} = 10\%$)")
    ax4.set_xscale("log")
    ax4.set_title("Primary User Interference Ingress vs. Scale", fontweight="bold")
    ax4.set_xlabel("Number of Secondary Users $N$ (Log Scale)")
    ax4.set_ylabel("PU Interference Probability (%)")
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend(loc="lower right")
    ax4.set_ylim(0, 15)

    plt.suptitle("Game-Theoretic Dynamic Spectrum Sharing & Nash Equilibrium (P3-004)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[OK] Nash Equilibrium verification plot saved to {plot_path}")


if __name__ == "__main__":
    run_nash_verification()
