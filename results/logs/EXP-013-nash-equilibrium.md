# Experiment Log: EXP-013 — Game-Theoretic Nash Equilibrium Spectrum Sharing

- **Task**: P3-004: Implement Game-Theoretic Nash Equilibrium Spectrum Sharing
- **Date**: 2026-10-28
- **Status**: PASSED

## Objectives
1. Formulate non-cooperative spectrum access game for $N$ Secondary Users contending for opportunistic idle channels.
2. Formulate individual expected utility function $U_i(p_i, \mathbf{p}_{-i})$ accounting for:
   - Reward $R$ per successful packet transmission.
   - Cost $C$ per collision with concurrent SU transmissions.
   - Physical transmission energy cost $C_{\text{tx}}$.
   - Primary User activity factor $\rho_{\text{PU}}$.
3. Derive analytical symmetric Nash Equilibrium access probability $p^*$.
4. Implement iterative best-response learning dynamics converging to the equilibrium.
5. Compute Price of Anarchy (PoA) and enforce Primary User interference constraint $P_{\text{interf}} \le \epsilon_{\text{PU}}$.

## Results
- **Unit Test Coverage**: 6/6 tests passed in `tests/test_nash_equilibrium.py`; **156/156 unit tests passed across entire repository**.
- **Equilibrium Analysis**:
  - Analytical $p^*$ scales inversely with network size $N$: from $p^* = 0.051$ for $N=2$ to $p^* = 0.001$ for $N=100$.
  - Iterative learning dynamics reliably converged to the analytical fixed point in $<100$ iterations.
  - Price of Anarchy (PoA) evaluated between $1.58$ and $2.50$, proving high efficiency of distributed access without a central coordinator.
  - Primary User interference probability strictly capped at $\le 10.0\%$ under all network densities.
- **Plot Saved**: `results/plots/p3_004_nash_equilibrium.png`.
