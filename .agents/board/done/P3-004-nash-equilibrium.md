---
id: P3-004
title: "Implement Game-Theoretic Nash Equilibrium Spectrum Sharing"
phase: 3 - Network Simulation
status: done
priority: high
depends_on: [P3-003]
tags: [game-theory, nash-equilibrium, spectrum-sharing, secondary-users]
---

## Objective
Model Secondary User spectrum contention as a non-cooperative game, derive and verify the
Nash Equilibrium where SU transmission probabilities optimize collective utility while strictly
bounding PU interference probability.

## Acceptance Criteria
- [x] Game-theoretic spectrum contention model in `/src/ns3_simulation/nash_equilibrium.py`
- [x] Utility function calculation incorporating reward per successful packet minus collision penalty
- [x] Analytical and iterative best-response solver converging to unique Nash Equilibrium
- [x] Verification that equilibrium strategy prevents Primary User degradation
- [x] Unit tests in `/tests/test_nash_equilibrium.py`
