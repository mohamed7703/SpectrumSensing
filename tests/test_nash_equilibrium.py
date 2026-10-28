"""Unit Tests for Game-Theoretic Nash Equilibrium Dynamic Spectrum Sharing.

Tests utility function calculation, analytical Nash Equilibrium derivations,
iterative best-response convergence, and Primary User interference constraints.
"""

import pytest
import numpy as np

from src.ns3_simulation.nash_equilibrium import SpectrumContentionGame


class TestSpectrumContentionGame:
    """Test game-theoretic spectrum sharing formulation."""

    def test_utility_monotonicity_with_reward(self) -> None:
        """Higher packet success reward must yield higher expected utility."""
        game_low = SpectrumContentionGame(num_sus=5, reward=5.0)
        game_high = SpectrumContentionGame(num_sus=5, reward=20.0)

        u_low = game_low.compute_utility(p_i=0.2, p_opponents=0.2)
        u_high = game_high.compute_utility(p_i=0.2, p_opponents=0.2)
        assert u_high > u_low

    def test_utility_decreases_with_collision_penalty(self) -> None:
        """Higher collision penalty must reduce utility under high contention."""
        game_c1 = SpectrumContentionGame(num_sus=5, collision_cost=1.0)
        game_c5 = SpectrumContentionGame(num_sus=5, collision_cost=10.0)

        u_c1 = game_c1.compute_utility(p_i=0.5, p_opponents=0.5)
        u_c5 = game_c5.compute_utility(p_i=0.5, p_opponents=0.5)
        assert u_c1 > u_c5

    def test_analytical_nash_equilibrium_bounds(self) -> None:
        """Equilibrium probability must be bounded in (0, 1) and scale down with N."""
        p_stars = []
        for n in [2, 5, 10, 20, 50]:
            game = SpectrumContentionGame(num_sus=n, reward=10.0, collision_cost=2.0, tx_energy_cost=0.5)
            p_star = game.compute_symmetric_nash_equilibrium()
            assert 0.0 < p_star < 1.0
            p_stars.append(p_star)

        # As number of competing SUs increases, individual access probability decreases
        for i in range(len(p_stars) - 1):
            assert p_stars[i] > p_stars[i + 1]

    def test_iterative_best_response_convergence(self) -> None:
        """Iterative best response must converge stably to the equilibrium."""
        game = SpectrumContentionGame(num_sus=5, reward=12.0, collision_cost=3.0, tx_energy_cost=1.0)
        p_star = game.compute_symmetric_nash_equilibrium()
        p_iter, num_iters, trajectory = game.solve_best_response_iterative(initial_p=0.8, tol=1e-4)

        assert num_iters < 100
        # Should converge close to analytical optimum
        assert np.isclose(p_iter, p_star, atol=0.05)

    def test_pu_protection_constraint_enforced(self) -> None:
        """Equilibrium strategy must strictly obey max PU interference probability."""
        eps_pu = 0.05
        game = SpectrumContentionGame(num_sus=10, max_pu_interference_prob=eps_pu)
        p_star = game.compute_symmetric_nash_equilibrium()

        # Cumulative interference probability: 1 - (1 - p*)^N <= eps_pu
        p_interf = 1.0 - (1.0 - p_star) ** game.n
        assert p_interf <= eps_pu + 1e-6

    def test_price_of_anarchy_ge_unity(self) -> None:
        """Price of Anarchy (PoA = Welfare_soc / Welfare_ne) must be >= 1.0."""
        game = SpectrumContentionGame(num_sus=4, reward=10.0, collision_cost=2.0, tx_energy_cost=1.0)
        poa = game.compute_price_of_anarchy()
        assert poa >= 0.999
