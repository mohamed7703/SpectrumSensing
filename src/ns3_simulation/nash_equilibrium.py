r"""Game-Theoretic Nash Equilibrium Analysis for Dynamic Spectrum Sharing.

Models contention among $N$ Secondary Users as a non-cooperative game,
computes individual utility functions with collision penalties, derives the
symmetric Nash Equilibrium access probability $p^*$, and solves for the
best-response dynamics under Primary User protection constraints.

Mathematical Notation:
    For $N$ SUs choosing access probabilities $\mathbf{p} = (p_1, \dots, p_N)$,
    the expected utility of SU $i$ given symmetric opponent strategy $p$ is:
    $$U_i(p_i, p) = R \cdot p_i (1 - p)^{N-1} - C \cdot p_i \left[1 - (1 - p)^{N-1}\right] - C_{\text{tx}} p_i$$
    where:
    - $R > 0$: Reward gained per successful packet transmission.
    - $C \ge 0$: Penalty incurred per collision with another SU.
    - $C_{\text{tx}} \ge 0$: Energy cost per transmission attempt.

    The unique symmetric Nash Equilibrium strategy $p^*$ satisfies:
    $$\frac{\partial U_i}{\partial p_i}\Bigg|_{p_i = p = p^*} = 0 \implies p^* = 1 - \left( \frac{C + C_{\text{tx}}}{R + C} \right)^{\frac{1}{N-1}}$$
    subject to the Primary User interference constraint:
    $$P_{\text{interf}}(\mathbf{p}^*) = 1 - (1 - p^*)^N \le \epsilon_{\text{PU}}$$
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SpectrumContentionGame:
    """Non-cooperative game-theoretic model for CRN spectrum access."""

    def __init__(
        self,
        num_sus: int = 10,
        reward: float = 10.0,
        collision_cost: float = 3.0,
        tx_energy_cost: float = 1.0,
        pu_activity_factor: float = 0.4,
        max_pu_interference_prob: float = 0.1,
    ) -> None:
        r"""Initialize spectrum contention game.

        Args:
            num_sus: Number of competing Secondary Users ($N \ge 2$).
            reward: Payoff $R$ per successfully received packet.
            collision_cost: Penalty $C$ per collision with other SUs.
            tx_energy_cost: Physical transmission energy cost $C_{\text{tx}}$.
            pu_activity_factor: Probability PU is active in the band.
            max_pu_interference_prob: Maximum allowable interference into PU ($\epsilon_{\text{PU}}$).
        """
        if num_sus < 2:
            raise ValueError(f"num_sus must be >= 2, got {num_sus}")
        if reward <= 0:
            raise ValueError(f"reward must be > 0, got {reward}")

        self.n = num_sus
        self.r = float(reward)
        self.c = float(collision_cost)
        self.c_tx = float(tx_energy_cost)
        self.pu_activity = float(pu_activity_factor)
        self.eps_pu = float(max_pu_interference_prob)

    def compute_utility(self, p_i: float, p_opponents: float) -> float:
        """Compute expected utility of SU $i$ given opponent transmission probability.

        Args:
            p_i: Transmission probability of SU $i$ in [0, 1].
            p_opponents: Transmission probability of each of the other $N-1$ SUs.

        Returns:
            Expected utility $U_i(p_i, p_{\text{opp}})$.
        """
        p_i = float(np.clip(p_i, 0.0, 1.0))
        p_opp = float(np.clip(p_opponents, 0.0, 1.0))

        # Probability no other SU transmits
        p_no_su_collision = (1.0 - p_opp) ** (self.n - 1)

        # Expected reward from successful transmissions (only when PU is idle)
        p_pu_idle = 1.0 - self.pu_activity
        expected_reward = self.r * p_i * p_no_su_collision * p_pu_idle

        # Expected cost from collisions
        p_collision = p_i * (1.0 - p_no_su_collision)
        expected_collision_cost = self.c * p_collision

        # Transmission energy cost
        tx_cost = self.c_tx * p_i

        return expected_reward - expected_collision_cost - tx_cost

    def compute_symmetric_nash_equilibrium(self) -> float:
        r"""Compute the analytical unique symmetric Nash Equilibrium strategy $p^*$.

        $$p^* = 1 - \left( \frac{C + C_{\text{tx}}}{R(1 - \rho_{\text{PU}}) + C} \right)^{\frac{1}{N-1}}$$
        """
        effective_r = self.r * (1.0 - self.pu_activity)
        ratio = (self.c + self.c_tx) / (effective_r + self.c)

        if ratio >= 1.0:
            # Transmission cost exceeds maximum possible reward -> no transmission
            return 0.0

        p_star = 1.0 - (ratio ** (1.0 / (self.n - 1)))
        p_star = float(np.clip(p_star, 0.0, 1.0))

        # Enforce PU protection constraint: 1 - (1 - p*)^N <= eps_pu
        # => (1 - p*)^N >= 1 - eps_pu => p* <= 1 - (1 - eps_pu)^(1/N)
        p_max_pu = 1.0 - (1.0 - self.eps_pu) ** (1.0 / self.n)
        return float(min(p_star, p_max_pu))

    def marginal_utility(self, p_opponents: float) -> float:
        r"""Compute marginal utility $\frac{\partial U_i}{\partial p_i}$ at opponent belief $p_{\text{opp}}$."""
        p_opp = float(np.clip(p_opponents, 0.0, 1.0))
        p_no_su = (1.0 - p_opp) ** (self.n - 1)
        p_pu_idle = 1.0 - self.pu_activity
        return self.r * p_no_su * p_pu_idle - self.c * (1.0 - p_no_su) - self.c_tx

    def solve_best_response_iterative(
        self,
        initial_p: float = 0.5,
        learning_rate: float = 0.05,
        max_iterations: int = 150,
        tol: float = 1e-4,
    ) -> Tuple[float, int, List[float]]:
        """Find Nash Equilibrium via learning gradient dynamics.

        Args:
            initial_p: Initial belief about opponent strategy.
            learning_rate: Gradient ascent step size.
            max_iterations: Maximum update rounds.
            tol: Convergence tolerance $|p^{(k+1)} - p^{(k)}| < \text{tol}$.

        Returns:
            Tuple of (equilibrium_p, num_iterations, trajectory_history).
        """
        p_current = float(initial_p)
        trajectory = [p_current]

        for it in range(max_iterations):
            grad = self.marginal_utility(p_current)
            p_next = float(np.clip(p_current + learning_rate * grad, 0.0, 1.0))

            # Enforce PU protection constraint
            p_max_pu = 1.0 - (1.0 - self.eps_pu) ** (1.0 / self.n)
            p_next = min(p_next, p_max_pu)
            trajectory.append(p_next)

            if abs(p_next - p_current) < tol:
                return p_next, it + 1, trajectory

            p_current = p_next

        return p_current, max_iterations, trajectory

    def compute_social_optimum(self) -> float:
        r"""Compute the socially optimal transmission probability maximizing total welfare.

        $$\mathbf{p}_{\text{soc}} = \arg\max_p \sum_{i=1}^N U_i(p, p) = \arg\max_p N \cdot U(p, p)$$
        """
        candidates = np.linspace(0.0, 1.0, 2000)
        social_welfares = [self.n * self.compute_utility(p, p) for p in candidates]
        best_p = float(candidates[np.argmax(social_welfares)])
        return best_p

    def compute_price_of_anarchy(self) -> float:
        r"""Compute the Price of Anarchy (PoA) quantifying efficiency loss from selfish contention.

        $$\text{PoA} = \frac{\text{Social Welfare}(\mathbf{p}_{\text{soc}})}{\text{Social Welfare}(\mathbf{p}^*)}$$
        """
        p_ne = self.compute_symmetric_nash_equilibrium()
        p_soc = self.compute_social_optimum()

        w_ne = self.n * self.compute_utility(p_ne, p_ne)
        w_soc = self.n * self.compute_utility(p_soc, p_soc)

        if w_ne <= 0 or w_soc <= 0:
            return 1.0

        return float(w_soc / w_ne)
