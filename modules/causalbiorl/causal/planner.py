"""
Causal planner — action selection via do-calculus on the learned SCM.

Given an SCM, the planner evaluates candidate actions by simulating
their *interventional* effects (Pearl's do-operator) and selects the
action that maximises the expected reward.

Two planning strategies are provided:

* **Grid search** — discretise the action space and evaluate each point.
* **CEM (Cross-Entropy Method)** — iterative sampling for high-dimensional
  action spaces.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from modules.causalbiorl.causal.scm import StructuralCausalModel


class CausalPlanner:
    """Select actions by reasoning about interventions through the SCM.

    Parameters
    ----------
    scm : StructuralCausalModel
        Fitted structural causal model.
    reward_fn : callable
        ``reward_fn(state, action) → float``.  Used to score predicted
        outcomes.
    method : ``"grid"`` | ``"cem"``
        Planning strategy.
    action_dim : int
        Dimensionality of the action space.
    n_samples : int
        For grid: total grid points.  For CEM: samples per iteration.
    horizon : int
        Multi-step look-ahead depth (default 1 = greedy).
    cem_iterations : int
        Number of CEM refinement rounds.
    cem_elite_frac : float
        Fraction of top samples retained in CEM.
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        reward_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float],
        action_dim: int,
        method: Literal["grid", "cem"] = "cem",
        n_samples: int = 200,
        horizon: int = 1,
        cem_iterations: int = 5,
        cem_elite_frac: float = 0.1,
    ) -> None:
        self.scm = scm
        self.reward_fn = reward_fn
        self.action_dim = action_dim
        self.method = method
        self.n_samples = n_samples
        self.horizon = horizon
        self.cem_iterations = cem_iterations
        self.cem_elite_frac = cem_elite_frac

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def plan(
        self,
        state: NDArray[np.floating],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        """Return the best action for the given state.

        Uses the SCM's ``do`` operator to predict the interventional
        effect of each candidate action.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.method == "grid":
            return self._plan_grid(state)
        return self._plan_cem(state, rng)

    # ------------------------------------------------------------------ #
    #  Grid search (low-dim actions)                                       #
    # ------------------------------------------------------------------ #

    def _plan_grid(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        # Uniform grid over [0, 1]^d
        n_per_dim = max(int(self.n_samples ** (1.0 / self.action_dim)), 2)
        grids = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(self.action_dim)]
        mesh = np.meshgrid(*grids, indexing="ij")
        candidates = np.stack([g.ravel() for g in mesh], axis=-1).astype(np.float32)

        best_reward = -np.inf
        best_action = candidates[0]

        for action in candidates:
            reward = self._rollout(state, action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action

    # ------------------------------------------------------------------ #
    #  Cross-Entropy Method (high-dim actions)                             #
    # ------------------------------------------------------------------ #

    def _plan_cem(
        self,
        state: NDArray[np.floating],
        rng: np.random.Generator,
    ) -> NDArray[np.floating]:
        mean = np.full(self.action_dim, 0.5, dtype=np.float32)
        std = np.full(self.action_dim, 0.25, dtype=np.float32)
        n_elite = max(int(self.n_samples * self.cem_elite_frac), 1)

        for _ in range(self.cem_iterations):
            samples = rng.normal(loc=mean, scale=std, size=(self.n_samples, self.action_dim))
            samples = np.clip(samples, 0.0, 1.0).astype(np.float32)

            rewards = np.array([self._rollout(state, a) for a in samples])
            elite_idx = np.argsort(rewards)[-n_elite:]
            elite = samples[elite_idx]

            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6  # prevent collapse

        return mean

    # ------------------------------------------------------------------ #
    #  Multi-step rollout                                                  #
    # ------------------------------------------------------------------ #

    def _rollout(self, state: NDArray[np.floating], action: NDArray[np.floating]) -> float:
        """Simulate *horizon* steps with constant action, summing reward."""
        total_reward = 0.0
        s = state.copy()
        for _ in range(self.horizon):
            # Build intervention dict — action dimensions
            intervention: dict[str, float] = {}
            action_start = self.scm.state_dim
            for k in range(self.action_dim):
                idx = action_start + k
                if idx < len(self.scm.all_names):
                    intervention[self.scm.all_names[idx]] = float(action[k])

            next_s = self.scm.do(intervention, s, action)
            total_reward += self.reward_fn(next_s, action)
            s = next_s
        return total_reward


# -------------------------------------------------------------------- #
#  Hierarchical Planner (Drug Discovery)                                #
# -------------------------------------------------------------------- #


class HierarchicalPlanner:
    """Two-level planner for drug discovery environments.

    Level 1: **Target selection** — discrete choice over identified
    causal targets.  Uses upper confidence bound (UCB) on per-target
    expected reward to balance exploration vs exploitation.

    Level 2: **Molecule generation** — continuous delta-z vector in
    GenMol’s latent space.  Uses CEM (Cross-Entropy Method) conditioned
    on the chosen target’s embedding.

    Parameters
    ----------
    scm : StructuralCausalModel | None
        Fitted SCM for causal planning.  If ``None``, target selection
        falls back to UCB without causal reasoning.
    reward_fn : callable
        ``(state, action) → float``.
    n_targets : int
        Number of available targets.
    latent_dim : int
        GenMol latent space dimension.
    cem_samples : int
        CEM samples per iteration.
    cem_iterations : int
        CEM refinement rounds.
    cem_elite_frac : float
        Fraction of top samples retained.
    ucb_c : float
        UCB exploration constant.
    """

    def __init__(
        self,
        scm: StructuralCausalModel | None = None,
        reward_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float] | None = None,
        n_targets: int = 5,
        latent_dim: int = 128,
        cem_samples: int = 200,
        cem_iterations: int = 5,
        cem_elite_frac: float = 0.1,
        ucb_c: float = 2.0,
    ) -> None:
        self.scm = scm
        self.reward_fn = reward_fn
        self.n_targets = n_targets
        self.latent_dim = latent_dim
        self.cem_samples = cem_samples
        self.cem_iterations = cem_iterations
        self.cem_elite_frac = cem_elite_frac
        self.ucb_c = ucb_c

        # Per-target statistics for UCB
        self._target_counts = np.zeros(n_targets, dtype=np.float64)
        self._target_rewards = np.zeros(n_targets, dtype=np.float64)
        self._total_steps = 0

    def plan(
        self,
        state: NDArray[np.floating],
        target_embeddings: NDArray[np.floating] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        """Select target + generate molecule (full hierarchical action).

        Returns
        -------
        ndarray of shape ``(2 + latent_dim,)``
            ``[target_selector, stop_signal, delta_z_0, …, delta_z_127]``
        """
        if rng is None:
            rng = np.random.default_rng()

        # Level 1: Select target via UCB
        target_idx = self._select_target_ucb(rng)

        # Level 2: Generate delta-z via CEM
        delta_z = self._plan_molecule_cem(
            state, target_idx, target_embeddings, rng,
        )

        # Compose hierarchical action
        action = np.zeros(2 + self.latent_dim, dtype=np.float32)
        # Map target index to continuous action in [-1, 1]
        action[0] = (2.0 * target_idx / max(self.n_targets - 1, 1)) - 1.0
        action[1] = -1.0  # no stop signal
        action[2:] = delta_z

        return action

    def update(
        self,
        target_idx: int,
        reward: float,
    ) -> None:
        """Update target statistics after observing a reward."""
        if 0 <= target_idx < self.n_targets:
            self._target_counts[target_idx] += 1
            self._target_rewards[target_idx] += reward
            self._total_steps += 1

    def _select_target_ucb(
        self,
        rng: np.random.Generator,
    ) -> int:
        """Select target using Upper Confidence Bound (UCB1).

        UCB1: ``x_bar_i + c * sqrt(ln(t) / n_i)``

        Balances exploitation (high mean reward targets) with
        exploration (under-visited targets).
        """
        if self._total_steps < self.n_targets:
            # Visit each target at least once
            unvisited = np.where(self._target_counts == 0)[0]
            return int(rng.choice(unvisited))

        mean_rewards = self._target_rewards / np.maximum(self._target_counts, 1)
        exploration = self.ucb_c * np.sqrt(
            np.log(self._total_steps + 1) / np.maximum(self._target_counts, 1)
        )
        ucb_scores = mean_rewards + exploration
        return int(np.argmax(ucb_scores))

    def _plan_molecule_cem(
        self,
        state: NDArray[np.floating],
        target_idx: int,
        target_embeddings: NDArray[np.floating] | None,
        rng: np.random.Generator,
    ) -> NDArray[np.floating]:
        """CEM planning in GenMol’s latent space.

        Generates candidate delta-z vectors and scores them using
        the reward function.  Iteratively refines the distribution
        toward high-reward regions.
        """
        if self.reward_fn is None:
            return rng.standard_normal(self.latent_dim).astype(np.float32) * 0.3

        mean = np.zeros(self.latent_dim, dtype=np.float32)
        std = np.full(self.latent_dim, 0.3, dtype=np.float32)
        n_elite = max(int(self.cem_samples * self.cem_elite_frac), 1)

        for _ in range(self.cem_iterations):
            samples = rng.normal(
                loc=mean, scale=std,
                size=(self.cem_samples, self.latent_dim),
            ).astype(np.float32)
            samples = np.clip(samples, -1.0, 1.0)

            # Score each candidate
            rewards = np.zeros(self.cem_samples, dtype=np.float32)
            for j, dz in enumerate(samples):
                # Build a candidate action
                action = np.zeros(2 + self.latent_dim, dtype=np.float32)
                action[0] = (2.0 * target_idx / max(self.n_targets - 1, 1)) - 1.0
                action[1] = -1.0
                action[2:] = dz
                rewards[j] = self.reward_fn(state, action)

            # Select elite
            elite_idx = np.argsort(rewards)[-n_elite:]
            elite = samples[elite_idx]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6

        return mean
