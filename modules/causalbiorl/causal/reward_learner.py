"""
Adaptive Reward Learner — hindsight reward shaping for multi-objective RL.

In drug discovery, the agent must optimise multiple objectives
simultaneously:

    1. Binding affinity   (DockBot / surrogate)
    2. Drug-likeness      (QED)
    3. Synthetic access.  (SA score)
    4. Novelty            (Tanimoto distance to known drugs)
    5. Causal confidence  (from SCM do-calculus)
    6. Structural stab.   (MirrorFold therapeutic score)

A naïve approach uses fixed weights:  ``r = Σ wᵢ rᵢ``.
But fixed weights:
    - Collapse to the easiest objective (the agent games it)
    - Ignore disease-specific difficulty profiles
    - Require manual tuning per disease

**Hindsight Reward Shaping (Option C)**:
    - A critic network estimates objective-specific difficulty
    - Objectives that are underperforming get up-weighted
    - Objectives that are easily satisfied get down-weighted
    - The agent naturally focuses on the bottleneck

This is a form of **curriculum learning** applied to reward
composition — the agent adapts its focus as it improves.

Reference:
    Adapted from Abels et al. (2019) "Dynamic Weights in
    Multi-Objective Deep Reinforcement Learning", ICML.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
#  Objective Definitions                                                     #
# ────────────────────────────────────────────────────────────────────────── #


OBJECTIVE_NAMES = [
    "binding",     # Binding affinity (kcal/mol, normalised)
    "qed",         # Quantitative Estimate of Drug-likeness [0,1]
    "sa",          # Synthetic Accessibility (normalised 0-1, higher=easier)
    "novelty",     # Tanimoto novelty vs known drugs [0,1]
    "causal",      # Causal confidence from SCM [0,1]
    "stability",   # MirrorFold therapeutic score [0,1]
]

N_OBJECTIVES = len(OBJECTIVE_NAMES)


def normalise_binding(affinity: float | None) -> float:
    """Normalise binding affinity to [0, 1] where higher = better.

    Vina scores are negative (better) kcal/mol.  Typical range
    is -12 (excellent) to 0 (no binding).

    Mapping: ``score = clamp(-affinity / 12, 0, 1)``
    """
    if affinity is None:
        return 0.3  # neutral prior
    return float(np.clip(-affinity / 12.0, 0.0, 1.0))


def normalise_sa(sa_score: float | None) -> float:
    """Normalise SA score to [0, 1] where higher = easier to synthesise.

    SA scores range from 1 (easy) to 10 (hard).
    Mapping: ``score = (10 - sa) / 9``
    """
    if sa_score is None:
        return 0.5
    return float(np.clip((10.0 - sa_score) / 9.0, 0.0, 1.0))


# ────────────────────────────────────────────────────────────────────────── #
#  Per-Objective Critic                                                      #
# ────────────────────────────────────────────────────────────────────────── #


class _ObjectiveCritic(nn.Module):
    """Small MLP that estimates expected return for one objective.

    Predicts ``V_i(s)`` — the expected value of objective ``i``
    given the current state.
    """

    def __init__(self, state_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────── #
#  Adaptive Reward Learner                                                   #
# ────────────────────────────────────────────────────────────────────────── #


class AdaptiveRewardLearner:
    """Learns dynamic reward weights via per-objective difficulty estimation.

    Architecture:
        - One critic per objective: ``V_i(state) → expected_score_i``
        - Difficulty = ``1 - V_i(state)`` (how far from perfect)
        - Weight for objective ``i`` ∝ ``difficulty_i^temperature``
        - Weights are normalised to sum to 1

    The ``temperature`` parameter controls adaptation strength:
        - ``temperature = 0`` → equal weights (fixed)
        - ``temperature = 1`` → linear difficulty weighting
        - ``temperature > 1`` → aggressive focus on bottleneck

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector (graph embedding + mol features).
    base_weights : ndarray | None
        Initial (prior) weights for each objective.  If ``None``,
        uses domain-informed defaults.
    temperature : float
        Controls how aggressively to up-weight bottleneck objectives.
    lr : float
        Learning rate for critic training.
    gamma : float
        Discount factor for temporal difference learning.
    device : str | None
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        state_dim: int = 128,
        base_weights: NDArray[np.floating] | None = None,
        temperature: float = 1.5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.temperature = temperature
        self.gamma = gamma

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Base (prior) weights — domain-informed
        if base_weights is not None:
            self.base_weights = np.array(base_weights, dtype=np.float32)
        else:
            # Default: binding + causal get slight prior emphasis
            self.base_weights = np.array(
                [0.25, 0.15, 0.10, 0.10, 0.25, 0.15],
                dtype=np.float32,
            )

        assert len(self.base_weights) == N_OBJECTIVES
        self.base_weights /= self.base_weights.sum()  # normalise

        # Per-objective critics
        self.critics = nn.ModuleList([
            _ObjectiveCritic(state_dim).to(self.device)
            for _ in range(N_OBJECTIVES)
        ])

        self.optimiser = torch.optim.Adam(self.critics.parameters(), lr=lr)

        # Running statistics for logging
        self._weight_history: list[NDArray[np.floating]] = []
        self._objective_history: list[NDArray[np.floating]] = []

    # ------------------------------------------------------------------ #
    #  Reward Computation                                                  #
    # ------------------------------------------------------------------ #

    def compute_reward(
        self,
        state: NDArray[np.floating],
        objective_scores: dict[str, float],
    ) -> float:
        """Compute the adaptively-weighted scalar reward.

        Parameters
        ----------
        state : ndarray
            Current state vector (graph embedding + mol features).
        objective_scores : dict[str, float]
            Per-objective scores, keyed by objective name.
            All values should be in [0, 1] (higher = better).

        Returns
        -------
        float
            Scalar reward.
        """
        weights = self._compute_weights(state)
        scores = np.array([
            objective_scores.get(name, 0.0)
            for name in OBJECTIVE_NAMES
        ], dtype=np.float32)

        # Track history
        self._weight_history.append(weights.copy())
        self._objective_history.append(scores.copy())

        return float(np.dot(weights, scores))

    def _compute_weights(
        self,
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute dynamic weights based on objective difficulty.

        Returns
        -------
        ndarray ``(N_OBJECTIVES,)`` summing to 1.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        # Predict expected score for each objective
        with torch.no_grad():
            expected = np.array([
                critic(state_t).item()
                for critic in self.critics
            ], dtype=np.float32)

        # Difficulty = 1 - expected_score (how far from perfect)
        difficulty = 1.0 - expected

        # Combine base weights with difficulty
        # Higher difficulty → higher weight
        raw_weights = self.base_weights * (difficulty ** self.temperature + 1e-6)

        # Normalise
        weights = raw_weights / raw_weights.sum()
        return weights

    # ------------------------------------------------------------------ #
    #  Critic Training                                                     #
    # ------------------------------------------------------------------ #

    def update(
        self,
        state: NDArray[np.floating],
        objective_scores: dict[str, float],
        next_state: NDArray[np.floating] | None = None,
    ) -> float:
        """Update objective critics with observed scores.

        Uses a simple TD(0) update:
            ``V_i(s) ← r_i + γ · V_i(s')``

        Parameters
        ----------
        state : ndarray
            Current state.
        objective_scores : dict[str, float]
            Observed objective scores.
        next_state : ndarray | None
            Next state (for TD target).  If ``None``, treats as terminal.

        Returns
        -------
        float
            Mean critic loss across objectives.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        if next_state is not None:
            next_state_t = torch.tensor(
                next_state, dtype=torch.float32, device=self.device,
            ).unsqueeze(0)
        else:
            next_state_t = None

        total_loss = torch.tensor(0.0, device=self.device)

        for i, name in enumerate(OBJECTIVE_NAMES):
            r_i = objective_scores.get(name, 0.0)

            # TD target
            if next_state_t is not None:
                with torch.no_grad():
                    v_next = self.critics[i](next_state_t)
                target = r_i + self.gamma * v_next
            else:
                target = torch.tensor(r_i, device=self.device).unsqueeze(0)

            # Prediction
            v_pred = self.critics[i](state_t)
            loss = nn.functional.mse_loss(v_pred, target)
            total_loss = total_loss + loss

        self.optimiser.zero_grad()
        total_loss.backward()
        self.optimiser.step()

        return total_loss.item() / N_OBJECTIVES

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_current_weights(
        self,
        state: NDArray[np.floating],
    ) -> dict[str, float]:
        """Return current adaptive weights as a readable dict."""
        weights = self._compute_weights(state)
        return {
            name: float(w)
            for name, w in zip(OBJECTIVE_NAMES, weights)
        }

    def get_weight_history(self) -> dict[str, list[float]]:
        """Return weight trajectory for analysis/plotting."""
        if not self._weight_history:
            return {name: [] for name in OBJECTIVE_NAMES}

        arr = np.array(self._weight_history)
        return {
            name: arr[:, i].tolist()
            for i, name in enumerate(OBJECTIVE_NAMES)
        }

    def get_objective_history(self) -> dict[str, list[float]]:
        """Return per-objective score trajectory."""
        if not self._objective_history:
            return {name: [] for name in OBJECTIVE_NAMES}

        arr = np.array(self._objective_history)
        return {
            name: arr[:, i].tolist()
            for i, name in enumerate(OBJECTIVE_NAMES)
        }

    def get_difficulty_profile(
        self,
        state: NDArray[np.floating],
    ) -> dict[str, float]:
        """Return the estimated difficulty for each objective.

        Values in [0, 1] — higher = harder to satisfy.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            expected = np.array([
                critic(state_t).item()
                for critic in self.critics
            ], dtype=np.float32)

        difficulty = 1.0 - expected
        return {
            name: float(d)
            for name, d in zip(OBJECTIVE_NAMES, difficulty)
        }
