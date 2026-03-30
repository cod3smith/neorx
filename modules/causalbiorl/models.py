"""
Pydantic data models for CausalBioRL configuration and results.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EnvConfig(BaseModel):
    """Configuration for a CausalBioRL environment."""

    env_id: str = Field(..., description="Gymnasium environment ID, e.g. 'GeneticToggle-v0'")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    extra_kwargs: dict[str, object] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    agent_type: Literal["causal", "ppo", "sac", "random"] = "causal"
    n_episodes: int = 500
    seed: int | None = None
    extra_kwargs: dict[str, object] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Configuration for a full benchmark run."""

    envs: list[EnvConfig] = Field(default_factory=list)
    agents: list[AgentConfig] = Field(default_factory=list)
    n_seeds: int = 10
    output_dir: str = "results"


class EpisodeResult(BaseModel):
    """Result of a single training run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_id: str
    agent_type: str
    seed: int
    episode_rewards: list[float]
    episode_lengths: list[int]
    total_steps: int


class BenchmarkResult(BaseModel):
    """Aggregated results of a benchmark experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_id: str
    agent_type: str
    mean_reward: float
    std_reward: float
    mean_total_steps: float
    std_total_steps: float
    per_seed: list[EpisodeResult] = Field(default_factory=list)


class TrainConfig(BaseModel):
    """CLI training configuration."""

    env: str = "GeneticToggle-v0"
    agent: Literal["causal", "ppo", "sac", "random"] = "causal"
    episodes: int = 500
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    seed: int | None = None
    output_dir: str = "results"


class DrugDiscoveryConfig(BaseModel):
    """Configuration for the DrugDiscovery-v0 environment.

    Extends the standard env config with drug-discovery-specific
    parameters: disease name, target budget, surrogate usage, etc.
    """

    disease: str = Field("Malaria", description="Disease to target")
    top_n_targets: int = Field(5, ge=1, le=10, description="Max causal targets")
    max_steps: int = Field(50, ge=10, description="Generate-screen budget per episode")
    latent_dim: int = Field(128, description="GenMol latent space dimension")
    use_surrogate: bool = Field(True, description="Use surrogate docking (fast) vs real DockBot")
    recalibrate_interval: int = Field(10, ge=1, description="Recalibrate surrogate every N steps")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    agent: Literal["causal", "ppo", "sac", "random"] = "causal"
    n_episodes: int = Field(100, ge=1, description="Training episodes")
    seed: int | None = None
    output_dir: str = "results"
