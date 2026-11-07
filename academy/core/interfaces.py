"""METARGON Academy — core interfaces (Phase 1).

Strict separation contract:
  * ``Environment`` exposes reset/observe/act/reward/termination only.
  * ``TaskGenerator`` produces episode specifications (configs are opaque
    to agents; only public facts and difficulty metadata are exposed).
  * ``StudentAgent`` observes, decides, learns, and can persist/restore
    its internal learning state.
  * ``CurriculumManager`` selects training tasks from measured progress.
  * ``ExperienceMemory`` stores and replays bounded experiences.
  * ``Evaluator`` measures policies on frozen task sets.
  * ``CheckpointManager`` writes/reads reproducible snapshots.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Environment(abc.ABC):
    @abc.abstractmethod
    def reset(self, ep_seed: int) -> object:
        """Start an episode; return the first public observation."""

    @abc.abstractmethod
    def step(self, action: str) -> object:
        """Apply an action; return a StepResult (observation, valid,
        events, terminated)."""

    @abc.abstractmethod
    def observe(self) -> object:
        """Current public observation."""

    @property
    @abc.abstractmethod
    def geom(self) -> object:
        """Public static world geometry (walls, cells, goal, start)."""


# ---------------------------------------------------------------------------
# Task generator / episode spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    env_config: object                # EnvConfig (opaque to agents)
    max_steps: int
    noise_q: float
    spec_seed: int
    pool_tag: str
    difficulty: object = field(default_factory=lambda: None)


class TaskGenerator(abc.ABC):
    @abc.abstractmethod
    def sample(self, difficulty_bin: Optional[str] = None) -> TaskSpec:
        """Sample a training task (optionally constrained to a difficulty
        bin)."""

    @abc.abstractmethod
    def holdout(self, kind: str = "test") -> list:
        """Frozen evaluation pool: "val" | "test" | "ood"."""


# ---------------------------------------------------------------------------
# Student agent
# ---------------------------------------------------------------------------
class StudentAgent(abc.ABC):
    name: str = "agent"

    @abc.abstractmethod
    def episode_start(self, spec: TaskSpec, geom: object = None) -> None:
        """Called at the start of each episode. Persistent learning state
        (memory, model) must survive across episodes. `geom` is the public
        static world geometry (never hidden rule state)."""

    @abc.abstractmethod
    def act(self, obs) -> str:
        """Return an action for the current public observation."""

    @abc.abstractmethod
    def learn(self, obs_before, action: str, result) -> None:
        """Consume feedback from the last step (online update)."""

    @abc.abstractmethod
    def save_state(self) -> dict:
        """Serialize the full persistent learning state."""

    @abc.abstractmethod
    def load_state(self, state: dict) -> None:
        """Restore persistent learning state from a snapshot."""

    @abc.abstractmethod
    def diagnostics(self) -> dict:
        """Training-state telemetry (used by the loop and evaluator)."""


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
class CurriculumManager(abc.ABC):
    @abc.abstractmethod
    def record_episode(self, spec: TaskSpec, success: bool, steps: int,
                       info: dict) -> None:
        """Feed measured outcome of a finished training episode."""

    @abc.abstractmethod
    def propose(self, rng: random.Random) -> Optional[str]:
        """Choose the next difficulty bin based on measured progress."""

    @abc.abstractmethod
    def progress(self) -> dict:
        """Measured progress per bin (success, confidence, count)."""

    @abc.abstractmethod
    def save_state(self) -> dict:
        """Serialize curriculum state."""

    @abc.abstractmethod
    def load_state(self, state: dict) -> None:
        """Restore curriculum state."""


# ---------------------------------------------------------------------------
# Experience memory
# ---------------------------------------------------------------------------
@dataclass
class Experience:
    obs_before: object
    action: str
    obs_after: object
    reward: float
    terminated: bool
    events: tuple = ()
    step_index: int = 0
    spec_seed: int = 0
    pool_tag: str = ""


class ExperienceMemory(abc.ABC):
    @abc.abstractmethod
    def push(self, exp: Experience) -> None:
        """Store one transition (bounded)."""

    @abc.abstractmethod
    def sample_batch(self, rng: random.Random, batch: int) -> list:
        """Uniformly sample a batch of stored experiences."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Current number of stored experiences."""

    @abc.abstractmethod
    def save_state(self) -> dict:
        """Serialize buffer contents."""

    @abc.abstractmethod
    def load_state(self, state: dict) -> None:
        """Restore buffer contents."""
