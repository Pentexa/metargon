"""METARGON Academy — bounded experience memory with replay."""

from __future__ import annotations

import random

from ..core.interfaces import Experience, ExperienceMemory
from ..ckpt.checkpoint import rng_from_state, rng_state_safe


class BoundedExperienceMemory(ExperienceMemory):
    def __init__(self, capacity: int = 4000, seed: int = 0):
        self.capacity = capacity
        self.seed = seed
        self.rng = random.Random(seed)
        self.buffer = []
        self._cursor = 0

    def push(self, exp: Experience) -> None:
        if self.capacity <= 0:
            return
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self._cursor] = exp
            self._cursor = (self._cursor + 1) % self.capacity

    def sample_batch(self, rng: random.Random = None, batch: int = 64) -> list:
        rng = rng or self.rng
        if not self.buffer:
            return []
        n = min(batch, len(self.buffer))
        return rng.sample(self.buffer, n)

    def __len__(self) -> int:
        return len(self.buffer)

    def save_state(self) -> dict:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "rng": rng_state_safe(self.rng),
            "cursor": self._cursor,
            "buffer": [
                {
                    "obs_before": list(a.obs_before),
                    "action": a.action,
                    "obs_after": list(a.obs_after),
                    "reward": a.reward,
                    "terminated": a.terminated,
                    "events": [list(e) for e in a.events],
                    "step_index": a.step_index,
                    "spec_seed": a.spec_seed,
                    "pool_tag": a.pool_tag,
                } for a in self.buffer
            ],
        }

    def load_state(self, state: dict) -> None:
        from ..envs.gridworld import Observation
        self.capacity = state["capacity"]
        self.seed = state["seed"]
        rng_from_state(self.rng, state)
        self._cursor = state["cursor"]
        self.buffer = []
        for d in state["buffer"]:
            self.buffer.append(Experience(
                obs_before=Observation(*d["obs_before"]),
                action=d["action"],
                obs_after=Observation(*d["obs_after"]),
                reward=d["reward"],
                terminated=d["terminated"],
                events=tuple(map(tuple, d["events"])),
                step_index=d["step_index"],
                spec_seed=d["spec_seed"],
                pool_tag=d["pool_tag"],
            ))
