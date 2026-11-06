"""METARGON Academy — curriculum manager.

Selects training tasks based on *measured* progress, not on an arbitrary
fixed schedule. Each difficulty bin keeps a bounded sliding-window record
of success; bins with success below target are promoted for training, and
the manager alternates among the least-confident active bins. When a bin's
measured success exceeds the target with enough confidence, the manager
proposes the next harder bin.
"""

from __future__ import annotations

import random
from collections import deque

BINS = ("easy", "medium", "hard")


class CurriculumManager:
    def __init__(self, success_target: float = 0.85,
                 min_confidence: float = 0.60,
                 window: int = 20,
                 harder_epsilon: float = 0.10,
                 easier_epsilon: float = 0.20,
                 seed: int = 0):
        self.success_target = success_target
        self.min_confidence = min_confidence
        self.window = window
        self.harder_epsilon = harder_epsilon
        self.easier_epsilon = easier_epsilon
        self.rng = random.Random(seed)
        self.seed = seed
        self.windows = {b: deque(maxlen=window) for b in BINS}
        self.counts = {b: 0 for b in BINS}
        self.active = "easy"
        self.history = []          # (bin, success) per recorded episode
        self.last_proposal = None

    # -- measured progress ----------------------------------------------
    def record_episode(self, spec, success: bool, steps: int, info: dict = None):
        b = spec.difficulty.bin if spec.difficulty is not None else self.active
        if b not in self.windows:
            b = "easy"
        self.windows[b].append(1.0 if success else 0.0)
        self.counts[b] += 1
        self.history.append((b, bool(success)))

    def bin_stats(self, b: str) -> dict:
        w = list(self.windows[b])
        n = len(w)
        if n == 0:
            return {"bin": b, "n": 0, "mean": None, "std": None,
                    "confidence": 0.0, "success_rate": 0.0}
        mean = sum(w) / n
        var = sum((x - mean) ** 2 for x in w) / n
        std = var ** 0.5
        return {"bin": b, "n": n, "mean": mean, "std": std,
                "confidence": max(0.0, 1.0 - std),
                "success_rate": mean}

    def progress(self) -> dict:
        return {b: self.bin_stats(b) for b in BINS}

    # -- proposal --------------------------------------------------------
    def propose(self, rng: random.Random = None) -> str:
        rng = rng or self.rng
        stats = self.progress()
        # active bin measured success
        active_stats = stats[self.active]
        if active_stats["n"] >= 4:
            rate = active_stats["success_rate"]
            # promote when clearly above target with reasonable evidence
            if (rate >= self.success_target - self.harder_epsilon
                    and active_stats["confidence"] >= self.min_confidence):
                idx = BINS.index(self.active)
                if idx < len(BINS) - 1:
                    self.active = BINS[idx + 1]
            elif rate < self.success_target - self.easier_epsilon:
                idx = BINS.index(self.active)
                if idx > 0:
                    self.active = BINS[idx - 1]
        # spread exposure: pick among bins with the least confident measures
        # (or unvisited bins first), weighted toward the active bin.
        candidates = []
        for b in BINS:
            st = stats[b]
            if st["n"] == 0:
                candidates.append((b, 100.0))
            else:
                candidates.append((b, st["confidence"]))
        candidates.sort(key=lambda x: (-x[1], BINS.index(x[0])))
        # 70% active, 30% least-confident exploration
        self.last_proposal = self.active
        if rng.random() < 0.30:
            b = candidates[0][0]
            if stats[b]["n"] > 0 and b != self.active:
                self.last_proposal = b
        return self.last_proposal

    # -- checkpointing ---------------------------------------------------
    def save_state(self) -> dict:
        return {
            "success_target": self.success_target,
            "min_confidence": self.min_confidence,
            "window": self.window,
            "harder_epsilon": self.harder_epsilon,
            "easier_epsilon": self.easier_epsilon,
            "seed": self.seed,
            "rng": self.rng.getstate(),
            "windows": {b: list(w) for b, w in self.windows.items()},
            "counts": dict(self.counts),
            "active": self.active,
            "history": self.history[-500:],
            "last_proposal": self.last_proposal,
        }

    def load_state(self, state: dict) -> None:
        self.success_target = state["success_target"]
        self.min_confidence = state["min_confidence"]
        self.window = state["window"]
        self.harder_epsilon = state["harder_epsilon"]
        self.easier_epsilon = state["easier_epsilon"]
        self.seed = state["seed"]
        self.rng.setstate(tuple(state["rng"]))
        self.windows = {b: deque(w, maxlen=self.window)
                        for b, w in state["windows"].items()}
        self.counts = dict(state["counts"])
        self.active = state["active"]
        self.history = [tuple(h) for h in state["history"]]
        self.last_proposal = state["last_proposal"]
