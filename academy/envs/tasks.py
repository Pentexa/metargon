"""METARGON Academy — procedural task families and immutable splits.

Two published task families feed the learning protocol:

  * **core family** — conjunctive hidden rules over the shared atom
    vocabulary on two public topologies (TP1 corridor, TP2 switchback).
    The full distinct configuration space is enumerated, filtered by
    ground-truth solvability, and split deterministically 60/20/20 into
    training / validation / held-out evaluation pools.
  * **OOD family** — unseen combinations of mechanisms: the OOD topology
    (TP2) plus rules that use *absence atoms* (empty plates) in a role the
    training pool never rewards, acting as correlated distractors for
    agents that learned a monotone "load plates helps" bias.

Splits are immutable for a research run: they are a pure function of
``pool_split_seed`` and are persisted as JSON. Evaluators verify disjointness
with the training pool before running.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from .gridworld import (
    DOORS, EnvConfig, TOPOLOGIES, is_solvable,
)

# ---------------------------------------------------------------------------
# Rule template pools per topology.
# All rules are conjunctive frozensets of atoms; OOD rules may be too but
# use absence atoms ("empty") in a role never seen in the core family.
# ---------------------------------------------------------------------------

# -- core family ------------------------------------------------------------
D1_TP1 = (
    frozenset({("sw", "S2", True)}),
    frozenset({("sw", "S2", False)}),
    frozenset({("load", "P1", "A")}),
    frozenset({("loadany", "P1")}),
    frozenset({("empty", "P1")}),
    frozenset({("sw", "S2", True), ("load", "P1", "A")}),
    frozenset({("sw", "S2", False), ("loadany", "P1")}),
    frozenset({("sw", "S2", True), ("empty", "P1")}),
    frozenset({("sw", "S2", False), ("empty", "P1")}),
)

D2_TP1 = (
    frozenset({("sw", "S1", True)}),
    frozenset({("sw", "S1", False)}),
    frozenset({("load", "P2", "B")}),
    frozenset({("loadany", "P2")}),
    frozenset({("empty", "P2")}),
    frozenset({("sw", "S1", True), ("loadany", "P2")}),
    frozenset({("sw", "S1", False), ("load", "P2", "B")}),
    frozenset({("sw", "S2", True), ("sw", "S1", False)}),
    frozenset({("load", "P1", "A"), ("load", "P2", "B")}),
    frozenset({("sw", "S1", True), ("load", "P1", "A")}),
    frozenset({("sw", "S2", False), ("loadany", "P2")}),
    frozenset({("sw", "S1", False), ("empty", "P2")}),
)

# TP2: D1 sits before any plate, so D1 rules can only reference switches.
D1_TP2 = (
    frozenset({("sw", "S2", True)}),
    frozenset({("sw", "S2", False)}),
    frozenset({("sw", "S2", True), ("empty", "P1")}),
    frozenset({("sw", "S2", False), ("empty", "P1")}),
)

D2_TP2 = (
    frozenset({("sw", "S1", True)}),
    frozenset({("sw", "S1", False)}),
    frozenset({("load", "P1", "A")}),
    frozenset({("loadany", "P1")}),
    frozenset({("load", "P2", "B")}),
    frozenset({("loadany", "P2")}),
    frozenset({("sw", "S1", True), ("loadany", "P1")}),
    frozenset({("sw", "S1", False), ("load", "P2", "B")}),
    frozenset({("load", "P1", "A"), ("load", "P2", "B")}),
    frozenset({("sw", "S2", True), ("sw", "S1", False)}),
    frozenset({("sw", "S1", True), ("loadany", "P2")}),
    frozenset({("sw", "S1", True), ("load", "P1", "A")}),
)

CORE_RULESETS = {
    "TP1": (D1_TP1, D2_TP1),
    "TP2": (D1_TP2, D2_TP2),
}

# -- OOD family -------------------------------------------------------------
# Absence atoms in a causal role never seen in the core family. All OOD
# tasks use TP2. Loading plates is never useful for D2 here, and the
# obvious "load a plate" action is actively harmful for the emptiness rules.
OOD_D1_TP2 = D1_TP2  # S2 switch rules with emptiness conjuncts

OOD_D2_TP2 = (
    frozenset({("sw", "S1", True), ("empty", "P2")}),
    frozenset({("sw", "S1", False), ("empty", "P1")}),
    frozenset({("load", "P1", "A"), ("empty", "P2")}),
    frozenset({("empty", "P1"), ("empty", "P2")}),
    frozenset({("sw", "S2", True), ("empty", "P2")}),
    frozenset({("sw", "S1", True), ("empty", "P1")}),
)

OOD_RULESETS = {
    "TP2": (OOD_D1_TP2, OOD_D2_TP2),
}


# ---------------------------------------------------------------------------
# Difficulty features (used by the curriculum; public metadata).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Difficulty:
    n_atoms: int
    delay: bool
    topology: str
    uses_absence: bool

    @property
    def bin(self) -> str:
        base = self.n_atoms + (2 if self.delay else 0)
        if base <= 3:
            return "easy"
        if base <= 5:
            return "medium"
        return "hard"

    def as_dict(self):
        return {"n_atoms": self.n_atoms, "delay": self.delay,
                "topology": self.topology, "uses_absence": self.uses_absence,
                "bin": self.bin}


def difficulty_of(config: EnvConfig) -> Difficulty:
    def uses_absence(rule):
        if isinstance(rule, tuple):
            return any(any(a[0] == "empty" for a in clause) for clause in rule)
        return any(a[0] == "empty" for a in rule)
    uses_absence_flag = uses_absence(config.d1_rule) or uses_absence(config.d2_rule)
    return Difficulty(
        n_atoms=config.n_atoms(),
        delay=config.has_delay(),
        topology=config.topology,
        uses_absence=uses_absence_flag,
    )


# ---------------------------------------------------------------------------
# Episode specification.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeSpec:
    config: EnvConfig
    max_steps: int
    noise_q: float
    spec_seed: int           # noise-stream seed for this episode
    pool_tag: str            # "train" | "val" | "test" | "ood"
    difficulty: Difficulty = field(default_factory=lambda: Difficulty(1, False, "TP1", False))

    @property
    def key(self):
        return self.config.key


# ---------------------------------------------------------------------------
# Split computation.
# ---------------------------------------------------------------------------
def _enumerate_configs(rulesets, topology: str) -> list:
    d1_pool, d2_pool = rulesets[topology]
    out = []
    for c1 in d1_pool:
        for c2 in d2_pool:
            for b1 in (0, 1):
                for b2 in (0, 1):
                    out.append(EnvConfig(
                        topology=topology,
                        d1_rule=c1, d2_rule=c2,
                        d1_delay=bool(b1), d2_delay=bool(b2)))
    return out


def compute_splits(pool_split_seed: int = 20260901):
    """Enumerate and deterministically split the core config space.
    Returns {"train": [...], "val": [...], "test": [...], "ood": [...],
             "counts": {...}} of EnvConfig lists."""
    core_all = []
    for tp in ("TP1", "TP2"):
        core_all.extend(_enumerate_configs(CORE_RULESETS, tp))
    prng = random.Random(pool_split_seed)
    solvable = [c for c in core_all if is_solvable(c)]
    prng.shuffle(solvable)
    n = len(solvable)
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    train = solvable[:train_n]
    val = solvable[train_n:train_n + val_n]
    test = solvable[train_n + val_n:]

    # OOD: unseen mechanisms on unseen topology, filtered and downsampled.
    ood_all = _enumerate_configs(OOD_RULESETS, "TP2")
    ood_solvable = [c for c in ood_all if is_solvable(c)]
    prng.shuffle(ood_solvable)
    ood = ood_solvable[:25]

    return {
        "train": train, "val": val, "test": test, "ood": ood,
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "ood": len(ood), "solvable": n, "enumerated": len(core_all),
        },
    }


def _config_to_dict(c: EnvConfig) -> dict:
    """Serialize a config to JSON (internal splits format).

    Encoding is *unambiguous*: clause lists plus a conjunctive flag per
    rule (the teacher-proposal grammar is distinct from this internal
    format). A conjunctive rule has exactly one clause.
    """
    def ser(r):
        if isinstance(r, tuple):          # disjunctive
            return [sorted(x) for x in r]
        return [sorted(r)]                # conjunctive: single clause
    return {
        "topology": c.topology,
        "d1_rule": ser(c.d1_rule),
        "d1_conjunctive": not isinstance(c.d1_rule, tuple),
        "d2_rule": ser(c.d2_rule),
        "d2_conjunctive": not isinstance(c.d2_rule, tuple),
        "d1_delay": c.d1_delay,
        "d2_delay": c.d2_delay,
    }


def _dict_to_config(d: dict) -> EnvConfig:
    def deser(r, conjunctive):
        clauses = [frozenset(map(tuple, clause)) for clause in r]
        if conjunctive:
            return clauses[0]             # exactly one clause
        return tuple(clauses)

    # legacy files (pre-flag) fall back to a shape heuristic
    def legacy_flag(r):
        return not (r and isinstance(r[0], list)
                    and r[0] and isinstance(r[0][0], list))

    return EnvConfig(
        topology=d["topology"],
        d1_rule=deser(d["d1_rule"], d.get("d1_conjunctive", legacy_flag(d["d1_rule"]))),
        d2_rule=deser(d["d2_rule"], d.get("d2_conjunctive", legacy_flag(d["d2_rule"]))),
        d1_delay=d["d1_delay"],
        d2_delay=d["d2_delay"],
    )


DEFAULT_SPLITS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "academy_v01",
    "splits.json")


def save_splits(splits, path: str = DEFAULT_SPLITS_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "train": [_config_to_dict(c) for c in splits["train"]],
        "val": [_config_to_dict(c) for c in splits["val"]],
        "test": [_config_to_dict(c) for c in splits["test"]],
        "ood": [_config_to_dict(c) for c in splits["ood"]],
        "counts": splits["counts"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_splits(path: str = DEFAULT_SPLITS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "train": [_dict_to_config(d) for d in data["train"]],
        "val": [_dict_to_config(d) for d in data["val"]],
        "test": [_dict_to_config(d) for d in data["test"]],
        "ood": [_dict_to_config(d) for d in data["ood"]],
        "counts": data["counts"],
    }


def get_splits(pool_split_seed: int = 20260901, cache_path: Optional[str] = None,
               force_compute: bool = False):
    """Return splits, computing (and caching to disk) on first use."""
    path = cache_path or DEFAULT_SPLITS_PATH
    if not force_compute and os.path.exists(path):
        return load_splits(path)
    splits = compute_splits(pool_split_seed)
    save_splits(splits, path)
    return splits


# ---------------------------------------------------------------------------
# Episode streams.
# ---------------------------------------------------------------------------
def make_episodes(pool, pool_tag: str, seed: int, episodes: int,
                  max_steps: int = 80, noise_q: float = 0.0,
                  switch_half: bool = False) -> list:
    """Deterministic stream of episodes from a pool. If ``switch_half``,
    the configuration switches to a fresh draw at the halfway point (used
    by the rule-change / adaptation protocol)."""
    rng = random.Random(seed)
    pool = list(pool)
    out = []
    cfg = rng.choice(pool)
    for e in range(episodes):
        if switch_half and e == episodes // 2:
            cfg = rng.choice(pool)
        out.append(EpisodeSpec(
            config=cfg,
            max_steps=max_steps,
            noise_q=noise_q,
            spec_seed=seed * 1000 + e,
            pool_tag=pool_tag,
            difficulty=difficulty_of(cfg),
        ))
    return out
