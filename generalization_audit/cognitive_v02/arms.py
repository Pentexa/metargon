"""Arm factory and negative-control transforms for Cognitive Evaluation v0.2.

An *arm* is either:

- a trained configuration (a kwargs dict for ``MetargonAcademyAgent``),
- a fresh/untrained configuration, or
- a transform applied after training a base arm (wipe / shuffle).

Candidate branches (e.g. Cognitive Memory v0.2, Exploration v0.2) can add
arms without changing this repository's code: drop a JSON spec into
``candidate_specs/`` (see README) or call ``register_candidate``. Their
policy code must remain on their own branch — only the spec is read here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from academy.agents.metargon_agent import MetargonAcademyAgent

from . import config

_CANDIDATE_DIR = Path(__file__).resolve().parent / "candidate_specs"


def make_policy(name, seed):
    spec = _resolve_spec(name)
    agent = MetargonAcademyAgent(seed=seed, **spec.get("kwargs", {}))
    agent._cognitive_arm = name
    return agent


def make_fresh(name, seed):
    spec = _resolve_spec(name)
    if spec.get("kind") == "transform":
        base = make_policy(spec["base"], seed)
        # transforms are applied after training; fresh copy of base
        return base
    return make_policy(name, seed)


def _resolve_spec(name):
    if name in config.ARM_SPECS:
        return dict(config.ARM_SPECS[name], kind="trained")
    if name in config.CONTROL_SPECS:
        return dict(config.CONTROL_SPECS[name])
    raise ValueError(f"unknown arm: {name}")


def apply_transform(agent, transform, seed, assert_changed=False):
    """Apply a negative-control transform to a trained agent's declarative
    memory. Returns (agent, changed_flag)."""
    if transform == "wipe":
        wipe_memory(agent)
        return agent, True
    if transform == "shuffle":
        return shuffle_memory(agent, seed, assert_changed=assert_changed)
    raise ValueError(f"unknown transform: {transform}")


def wipe_memory(agent):
    m = agent.memory
    m.hypotheses = []
    m.evidence_bank = []
    m.bank_retrievals = 0
    # atom-transfer profiles are also declarative knowledge
    m._atom_stats = {}
    m._atom_ctxs = {}
    m._pair_stats = {}


def shuffle_memory(agent, seed, assert_changed=False):
    """Randomly permute antecedent associations among same-door hypotheses
    (and bank consequences), breaking the true structure while preserving
    memory size, counts and support levels."""
    rng = random.Random(1000003 + seed)
    m = agent.memory
    changed = False
    by_door = {}
    for h in m.hypotheses:
        by_door.setdefault(h.door, []).append(h)
    for door, hyps in by_door.items():
        if len(hyps) < 2:
            continue
        antes = [h.antecedent for h in hyps]
        shuffled = antes[:]
        rng.shuffle(shuffled)
        for h, ante in zip(hyps, shuffled):
            if ante != h.antecedent:
                changed = True
            h.antecedent = ante
            # keep support/contradiction counts as-is (content moved)
    if m.evidence_bank:
        bank = list(m.evidence_bank)
        rng.shuffle(bank)
        m.evidence_bank = bank
        changed = True
    if assert_changed and not changed:
        raise AssertionError("shuffle transform made no change (arm content "
                             "already unstructured?)")
    return agent, changed


# ---------------------------------------------------------------------------
# Candidate interface (upcoming v0.2 branches)
# ---------------------------------------------------------------------------
def register_candidate(name, kwargs=None, kind="trained", base=None,
                       transform=None):
    """Register an arm spec at runtime (e.g. from a candidate branch's
    harness before running the benchmark)."""
    kwargs = kwargs or {}
    if kind == "transform":
        config.CONTROL_SPECS[name] = {"kind": "transform", "base": base,
                                      "transform": transform}
    else:
        config.ARM_SPECS[name] = {"kwargs": kwargs}


def load_candidate_specs(directory=None):
    """Load ``*.json`` arm spec files from ``directory`` (default:
    ``candidate_specs/`` of this package). JSON shape:
    ``{"name": ..., "kwargs": {...}}`` for trained arms,
    ``{"name": ..., "kind": "transform", "base": ..., "transform": ...}``
    for controls. Returns list of registered names."""
    directory = Path(directory) if directory else _CANDIDATE_DIR
    registered = []
    if not directory.exists():
        return registered
    for p in sorted(directory.glob("*.json")):
        spec = json.loads(p.read_text(encoding="utf-8"))
        name = spec.get("name") or p.stem
        kind = spec.get("kind", "trained")
        if kind == "transform":
            register_candidate(name, kind="transform",
                               base=spec["base"], transform=spec["transform"])
        else:
            register_candidate(name, kwargs=spec.get("kwargs", {}))
        registered.append(name)
    return registered


def trained_arms():
    return tuple(config.ARM_SPECS.keys())


def all_arms():
    return tuple(config.ARM_SPECS.keys()) + tuple(config.CONTROL_SPECS.keys())


def arm_label(name):
    if name in config.ARM_SPECS:
        return name
    return f"{name} ({config.CONTROL_SPECS[name].get('base', '')})"
