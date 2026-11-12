"""Deterministic episode streams for every cognitive protocol.

All streams are pure functions of (seed, frozen pools) — the same stream is
handed to every policy so budgets and worlds are perfectly matched. No
stream is selected based on measured outcomes.
"""
from __future__ import annotations

import random

from academy.envs.tasks import EpisodeSpec, difficulty_of, make_episodes

from . import config
from .splits import (
    compute_live_splits,
    load_frozen_splits,
    rule_from_canonical,
)

# ---------------------------------------------------------------------------
# Rule / config canonical forms (hash-order independent)
# ---------------------------------------------------------------------------
def _rule_clauses(rule):
    """True-rule clauses, each as a frozenset of atom tuples.

    A conjunctive rule contributes one clause; a disjunctive tuple
    contributes each clause. Used for ground-truth comparisons and
    canonical signatures.
    """
    clauses = list(rule) if isinstance(rule, tuple) else [rule]
    return [frozenset(tuple(a) for a in clause) for clause in clauses]


def config_rule_clauses(config):
    return {
        "D1": _rule_clauses(config.d1_rule),
        "D2": _rule_clauses(config.d2_rule),
    }


def config_signature(config) -> str:
    """Stable string signature of a config (independent of set iteration)."""
    def clauses_sig(rule):
        return sorted(tuple(sorted(_clause_key(c))) for c in _rule_clauses(rule))
    d = {
        "topology": config.topology,
        "D1": clauses_sig(config.d1_rule),
        "D2": clauses_sig(config.d2_rule),
        "delay1": config.d1_delay,
        "delay2": config.d2_delay,
    }
    return repr(d)


def _clause_key(clause):
    """Canonical comparable key for a clause (frozenset of atoms)."""
    return tuple(sorted(tuple(a) for a in clause))


# ---------------------------------------------------------------------------
# Pool access
# ---------------------------------------------------------------------------
def pools(frozen=True):
    """Return the frozen (canonical) pools as real EnvConfig lists."""
    if frozen:
        frozen_pools, _ = load_frozen_splits()
        return {tag: [rule_from_canonical(c) for c in pool]
                for tag, pool in frozen_pools.items()}
    _, raw = compute_live_splits()
    return raw


# ---------------------------------------------------------------------------
# Standard streams
# ---------------------------------------------------------------------------
def _distinct_config_stream(pool, pool_tag, seed, episodes):
    """Deterministic stream over distinct configs.

    The pool is canonical-sorted; the stream starts at a seed-rotated
    offset and cycles through the (up to ``episodes``) first configs so a
    stream exposes many distinct hidden-rule configurations. Matched
    across policies; no outcome-based selection.
    """
    ordered = sorted(pool, key=config_signature)
    start = seed % len(ordered)
    seq = ordered[start:] + ordered[:start]
    out = []
    for e in range(episodes):
        cfg = seq[e % len(seq)]
        out.append(EpisodeSpec(
            config=cfg,
            max_steps=config.MAX_STEPS,
            noise_q=config.NOISE_Q,
            spec_seed=seed * 1000 + e,
            pool_tag=pool_tag,
            difficulty=difficulty_of(cfg),
        ))
    return out


def train_stream(seed, episodes=config.TRAIN_EPISODES):
    return _distinct_config_stream(pools()["train"], "train", seed, episodes)


def eval_stream(pool_tag, seed, episodes=config.EVAL_EPISODES):
    return _distinct_config_stream(pools()[pool_tag], pool_tag, seed,
                                   episodes)


# ---------------------------------------------------------------------------
# Revisit (continual retention) stream: A,B,C,A,B,C,A
# ---------------------------------------------------------------------------
def _pick_configs(pool, n, order_seed):
    """First n configs of the canonical-sorted pool (stable, outcome-free)."""
    ordered = sorted(pool, key=config_signature)
    return ordered[:n]


def revisit_stream(seed, n_configs=3):
    pool = pools()["train"]
    configs = _pick_configs(pool, n_configs, seed)
    cycle = configs * 2 + configs[:1]     # A,B,C,A,B,C,A
    out = []
    for e, cfg in enumerate(cycle):
        out.append(EpisodeSpec(
            config=cfg,
            max_steps=config.MAX_STEPS,
            noise_q=config.NOISE_Q,
            spec_seed=seed * 1000 + e,
            pool_tag="train",
            difficulty=difficulty_of(cfg),
        ))
    return out


def revisit_occurrences(stream):
    """Map config signature -> list of episode indices (in stream order)."""
    occ = {}
    for e, spec in enumerate(stream):
        occ.setdefault(config_signature(spec.config), []).append(e)
    return occ


# ---------------------------------------------------------------------------
# Switch (adaptability) stream: rule A for half, rule B for half
# ---------------------------------------------------------------------------
def _rule_polarity(rule):
    """If the rule is a single switch atom, return (switch, value)."""
    if isinstance(rule, tuple) or len(rule) != 1:
        return None
    atom = next(iter(rule))
    if atom[0] == "sw":
        return atom[1], bool(atom[2])
    return None


def switch_pair(seed=0):
    """Deterministically pick (cfgA, cfgB) from the train pool.

    Pair property: same topology, same rule and delay for the *other*
    door, and a *different* rule (preferably opposite switch polarity) for
    the target door. Candidates are ordered canonically; the first match
    is chosen — never by outcome.
    """
    pool = pools()["train"]
    ordered = sorted(pool, key=config_signature)

    def same_other(a, b):
        return (a.topology == b.topology
                and a.d2_rule == b.d2_rule
                and a.d1_rule != b.d1_rule
                and a.d1_delay == b.d1_delay
                and a.d2_delay == b.d2_delay)

    best = None
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if not same_other(a, b):
                continue
            pa, pb = _rule_polarity(a.d1_rule), _rule_polarity(b.d1_rule)
            if pa and pb and pa[0] == pb[0] and pa[1] != pb[1]:
                return a, b      # opposite polarity switch rules
            if best is None:
                best = (a, b)
    if best is None:
        raise ValueError("no switch pair found in train pool")
    return best


def switch_stream(seed, half=config.SWITCH_HALF):
    a, b = switch_pair(seed)
    out = []
    e = 0
    for _ in range(half):
        out.append(EpisodeSpec(config=a, max_steps=config.MAX_STEPS,
                               noise_q=config.NOISE_Q,
                               spec_seed=seed * 1000 + e,
                               pool_tag="train", difficulty=difficulty_of(a)))
        e += 1
    for _ in range(half):
        out.append(EpisodeSpec(config=b, max_steps=config.MAX_STEPS,
                               noise_q=config.NOISE_Q,
                               spec_seed=seed * 1000 + e,
                               pool_tag="train", difficulty=difficulty_of(b)))
        e += 1
    return out


def switch_split_point(stream):
    sigs = [config_signature(s.config) for s in stream]
    return len(sigs) // 2


# ---------------------------------------------------------------------------
# Stream signatures (determinism verification)
# ---------------------------------------------------------------------------
def stream_signature(stream):
    return [
        (config_signature(spec.config), spec.spec_seed, spec.max_steps,
         spec.noise_q)
        for spec in stream
    ]
