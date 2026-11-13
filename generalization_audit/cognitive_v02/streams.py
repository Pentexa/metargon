"""Deterministic episode streams for Cognitive Evaluation v0.2.

All streams are pure functions of (seed, frozen pools). The same stream is
handed to every arm, so action and computation budgets are identical.
"""
from __future__ import annotations

from academy.envs.gridworld import EnvConfig, is_solvable
from academy.envs.tasks import EpisodeSpec, difficulty_of

from . import config
from .splits import pools_dict


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------
def _rule_clauses(rule):
    clauses = list(rule) if isinstance(rule, tuple) else [rule]
    return [frozenset(tuple(a) for a in clause) for clause in clauses]


def rule_sig(rule):
    return frozenset(tuple(sorted(tuple(a) for a in c))
                     for c in _rule_clauses(rule))


def config_signature(cfg) -> str:
    return repr({
        "topology": cfg.topology,
        "D1": sorted(tuple(sorted(tuple(a) for a in c))
                     for c in _rule_clauses(cfg.d1_rule)),
        "D2": sorted(tuple(sorted(tuple(a) for a in c))
                     for c in _rule_clauses(cfg.d2_rule)),
        "d1": cfg.d1_delay, "d2": cfg.d2_delay,
    })


# ---------------------------------------------------------------------------
# Pool access
# ---------------------------------------------------------------------------
_POOLS_CACHE = None


def _pools():
    global _POOLS_CACHE
    if _POOLS_CACHE is None:
        _POOLS_CACHE = pools_dict()["pools"]
    return _POOLS_CACHE


def _distinct_config_stream(pool, pool_tag, seed, episodes, noise_q):
    ordered = sorted(pool, key=config_signature)
    start = seed % len(ordered)
    seq = ordered[start:] + ordered[:start]
    out = []
    for e in range(episodes):
        cfg = seq[e % len(seq)]
        out.append(EpisodeSpec(
            config=cfg, max_steps=config.MAX_STEPS, noise_q=noise_q,
            spec_seed=seed * 1000 + e, pool_tag=pool_tag,
            difficulty=difficulty_of(cfg)))
    return out


def train_stream(seed, episodes=config.TRAIN_EPISODES):
    return _distinct_config_stream(_pools()["train"], "train", seed,
                                   episodes, config.NOISE_Q_CLEAN)


def eval_stream(pool_tag, seed, episodes=config.EVAL_EPISODES,
                noise_q=config.NOISE_Q_CLEAN):
    return _distinct_config_stream(_pools()[pool_tag], pool_tag, seed,
                                   episodes, noise_q)


def single_config_stream(cfg, seed, episodes, noise_q=0.0, pool_tag="train"):
    out = []
    for e in range(episodes):
        out.append(EpisodeSpec(
            config=cfg, max_steps=config.MAX_STEPS, noise_q=noise_q,
            spec_seed=seed * 1000 + e, pool_tag=pool_tag,
            difficulty=difficulty_of(cfg)))
    return out


# ---------------------------------------------------------------------------
# Conflict pair (A/B) construction
# ---------------------------------------------------------------------------
_SOLVABLE_CACHE = {}


def _is_solvable_cached(cfg):
    key = config_signature(cfg)
    if key not in _SOLVABLE_CACHE:
        _SOLVABLE_CACHE[key] = is_solvable(cfg)
    return _SOLVABLE_CACHE[key]


def _flip_d1(cfg):
    r = cfg.d1_rule
    if isinstance(r, tuple) or len(r) != 1:
        return None
    atom = next(iter(r))
    if atom[0] != "sw":
        return None
    return EnvConfig(
        topology=cfg.topology,
        d1_rule=frozenset({(atom[0], atom[1], not atom[2])}),
        d2_rule=cfg.d2_rule,
        d1_delay=cfg.d1_delay,
        d2_delay=cfg.d2_delay,
    )


def conflict_pair(seed=0, min_shared=4):
    """Deterministically pick (cfgA, cfgB) from the train pool where B is
    the flipped-polarity D1 variant of A (same other-door rule, same
    delays), both solvable, and A's D1 rule recurs in >= min_shared
    held-out (test/val) configs for retention evaluation."""
    train = sorted(_pools()["train"], key=config_signature)
    held = _pools()["test"] + _pools()["val"]
    start = seed % len(train)
    seq = train[start:] + train[:start]
    for a in seq:
        b = _flip_d1(a)
        if b is None or not _is_solvable_cached(b):
            continue
        n_shared = sum(1 for x in held
                       if x.topology == a.topology
                       and rule_sig(x.d1_rule) == rule_sig(a.d1_rule))
        if n_shared >= min_shared:
            return a, b
    raise ValueError("no feasible conflict pair found")


def retention_eval_configs(cfg, max_configs=config.RETENTION_EVAL_MAX_CONFIGS):
    """Held-out configs sharing cfg's D1 rule (test+val), canonical
    ordered; falls back to the config itself when none are available."""
    held = _pools()["test"] + _pools()["val"]
    sig = rule_sig(cfg.d1_rule)
    matches = [x for x in held
               if x.topology == cfg.topology and rule_sig(x.d1_rule) == sig]
    if not matches:
        return [cfg]
    return sorted(matches, key=config_signature)[:max_configs]


def retention_eval_stream(cfg, seed, episodes):
    configs = retention_eval_configs(cfg)
    out = []
    for e in range(episodes):
        c = configs[e % len(configs)]
        out.append(EpisodeSpec(
            config=c, max_steps=config.MAX_STEPS, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag="heldout",
            difficulty=difficulty_of(c)))
    return out


def phase_train_stream(cfg, seed, episodes, phase_tag):
    return single_config_stream(cfg, seed, episodes, pool_tag=phase_tag)


def stream_signature(stream):
    return [(config_signature(s.config), s.spec_seed, s.max_steps,
             s.noise_q) for s in stream]


def stream_distinct_configs(stream):
    return sorted({config_signature(s.config) for s in stream})
