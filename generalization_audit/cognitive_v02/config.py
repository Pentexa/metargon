"""Cognitive Evaluation v0.2 — fixed configuration.

All sizes, seeds and arm definitions live here (plus CLI overrides) so a
run is fully reproducible. Evaluation streams are deterministic pure
functions of (seed, frozen pools); no task is selected by outcome.
"""
from __future__ import annotations

# --- seeded protocol sizes -------------------------------------------------
DEFAULT_SEEDS = (0, 1, 2, 3)
STREAM_SEED_BASE = 20261121

MAX_STEPS = 80
NOISE_Q_CLEAN = 0.0
NOISE_Q_NOISY = 0.08

TRAIN_EPISODES = 12
EVAL_EPISODES = 6
AB_TRAIN_PER_PHASE = 8
AB_EVAL_EPISODES = 4
RETENTION_EVAL_MAX_CONFIGS = 3

# --- arm definitions -------------------------------------------------------
# A "trained arm" is a full agent configuration that gets trained on the
# training stream and then evaluated. Controls are derived from the
# trained baseline (fresh = untrained; muted = declarative memory wiped;
# shuffled = associations randomized).
ARM_SPECS = {
    "trained": {"kwargs": {}},
    "epoch_reset": {
        # v0.7 "best known configuration": episode-scoped evidence +
        # per-episode probe reset (see Cognitive Memory v0.1 report).
        "kwargs": {"memory_episode_scoped": True,
                   "probe_reset_per_episode": True},
    },
    "explore": {
        # Cognitive Exploration v0.1 (master switch, merged PR #49).
        "kwargs": {"cognitive_exploration": True},
    },
    "explore_epoch_reset": {
        "kwargs": {"cognitive_exploration": True,
                   "memory_episode_scoped": True,
                   "probe_reset_per_episode": True},
    },
}

CONTROL_SPECS = {
    "fresh": {"kind": "fresh", "base": None},
    "muted": {"kind": "transform", "base": "trained",
              "transform": "wipe"},
    "shuffled": {"kind": "transform", "base": "trained",
                 "transform": "shuffle"},
}

# NOTE: the arm registry is dynamic (candidate branches may add arms via
# ``arms.register_candidate`` / ``load_candidate_specs``). Use the
# ``arms.trained_arms()`` / ``arms.all_arms()`` helpers for the current
# registry instead of caching these tuples at import time.

# --- evaluation pools ------------------------------------------------------
EVAL_POOLS = (
    ("val_clean", "val", NOISE_Q_CLEAN),
    ("test_clean", "test", NOISE_Q_CLEAN),
    ("test_noisy", "test", NOISE_Q_NOISY),
    ("ood", "ood", NOISE_Q_CLEAN),
)
