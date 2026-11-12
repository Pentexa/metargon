"""Cognitive benchmark v0.1 — fixed configuration.

All experiment knobs live here so every run is reproducible from the CLI
flags plus these defaults. Streams are drawn deterministically from the
frozen Academy splits; nothing is selected based on measured outcomes.
"""
from __future__ import annotations

# Frozen split source (same pure function the Academy uses).
POOL_SPLIT_SEED = 20260901

# Episode budget shared by every policy (matched budgets).
MAX_STEPS = 80
NOISE_Q = 0.0

# Seeds over which confidence intervals are computed.
DEFAULT_SEEDS = (0, 1, 2, 3)

# Protocol sizes (baseline preset).
TRAIN_EPISODES = 24          # training stream length per seed
EVAL_EPISODES = 8            # per eval pool (val / test / ood)
REVISIT_EPISODES = 7         # A,B,C,A,B,C,A continual stream
SWITCH_HALF = 10             # episodes per rule (A then B) in switch stream

# Deterministic base seeds used to derive per-protocol episode streams.
# (Incremented per seed value passed on the CLI.)
STREAM_SEED_BASE = 20261101

# Policies in the roster. Root agent = unmodified Academy student.
POLICIES_TRAINED = ("metargon_trained", "metargon_muted")
POLICIES_FRESH = ("metargon_fresh", "random", "episodic", "qlearning")
POLICIES_DIAGNOSTIC = ("oracle",)
ALL_POLICIES = (
    "metargon_trained",
    "metargon_muted",
    "metargon_fresh",
    "random",
    "episodic",
    "qlearning",
    "oracle",
)

# Which protocols each policy participates in.
PROTOCOLS = (
    "train",       # training stream (shared across policies needing it)
    "eval_val",    # zero-shot on validation pool
    "eval_test",   # zero-shot on held-out test pool
    "eval_ood",    # zero-shot on OOD pool
    "revisit",     # continual retention across intervening tasks
    "switch",      # adaptability under a mid-stream rule change
    "restart",     # retention across save/restore
)
