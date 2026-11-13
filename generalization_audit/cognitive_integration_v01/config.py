"""Cognitive Integration v0.1 — protocol configuration (preregistered).

The confirmatory seed set below is fixed by the preregistration document
and is untouched by any prior committed evaluation. Smoke runs use only
seed 0.
"""
from __future__ import annotations

from generalization_audit.cognitive_v02 import config as V2

# --- preregistered confirmatory seeds ---------------------------------------
CONFIRMATORY_SEEDS = (101, 103, 107, 109)
SMOKE_SEED = 0

# --- protocol sizes (same defaults as v0.2; may be reduced by --quick) ------
TRAIN_EPISODES = V2.TRAIN_EPISODES
EVAL_EPISODES = V2.EVAL_EPISODES
AB_TRAIN_PER_PHASE = V2.AB_TRAIN_PER_PHASE
AB_EVAL_EPISODES = V2.AB_EVAL_EPISODES

EVAL_POOLS = V2.EVAL_POOLS
PHASES = ("standard", "ab")

# --- extra control arms (registered at import; never trained configurations)
EXTRA_CONTROLS = {
    "muted_epoch_reset": {"kind": "transform", "base": "epoch_reset",
                          "transform": "wipe"},
    "muted_explore_epoch_reset": {"kind": "transform",
                                  "base": "explore_epoch_reset",
                                  "transform": "wipe"},
    "shuffled_explore_epoch_reset": {"kind": "transform",
                                     "base": "explore_epoch_reset",
                                     "transform": "shuffle"},
}

# --- arm ordering for reports ------------------------------------------------
ARM_ORDER = (
    "fresh", "trained", "muted", "shuffled",
    "epoch_reset", "muted_epoch_reset",
    "explore", "explore_epoch_reset",
    "muted_explore_epoch_reset", "shuffled_explore_epoch_reset",
)

PRIMARY_POOL = "val_clean"
SECONDARY_POOLS = ("test_clean", "test_noisy", "ood")
