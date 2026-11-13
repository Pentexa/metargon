"""Interaction and attribution analysis for Cognitive Integration v0.1.

All analyses consume per-seed raw results produced by the shared v0.2
runner (the same streams for every arm). Proportions are converted to
rates before paired deltas.
"""
from __future__ import annotations

from generalization_audit.cognitive_v02.metrics import paired_deltas

# blocks / keys used by the v0.2 runner
_EVAL_BLOCK_KEYS = ("success_rate", "zero_shot_coverage",
                    "within_episode_discovery", "door_open_rate",
                    "fallback_rate", "invalid_rate",
                    "planner_ops_per_step", "mean_steps_to_success")


def _rate(v):
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return v[0] / v[1] if v[1] else None
    return v


def per_seed_value(seed_results, arm, block, key):
    """Value of metric `key` for `arm` in `block` across seeds."""
    out = []
    for seed in sorted(seed_results):
        node = seed_results[seed].get(arm, {})
        for part in block.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        v = node.get("metrics", {}).get(key) if isinstance(node, dict) else None
        out.append(_rate(v))
    return out


def interaction_term(seed_results, pool, key="success_rate"):
    """Per-seed interaction: (E+Ex − E) − (Ex − T) on `pool`."""
    a = per_seed_value(seed_results, "explore_epoch_reset", pool, key)
    b = per_seed_value(seed_results, "epoch_reset", pool, key)
    c = per_seed_value(seed_results, "explore", pool, key)
    d = per_seed_value(seed_results, "trained", pool, key)
    out = []
    for x, y, z, w in zip(a, b, c, d):
        if None in (x, y, z, w):
            out.append(None)
        else:
            out.append((x - y) - (z - w))
    return out


def delta_summary(seed_results, arm_a, arm_b, pool, key="success_rate"):
    va = per_seed_value(seed_results, arm_a, pool, key)
    vb = per_seed_value(seed_results, arm_b, pool, key)
    return paired_deltas(va, vb)


def interaction_summary(seed_results, pool, key="success_rate"):
    terms = interaction_term(seed_results, pool, key)
    import statistics
    vals = [t for t in terms if t is not None]
    if not vals:
        return {"mean": None, "ci": None, "n": 0}
    mean = statistics.mean(vals)
    # bootstrap CI on interaction terms (same convention as v0.2)
    import random
    rng = random.Random(20261140)
    boot = []
    for _ in range(2000):
        boot.append(statistics.mean(rng.choice(vals) for _ in vals))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot)) - 1]
    return {"mean": mean, "ci": (lo, hi), "n": len(vals)}


def build_analysis(seed_results, pools=("val_clean", "test_clean",
                                        "test_noisy", "ood")):
    analysis = {"deltas": {}, "interactions": {}, "per_seed": {}}
    deltas = analysis["deltas"]
    for pool in pools:
        for label, a, b in (
            ("explore_epoch_reset-epoch_reset",
             "explore_epoch_reset", "epoch_reset"),
            ("explore_epoch_reset-explore",
             "explore_epoch_reset", "explore"),
            ("explore_epoch_reset-trained",
             "explore_epoch_reset", "trained"),
            ("epoch_reset-trained", "epoch_reset", "trained"),
            ("explore-trained", "explore", "trained"),
            ("muted_explore_epoch_reset-muted_epoch_reset",
             "muted_explore_epoch_reset", "muted_epoch_reset"),
            ("muted_explore_epoch_reset-explore_epoch_reset",
             "muted_explore_epoch_reset", "explore_epoch_reset"),
            ("shuffled_explore_epoch_reset-explore_epoch_reset",
             "shuffled_explore_epoch_reset", "explore_epoch_reset"),
        ):
            deltas[f"{label}.{pool}.success_rate"] = delta_summary(
                seed_results, a, b, pool)
            deltas[f"{label}.{pool}.zero_shot_coverage"] = delta_summary(
                seed_results, a, b, pool, "zero_shot_coverage")
        analysis["interactions"][f"{pool}.success_rate"] = (
            interaction_summary(seed_results, pool))
    # A/B/A retention comparisons (interference)
    for label, a, b in (
        ("explore_epoch_reset-epoch_reset",
         "explore_epoch_reset", "epoch_reset"),
        ("explore_epoch_reset-explore",
         "explore_epoch_reset", "explore"),
        ("explore_epoch_reset-trained",
         "explore_epoch_reset", "trained"),
    ):
        for key in ("a2_success", "b_success"):
            va = per_seed_value(seed_results, a, "ab.retention", key)
            vb = per_seed_value(seed_results, b, "ab.retention", key)
            deltas[f"{label}.ab.retention.{key}"] = paired_deltas(va, vb)
    for seed in sorted(seed_results):
        analysis["per_seed"][str(seed)] = {
            arm: {
                p: per_seed_value({seed: seed_results[seed]}, arm, p,
                                   "success_rate")[0]
                for p in pools
            }
            for arm in sorted(seed_results[seed])
            if arm != "_floors"
        }
    return analysis
