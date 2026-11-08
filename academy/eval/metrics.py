"""METARGON Academy — evaluation metrics.

Success rates, steps, sample efficiency, adaptation latency, forgetting,
bootstrap confidence intervals (paired where appropriate), and learning
curves.
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, pstdev


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95,
                 seed: int = 0) -> tuple:
    """Bootstrap confidence interval for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo_i = int((1.0 - ci) / 2.0 * n_boot)
    hi_i = int((1.0 + ci) / 2.0 * n_boot) - 1
    return (means[max(0, lo_i)], means[min(n_boot - 1, hi_i)])


def summarize_episodes(records) -> dict:
    """records: list of per-episode dicts with 'success' and 'steps'."""
    n = len(records)
    if n == 0:
        return {"n": 0, "success_rate": 0.0, "mean_steps": 0.0,
                "ci": (0.0, 0.0)}
    succ = [1.0 if r["success"] else 0.0 for r in records]
    steps = [r["steps"] for r in records]
    mean_succ = mean(succ)
    return {
        "n": n,
        "success_rate": mean_succ,
        "success_std": pstdev(succ),
        "mean_steps": mean(steps),
        "ci": bootstrap_ci(succ),
        "successes": int(sum(succ)),
    }


def learning_curve(records, snapshot_points) -> dict:
    """Success rate in windows ending at each snapshot point."""
    out = {}
    for p in snapshot_points:
        window = [r for r in records if r["episode"] < p]
        out[str(p)] = summarize_episodes(window)["success_rate"] if window else None
    return out


def adaptation_latency(records, switch_episode: int) -> dict:
    """After a rule change at switch_episode: episodes until the first
    success after the switch (None if never)."""
    after = [r for r in records if r["episode"] >= switch_episode]
    first_success = None
    for r in after:
        if r["success"]:
            first_success = r["episode"]
            break
    return {"switch_episode": switch_episode,
            "first_success_after_switch": first_success,
            "post_switch_success": summarize_episodes(after)["success_rate"]}


def forgetting_metrics(early_records, late_records) -> dict:
    """Retention: success on an earlier family measured before vs after
    training on later families."""
    early = summarize_episodes(early_records)
    late = summarize_episodes(late_records)
    return {
        "early_success": early["success_rate"],
        "late_success": late["success_rate"],
        "forgetting_delta": early["success_rate"] - late["success_rate"],
    }


def paired_difference(a_succ, b_succ, seed: int = 0, n_boot: int = 2000) -> dict:
    """Paired bootstrap over per-seed success rates, ordered (a − b).

    Convention: call with ``(candidate, baseline)`` so a positive mean
    means the candidate improved on the baseline. Reversing the
    arguments flips the sign of the result.
    """
    assert len(a_succ) == len(b_succ)
    n = len(a_succ)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_succ[i] for i in idx) / n
        db = sum(b_succ[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    return {"mean": mean(diffs), "ci": (lo, hi)}
