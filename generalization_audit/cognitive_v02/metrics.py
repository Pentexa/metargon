"""Metric computation for Cognitive Evaluation v0.2.

Per-seed protocols compute *raw* metric values:
- proportions -> ``(k, n)`` counts;
- unbounded scalars -> numbers (steps, planner ops, hypothesis counts).

The scorecard then aggregates across seeds: pooled 95% Wilson interval
for proportions, mean ± 95% Student-t CI for scalars, and bootstrap
paired deltas for comparisons.
"""
from __future__ import annotations

import math
import random
import statistics

from .streams import rule_sig  # noqa: F401 (kept for config signatures)

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t(n):
    return _T_TABLE.get(n, 1.96)


def summarize(values):
    if isinstance(values, (int, float)):
        values = [values]
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sem": None, "ci": None, "n": 0,
                "method": "t"}
    n = len(vals)
    mean = statistics.mean(vals)
    if n < 2:
        return {"mean": mean, "sem": None, "ci": None, "n": n,
                "method": "t"}
    sem = statistics.stdev(vals) / math.sqrt(n)
    half = _t(n) * sem
    return {"mean": mean, "sem": sem, "ci": (mean - half, mean + half),
            "n": n, "method": "t"}


def _wilson_ci(k, n, z=1.96):
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
            / denom)
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    if lo < 1e-12:
        lo = 0.0
    if 1.0 - hi < 1e-12:
        hi = 1.0
    return (lo, hi)


def summarize_proportion(counts):
    if isinstance(counts, tuple) and len(counts) == 2 and not isinstance(
            counts[0], (tuple, list)):
        counts = [counts]
    pairs = [tuple(c) for c in counts if c is not None and c[1] > 0]
    if not pairs:
        return {"mean": None, "sem": None, "ci": None, "n": 0,
                "n_events": 0, "method": "wilson"}
    k = sum(p[0] for p in pairs)
    n = sum(p[1] for p in pairs)
    return {
        "mean": k / n,
        "sem": None,
        "ci": _wilson_ci(k, n),
        "n": len(pairs),
        "n_events": n,
        "method": "wilson",
    }


def paired_deltas(candidate, reference, n_boot=2000, seed=0):
    pairs = [(c, r) for c, r in zip(candidate, reference)
             if c is not None and r is not None]
    if not pairs:
        return {"mean": None, "ci": None, "n": 0}
    deltas = [c - r for c, r in pairs]
    mean = statistics.mean(deltas)
    rng = random.Random(20261121 + seed)
    boot = []
    for _ in range(n_boot):
        sample = [rng.choice(deltas) for _ in deltas]
        boot.append(statistics.mean(sample))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot)) - 1]
    return {"mean": mean, "ci": (lo, hi), "n": len(pairs),
            "method": "bootstrap"}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def _rule_clauses(rule):
    """Rule -> list of clause frozensets (atoms), like the stream module."""
    clauses = list(rule) if isinstance(rule, tuple) else [rule]
    return [frozenset(tuple(a) for a in clause) for clause in clauses]


def truth_rules(stream):
    truths = set()
    for spec in stream:
        for door in ("D1", "D2"):
            rule = spec.config.rule_for(door)
            for clause in _rule_clauses(rule):
                truths.add((door, clause))
    return truths


def _ante_key(ante_list):
    return frozenset(tuple(a) for a in ante_list)


def match_hypotheses(mem, truths):
    correct, verified_correct, verified_wrong = set(), set(), set()
    contradicted_truth = set()
    for h in mem:
        ante = _ante_key(h["antecedent"])
        is_true = (h["door"], ante) in truths
        if is_true:
            correct.add((h["door"], ante))
            if h["status"] == "verified":
                verified_correct.add((h["door"], ante))
        elif h["status"] == "verified":
            verified_wrong.add((h["door"], ante))
        if h["status"] == "contradicted" and is_true:
            contradicted_truth.add((h["door"], ante))
    return {
        "correct_any": correct,
        "verified_correct": verified_correct,
        "verified_wrong": verified_wrong,
        "contradicted_truth": contradicted_truth,
    }


def discovery_counts(mem, truths):
    m = match_hypotheses(mem, truths)
    n_truths = len(truths) or 1
    n_verified = len(m["verified_correct"]) + len(m["verified_wrong"])
    return {
        "verified_correct": len(m["verified_correct"]),
        "verified_wrong": len(m["verified_wrong"]),
        "contradicted_truth": len(m["contradicted_truth"]),
        "hypotheses_total": len(mem),
        "coverage_counts": (len(m["correct_any"]), n_truths),
        "precision_counts": ((len(m["verified_correct"]), n_verified)
                             if n_verified else None),
    }


# ---------------------------------------------------------------------------
# Per-record helpers
# ---------------------------------------------------------------------------
def _rate_counts(recs, key):
    if not recs:
        return None
    return sum(1 for r in recs if r.get(key)), len(recs)


def _mean(recs, key, only_success=True):
    vals = [r.get(key) for r in recs if r.get("success") or not only_success]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _sum_delta(recs, key):
    return sum(r.get("trace_delta", {}).get(key, 0) for r in recs)


def _steps_total(recs):
    return sum(r["steps"] for r in recs) or 1


# ---------------------------------------------------------------------------
# Raw per-block metrics
# ---------------------------------------------------------------------------
def standard_train_raw(recs, truths):
    final = discovery_counts(recs[-1].get("mem", []) if recs else [], truths)
    steps = _steps_total(recs)
    return {
        "success_rate": _rate_counts(recs, "success"),
        "within_episode_discovery": _rate_counts(
            recs, "discovered_this_episode"),
        "verified_correct": final["verified_correct"],
        "verified_wrong": final["verified_wrong"],
        "hypotheses_total": final["hypotheses_total"],
        "coverage": final["coverage_counts"],
        "precision": final["precision_counts"],
        "planner_ops_per_step": _sum_delta(recs, "planner_ops") / steps,
        "fallback_rate": (_sum_delta(recs, "random_fallback_steps"), steps),
        "invalid_rate": (sum(r["invalid"] for r in recs), steps),
        "mean_steps_to_success": _mean(recs, "steps"),
    }


def eval_pool_raw(recs, pre_mem, truths):
    steps = _steps_total(recs)
    m = match_hypotheses(pre_mem or [], truths)
    n_true = len(truths) or 1
    cov = (len(m["verified_correct"]), n_true)
    return {
        "success_rate": _rate_counts(recs, "success"),
        "door_open_rate": _rate_counts(recs, "door_open"),
        "within_episode_discovery": _rate_counts(
            recs, "discovered_this_episode"),
        "zero_shot_coverage": cov,
        "planner_ops_per_step": _sum_delta(recs, "planner_ops") / steps,
        "fallback_rate": (_sum_delta(recs, "random_fallback_steps"), steps),
        "invalid_rate": (sum(r["invalid"] for r in recs), steps),
        "plan_cache_hits_per_step": (
            _sum_delta(recs, "plan_cache_hits") / steps),
        "plans_used_per_step": _sum_delta(recs, "plans_used") / steps,
        "mean_steps_to_success": _mean(recs, "steps"),
    }


def ab_phase_raw(recs):
    steps = _steps_total(recs)
    return {
        "success_rate": _rate_counts(recs, "success"),
        "within_episode_discovery": _rate_counts(
            recs, "discovered_this_episode"),
        "mean_steps_to_success": _mean(recs, "steps"),
        "planner_ops_per_step": _sum_delta(recs, "planner_ops") / steps,
        "fallback_rate": (_sum_delta(recs, "random_fallback_steps"), steps),
    }


def retention_raw(a1_recs, a2_recs, b_recs, a_cfg, mem_after):
    truths_a = {(door, clause) for door in ("D1", "D2")
                for clause in _rule_clauses(a_cfg.rule_for(door))}
    m = match_hypotheses(mem_after or [], truths_a)
    return {
        "a1_success": _rate_counts(a1_recs, "success"),
        "a2_success": _rate_counts(a2_recs, "success"),
        "b_success": _rate_counts(b_recs, "success"),
        "a_rule_retained_final": (
            len(m["verified_correct"]), len(truths_a)),
    }
