#!/usr/bin/env python3
"""Offline generalization metrics for the audit benchmark.

All metrics are computed offline from per-episode snapshots after the
run; they are never available to any policy during interaction.

* exact discovery   — the verified/predicted hypothesis antecedent equals
                      the true door condition.
* coverage          — true condition is a subset of the discovered
                      antecedent (superset hypothesis).
* false-verified    — fraction of verified hypotheses whose antecedent is
                      not a subset of the true condition.
* forgetting        — after a mechanism shift, how much of the pre-shift
                      verified knowledge disappears (Jaccard complement on
                      verified antecedents per door).
* adaptation        — first post-shift episode with a success, and the
                      change in success rate across the shift.
"""
from __future__ import annotations

import statistics


def _true_cond(spec, door: str):
    cfg = spec.config
    if hasattr(cfg, "cond_for"):
        return cfg.cond_for(door)
    return None


def discovery_from_rows(rows):
    """rows: list of (spec, hypotheses_snapshot). Returns per-door stats."""
    out = {"D1": {"n": 0, "exact": 0, "covered": 0, "false_verified": 0},
           "D2": {"n": 0, "exact": 0, "covered": 0, "false_verified": 0}}
    for spec, hs in rows:
        for door in ("D1", "D2"):
            true_cond = _true_cond(spec, door)
            if true_cond is None:
                continue
            out[door]["n"] += 1
            exact = 0
            covered = 0
            false_ver = 0
            for h in hs:
                if getattr(h, "door", None) != door:
                    continue
                if h.status not in ("verified", "predicted"):
                    continue
                ante = frozenset(h.antecedent)
                if ante == frozenset(true_cond):
                    exact = 1
                if frozenset(true_cond) <= ante:
                    covered = 1
                if not (frozenset(true_cond) <= ante):
                    false_ver += 1
            out[door]["exact"] += exact
            out[door]["covered"] += covered
            out[door]["false_verified"] += false_ver
    return out


def summarize_discovery(stats):
    s = {}
    for door, st in stats.items():
        n = max(1, st["n"])
        s[door] = {
            "n": st["n"],
            "exact_rate": st["exact"] / n,
            "covered_rate": st["covered"] / n,
            "false_verified_per_episode": st["false_verified"] / n,
        }
    return s


def verified_antecedents_by_door(hs, door: str):
    return {frozenset(h.antecedent) for h in hs
            if getattr(h, "door", None) == door
            and h.status in ("verified", "predicted")}


def forgetting_between(pre_hs, post_hs, door: str):
    """Jaccard complement of verified antecedents between two episode
    windows for a door. 1.0 = all pre-shift knowledge gone; 0.0 = none
    forgotten."""
    pre = verified_antecedents_by_door(pre_hs, door)
    post = verified_antecedents_by_door(post_hs, door)
    union = pre | post
    if not union:
        return None
    return len(pre - post) / len(union)


def first_recovery_episode(successes, shift_at: int):
    """Index (0-based) of the first success on/after the shift episode."""
    for i in range(shift_at, len(successes)):
        if successes[i]:
            return i
    return None


def adaptation_summary(pre_success, post_success, shift_at: int):
    if not post_success:
        return {"pre_success_rate": None, "post_success_rate": None,
                "delta": None, "first_recovery_episode": None}
    pre_rate = statistics.fmean(pre_success) if pre_success else None
    post_rate = statistics.fmean(post_success)
    return {
        "pre_success_rate": pre_rate,
        "post_success_rate": post_rate,
        "delta": (None if pre_rate is None else post_rate - pre_rate),
        "first_recovery_episode": first_recovery_episode(
            list(pre_success) + list(post_success), shift_at),
    }
