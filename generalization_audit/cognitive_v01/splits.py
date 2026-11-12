"""Frozen splits for the cognitive benchmark.

The Academy ships a deterministic split function
(``academy.envs.tasks.compute_splits``). This module snapshots the result
into a committed JSON under ``cognitive_v01/configs/`` so evaluation never
depends on later edits to shared files, verifies structural equality with
the live canonical computation, and checks disjointness + solvability.

Canonical serialization is hash-order independent (atom lists are sorted
lexicographically), so artifacts are stable across runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from academy.envs.gridworld import is_solvable
from academy.envs.tasks import compute_splits

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
FROZEN_SPLITS = _CONFIGS_DIR / "frozen_split_v01.json"


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------
def _canonical_rule(rule):
    """Rule -> canonical JSON (list of sorted atom lists)."""
    clauses = list(rule) if isinstance(rule, tuple) else [rule]
    return [sorted((list(a) for a in clause), key=repr) for clause in clauses]


def canonical_config_dict(config) -> dict:
    return {
        "topology": config.topology,
        "d1_rule": _canonical_rule(config.d1_rule),
        "d1_delay": config.d1_delay,
        "d2_rule": _canonical_rule(config.d2_rule),
        "d2_delay": config.d2_delay,
    }


def rule_from_canonical(d):
    from academy.envs.gridworld import EnvConfig
    def deser(rows):
        clauses = tuple(frozenset(map(tuple, row)) for row in rows)
        return clauses[0] if len(clauses) == 1 else clauses
    return EnvConfig(
        topology=d["topology"],
        d1_rule=deser(d["d1_rule"]),
        d1_delay=d["d1_delay"],
        d2_rule=deser(d["d2_rule"]),
        d2_delay=d["d2_delay"],
    )


# ---------------------------------------------------------------------------
# Freeze / load / verify
# ---------------------------------------------------------------------------
_REPO_SPLITS = (Path(__file__).resolve().parent.parent.parent /
                "results" / "academy_v01" / "splits.json")


def compute_live_splits():
    """Canonical split computation (same pure function as the Academy)."""
    raw = compute_splits(pool_split_seed=20260901)
    return {tag: [canonical_config_dict(c) for c in pool]
            for tag, pool in raw.items() if tag != "counts"}, raw


def load_repo_canonical_pools():
    """Read the Academy's canonical split file (cheap; no recompute)."""
    if not _REPO_SPLITS.exists():
        return None
    from academy.envs.tasks import load_splits
    raw = load_splits(str(_REPO_SPLITS))
    return {tag: [canonical_config_dict(c) for c in pool]
            for tag, pool in raw.items() if tag != "counts"}


def _frozen_payload(pools, raw_counts):
    return {
        "version": 1,
        "pool_split_seed": 20260901,
        "counts": raw_counts,
        "pools": pools,
    }


def freeze_splits(path=FROZEN_SPLITS):
    pools, raw = compute_live_splits()
    payload = _frozen_payload(pools, raw["counts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True),
                    encoding="utf-8")
    return path


def load_frozen_splits(path=FROZEN_SPLITS):
    if not path.exists():
        freeze_splits(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["pools"], payload["counts"]


def _set_of_pool(pool):
    return frozenset(
        (p["topology"],
         frozenset(tuple(map(tuple, c)) for c in p["d1_rule"]),
         p["d1_delay"],
         frozenset(tuple(map(tuple, c)) for c in p["d2_rule"]),
         p["d2_delay"])
        for p in pool)


def verify_frozen_splits(path=FROZEN_SPLITS, raise_on_mismatch=True,
                         light=True):
    """Structural equality between the committed snapshot and the canonical
    computation (order-insensitive rule equality).

    ``light=True`` compares against the Academy's committed canonical splits
    JSON (no recomputation); ``light=False`` recomputes the pure function.
    """
    frozen, counts = load_frozen_splits(path)
    if light:
        repo = load_repo_canonical_pools()
        if repo is None:
            return verify_frozen_splits(path, raise_on_mismatch, light=False)
        live = repo
        raw_counts = counts
        # repo file has no counts; require pool sizes to match
        ok = all(len(live[tag]) == len(frozen[tag])
                 for tag in ("train", "val", "test", "ood"))
    else:
        live, raw = compute_live_splits()
        raw_counts = raw["counts"]
        ok = counts == raw_counts

    ok = ok and all(_set_of_pool(live[tag]) == _set_of_pool(frozen[tag])
                    for tag in ("train", "val", "test", "ood"))
    if not ok and raise_on_mismatch:
        raise AssertionError("frozen splits diverge from canonical computation")
    return ok


def verify_solvable(pool):
    for c in pool:
        cfg = rule_from_canonical(c)
        if not is_solvable(cfg):
            raise AssertionError(f"frozen config not solvable: {cfg.key}")
    return True


def verify_disjoint(*pools):
    seen = set()
    for pool in pools:
        for c in pool:
            sig = tuple(
                (c["topology"],
                 frozenset(tuple(map(tuple, r)) for r in c["d1_rule"]),
                 c["d1_delay"],
                 frozenset(tuple(map(tuple, r)) for r in c["d2_rule"]),
                 c["d2_delay"]))
            if sig in seen:
                raise AssertionError("pool overlap detected")
            seen.add(sig)
    return True


def splits_snapshot() -> dict:
    """Public summary used in reports."""
    pools, counts = load_frozen_splits()
    return {
        "counts": counts,
        "pools": {tag: [canonical_config_dict(rule_from_canonical(c))
                         for c in pool] for tag, pool in pools.items()},
    }
