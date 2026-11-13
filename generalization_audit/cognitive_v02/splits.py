"""Frozen splits for Cognitive Evaluation v0.2.

Snapshots the Academy's canonical split computation into a committed JSON
(`configs/frozen_split_v02.json`) and verifies disjointness, solvability
(sample) and equality with the canonical computation. Serialization is
hash-order independent (atom lists sorted lexicographically).
"""
from __future__ import annotations

import json
from pathlib import Path

from academy.envs.gridworld import is_solvable
from academy.envs.tasks import compute_splits

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
FROZEN_SPLITS = _CONFIGS_DIR / "frozen_split_v02.json"

_REPO_SPLITS = (Path(__file__).resolve().parent.parent.parent /
                "results" / "academy_v01" / "splits.json")


def _canonical_rule(rule):
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


def config_from_canonical(d):
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


def load_repo_canonical_pools():
    if not _REPO_SPLITS.exists():
        return None
    from academy.envs.tasks import load_splits
    raw = load_splits(str(_REPO_SPLITS))
    return {tag: [canonical_config_dict(c) for c in pool]
            for tag, pool in raw.items() if tag != "counts"}


def freeze_splits(path=FROZEN_SPLITS):
    raw = compute_splits(pool_split_seed=20260901)
    pools = {tag: [canonical_config_dict(c) for c in pool]
             for tag, pool in raw.items() if tag != "counts"}
    payload = {"version": 2, "pool_split_seed": 20260901,
               "counts": raw["counts"], "pools": pools}
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


def verify_frozen_splits(path=FROZEN_SPLITS, raise_on_mismatch=True):
    frozen, counts = load_frozen_splits(path)
    repo = load_repo_canonical_pools()
    if repo is None:
        return True
    ok = all(len(repo[tag]) == len(frozen[tag])
             for tag in ("train", "val", "test", "ood"))
    ok = ok and all(_set_of_pool(repo[tag]) == _set_of_pool(frozen[tag])
                    for tag in ("train", "val", "test", "ood"))
    if not ok and raise_on_mismatch:
        raise AssertionError("frozen splits diverge from canonical computation")
    return ok


def verify_solvable_sample(pool, k=8):
    for c in pool[:k]:
        cfg = config_from_canonical(c)
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


def pools_dict():
    pools, counts = load_frozen_splits()
    return {
        "counts": counts,
        "pools": {tag: [config_from_canonical(c) for c in pool]
                  for tag, pool in pools.items()},
    }
