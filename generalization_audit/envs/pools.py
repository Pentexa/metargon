#!/usr/bin/env python3
"""Frozen configuration pools for the generalization-audit families.

Every configuration is filtered by an *independent* ground-truth
solvability search over the family's own hidden dynamics (never shared
with any policy). Pools are split deterministically into disjoint
train/val/test, and the resulting configurations are serialized to a
frozen JSON file BEFORE any benchmarking takes place. Benchmarks load the
frozen file, so the exact test set is fixed ahead of time.
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

from lab_env import ACTIONS, GOAL, SWITCHES, PLATES, make_state, public_next

from .base import (
    L1_FIRST, L1_SECOND, L2_FIRST, L2_SECOND,
    R1_ATOMS, R2_ATOMS, LConfig, RConfig,
)

# --------------------------------------------------------------------------
# Solvability search over the family's own dynamics
# --------------------------------------------------------------------------
def _state_key(st) -> tuple:
    return (st.pos, st.objs["A"], st.objs["B"],
            tuple(st.switch_state[s] for s in SWITCHES),
            tuple(st.plate_load[p] for p in PLATES),
            tuple(st.door_open[d] for d in ("D1", "D2")))


def _r_transition(state, action, config: RConfig, hold_k: int, hold):
    """Family-R transition used ONLY by the independent solver."""
    nxt = public_next(state, action)
    if nxt is None:
        return None, [], hold
    events = []
    from lab_env import eval_atoms
    for door in ("D1", "D2"):
        cond = eval_atoms(config.cond_for(door), nxt)
        if not nxt.door_open[door]:
            if cond:
                hold[door] += 1
                if hold[door] >= hold_k:
                    nxt.door_open[door] = True
                    events.append((door, "open"))
            else:
                hold[door] = 0
        else:
            if not cond:
                if nxt.pos != _door_cell(door) and \
                        _occupied(nxt, _door_cell(door)) is None:
                    nxt.door_open[door] = False
                    hold[door] = 0
                    events.append((door, "close"))
    return nxt, events, hold


def _l_transition(state, action, config: LConfig, window: int, phase, phase_step, step):
    nxt = public_next(state, action)
    if nxt is None:
        return None, [], phase, phase_step
    events = []
    from lab_env import eval_atoms
    for door in ("D1", "D2"):
        first = eval_atoms(config.first_for(door), nxt)
        second = eval_atoms(config.second_for(door), nxt)
        if phase[door] == 0:
            if first and not second:
                phase[door] = 1
                phase_step[door] = step
        elif phase[door] == 1:
            if phase_step[door] is not None and step - phase_step[door] > window:
                phase[door] = 0
                phase_step[door] = None
            elif second:
                if not nxt.door_open[door]:
                    nxt.door_open[door] = True
                    events.append((door, "open"))
                phase[door] = 2
    return nxt, events, phase, phase_step


def _door_cell(door):
    from lab_env import DOOR_CELL
    return DOOR_CELL[door]


def _occupied(state, cell):
    return _occupied_by_object(state, cell)


def _occupied_by_object(state, cell):
    for o, c in state.objs.items():
        if c == cell:
            return o
    return None


def solvable_r(config: RConfig, hold_k: int = 3, max_plan: int = 80) -> bool:
    start = make_state()
    if start.pos == GOAL:
        return True
    q = deque([(start, 0, {"D1": 0, "D2": 0})])
    seen = set()
    seen.add((_state_key(start), 0, 0))
    while q:
        st, depth, hold = q.popleft()
        if depth >= max_plan:
            continue
        for a in ACTIONS:
            nxt, _ev, nhold = _r_transition(st, a, config, hold_k, dict(hold))
            if nxt is None:
                continue
            k = (_state_key(nxt), nhold["D1"], nhold["D2"])
            if k in seen:
                continue
            seen.add(k)
            if nxt.pos == GOAL:
                return True
            q.append((nxt, depth + 1, nhold))
    return False


def solvable_l(config: LConfig, window: int = 30, max_plan: int = 80) -> bool:
    start = make_state()
    if start.pos == GOAL:
        return True
    q = deque([(start, 0, {"D1": 0, "D2": 0}, {"D1": None, "D2": None})])
    seen = set()
    seen.add((_state_key(start), 0, 0, None, None))
    while q:
        st, depth, phase, phase_step = q.popleft()
        if depth >= max_plan:
            continue
        for a in ACTIONS:
            nxt, _ev, nph, nps = _l_transition(st, a, config, window,
                                                dict(phase), dict(phase_step),
                                                depth + 1)
            if nxt is None:
                continue
            k = (_state_key(nxt), nph["D1"], nph["D2"],
                 nps["D1"], nps["D2"])
            if k in seen:
                continue
            seen.add(k)
            if nxt.pos == GOAL:
                return True
            q.append((nxt, depth + 1, nph, nps))
    return False


# --------------------------------------------------------------------------
# Pool construction
# --------------------------------------------------------------------------
def _config_to_json(cfg) -> dict:
    if isinstance(cfg, RConfig):
        return {"family": "r",
                "d1_cond": sorted(cfg.d1_cond), "d2_cond": sorted(cfg.d2_cond)}
    if isinstance(cfg, LConfig):
        return {"family": "l",
                "d1_first": sorted(cfg.d1_first), "d1_second": sorted(cfg.d1_second),
                "d2_first": sorted(cfg.d2_first), "d2_second": sorted(cfg.d2_second)}
    raise TypeError(cfg)


def config_from_json(d: dict):
    if d["family"] == "r":
        return RConfig(d1_cond=frozenset(map(tuple, d["d1_cond"])),
                       d2_cond=frozenset(map(tuple, d["d2_cond"])))
    return LConfig(d1_first=frozenset(map(tuple, d["d1_first"])),
                   d1_second=frozenset(map(tuple, d["d1_second"])),
                   d2_first=frozenset(map(tuple, d["d2_first"])),
                   d2_second=frozenset(map(tuple, d["d2_second"])))


def enumerate_configs(family: str):
    if family == "r":
        return [RConfig(d1_cond=c1, d2_cond=c2)
                for c1 in R1_ATOMS for c2 in R2_ATOMS]
    if family == "l":
        return [LConfig(d1_first=f1, d1_second=s1, d2_first=f2, d2_second=s2)
                for f1 in L1_FIRST for s1 in L1_SECOND
                for f2 in L2_FIRST for s2 in L2_SECOND]
    raise ValueError(family)


SOLVABLE = {"r": solvable_r, "l": solvable_l}


def build_pools(family: str, seed_pools: int = 20260910,
                split=(0.60, 0.20, 0.20)):
    """Deterministic, disjoint train/val/test pools of solvable configs."""
    prng = random.Random(seed_pools)
    all_configs = enumerate_configs(family)
    solvable = [c for c in all_configs if SOLVABLE[family](c)]
    prng.shuffle(solvable)
    n = len(solvable)
    train_n = int(n * split[0])
    val_n = int(n * split[1])
    return (solvable[:train_n], solvable[train_n:train_n + val_n],
            solvable[train_n + val_n:]), n


def freeze_pools(family: str, path: Path, seed_pools: int = 20260910) -> dict:
    """Build pools and write the frozen configuration file (train/val/test
    configs + split seed + family + solvability counts). Returns the doc."""
    (train, val, test), n_solvable = build_pools(family, seed_pools)
    doc = {
        "family": family,
        "split_seed": seed_pools,
        "n_solvable": n_solvable,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train": [_config_to_json(c) for c in train],
        "val": [_config_to_json(c) for c in val],
        "test": [_config_to_json(c) for c in test],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def load_frozen_pools(doc: dict):
    return ([config_from_json(d) for d in doc["train"]],
            [config_from_json(d) for d in doc["val"]],
            [config_from_json(d) for d in doc["test"]])
