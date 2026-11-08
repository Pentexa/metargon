#!/usr/bin/env python3
"""Shortcut and leakage checks for the audit benchmark.

* solvability check  — every frozen configuration must be solvable under
                        the family's own dynamics (independent solver).
* no-hidden-access    — after an episode, no policy object may hold a
                        reference to any configuration object; run-time
                        policy state is scanned one level deep.
* noise parity        — at equal step indexes of the same episode seed,
                        all policies observe identical corrupted readings.
"""
from __future__ import annotations

import json


def assert_all_solvable(frozen_doc: dict, solvable_fn) -> None:
    from ..envs.pools import config_from_json
    bad = []
    for split in ("train", "val", "test"):
        for d in frozen_doc[split]:
            cfg = config_from_json(d)
            if not solvable_fn(cfg):
                bad.append((split, d))
    if bad:
        raise AssertionError(
            f"{len(bad)} frozen configuration(s) are not solvable: {bad[:5]}")


def assert_no_hidden_access(policy, config) -> None:
    """Scan a policy's state after an episode for hidden-config references.
    A policy that stored the config object would leak hidden rules."""
    cfg_types = (type(config),)
    for value in vars(policy).values():
        if isinstance(value, cfg_types):
            raise AssertionError(
                f"policy {getattr(policy, 'name', policy)} stored a hidden "
                f"config object")
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, cfg_types):
                    raise AssertionError(
                        f"policy {getattr(policy, 'name', policy)} stored a "
                        f"hidden config object in a container")


def noise_parity_check(world_cls, config, ep_seed, steps: int = 40):
    """Verify deterministic per-step corrupted readings for a given
    episode seed: two independent worlds produce identical readings at
    identical step indexes."""
    w1 = world_cls(config, ep_seed=ep_seed, noise_q=0.08)
    w2 = world_cls(config, ep_seed=ep_seed, noise_q=0.08)
    from lab_env import ACTIONS
    for i in range(steps):
        a = ACTIONS[i % len(ACTIONS)]
        r1 = w1.step(a) if not w1.finished else None
        r2 = w2.step(a) if not w2.finished else None
        if r1 is None or r2 is None:
            continue
        o1, o2 = r1.observation, r2.observation
        assert (o1.s1, o1.s2, o1.p1, o1.p2, o1.d1, o1.d2) == \
               (o2.s1, o2.s2, o2.p1, o2.p2, o2.d1, o2.d2), \
               f"noise parity broken at step {i}"
    return True
