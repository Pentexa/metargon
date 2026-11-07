"""Regenerate the immutable Academy task splits, parallelized.

The split computation itself is deterministic (same seed -> same pools);
parallelism is used only to evaluate ``is_solvable`` for independent
configurations, so results are bit-identical to the sequential version.

Usage:
    python -m academy.tools.regen_splits [--seed 20260901]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from academy.envs.tasks import (  # noqa: E402
    DEFAULT_SPLITS_PATH, _enumerate_configs, CORE_RULESETS, OOD_RULESETS,
    save_splits,
)
from academy.envs.gridworld import is_solvable  # noqa: E402


def _workers():
    return max(2, min(8, (os.cpu_count() or 4)))


def _solvable_flags(configs):
    with ProcessPoolExecutor(max_workers=_workers()) as ex:
        return list(ex.map(is_solvable, configs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default=DEFAULT_SPLITS_PATH)
    args = ap.parse_args(argv)

    import random
    from academy.envs.tasks import compute_splits

    core_all = []
    for tp in ("TP1", "TP2"):
        core_all.extend(_enumerate_configs(CORE_RULESETS, tp))
    ood_all = _enumerate_configs(OOD_RULESETS, "TP2")

    prng = random.Random(args.seed)
    solvable_flags = _solvable_flags(core_all)
    solvable = [c for c, ok in zip(core_all, solvable_flags) if ok]
    prng.shuffle(solvable)
    n = len(solvable)
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    train = solvable[:train_n]
    val = solvable[train_n:train_n + val_n]
    test = solvable[train_n + val_n:]

    ood_flags = _solvable_flags(ood_all)
    ood_solvable = [c for c, ok in zip(ood_all, ood_flags) if ok]
    prng.shuffle(ood_solvable)
    ood = ood_solvable[:25]

    splits = {
        "train": train, "val": val, "test": test, "ood": ood,
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "ood": len(ood), "solvable": n, "enumerated": len(core_all),
        },
    }
    path = save_splits(splits, args.out)
    print(f"splits written to {path}: {splits['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
