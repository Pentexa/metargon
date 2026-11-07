#!/usr/bin/env python3
"""Freeze audit configuration pools BEFORE benchmarking.

Usage (from repository root):

    python generalization_audit/freeze.py --family r
    python generalization_audit/freeze.py --family l

Writes generalization_audit/configs/frozen_<family>.json containing the
exact train/val/test pools, split seed, and solvability counts. The
benchmark loads these frozen files, so the test set is fixed in advance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from generalization_audit.envs.pools import freeze_pools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("r", "l"), default="r")
    parser.add_argument("--seed-pools", type=int, default=20260910)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (
        Path("generalization_audit/configs") / f"frozen_{args.family}.json")
    doc = freeze_pools(args.family, out, args.seed_pools)
    print(f"froze family={args.family}: solvable={doc['n_solvable']} "
          f"train={doc['n_train']} val={doc['n_val']} test={doc['n_test']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
