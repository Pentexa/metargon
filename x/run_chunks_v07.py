"""Run each memory-v0.7 variant in its own subprocess; append to JSONL.

Resumable: variants already recorded are skipped. Usage:
    python x/run_chunks_v07.py [--quick]
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.getcwd()
APPEND = os.path.join(ROOT, "results", "academy_v01", "memory_v07",
                      "memory_v07_runs.jsonl")

FULL_VARIANTS = ["fresh", "original", "memory_wiped", "atomtx",
                 "shuffled", "epoch"]
QUICK_VARIANTS = ["fresh", "original", "shuffled", "epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()
    variants = args.variants or (QUICK_VARIANTS if args.quick
                                 else FULL_VARIANTS)
    os.makedirs(os.path.dirname(APPEND), exist_ok=True)
    done = set()
    if os.path.exists(APPEND):
        for line in open(APPEND, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["variant"])

    fresh_cache = os.path.join(
        ROOT, "results", "academy_v01", "memory_v07", "fresh_control.json")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    for v in variants:
        if v in done:
            print(f"skip {v} (already recorded)")
            continue
        print(f"run {v} ...", flush=True)
        code = (
            "import sys, os\n"
            "sys.path.insert(0, os.getcwd())\n"
            "from academy.tools.memory_experiment_v07 import "
            "run_v07_experiment\n"
            "run_v07_experiment(variants=%r, append_path=%r, verbose=True,\n"
            "                   fresh_cache_path=%r)\n"
        ) % ([v], APPEND, fresh_cache)
        r = subprocess.run(
            [sys.executable, "-u", "-c", code],
            cwd=ROOT, capture_output=True, text=True, env=env)
        print(r.stdout[-2500:])
        if r.stderr.strip():
            print("STDERR:", r.stderr[-2500:])
        if r.returncode != 0:
            print(f"!! {v} failed rc={r.returncode}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
