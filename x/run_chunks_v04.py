"""Run each memory-v0.4 variant in its own subprocess (robust against
environmental kills); append results to JSONL; combine at the end."""
import json
import os
import subprocess
import sys

ROOT = os.getcwd()
APPEND = os.path.join(ROOT, "results", "academy_v01", "memory_v04",
                      "memory_v04_runs.jsonl")
VARIANTS = ["fresh", "original", "probe_local", "ctx", "ctx_reset",
            "ctx_gate", "ctx_full", "memory_wiped"]

os.makedirs(os.path.dirname(APPEND), exist_ok=True)
done = set()
if os.path.exists(APPEND):
    for line in open(APPEND, encoding="utf-8"):
        line = line.strip()
        if line:
            done.add(json.loads(line)["variant"])

code = (
    "import sys, os\n"
    "sys.path.insert(0, os.getcwd())\n"
    "from academy.tools.memory_experiment_v04 import run_v04_experiment\n"
    "run_v04_experiment(variants=%r, append_path=%r, verbose=True)\n"
)

for v in VARIANTS:
    if v in done:
        print(f"skip {v} (already recorded)")
        continue
    print(f"run {v} ...", flush=True)
    r = subprocess.run([sys.executable, "-u", "-c", code % ([v], APPEND)],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.stderr.strip():
        print("STDERR:", r.stderr[-2000:])
    if r.returncode != 0:
        print(f"!! {v} failed rc={r.returncode}", flush=True)
print("ALL DONE")
