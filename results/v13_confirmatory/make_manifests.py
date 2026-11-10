#!/usr/bin/env python3
"""Generate METARGON v0.13 confirmatory experiment manifest configs + stamped
manifests (Quality v0.5 verifier compatible).

Artifacts live under results/v13_confirmatory/ (kept separate from pilot
results/v13/). Every data artifact gets a covering manifest.

Usage:
  python results/v13_confirmatory/make_manifests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "generalization_audit"))

from generalization_audit.tools.verify_experiment import stamp_manifest  # noqa: E402

RESULTS = ROOT / "results" / "v13_confirmatory"
MANIFESTS = ROOT / "generalization_audit" / "manifests"

METRICS = {
    "success_rate": "fraction of episodes completed (100 per seed)",
    "exact_discovery_D1": "fraction of episodes where a verified temporal hypothesis exactly matches the true ordered rule (first, second) for D1; predicted hypotheses not counted",
    "exact_discovery_D2": "fraction of episodes where a verified hypothesis exactly matches the true rule for D2",
    "covered_discovery_D1": "fraction of episodes where a verified hypothesis's antecedent is a subset of the true D1 rule",
    "covered_discovery_D2": "fraction of episodes where a verified hypothesis's antecedent is a subset of the true D2 rule",
    "temporal_probes": "temporal probe count per seed-run",
    "cpu_s": "wall-clock CPU seconds for the seed-run",
}

SPECS = {
    "final_l_8x100.json": (
        "v13-confirmatory-l", "l", ["stable", "novel"],
        "METARGON v0.13 confirmatory study, family L (ordered sequence), "
        "8 independent seeds x 100 episodes. Preregistered protocol in "
        "docs/RESEARCH_V13_CONFIRMATORY.md. Policies: metargon12 "
        "(v0.12.1 baseline), metargon13 (candidate), metargon13-gen "
        "(rolling-generator ablation), metargon13-notp (temporal off). "
        "Seeds 21800-21807, PYTHONHASHSEED=0.",
        "metargon_v13 (frozen v0.13 defaults)",
        "python/benchmark_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 100"),
    "final_f_8x100.json": (
        "v13-confirmatory-f", "f", ["noisy_obs"],
        "METARGON v0.13 confirmatory preservation cell, family F noisy_obs, "
        "8 independent seeds x 100 episodes. Seeds 21800-21807, "
        "PYTHONHASHSEED=0.",
        "metargon_v13 (frozen v0.13 defaults)",
        "python/benchmark_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 100"),
    "final_h_8x100.json": (
        "v13-confirmatory-h", "h", ["stable"],
        "METARGON v0.13 confirmatory preservation cell, family H stable, "
        "8 independent seeds x 100 episodes. Seeds 21800-21807, "
        "PYTHONHASHSEED=0.",
        "metargon_v13 (frozen v0.13 defaults)",
        "python/benchmark_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 100"),
    "final_r_8x100.json": (
        "v13-confirmatory-r", "r", ["stable"],
        "METARGON v0.13 confirmatory preservation cell, family R stable, "
        "8 independent seeds x 100 episodes. Seeds 21800-21807, "
        "PYTHONHASHSEED=0.",
        "metargon_v13 (frozen v0.13 defaults)",
        "python/benchmark_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 100"),
    "final_r_nodc_8x100.json": (
        "v13-confirmatory-r-nodc", "r", ["stable"],
        "METARGON v0.13 exploratory R cell with dual-context off "
        "(-nodc), 8 independent seeds x 100 episodes. Exploratory only; "
        "not part of the frozen confirmatory policy.",
        "metargon_v13 (dual-context off)",
        "python/benchmark_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 100"),
    "diag_l_stable_8x40.json": (
        "v13-confirmatory-diag-l-stable", "l", ["stable"],
        "METARGON v0.13 confirmatory diagnostics (Mission C), family L "
        "stable, 8 independent seeds x 40 episodes, instrumented v13 "
        "policy. Non-benchmark diagnostic format.",
        "metargon_v13 + diagnose_v13.py",
        "python/diagnose_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 40 --policy v13"),
    "diag_l_stable_v12_8x40.json": (
        "v13-confirmatory-diag-l-stable-v12", "l", ["stable"],
        "METARGON v0.13 confirmatory diagnostics (Mission C), family L "
        "stable, 8 independent seeds x 40 episodes, instrumented v12 "
        "policy. Non-benchmark diagnostic format.",
        "metargon_v12 + diagnose_v13.py",
        "python/diagnose_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 40 --policy v12"),
    "diag_l_novel_8x40.json": (
        "v13-confirmatory-diag-l-novel", "l", ["novel"],
        "METARGON v0.13 confirmatory diagnostics (Mission C), family L "
        "novel, 8 independent seeds x 40 episodes, instrumented v13 "
        "policy. Non-benchmark diagnostic format.",
        "metargon_v13 + diagnose_v13.py",
        "python/diagnose_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 40 --policy v13"),
    "diag_l_novel_v12_8x40.json": (
        "v13-confirmatory-diag-l-novel-v12", "l", ["novel"],
        "METARGON v0.13 confirmatory diagnostics (Mission C), family L "
        "novel, 8 independent seeds x 40 episodes, instrumented v12 "
        "policy. Non-benchmark diagnostic format.",
        "metargon_v12 + diagnose_v13.py",
        "python/diagnose_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 40 --policy v12"),
    "diag_h_stable_8x40.json": (
        "v13-confirmatory-diag-h-stable", "h", ["stable"],
        "METARGON v0.13 confirmatory diagnostics (Mission D), family H "
        "stable, 8 independent seeds x 40 episodes, instrumented v13 "
        "policy. Non-benchmark diagnostic format.",
        "metargon_v13 + diagnose_v13.py",
        "python/diagnose_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 40 --policy v13"),
    "diag_r_stable_8x40.json": (
        "v13-confirmatory-diag-r-stable", "r", ["stable"],
        "METARGON v0.13 confirmatory diagnostics (Mission D), family R "
        "stable, 8 independent seeds x 40 episodes, instrumented v13 "
        "policy. Non-benchmark diagnostic format.",
        "metargon_v13 + diagnose_v13.py",
        "python/diagnose_v13.py --seed-base 21800 --seeds 8 "
        "--episodes 40 --policy v13"),
}

import subprocess
_COMMIT = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
    check=False).stdout.strip() or "a617365"
COMMIT = _COMMIT  # starting main SHA (branch base); actual HEAD at stamp time


def make_cfg(fname, spec):
    art = json.loads((RESULTS / fname).read_text(encoding="utf-8"))
    meta = art.get("metadata", {})
    name, family, conds, desc, agent, harness = spec
    return {
        "schema_version": "experiment-manifest-v01",
        "protocol_version": "eval-protocol-v01",
        "experiment": {
            "name": name,
            "description": desc,
            "format": "benchmark_v12" if "diag" not in fname else "generic",
            "agent_version": agent,
            "harness": harness,
            "commit_sha": COMMIT,
            "python_version": "3.11.0",
            "platform": "windows",
            "hash_seed": "0",
            "family": family,
            "conditions": conds,
            "seeds": [i + 21800 for i in range(8)],
            "episodes_per_seed": meta.get("episodes_per_seed", 100),
            "action_budget_max_steps": 80,
            "policies": list(meta.get("policies", [])),
            "ablation_flags": {
                "use_recent_credit": False,
                "use_temporal": True,
                "PYTHONHASHSEED": 0,
                "study": "confirmatory-8x100",
            },
            "parameters": {
                "policy_seed_offset": 1000003,
                "world_seed_base": 21800,
            },
            "resources": {
                "planner_max_expand": 700,
                "probe_max_plans": 3,
                "memory_limit_bytes": None,
                "per_policy_cpu_s": True,
            },
            "evaluation": {
                "comparison_convention": "candidate_minus_baseline",
                "bootstrap": {
                    "method": "paired",
                    "seed": 4301,
                    "rounds": 1500,
                    "alpha": 0.05,
                },
                "metrics": METRICS,
            },
        },
        "artifacts": [
            {
                "path": f"../../results/v13_confirmatory/{fname}",
                "role": "raw" if "diag" not in fname else "summary",
                "generated_by": harness,
                "contains_per_seed": True,
            },
        ],
    }


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for fname, spec in SPECS.items():
        if not (RESULTS / fname).exists():
            print(f"skip (artifact not present yet): {fname}")
            continue
        cfg = make_cfg(fname, spec)
        if fname.startswith("final"):
            stem = f"v13_confirmatory_{fname.split('_')[1]}"
            if "nodc" in fname:
                stem += "_nodc"
        else:
            # diag_l_stable_8x40.json -> v13_confirmatory_diag_l_stable
            # diag_h_stable_8x40.json -> v13_confirmatory_diag_h_stable
            # diag_l_stable_v12_8x40.json -> v13_confirmatory_diag_l_stable_v12
            parts = fname.split("_")
            stem = f"v13_confirmatory_diag_{parts[1]}_{parts[2]}"
            if "v12" in fname:
                stem += "_v12"
        cfg_path = MANIFESTS / f"{stem}.cfg.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n",
                            encoding="utf-8")
        manifest = stamp_manifest(cfg, MANIFESTS)
        man_path = MANIFESTS / f"{stem}.manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2) + "\n",
                            encoding="utf-8")
        print(f"stamped {man_path.relative_to(ROOT)}")
        for a in manifest["artifacts"]:
            print(f"    {a['path']}: {a['sha256'][:12]}… ({a['size_bytes']} B)")


if __name__ == "__main__":
    main()
