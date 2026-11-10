#!/usr/bin/env python3
"""Generate v0.13 experiment manifest configs + stamped manifests.

Reads each committed raw artifact's metadata (policies, conditions, seeds)
directly, so the manifest cannot drift from the artifact. Stamping uses
the merged Quality v0.5 verifier (`stamp_manifest`).

Usage:
  python results/v13/make_manifests.py

Writes:
  generalization_audit/manifests/v13_*.cfg.json      (config skeletons)
  generalization_audit/manifests/v13_*.manifest.json (stamped)
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

RESULTS = ROOT / "results" / "v13"
MANIFESTS = ROOT / "generalization_audit" / "manifests"

# artifact file -> (experiment name, family, description, agent_version,
#                   harness, ablation_flags, note)
SPECS = {
    "final_l.json": (
        "v13-final-l",
        "l",
        "METARGON v0.13 final paired runs, family L (ordered sequence), "
        "4 seeds x 40 episodes. v0.13 (rolling temporal completion) vs "
        "v0.12.1 and ablations; temporal exact-D1 claim.",
        "metargon_v13 (v0.13 defaults: rolling temporal completion; "
        "recent-credit off; dual context; adaptive confirm)",
        "python/benchmark_v13.py",
        {"use_recent_credit": False, "use_temporal": True,
         "temporal_completion_mode": "rolling",
         "family_l_frozen_pool": "generalization_audit/configs/frozen_l.json"},
        "temporal delta v13 vs v13-notp; v12 temporal delta reference",
    ),
    "final_f.json": (
        "v13-final-f",
        "f",
        "METARGON v0.13 final paired runs, family F (declared), all 5 "
        "conditions, 4 seeds x 40 episodes. Regression preservation vs "
        "v0.12.1 and v0.10/v0.11.",
        "metargon_v13 (v0.13 defaults; temporal off on F)",
        "python/benchmark_v13.py",
        {"use_recent_credit": False, "use_temporal": False,
         "declared_family_pools": "lab_env split_pools()"},
        "preservation matrix: v13 == v12 on every cell",
    ),
    "final_h.json": (
        "v13-final-h",
        "h",
        "METARGON v0.13 final paired runs, family H (structurally novel), "
        "stable/novel, 4 seeds x 40 episodes. Regression preservation.",
        "metargon_v13 (v0.13 defaults; temporal off on H)",
        "python/benchmark_v13.py",
        {"use_recent_credit": False, "use_temporal": False,
         "h_pools_seed": 20261107},
        "preservation matrix: v13 == v12 on every cell",
    ),
    "final_r.json": (
        "v13-final-r",
        "r",
        "METARGON v0.13 final paired runs, family R (sustained-hold, "
        "audit frozen pool read-only), stable/novel, 4 seeds x 40 "
        "episodes. Regression preservation incl. -nodc reference.",
        "metargon_v13 (v0.13 defaults; temporal off on R)",
        "python/benchmark_v13.py",
        {"use_recent_credit": False, "use_temporal": False,
         "family_r_frozen_pool": "generalization_audit/configs/frozen_r.json"},
        "preservation matrix; nodc column documents the dual-context cost",
    ),
    "baseline_l.json": (
        "v13-baseline-l",
        "l",
        "METARGON v0.12.1 baseline reproduction, family L, stable/novel, "
        "4 seeds x 40 episodes (metargon12 + metargon12-notp).",
        "metargon_v12 (v0.12.1 defaults: recent-credit off, temporal on "
        "for L)",
        "python/benchmark_v12.py",
        {"use_recent_credit": False, "use_temporal": True,
         "family_l_frozen_pool": "generalization_audit/configs/frozen_l.json"},
        "baseline matches v0.12.1 study means (16.9/23.1 exact D1)",
    ),
    "pilot_l.json": (
        "v13-pilot-l",
        "l",
        "METARGON v0.13 pilot, family L, stable/novel, 4 seeds x 40 "
        "episodes, full ablation set (-gen/-tw/-tpex/-notp).",
        "metargon_v13 (pilot config: rolling + proximity + tpex)",
        "python/benchmark_v13.py",
        {"use_recent_credit": False, "use_temporal": True,
         "temporal_completion_mode": "rolling",
         "temporal_support_weighting": "proximity",
         "temporal_target_exempt": True,
         "family_l_frozen_pool": "generalization_audit/configs/frozen_l.json"},
        "pilot: affinity knobs measured neutral; adopted defaults are "
        "rolling + flat + no exemption",
    ),
}

METRICS = {
    "success_rate": "fraction of episodes completed (40 per seed)",
    "exact_discovery_D1": "fraction of episodes where a verified temporal hypothesis exactly matches the true ordered rule (first, second) for D1; predicted hypotheses not counted",
    "exact_discovery_D2": "fraction of episodes where a verified hypothesis (compositional or temporal) exactly matches the true rule for D2",
    "covered_discovery_D1": "fraction of episodes where a verified hypothesis's antecedent is a subset of the true D1 rule",
    "covered_discovery_D2": "fraction of episodes where a verified hypothesis's antecedent is a subset of the true D2 rule",
    "temporal_probes": "temporal probe count per seed-run",
    "cpu_s": "wall-clock CPU seconds for the seed-run",
}


def make_cfg(fname, spec):
    art = json.loads((RESULTS / fname).read_text(encoding="utf-8"))
    meta = art["metadata"]
    name, family, desc, agent, harness, abl, note = spec
    return {
        "schema_version": "experiment-manifest-v01",
        "protocol_version": "eval-protocol-v01",
        "experiment": {
            "name": name,
            "description": desc,
            "format": "benchmark_v12",
            "agent_version": agent,
            "harness": harness,
            "commit_sha": "437cf9d",
            "python_version": "3.11.0",
            "platform": "windows",
            "hash_seed": "unset (randomized per process)",
            "family": family,
            "conditions": list(meta["conditions"]),
            "seeds": [i + 11800 for i in range(meta["seeds"])],
            "episodes_per_seed": meta["episodes_per_seed"],
            "action_budget_max_steps": 80,
            "policies": list(meta["policies"]),
            "ablation_flags": abl,
            "parameters": {
                "policy_seed_offset": 1000003,
                "world_seed_base": 11800,
                "note": note + "; produced with code byte-identical to "
                        "commit 437cf9d (git diff 4e61903..437cf9d empty)",
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
                "path": f"../../results/v13/{fname}",
                "role": "raw",
                "generated_by": f"{harness} --seeds 4 --episodes 40 "
                                f"--family {family} --no-mem",
                "contains_per_seed": True,
            },
        ] + ([
            {
                "path": f"../../results/v13/rendered_final_{family}.md",
                "role": "table",
                "generated_by": "python/v13_postprocess.py",
                "contains_per_seed": True,
            },
        ] if fname in ("final_l.json", "final_f.json", "final_h.json",
                       "final_r.json") else []),
    }


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for fname, spec in SPECS.items():
        cfg = make_cfg(fname, spec)
        parts = spec[0].split("-")
        stem = f"v13_{parts[1]}_{parts[2]}"
        cfg_path = MANIFESTS / f"{stem}.cfg.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        manifest = stamp_manifest(cfg, MANIFESTS)
        man_path = MANIFESTS / f"{stem}.manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2) + "\n",
                            encoding="utf-8")
        print(f"stamped {man_path.relative_to(ROOT)}")
        for a in manifest["artifacts"]:
            print(f"    {a['path']}: {a['sha256'][:12]}… ({a['size_bytes']} B)")


if __name__ == "__main__":
    main()
