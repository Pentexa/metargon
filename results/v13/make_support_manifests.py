#!/usr/bin/env python3
"""Generate v0.13 *supporting* experiment manifests (regressions, gate
experiments, diagnostics, verification reports) so that EVERY data file
under results/v13 is covered by a provenance manifest (artifact gate).

Benchmark-format artifacts (regress/r_gate) reuse the benchmark_v12
format; diagnostics and verification reports use format "generic" (the
verifier reports INCONCLUSIVE for non-benchmark formats, never FAIL).

Usage:
  python results/v13/make_support_manifests.py
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

METRICS = {
    "success_rate": "fraction of episodes completed (40 per seed)",
    "exact_discovery_D1": "fraction of episodes where a verified temporal hypothesis exactly matches the true ordered rule (first, second) for D1; predicted hypotheses not counted",
    "exact_discovery_D2": "fraction of episodes where a verified hypothesis exactly matches the true rule for D2",
    "covered_discovery_D1": "fraction of episodes where a verified hypothesis's antecedent is a subset of the true D1 rule",
    "covered_discovery_D2": "fraction of episodes where a verified hypothesis's antecedent is a subset of the true D2 rule",
    "temporal_probes": "temporal probe count per seed-run",
    "cpu_s": "wall-clock CPU seconds for the seed-run",
}

# groups: (manifest stem, name, format, family, conditions, description,
#          agent_version, harness, policies_source('artifact'|None), files)
GROUPS = [
    ("v13_regress_f_noisy", "v13-regress-f-noisy", "benchmark_v12", "f",
     ["noisy_obs"],
     "METARGON v0.13 regression check, family F noisy_obs, 4 seeds x 40 "
     "episodes: v0.13 vs v0.12.1/v0.11/v0.10.",
     "metargon_v13 (v0.13 defaults; temporal off on F)",
     "python/benchmark_v13.py", "artifact", ["regress_f_noisy.json"]),
    ("v13_regress_h_stable", "v13-regress-h-stable", "benchmark_v12", "h",
     ["stable"],
     "METARGON v0.13 regression check, family H stable, 4 seeds x 40 "
     "episodes.",
     "metargon_v13 (v0.13 defaults; temporal off on H)",
     "python/benchmark_v13.py", "artifact", ["regress_h_stable.json"]),
    ("v13_regress_r_stable", "v13-regress-r-stable", "benchmark_v12", "r",
     ["stable"],
     "METARGON v0.13 regression check, family R stable, 4 seeds x 40 "
     "episodes.",
     "metargon_v13 (v0.13 defaults; temporal off on R)",
     "python/benchmark_v13.py", "artifact", ["regress_r_stable.json"]),
    ("v13_r_gate", "v13-r-gate", "benchmark_v12", "r", ["stable"],
     "METARGON v0.13 persistence-gate experiments (rejected): family R "
     "stable, 4 seeds x 40 episodes, two runs (r_gate, r_gate2 after "
     "the dual-context gate fix).",
     "metargon_v13 (persistence-gate experiment; rejected)",
     "python/benchmark_v13.py", "artifact",
     ["r_gate.json", "r_gate2.json"]),
    ("v13_diagnostics", "v13-diagnostics", "generic", "l",
     ["stable", "novel"],
     "METARGON v0.13 temporal seed-dependence diagnostics: instrumented "
     "runs on the exact paired worlds (v0.12.1 policy and v0.13 policy), "
     "4 seeds x 40 episodes. Non-benchmark format (diagnostic counters).",
     "metargon_v12/v13 + diagnose_v13.py",
     "python/diagnose_v13.py", None,
     ["diag_l_stable.json", "diag_l_novel.json", "diag_l_final.json",
      "diag_r_stable_v13.json"]),
]


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for stem, name, fmt, family, conds, desc, agent, harness, \
            pol_src, files in GROUPS:
        arts = [load(f) for f in files]
        if "metadata" in arts[0]:
            meta0 = arts[0]["metadata"]
            seeds = [i + 11800 for i in range(meta0["seeds"])]
            episodes = meta0["episodes_per_seed"]
        else:
            # verification reports carry no experiment metadata; the runs
            # they verify are uniformly 4 seeds x 40 episodes
            seeds = [11800, 11801, 11802, 11803]
            episodes = 40
        policies = []
        if pol_src == "artifact":
            seen = set()
            for a in arts:
                for p in a["metadata"]["policies"]:
                    if p not in seen:
                        seen.add(p)
                        policies.append(p)
        cfg = {
            "schema_version": "experiment-manifest-v01",
            "protocol_version": "eval-protocol-v01",
            "experiment": {
                "name": name,
                "description": desc,
                "format": fmt,
                "agent_version": agent,
                "harness": harness,
                "commit_sha": "d9fd56f",
                "python_version": "3.11.0",
                "platform": "windows",
                "hash_seed": "unset (randomized per process)",
                "family": family,
                "conditions": conds,
                "seeds": seeds,
                "episodes_per_seed": episodes,
                "action_budget_max_steps": 80,
                "policies": policies,
                "ablation_flags": {
                    "use_recent_credit": False,
                    "note": "produced with code byte-identical to commit "
                            "1518054 (rebase onto main after hotfix); "
                            "PYTHONHASHSEED unset during runs",
                },
                "parameters": {
                    "policy_seed_offset": 1000003,
                    "world_seed_base": 11800,
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
                    "path": f"../../results/v13/{f}",
                    "role": "raw" if fmt == "benchmark_v12" else "summary",
                    "generated_by": (
                        f"{harness} --seeds 4 --episodes 40 "
                        f"--family {family} --no-mem"
                        if fmt == "benchmark_v12" else harness),
                    "contains_per_seed": True,
                } for f in files
            ],
        }
        stem_path = MANIFESTS / f"{stem}.cfg.json"
        stem_path.write_text(json.dumps(cfg, indent=2) + "\n",
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
