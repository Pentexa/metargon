# METARGON Cognitive Intelligence Benchmark v0.1

Independent cognitive evaluation of the **actual** Academy student agent
(`MetargonAcademyAgent`), without modifying it. Measures six dimensions:

| # | Dimension | Definition |
|---|-----------|------------|
| 1 | Causal discovery | Can the student discover hidden mechanisms (verified rules matching ground truth)? |
| 2 | Learning efficiency | How much experience is required (episodes-to-first-success / -verified-rule, learning curves)? |
| 3 | Knowledge retention | Does learning survive new episodes (continual revisit) and restart (save/restore)? |
| 4 | Generalization | Does knowledge transfer to unseen rules/configurations (zero-shot on val/test/OOD)? |
| 5 | Adaptability | Can the student detect and recover from wrong beliefs (rule-switch protocol)? |
| 6 | Planning efficiency | Does experience lead to better action choices (steps, fallback, invalid actions, planner cost)? |

## Principles

- **Frozen splits** — train/val/test/OOD pools are snapshotted from the
  Academy's canonical split function
  (`generalization_audit/cognitive_v01/configs/frozen_split_v01.json`) and
  verified (disjointness, solvability, equality with the canonical
  computation) before every run.
- **Matched streams** — every policy sees the *same* deterministic episode
  stream per seed (same configs, seeds, noise, budgets).
- **No outcome-based selection** — configs are sampled by fixed rules
  (canonical ordering + seeded streams), never by measured performance.
- **No hidden leakage** — the student only receives public observations;
  hidden dynamics are used only by the privileged *oracle* (diagnostic,
  excluded from rankings). A leakage test audits this.
- **Multiple seeds + CIs** — bounded probability metrics (success rates,
  coverage, precision, correction fractions) report a pooled 95% Wilson
  interval (always inside [0, 1]); unbounded metrics (steps,
  episodes-to-event, planner counters) report mean ± 95% Student-t CI
  over independent seeds. The method is recorded per metric in the
  scorecard and report.
- **Performance vs memory vs search separated** — the scorecard reports
  task success, verified-knowledge quality (precision/coverage), and
  planner/search counters independently, plus a memory-muted control
  (identical training exposure, declarative memory emptied before
  evaluation) to isolate the contribution of learned knowledge.
- **Planning counters are named accurately** — `plan_cache_hits` counts
  cache lookups that found an entry, while `plans_used` counts executed
  plans; they are not a hit/miss pair, so no "hit rate" is computed.
  The scorecard reports `plan_cache_hits_per_step` and
  `plans_used_per_step` separately.

## Policies

| Policy | Role |
|--------|------|
| `metargon_trained` | unmodified student, trained on the train stream |
| `metargon_muted` | same training, declarative memory wiped before eval (control) |
| `metargon_fresh` | unmodified student, no prior training |
| `random` / `episodic` / `qlearning` | model-free baselines |
| `oracle` | privileged perfect-knowledge planner (diagnostic only) |

## Running

```bash
# baseline (4 seeds)
python -m generalization_audit.cognitive_v01.benchmark_cognitive \
    --seeds 0 1 2 3 \
    --out generalization_audit/cognitive_v01/results

# quick smoke (CI / tests)
python -m generalization_audit.cognitive_v01.benchmark_cognitive --quick
```

Outputs:

- `results/summary.json` — the scorecard (dimensions × metrics, mean ± CI).
- `results/per_seed.json` — full per-seed raw records.
- `docs/COGNITIVE_REPORT_v01.md` — rendered report.

## Tests

```bash
python -m unittest discover -s generalization_audit/cognitive_v01/tests -t . -v
```

Coverage: frozen-split integrity, protocol determinism/structure, no-leak
checks, driver/policy behavior, metric unit tests, and an end-to-end smoke
of the runner + scorecard + report.

## Reproducibility

- Streams are pure functions of `(seed, frozen pools)`.
- The git head and all sizes/seeds are recorded in the scorecard `meta`.
- The benchmark can be re-run on any candidate branch without touching its
  code: it only imports the public Academy APIs.

## Limitations (see report for full detail)

- Finite budgets, seeds, and one task family (Academy gridworld) — results
  describe this student *on this task distribution*.
- Success requires reaching the goal within 80 steps; hard configs may be
  unreachable within budget for any policy.
- The memory-muted control still allows within-episode learning during
  evaluation; it removes only cross-episode declarative knowledge.
