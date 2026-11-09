# METARGON Independent Audit v0.2 — Evaluation Integrity

**Auditor:** Independent evaluation & software-verification engineer
**Date:** 8 November 2026
**Branch:** `audit/evaluation-v02`
**Scope:** statistical reporting (Memory v0.3), public-observation
boundary, benchmark integrity, reproducibility, and frozen protocols for
future evaluations (v0.12 / Academy v0.4).

## Issue status summary

| # | Issue | Status |
|---|---|---|
| A1 | Memory v0.3 comparisons were reported with reversed difference sign (baseline − candidate) | **Confirmed → Corrected** |
| A2 | `paired_difference` convention was undocumented for ordering | **Corrected** (docstring) |
| A3 | `docs/MEMORY_V03.md` delta table and CIs had flipped signs | **Corrected** |
| A4 | Committed `memory_v03_summary.json` comparisons recomputed; raw evidence unchanged | **Corrected** |
| B1 | Hidden `EnvConfig` reachable by student policies through `episode_start(spec, …)` | **Confirmed → Corrected** (runner boundary) |
| B2 | No tests attempted hidden-config access through student-facing objects | **Corrected** (adversarial tests) |
| C1 | Missing policy checkpoint silently evaluated an untrained agent | **Confirmed → Corrected** (warning + marker) |
| C2 | `sample_substrate` whoseism depends on `PYTHONHASHSEED` | **Confirmed → Documented** (pinned) |
| C3 | Committed memory_v03 grid vs current splits | **Verified consistent** under pinned hash seed |
| C4 | Validation tests that fail when metadata/artifacts disagree | **Added** |
| D1 | Frozen Academy evaluation grid for future implementations | **Added** |
| D2 | Academy vs mechanism-lab separation | **Retained** |
| E1 | Cross-baseline compute equality in the Academy pipeline | **Not corrected** (out of scope; documented) |
| E2 | `Evaluator.compare` reference-order convention | **Not corrected** (backwards compat; documented) |

---

## Priority A — Correct statistical reporting

### Root cause

`academy/eval/metrics.py: paired_difference(a, b)` computes `a − b`.
The callers in `academy/tools/memory_experiment.py` and
`academy/tools/memory_merge.py` invoked it with `(original, variant)`,
so every reported "Classic Δ" was actually **baseline − candidate**.

Concretely, with original classic success 0.5625 and window 0.3125, the
pipeline reported **+0.2543** for window. A reader sees a positive delta
and concludes the variant improved, when in fact window *worsened*
(−0.25). The same sign error affected `ratio` (+0.2242 → −0.2242) and
`combined` (+0.1941 → −0.1941). `train_delta` and `retention_delta`
were already candidate − baseline and were untouched.

### Fixes

1. `academy/tools/memory_experiment.py` and `memory_merge.py`: call
   `paired_difference(variant, original)`; comment documents
   candidate − baseline.
2. `academy/eval/metrics.py`: updated docstring states the convention
   (positive = candidate improves).
3. `results/academy_v01/memory_v03/memory_v03_summary.json`: the
   `comparisons` block was recomputed with the corrected convention.
   Raw per-config vectors (`conditions_detail`), OOD, retention, and
   memory states are unchanged (no historical experiment data was
   rewritten).
4. `docs/MEMORY_V03.md`: delta table now shows corrected signs and CIs,
   with a one-line convention note.

### Verification

`generalization_audit/tests/test_integrity.py`:
- `test_signed_comparisons_are_negative_for_bad_variants` — window/ratio/
  combined must be negative;
- `test_comparisons_match_candidate_minus_baseline` — recomputes every
  comparison from raw per-config vectors (same seed) and asserts it
  equals the committed summary;
- `test_summary_conditions_detail_matches_jsonl` — commits summary
  `conditions_detail` must equal the JSONL records exactly.

---

## Priority B — Enforce the public-observation boundary

### Finding

`academy/eval/evaluator.py`, `academy/learning/loop.py`, and
`academy/evren/curriculum_experiment.py` passed the full `EpisodeSpec`
— including the hidden `EnvConfig` (door rules, delays) — to every
policy via `episode_start(spec, geom)`. Student policies did not read
it, but the boundary was convention, not enforcement.

### Fix (runner/protocol boundary only)

New `academy/eval/protocol.py`:

- `public_episode_spec(spec)` — a `SimpleNamespace` exposing only public
  fields (`max_steps`, `noise_q`, `spec_seed`, `pool_tag`,
  `difficulty`) with **no `config` attribute**;
- `spec_for(agent, spec)` — public view for all non-privileged
  policies; the full spec only for privileged diagnostic comparators
  (`OraclePolicy`, by `name == "oracle"`).

All three runner sites now call `agent.episode_start(spec_for(agent,
spec), env.geom)`. A student policy that attempts to read
`spec.config` now raises `AttributeError`.

### Tests (`test_academy_protocol.py`)

- public spec hides `config`; oracle is privileged; `spec_for` selects
  correctly;
- adversarial `_EvilStudent` cannot read `spec.config` through
  `run_one_episode`;
- `Evaluator.run_policy` passes public spec to a student and the full
  spec to the oracle.

---

## Priority C — Benchmark integrity

### C1. Silent untrained fallback (confirmed → corrected)

`academy/cli/main.py: cmd_eval` silently evaluated a *fresh, untrained*
agent when a policy checkpoint was missing. Now:

- `load_policy_state_or_warn(...)` prints a stderr warning naming the
  policy and stating the result is not a fair post-training measurement;
- the result dict is marked `"untrained_fallback": true` and persisted
  in `eval_<kind>.json`.

Tests cover both warning-on and warning-off behavior.

### C2. Hash-seed sensitivity of `sample_substrate` (confirmed → documented)

`academy/evren/retention.py: sample_substrate` sorts the pool with
`key=lambda c: str(c.key)`. `EnvConfig.key` contains `frozenset`
objects; `str(frozenset)` ordering is hash-seed dependent, so the
"deterministic" sampler yields *different samples* under different
`PYTHONHASHSEED` values. The memory_v03 experiment pinned
`PYTHONHASHSEED=0`, so its artifacts are internally consistent (verified
below); the code-level determinism claim is stronger than the code
guarantees.

Recorded in the frozen grid file (`hash_seed: "pinned (0)"`).

### C3. Committed memory_v03 evidence is reproducible (verified)

With `PYTHONHASHSEED=0` and the current `splits.json`, the committed
memory_v03 per-config grids reproduce exactly — classic 16/16, OOD
12/12. No stale artifact; this also cross-checks the frozen Academy grid
(see D1). A naive earlier comparison without the pinned hash seed
produced a misleading "stale" signal; the reproducible result is
reported here.

### C4. Summary `metadata` agreement checks (added)

`test_integrity.py` asserts committed summary metadata (`run_id`,
`seed`, `episodes`, `hash_randomization` false, `conditions_detail`,
`comparisons`) and recomputes comparisons from raw vectors.

---

## Priority D — Independent future evaluation

- LaboratoryFrozen Academy protocol** `generalization_audit/configs/academy_eval_grid_v01.json`:
  - classic: 16 test configs × 2 seeds × 4 episodes (sample seed
    20261108);
  - OOD: 12 configs × 1 seed × 3 episodes (sample seed 20261110);
  - matches the committed memory_v03 grid exactly under pinned hash seed;
  - future Academy implementations (v0.4 etc.) must be evaluated on these
    same configs/seeds, defined before observing their results.
- The mechanism-lab families R/L frozen protocols remain in
  `generalization_audit/configs/frozen_{r,l}.json`.
- Academy and mechanism-lab results are kept separate (different
  environments/protocols); no raw cross-environment success-rate
  comparisons.

---

## Not corrected / unresolved (documented)

1. **Academy compute equality across baselines (E1).** The Academy
   pipeline gives the Bayes baseline a larger per-decision planning
   budget than the student. Equalizing would require changing shared
   baseline code (out of scope). The audit framework provides
   compute-capped policy wrappers (`generalization_audit`) for future
   use; the Academy pipeline itself is not modified.
2. **`Evaluator.compare` order convention (E2).** `compare_all` names
   comparisons `reference_vs_name` and computes `results_a −
   results_b` (baseline − candidate). Preserved for backwards
   compatibility; documented here to prevent misreading as
   improvements. Use `paired_difference(candidate, baseline)` for any
   new reporting.
3. **R/L sample sizes.** Family R (2 seeds × 15 episodes) and family L
   (3 seeds × 20 episodes) remain preliminary; adequately replicated
   findings require 8+ seeds. This is stated in `AUDIT_REPORT.md`.

---

## Files changed (this audit)

| Path | Change |
|---|---|
| `academy/eval/protocol.py` | new public-observation boundary |
| `academy/eval/evaluator.py` | use `spec_for` |
| `academy/learning/loop.py` | use `spec_for` |
| `academy/evren/curriculum_experiment.py` | use `spec_for` |
| `academy/eval/metrics.py` | `paired_difference` docstring convention |
| `academy/tools/memory_experiment.py` | corrected comparison order |
| `academy/tools/memory_merge.py` | corrected comparison order |
| `academy/cli/main.py` | untrained-fallback warning + marker |
| `results/academy_v01/memory_v03/memory_v03_summary.json` | comparisons recomputed (candidate − baseline) |
| `docs/MEMORY_V03.md` | corrected delta table + convention note |
| `generalization_audit/configs/academy_eval_grid_v01.json` | frozen Academy grid |
| `generalization_audit/tests/test_academy_protocol.py` | new tests |
| `generalization_audit/tests/test_integrity.py` | new tests |
| `generalization_audit/docs/AUDIT_V02.md` | this report |
