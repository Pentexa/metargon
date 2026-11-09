# METARGON Independent Audit v0.3 — Evaluation Integrity

**Auditor:** Independent evaluation & software-verification engineer
**Date:** 9 November 2026
**Branch:** `audit/evaluation-v03`
**Scope:** v0.12 report arithmetic (RESEARCH_V12.md), Memory v0.4
report signs (MEMORY_V04.md) vs committed raw artifacts; frozen criteria
for v0.12.1 / Academy v0.5.

## Issue status summary

| # | Issue | Status |
|---|---|---|
| 1 | v0.12 L-family "temporal Δ" column values swapped (stable/novel) | **Confirmed → Corrected** |
| 2 | v0.12 F-family v0.11 exact-D2 shift value wrong (66.9 vs raw 65.0) | **Confirmed → Corrected** |
| 3 | Memory v0.4 classic deltas sign-flipped in code, summary and report | **Confirmed → Corrected** |
| 4 | v0.12 headline success tables (F/H/R/L) | **Verified** vs raw |
| 5 | Memory v0.4 train/memory/mid-curve numbers | **Verified** vs raw |
| 6 | v0.12 temporal exact-D1 "16.9/23.1%" provenance | **Was mis-reported as unresolved; corrected** — field lookup error; values reproduce exactly from `exact_discovery_D1` (seed-sensitive, see §5) |
| 7 | Frozen evaluation criteria for future v0.12.1 / v0.5 | **Added** |

---

## 1. v0.12 report — temporal exact-D1 deltas (confirmed, corrected)

`docs/RESEARCH_V12.md` §4.4 (family L) reported:

| condition | exact D1 (v10/v11) | exact D1 (v12) | temporal Δ |
|---|---:|---:|---:|
| stable | 0.0% | 16.9% | +23.12 [0, 50.6] |
| novel | 0.0% | 23.1% | +16.88 [0, 50.6] |

Raw committed artifact (`results/v12/summary_l_small.json`):

```
stable comparison:temporal_minus_notp_exact_D1: delta=0.16875 CI=[0, 0.50625]
novel comparison:temporal_minus_notp_exact_D1: delta=0.23125 CI=[0, 0.50625]
```

The Δ column values were swapped: stable must be +16.88, novel +23.12.
The exact-D1 percentages (16.9 / 23.1) are correct and equal the deltas
because the notp ablation is 0 on both conditions. Corrected in the
report with provenance note.

## 2. v0.12 report — F-family v0.11 exact D2 (confirmed, corrected)

`RESEARCH_V12.md` §4.1: "v0.11 was 65.6/66.9/31.9/66.9/64.4 on D2
across those conditions".

Raw committed artifact (`results/v12/summary_f_small.json`):

```
metargon11 exact_discovery_D2: stable=0.65625 shift=0.6500 noisy=0.31875
                               novel=0.66875 tight=0.64375
```

The **shift** value is 65.0, not 66.9. All other four values match.
Corrected in the report with provenance note. v0.12 D2 values
(84.4/88.8/76.9/88.1/86.9) and D1 values all match raw exactly.

## 3. Memory v0.4 deltas (confirmed, corrected)

`docs/MEMORY_V04.md` §5.2 reported classic deltas vs original:

| variant | reported | recomputed from raw (candidate − baseline) |
|---|---|---|
| fresh | −0.100 [−0.211, 0] | +0.100 [0, +0.211] |
| probe_local | −0.032 [−0.094, 0] | +0.032 [0, +0.094] |
| ctx / ctx_reset / ctx_gate | −0.030 | +0.030 |
| ctx_full | +0.030 [0, +0.094] | −0.030 [−0.094, 0] |
| memory_wiped | 0.000 | 0.000 |

All signs were flipped. Root cause: `academy/tools/memory_merge_v04.py`
called `paired_difference(original, variant)` — the exact bug corrected
for Memory v0.3 in audit v0.2 but reintroduced in the v0.4 merge tool.
The report's own interpretation text ("directionally positive … recover
~3 pp") agreed with the *correct* signs, so the table contradicted its
own prose.

Fixes:
- `memory_merge_v04.py`: argument order → `(candidate, baseline)`.
- Committed `memory_v04_summary.json` comparisons recomputed (raw
  per-config vectors unchanged).
- `docs/MEMORY_V04.md` §5.2 table corrected with convention note.

Train deltas, memory-status means and mid-curves (§5.1/§5.3) all match
raw exactly and were untouched.

## 4. Verified claims (no change needed)

- v0.12 F success table: all 20 cells match `summary_f_small.json`.
- v0.12 H success: stable/novel = v0.10 83.75 / v0.11 73.12, 79.38 /
  v0.12 73.12, 79.38; exact D2 63.1/61.9, 41.2/43.8, 70.0/76.9 — all match.
- v0.12 R success + exact D1 rows match.
- Memory v0.4 train/memory/mid-curve values match JSONL exactly.

## 5. Temporal exact-D1 provenance — corrected (was incorrectly "unresolved")

**Correction:** the previous audit draft claimed the v0.12 exact-D1
values (16.9% stable / 23.1% novel) could not be reproduced from the
committed artifacts. **That claim was wrong** — it resulted from
reading the wrong JSON field: the draft looked for
`discovery.D1.exact_rate` while the committed summary stores the metric
as a flat per-policy field `exact_discovery_D1`.

Verified from committed `results/v12/summary_l_small.json`
(`raw` per-seed arrays vs `summary` means):

| Seed | Stable exact D1 | Novel exact D1 |
|---|---:|---:|
| 0 | 0.000 | 0.000 |
| 1 | 0.000 | 0.250 |
| 2 | 0.675 | 0.675 |
| 3 | 0.000 | 0.000 |
| Mean | 0.16875 | 0.23125 |

- All `metargon12-notp` exact-D1 values are 0.000, so
  `temporal_minus_notp_exact_D1` deltas equal the v0.12 means:
  **+16.875 pp** (stable) and **+23.125 pp** (novel), matching the
  report's 16.9% / 23.1%.
- Recorded metric definition: fraction of episodes (per seed) in which
  a temporal hypothesis with `status == "verified"` exactly matches the
  true ordered rule `(first, second)` for D1 (`benchmark_v12.py`
  `discovery_row`). It counts verified temporal hypotheses only;
  `predicted` hypotheses are not counted.

**Seed sensitivity (honest limitation):** discovery is driven by a
single seed per condition — stable: seed 2 only (0.675); novel: seeds 1
(0.250) and 2 (0.675). Population stddev across the 4 seeds is 0.292
(stable) and 0.276 (novel). The *recorded metrics* are reproducible
(means match committed summaries exactly), but the *generalization
across seeds* is not established: with n=4 seeds, most observed success
originates from one or two seeds. The report's "16.9–23.1% exact D1"
should be read as a seed-sensitive preliminary estimate, not a robust
population estimate. The frozen protocol (below) therefore requires
per-seed reporting and larger seed sets for any temporal-learning claim.

## 6. Frozen criteria for future evaluations

`generalization_audit/configs/frozen_eval_criteria_v01.json` defines,
without inspecting v0.12.1 / Academy v0.5 results:

- **Zero-shot vs within-evaluation-learning separation:** report both
  frozen (learning disabled during evaluation) and within-eval scores;
  "transfer/retention" claims must use the zero-shot score. Fresh
  controls under the same protocol are mandatory.
- **Resource-budget fairness:** identical per-decision compute caps;
  baselines with larger budgets must be capped or reported alongside CPU
  seconds; invalid actions reported per policy.
- Statistics: candidate − baseline, per-seed paired bootstrap 95% CIs,
  attached raw vectors, recorded hash seed.
- Seed/config alignment: record split source/seed, eval seeds, noise,
  max steps; frozen pools read-only.
- v0.12.1: 8×100 runs expected (committed v0.12 is 4×40 = preliminary);
  per-condition exact discovery fields in the artifact metadata.
- Academy v0.5: frozen grids as in `academy_eval_grid_v01.json`; no
  cross-environment success comparisons.

## 7. Files changed (this audit)

| Path | Change |
|---|---|
| `academy/tools/memory_merge_v04.py` | comparison order fixed (candidate − baseline) |
| `results/academy_v01/memory_v04/memory_v04_summary.json` | comparisons recomputed |
| `docs/MEMORY_V04.md` | §5.2 corrected + convention note |
| `docs/RESEARCH_V12.md` | §4.4 temporal Δ swapped→corrected; §4.1 D2 shift 66.9→65.0; provenance notes |
| `generalization_audit/configs/frozen_eval_criteria_v01.json` | new frozen criteria |
| `generalization_audit/tests/test_artifact_integrity_v03.py` | new tests (15) |
| `generalization_audit/docs/AUDIT_V03.md` | this report |
