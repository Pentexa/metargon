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
| 6 | v0.12 temporal exact-D1 "16.9/23.1%" provenance | **Not reproducible from committed summary (unresolved)** |
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

## 5. Unresolved — temporal exact-D1 provenance

The report claims v0.12 exact D1 = 16.9% (stable) / 23.1% (novel) on L.
The committed `summary_l_small.json` contains only the *comparison*
deltas (`temporal_minus_notp_exact_D1`); its per-policy
`exact_discovery_D1` fields are 0.0 for every policy including v0.12
(those fields count only `verified` compositional hypotheses, and the
temporal branch is counted only when `verified` — the committed run
reports 0.0). The 16.9/23.1 numbers cannot be reproduced from the
committed summary alone except as the notp delta; the generating
procedure for those percentages (likely a separate analysis including
`predicted` temporal hypotheses) is not committed. The new v0.12.1
protocol must record exact-discovery per policy per condition directly
(and define whether predicted temporal hypotheses count as discovery).

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
