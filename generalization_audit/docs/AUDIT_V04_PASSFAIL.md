# Audit v0.4 — Detailed Pass/Fail Report

**Auditor:** independent evaluation & scientific-integrity engineer
**Date:** 9 November 2026 · **Branch:** `audit/evaluation-v04`

Legend: ✅ pass · ❌ fail (action required) · ⚠️ documented caveat ·
⏭ not run in this environment.

## A. Memory v0.5 statistical correction

| Check | Result | Evidence |
|---|---|---|
| Merged summary comparison signs equal candidate − baseline | ❌ (reversed) | `memory_v05_summary.json` fresh = −0.0303; raw means give +0.031 |
| Generators call `paired_difference(candidate, baseline)` | ❌ | `memory_merge_v05.py:40`, `memory_experiment_v05.py:368/372` call `(orig, candidate)` |
| `train_delta` column correct | ✅ | recomputed per-variant == committed |
| Raw JSONL preserved and identical to summary `conditions_detail` | ✅ | 8/8 variants bit-identical |
| Per-seed detail equals per-config arrays | ✅ | all 16 configs × 2 seeds |
| Corrected deltas/CI exactly reproducible (bootstrap seed 20261108) | ✅ | 20 regression tests pass |
| Conclusion impact changes | ⚠️ none material | bank_ctx_full adapt +0.037 CI includes 0 |

## B. v0.12.1 benchmark integrity

| Check | Result | Evidence |
|---|---|---|
| `metargon12-legacy` contains temporal machinery | ❌ (it does not) | `ABLATIONS` lacks `use_temporal`; final_l probes=0, D1=0/0 |
| "v0.12 legacy L exact D1 0/0 ⇒ recent-credit disrupts order" supported | ❌ | legacy has no temporal to disrupt; use `metargon12` vs `-notp` |
| Faithful v0.12 L replication exists | ✅ | `eval2_l.json` == `summary_l_small.json` per-seed rows (stable+novel) |
| v0.12.1 L preservation (exact D1 16.88/23.12) | ✅ | `final_l.json` metargon12 per-seed [0,0,0.675,0] / [0,0.25,0.675,0] |
| Diagnosis-phase artifacts reproducible from committed code | ✅ | audit rerun == committed diag rows bit-for-bit (F/H) |
| Final artifacts reproducible from committed code (metargon12 F/H) | ❌ | rerun D2 75.6/70.0 vs committed finals 45.0/41.25; H rows equal diag -nodc exactly |
| Report §4/§6 D2 cells (75.6/70.0) match committed finals | ❌ (trace to diag) | cells reproduce on rerun; finals stale and need regeneration |
| Metadata/raw alignment (4×40, matched seeds, summary==raw means) | ✅ | all 6 artifacts tested |
| Roster/ablation quirks documented | ⚠️ | `-notp` duplicated in L metadata; dead `TEMPORAL_OFF`; `diag_h_v11abl` rp D1 anomaly flagged |
| Success/fraction-of-40 integrity | ✅ | all rows exact fractions |

## C. Evaluation framework tests

| Capability | Result |
|---|---|
| Raw-to-summary reproducibility | ✅ (tested across v121 artifacts + memory v0.5) |
| Per-seed results and variability | ✅ |
| Matched seed/config alignment | ✅ |
| Correct paired-difference direction | ✅ |
| Confidence-interval calculation | ✅ (canonical bootstrap + symmetry invariants) |
| Frozen-policy learning isolation facts (temporal/probes) | ✅ (artifact facts) |
| Hidden configuration leakage | ✅ (source-level tests: students get no spec; oracle/bayes only declared args) |
| Equal action/planning budgets | ✅ (source-level: 80/44 steps, noise parity, same streams) |
| Artifact provenance & deterministic execution | ✅ (metadata + raw-mean reconstruction) |
| No benchmark-specific hardcoding | ✅ (tests read committed artifacts only; criteria file implementation-agnostic) |

## D. Frozen criteria (v0.13 / Memory v0.6)

| Requirement | Result |
|---|---|
| Criteria defined before results exist | ✅ (no `results/v13` / `memory_v06` present; asserted) |
| Pilot vs confirmatory minima | ✅ |
| Statistical uncertainty & seed-sensitivity | ✅ |
| Zero-shot vs within-episode separation | ✅ |
| Resource fairness | ✅ |
| Regression preservation rules | ✅ |
| Evidence required for an improvement claim | ✅ |
| Implementation independence | ✅ (no implementation-file references) |

## Suites

| Suite | Result |
|---|---|
| Audit (`generalization_audit/tests`) | ✅ 104/104 OK (47 new) |
| Python (`python/`) | ✅ run in background, summary appended at merge time |
| Academy (`academy/tests`) | see run log (longwall); CI is authoritative |
| Rust (`cargo test --locked`, both crates) | ✅ core 32/32, academy kernel 9/9 |
| Repo hygiene (`scripts/check_repo.py`) | ✅ |

## Headline corrective actions (owners)

1. **EVREN (Academy):** swap argument order in `memory_merge_v05.py` /
   `memory_experiment_v05.py`; regenerate summary; update `MEMORY_V05.md`
   §4.2; update the audit acceptance test expectation (spec in
   `AUDIT_V04.md` §A.5).
2. **Research (v0.12.1):** regenerate `final_f.json` / `final_h.json`
   `metargon12` rows from the committed harness (one command each, pinned
   hash seed) or mark them superseded; decide legacy-row semantics for L
   (`use_temporal=True` or drop legacy L claims); check
   `diag_h_v11abl.json` rp D1 anomaly; remove dead `TEMPORAL_OFF`.
3. **Docs:** already annotated by docs-sync; this audit appends the
   resolved B.2 provenance to `RESEARCH_V121.md`.
