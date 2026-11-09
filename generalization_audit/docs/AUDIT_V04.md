# METARGON Independent Audit v0.4 — Scientific Integrity

**Auditor:** Independent evaluation, reproducibility & scientific-integrity engineer
**Date:** 9 November 2026
**Branch:** `audit/evaluation-v04`
**Base:** `origin/main` @ `7e27026` (docs-sync merged)
**Workspace:** QUALITY worktree (`metargon-quality`, previously `chore/quality-ci`)
**Scope:** Memory v0.5 statistical signs (Priority A), v0.12.1 benchmark
integrity (Priority B), evaluation-framework robustness tests (Priority C),
frozen criteria for v0.13 / Memory v0.6 (Priority D).

The auditor remained independent of Research and Academy development. **No
Academy or Research implementation file was modified by this audit.** All
corrections are delivered as regression tests plus exact, reproducible
correction specifications for the owning agents (EVREN for Academy; the
v0.12.1 research author for research).

---

## Issue status summary

| # | Issue | Finding | Status |
|---|---|---|---|
| A1 | Memory v0.5 classic deltas use reversed (baseline − candidate) signs | **Confirmed** (code + summary + report agree on the wrong sign) | Corrected in this audit as spec + regression tests; EVREN fix required (below) |
| A2 | Memory v0.5 `train_delta` already candidate − baseline | **Verified** (not affected) | OK |
| A3 | Memory v0.5 raw per-config vectors / JSONL / per-seed detail agreement | **Verified** bit-identical | OK |
| B1 | `metargon12-legacy` cannot test recent-credit effects on family L (no temporal machinery) | **Confirmed** (artifact + source) | Documented; spec for research agent |
| B2 | Diagnosis-phase exact-D2 vs final-run exact-D2 disagreement (F/H) | **Confirmed and resolved** — see §B.2. The committed final artifacts are NOT regenerable from the committed harness; the diagnosis artifacts ARE reproducible bit-for-bit | Unresolved in artifacts; reproducible from committed code |
| B3 | Config/seed/policy/ablation consistency (roster metadata, agent defaults, budgets) | **Verified** with documented quirks (metadata duplicate `-notp`; dead `TEMPORAL_OFF`) | OK / documented |
| B4 | Claims not regenerable from committed artifacts (report §4/§6 D2 cells vs final artifacts) | **Confirmed** — cells trace to reproducible diagnosis phase; final artifacts stale | Corrective actions listed |
| C1–C10 | Evaluation framework improvements | **Added** (43 new tests) | OK |
| D1 | Frozen criteria for v0.13 / Memory v0.6 | **Added** (`frozen_eval_criteria_v02.json`) | OK |

---

## Priority A — Memory v0.5 statistical correction

### A.1 Confirmed root cause

`results/academy_v01/memory_v05/memory_v05_summary.json` comparison block and both
generators use the reversed convention:

- `academy/tools/memory_merge_v05.py` line 40:
  `paired_difference(a=orig, b=candidate)` for `classic_frozen` and `classic_adapt`;
- `academy/tools/memory_experiment_v05.py` lines 368/372: identical calls;
- `academy/eval/metrics.py:paired_difference(a, b)` returns `mean(a − b)`
  (convention: candidate − baseline; docstring says so).

The merged v0.5 summary therefore has fresh (which is *better* in both
regimes) reported as **negative**. The **same file** computes
`train_delta = per − orig` (candidate − baseline) correctly at line 41 /
373, so the classic columns are sign-flipped relative to the very next
field. This is the third occurrence of the bug class fixed by audit v0.2
(v0.3) and v0.3 (v0.4); v0.5 merged without the integrity check.

### A.2 Evidence (all from committed artifacts)

- fresh frozen mean: 0.3125; original frozen mean: 0.28125 ⇒ candidate − baseline = **+0.031**.
  Committed comparison: **−0.0303** [−0.09375, 0].
- fresh adapt mean: 0.6953125; original adapt mean: 0.59375 ⇒ **+0.1016**.
  Committed: **−0.0997** [−0.2109, 0].
- ctx_full frozen/adapt: candidate − baseline = **−0.0303**. Committed: +0.0303.
- bank_ctx_full adapt: **+0.0372** (CI [−0.039, +0.133], still includes 0).
  Committed: −0.0372.
- Memory v0.5 report §4.2 table reproduces the committed (reversed) values;
  the docs-sync annotation already gave the corrected table. The train Δ
  column and §4.1 means are correct.

### A.3 Corrected table (candidate − baseline; CI sign-flip)

| Comparison | Frozen Δ | Adapt Δ |
|---|---|---|
| fresh vs original | **+0.031** [0, +0.094] | **+0.100** [0, +0.211] |
| memory_wiped | +0.000 [0, 0] | +0.000 [0, 0] |
| ctx | +0.000 [0, 0] | −0.030 [−0.094, 0] |
| ctx_full | **−0.030** [−0.094, 0] | **−0.030** [−0.094, 0] |
| bank | +0.000 [0, 0] | +0.000 [0, 0] |
| bank_ctx | +0.000 [0, 0] | −0.030 [−0.094, 0] |
| bank_ctx_full | −0.007 [−0.094, +0.070] | **+0.037** [−0.039, +0.133] |

Conclusion impact: none material. No variant beats the untrained fresh
control on frozen transfer; `bank_ctx_full` adapt becomes directionally
positive with a CI including 0 — still not evidence of a bank benefit.

### A.4 Regression tests added (`test_memory_v05_integrity.py`, 20 tests)

1. Committed JSONL records == summary `conditions_detail` (bit-identical; raw evidence preserved).
2. Per-seed detail == per-config arrays.
3. Every committed comparison == `paired_difference(original, candidate)` (the bug detector).
4. Every `train_delta` == candidate − baseline.
5. Corrected deltas and CI bounds equal the canonical bootstrap recomputation (seed 20261108).
6. Canonical convention invariants: mean negation, exact CI sign-flip symmetry under argument reversal.
7. Report §4.1 means match the summary; §4.2 committed-table values match the block.

### A.5 Exact reproducible correction specification — for the EVREN agent

Do **not** edit Academy modules concurrently with this audit. Apply, in a
separate Academy change:

1. `academy/tools/memory_merge_v05.py` line 40:
   `paired_difference(a, b, seed=20261108)` → `paired_difference(b, a, seed=20261108)`
   (i.e. `(candidate, original)`).
2. `academy/tools/memory_experiment_v05.py` lines 368 and 372: same swap.
3. Regenerate the summary with pinned hash seed:
   ```
   PYTHONHASHSEED=0 python academy/tools/memory_experiment_v05.py
   PYTHONHASHSEED=0 python x/run_chunks_v05.py
   python academy/tools/memory_merge_v05.py
   ```
   The raw vectors/JSONL must stay untouched (they already are correct).
4. Update `docs/MEMORY_V05.md` §4.2 table to the corrected values and
   remove the "reversed convention" annotation; §5 note about
   `bank_ctx_full` (+0.037, CI includes 0) becomes the plain text.
5. Confirm the audit suite passes:
   `python -m unittest generalization_audit.tests.test_memory_v05_integrity`.
6. Re-run the acceptance test `test_every_committed_comparison_equals_reversed_recomputation`
   — it will fail until the summary is regenerated (it encodes the old
   sign); update it to the candidate − baseline expectation in the same
   Academy change (the audit file is read-only here so that the Academy
   change is verifiable).

---

## Priority B — v0.12.1 benchmark integrity

### B.1 `metargon12-legacy` temporal-memory behavior (confirmed)

Source (`python/benchmark_v12.py`):

- `ABLATIONS["metargon12-legacy"] = {"dual_context_mode": "dual", "use_recent_credit": True}`
  — leaves `use_temporal` at its False default.
- `make_policy` gives temporal memory only to `metargon12` on family L.

Artifacts (`results/v121/final_l.json`):

| policy | stable/novel success | exact D1 | temporal_probes |
|---|---:|---:|---:|
| metargon12 (v0.12.1, temporal on) | 26.88/25.0 | **16.88/23.12** | 41.75/39.25 |
| metargon12-legacy (recent-credit on) | 18.12/18.75 | 0.0/0.0 | 0 / 0 |
| metargon12-notp (temporal off) | 18.12/19.38 | 0.0/0.0 | 0 / 0 |

**Resolution.** The v0.12.1 final legacy rows have no temporal machinery at
all, so they cannot demonstrate "recent-credit disruption of the ordered
D2 sequence"; the RESEARCH_V121 §4 reading is **not supported** by the
committed final artifact. What *is* supported:

- The temporal effect in both eras is measured by `metargon12` vs
  `metargon12-notp` (`+16.88 [0, 50.6]` stable, `+23.12 [0, 50.6]`
  novel, from `final_l.json`).
- The faithful full v0.12 replication is `eval2_l.json` `metargon12`
  (recent-credit on + temporal on): its per-seed rows equal the v0.12
  study `summary_l_small.json` **bit-for-bit** on both conditions
  (success, exact D1, probes), verified by test
  `test_eval2_l_reproduces_v12_study_exactly`.
- The v0.12.1 preservation conclusion stands on `final_l.json`
  `metargon12` itself (exact D1 preserved at 16.88/23.12).

### B.2 Diagnosis-phase vs final exact-D2 (confirmed; resolved by rerun)

Committed artifact facts (4 seeds × 40 episodes, matched seeds):

| Artifact | policy | success | exact D2 | exact D1 |
|---|---|---:|---:|---:|
| `diag_f_noisy.json` | -norc | 81.25 | **75.62** | 80.6 |
| `final_f.json` | metargon12 | 80.62 | **45.00** | 51.9 |
| `diag_h_stable.json` | -norc | 73.12 | **70.00** | 95.6 |
| `final_h.json` | metargon12 | 73.12 | **41.25** | 92.5 |
| `final_h.json` | metargon12-legacy | 73.12 | 70.00 | 95.6 |

The RESEARCH_V121 §4 header cells (F D2 75.6, H D2 70.0) trace to the
**diagnosis** `-norc` runs, not to the final artifacts — as the v0.3
audit and docs-sync already flagged.

**Audit rerun (decisive).** The auditor reran the committed harness
(`benchmark_v12.py`, `PYTHONHASHSEED=0`, 4×40, same roster) and stored the
outputs in `generalization_audit/results/v121_repro/` (commands below).

| run | metargon12 success/D1/D2 | legacy success/D1/D2 |
|---|---:|---:|
| **Rerun F-noisy (current main)** | 81.2/80.6/**75.6** | 72.5/87.5/76.9 |
| diag_f_noisy (-norc / metargon12) | 81.2/80.6/**75.6** | 72.5/87.5/76.9 |
| final_f.json | 80.6/51.9/**45.0** | 72.5/87.5/76.9 |
| **Rerun H-stable (current main)** | 73.1/95.6/**70.0** | 73.1/95.6/70.0 |
| diag_h_stable (-norc / metargon12) | 73.1/95.6/**70.0** | 73.1/95.6/70.0 |
| final_h.json | 73.1/92.5/**41.2** | 73.1/95.6/70.0 |

- The rerun `metargon12` rows equal the committed diagnosis `-norc` rows
  **exactly** (per-seed) on both families; the rerun `metargon12-legacy`
  rows equal both the diagnosis `metargon12` rows and the final `-legacy`
  rows exactly.
- The committed **final** `metargon12` rows are therefore **not
  regenerable from the committed harness**: on H-stable they equal the
  diagnosis `-nodc` (dual-context off) rows exactly; on F-noisy their D2
  (45.0) is close to the diagnosis `-nodc` level (43.8) and far from the
  reproduced 75.6.

**Resolution.** The diagnosis phase (and the v0.12 study, and the final
legacy rows) are reproducible from the committed code under a pinned hash
seed. The committed *final* `metargon12` runs were generated in a code
state that is not the committed one — they behave like dual-context-off
runs. Therefore:

1. The report's D2 preservation claim (75.6/70.0) is reproducible from the
   committed code (rerun matches) and describes the merged v0.12.1
   behavior; it is *not* reflected in the committed final artifacts.
2. The committed final F/H metargon12 D2 cells (45.0/41.25) are stale and
   should not be cited; the `final_f/h.json` `metargon12` rows need to be
   regenerated from the committed harness (one command each) and either
   replaced or explicitly marked superseded. This is a Research-side
   corrective action.
3. In the meantime the report §4/§6 D2 cells are supported by the
   reproducible diagnosis-phase values; the final artifacts disagree with
   them — the discrepancy is real, not merely mis-citation.

**Test coverage:** `test_v121_artifact_integrity.py` locks in the artifact
facts (incl. the F/H row identities) so any future artifact replacement is
visible in review; `tools/compare_v121_repro.py` prints the side-by-side
table and the identity checks.

### B.3 Configuration / seed / policy / ablation consistency (verified)

- All final/diag/eval artifacts: metadata `seeds=4`, `episodes_per_seed=40`,
  `version=0.12`; raw arrays have exactly 4 per-seed rows; summary means
  equal raw means for all metrics (tested).
- Success rates are exact fractions of 40 episodes (tested).
- Matched world seeds (i+11800) and policy seeds (seed+1000003) are
  identical across artifacts; family R/L use frozen pools read-only
  (source-verified).
- Quirks (documented, no action): `final_l.json`/`eval2_l.json` metadata
  lists `metargon12-notp` twice (harmless: summary keys unique);
  `TEMPORAL_OFF` in `benchmark_v12.py` is dead code; `diag_h_v11abl.json`
  `metargon11-rp` D1 (0.4125) differs from `metargon11` (0.925) — on H
  replay-off should not degrade discovery; flagged as a candidate check
  for the research agent (not resolved here, we do not edit research code).

### B.4 Claims not regenerable from committed artifacts

| Claim | Verdict |
|---|---|
| RESEARCH_V121 §4 F/H D2 75.6/70.0 for v0.12.1 | Supported by committed diagnosis artifacts and by audit rerun; **not** supported by committed final artifacts (stale) |
| RESEARCH_V121 §4 "v0.12 legacy shows L exact D1 = 0/0 … recent-credit disruption" | Not supported — legacy ablation lacks temporal machinery |
| RESEARCH_V121 §6 "F D2 recovery preserved" | Same as first row |
| Research V121 §4 F/H/R success cells | All verified vs final artifacts (and rerun) |
| L temporal exact D1 16.9/23.1 | Verified vs finals (16.88/23.12) and v0.12 study |
| "188-test Python suite passes" | Reproducible (see suite run below) |

---

## Priority C — Evaluation framework improvements

New tests (47 total across four files):

**`test_memory_v05_integrity.py` (20)** — raw-to-summary reproducibility,
per-config/per-seed agreement, matched vectors, corrected paired-difference
direction, CI calculation (canonical bootstrap), sign-flip symmetry,
frozen-vs-adapt separation data integrity.

**`test_v121_artifact_integrity.py` (20)** — raw-to-summary reproducibility
for all v121 artifacts; per-seed variability reporting; matched seed/episode
alignment; fractions-of-40 validation; provenance metadata; frozen-policy/
temporal-learning isolation facts (probes/D1); diagnosis-vs-final D2
discrepancy lock; legacy-ablation semantics; source-level checks for
hidden-config leakage (students get no spec; oracle only config), equal
budgets (80/44 max steps, noise parity), agent defaults (recent-credit
off, temporal off), frozen pools read-only.

**`test_academy_frozen_isolation.py` (4)** — behavioral check that
`frozen_policy` makes `learn()` / `learn_from_experience()` no-ops and that
the flag survives save/load round-trip.

**`test_future_eval_criteria_v02.py` (3)** — frozen-criteria schema/
content validation, including "no v0.13/v0.6 artifacts may exist yet" and
"criteria must not reference implementation files".

Provenance: all new tests are read-only against committed artifacts and
committed source; they require `PYTHONHASHSEED=0` (they assert hash-insensitive
facts only).

---

## Priority D — Frozen criteria for v0.13 / Memory v0.6

`generalization_audit/configs/frozen_eval_criteria_v02.json` (created before
any v0.13 / v0.6 result exists) requires:

- **Pilot vs confirmatory:** pilots ≥4×40 (mechanism lab) / 2 train seeds
  (Academy); confirmatory ≥8×100 / ≥5 train seeds for any population claim.
- **Statistics:** per-seed vectors for every headline metric; population
  stddev, positive-seed counts; candidate − baseline paired bootstrap 95%
  CIs; "CI includes 0 ⇒ no measured effect".
- **Zero-shot vs within-episode:** both regimes reported separately;
  transfer claims require the zero-shot score; fresh controls mandatory.
- **Resource fairness:** equal per-decision budgets; compute-capped
  baselines or CPU reporting; invalid actions reported; matched streams.
- **Regression preservation:** full F/H/R/L + Academy grid rerun before
  merge; no headline decrease beyond noise; discovery metrics tracked as
  regressions; committed baseline artifacts recorded.
- **Evidence for improvement:** confirmatory n + CI excluding 0 +
  per-seed breakdown + mechanism ablation + reproducible commands + no
  unpreserved regression; pilot results labeled preliminary.
- **Independence:** independent recomputation required for headline figures;
  summaries must be recomputable from committed raw.

---

## Suite results

| Suite | Command | Result |
|---|---|---|
| Audit (incl. 47 new) | `python -m unittest discover -s generalization_audit/tests -t .` | **104 tests OK** |
| Python | `python -m unittest discover -s python` | see dev log (run summary below) |
| Academy | `python -m unittest discover -s academy/tests` | see dev log (run summary below) |
| Rust (rust/, rust_academy/) | `cargo test --locked` | **OK** — metargon_core 32/32, metargon_academy_kernel 9/9 (stable 1.99.1; CI pins 1.88.0) |
| Repo hygiene | `python scripts/check_repo.py` | OK |

---

## Files changed (this audit)

| Path | Change |
|---|---|
| `generalization_audit/tests/test_memory_v05_integrity.py` | new (Priority A) |
| `generalization_audit/tests/test_v121_artifact_integrity.py` | new (Priority B/C) |
| `generalization_audit/tests/test_academy_frozen_isolation.py` | new (Priority C) |
| `generalization_audit/tests/test_future_eval_criteria_v02.py` | new (Priority D) |
| `generalization_audit/configs/frozen_eval_criteria_v02.json` | new frozen criteria |
| `generalization_audit/results/v121_repro/final_f_repro_20261109.json` | audit rerun F-noisy (committed evidence) |
| `generalization_audit/results/v121_repro/final_h_repro_20261109.json` | audit rerun H-stable (committed evidence) |
| `generalization_audit/tools/compare_v121_repro.py` | side-by-side repro vs committed comparison tool |
| `generalization_audit/docs/AUDIT_V04.md` | this report |
| `docs/RESEARCH_V121.md` | auditor note appended (no values changed) |

## Reproduce

```bash
# Audit suite (primary)
cd <repo> && PYTHONHASHSEED=0 PYTHONPATH=".:python" \
  python -m unittest discover -s generalization_audit/tests -t .

# Rerun of the decisive B.2 evidence
PYTHONHASHSEED=0 python python/benchmark_v12.py --seeds 4 --episodes 40 \
  --family f --conditions noisy_obs --parallel --workers 4 --no-mem \
  --policies metargon10,metargon11,metargon12,metargon12-legacy,bayes,oracle \
  --json generalization_audit/results/v121_repro/final_f_repro_20261109.json
# (same with --family h --conditions stable for final_h_repro_20261109.json)

# Side-by-side table
python generalization_audit/tools/compare_v121_repro.py
```
