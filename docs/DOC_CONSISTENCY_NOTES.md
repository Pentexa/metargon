# Documentation Consistency Notes — docs sync (9 November 2026)

This document records **cross-artifact inconsistencies discovered while
synchronizing the public documentation** with the merged implementation
(METARGON v0.12, v0.12.1, Academy Memory v0.4/v0.5, Audits v0.2/v0.3).

Ground rules followed by the sync (and recorded here so reviewers can
verify them):

- **No code, algorithm, benchmark, test, or raw result artifact was
  modified.** Committed reports keep their recorded values; corrections
  are added as clearly-marked annotations with provenance.
- Findings below were verified directly against committed artifacts
  (paths given). Where a conclusion in a report is unsupported by
  artifacts, that is stated explicitly rather than resolved silently.
- These are documentation-layer notes; the underlying issues need code
  or experiment-program fixes by people explicitly changing that code.
  Each issue lists a recommended fix.

---

## Issue 1 — Memory v0.5 comparison deltas use the reversed sign convention

**Where:** `academy/tools/memory_merge_v05.py` (line 40) and
`academy/tools/memory_experiment_v05.py` (lines 368/372): both call
`paired_difference(a, b)` with `a = original, b = candidate`;
`academy/eval/metrics.py:paired_difference` returns `mean(a − b)`.
Note the contrast inside the same file: `train_delta` is computed at
line 41 as `per − orig` (candidate − baseline), so the frozen/adapt
columns are sign-flipped **relative to the very next line**.

**Evidence (all from committed artifacts):**

- `results/academy_v01/memory_v05/memory_v05_summary.json` records
  `fresh_vs_original.classic_frozen.mean = −0.0303 [−0.09375, 0]`, while
  the §4.1 means give fresh frozen 0.312 vs original 0.281 — the value
  under the project-wide **candidate − baseline** convention (fixed by
  Independent Audit v0.2 for Memory v0.3 and Audit v0.3 for v0.4) is
  **+0.031**.
- This is the same bug class Audit v0.2 issue A1 (Memory v0.3) and
  Audit v0.3 issue 3 (Memory v0.4) confirmed and fixed — Memory v0.5
  was merged (PR #32) without re-running that check.

**Corrected values under candidate − baseline** (sign-flip of the
recorded CI bounds):

| Comparison | Frozen Δ | Adapt Δ |
|---|---|---|
| fresh vs original | **+0.031** [0, +0.094] | **+0.100** [0, +0.211] |
| memory_wiped | +0.000 [0, 0] | +0.000 [0, 0] |
| ctx | +0.000 [0, 0] (frozen) | −0.030 [−0.094, 0] |
| ctx_full | **−0.030** [−0.094, 0] | **−0.030** [−0.094, 0] |
| bank | +0.000 [0, 0] | +0.000 [0, 0] |
| bank_ctx | +0.000 [0, 0] (frozen) | −0.030 [−0.094, 0] |
| bank_ctx_full | −0.007 [−0.094, +0.070] | **+0.037** [−0.039, +0.133] |

The Train Δ column is already in the candidate − baseline direction and
was **not** affected.

**Impact on conclusions:** none that matters — no variant beats the
untrained fresh control on frozen transfer; `bank_ctx_full`'s adapt
delta becomes directionally positive but its CI still includes 0;
"zero-shot transfer absent" and "memory_wiped ≡ original" are
convention-independent. The only sentence that flips is `MEMORY_V05.md`
§5's "adaptation is directionally negative (−0.037)" — annotated
in-place in that report (§4.2 note + §5 inline note).

**Recommended fix (code owner):** swap the argument order in
`memory_merge_v05.py`/`memory_experiment_v05.py` to
`paired_difference(candidate, original)`, regenerate
`memory_v05_summary.json`, and add an artifact-integrity test in the
style of `generalization_audit/tests/test_artifact_integrity_v03.py`.

---

## Issue 2 — `metargon12-legacy` (and `-norc`) rows cannot test recent-credit effects on family L

**Where:** `python/benchmark_v12.py` `ABLATIONS` (lines ~62–75):
`metargon12-legacy = {dual_context_mode: "dual", use_recent_credit: True}`
leaves `use_temporal` at its default (False). Only the `metargon12`
**policy** gets temporal memory, and only on family L
(`make_policy`, lines 78–82). Confirmed identical in the v0.12-era
harness (`git show 5182bcc:python/benchmark_v12.py`) and agent defaults
(`metargon_v12.py`: `use_temporal=False` default, and
`use_recent_credit=True` at the time of the study).

**Evidence:** `results/v121/final_l.json` — `metargon12-legacy`:
success 18.12/18.75, exact D1 **0.0/0.0**, `temporal_probes` **0**
(vs `metargon12` (v0.12.1, temporal on): 26.88/25.0, **16.88/23.12**,
probes 41.75/39.25). The v0.12 study's own `-norc`/`-nodc`/`-noad`
L rows also lack temporal, so no v0.12-era run ever measured
"recent-credit off **with** temporal on".

**Consequences:**

1. The RESEARCH_V121 §4 interpretation ("v0.12 legacy shows L exact D1
   = 0/0 … recent-credit protection disrupts the ordered D2 sequence …
   recent-credit off is *required* to preserve temporal L discovery")
   is **not supported** by the final-run artifact — the legacy
   ablation has no temporal machinery to disrupt.
2. What IS artifact-supported: (a) the temporal effect in both eras
   is measured by `metargon12` vs `metargon12-notp` (Δ +16.88 [0, 50.6]
   stable / +23.12 [0, 50.6] novel); (b) the v0.12.1 preservation
   conclusion is directly supported by `final_l.json` `metargon12`
   (the only committed recent-credit-OFF + temporal-ON runs), which
   preserves exact D1 at 16.88/23.12; (c) a faithful full-v0.12
   (recent-credit ON + temporal ON) replication exists in the
   diagnosis-phase `eval2_l.json` `metargon12` run (success 27.5/29.4,
   exact D1 16.9/23.1), pre-dating the default flip.

The mixed provenance of the RESEARCH_V121 §4 table (success cells from
final runs; the "v12 legacy" L cells from the v0.12 study) is annotated
in that report.

**Recommended fix (code owner):** either include
`use_temporal=True` in the `metargon12-legacy` ablation (then rerun the
final L runs if that specific comparison is decision-relevant) or drop
legacy rows from L-family claims; document the ABLATIONS semantics in
the harness docstring.

---

## Issue 3 — Exact-D2 values disagree between diagnosis-phase and final runs

**Where:** `results/v121/` — diagnosis-phase run(s) vs the final runs.

**Evidence (4 seeds × 40 episodes, same/nominal settings):**

| Artifact | Policy | Success | Exact D2 |
|---|---|---:|---:|
| `diag_f_noisy.json` | metargon12-norc | 81.25 | **75.62** |
| `final_f.json` | metargon12 (v0.12.1) | 80.62 | **45.00** |
| `diag_h_stable.json` | metargon12-norc | 73.12 | **70.00** |
| `final_h.json` | metargon12 (v0.12.1) | 73.12 | **41.25** |
| `final_h.json` | metargon12-legacy (recent-credit ON) | 73.12 | 70.00 |

Success rates agree closely (81.25 vs 80.62 noisy; 73.12 vs 73.12
stable), but the exact-D2 discovery counters differ by 30–29pp between
runs that should be identical in configuration and seeds. The
RESEARCH_V121 §4 header cells `F-noisy D2 75.6` / `H-stable D2 70.0`
reproduce the diagnosis `-norc` values, not the final artifacts; the
v0.11-row D2 values (31.9 / 41.2) DO match both final artifacts.

**Consequence:** the "F D2 recovery preserved (75.6 vs v11's 31.9)"
preservation check in RESEARCH_V121 §6 reproduces in the diagnosis
harness but not in the committed final runs (45.0, still above v11's
31.9 on noisy but far below the reported 75.6). Public-facing docs
(README, RESEARCH_INDEX) therefore only carry success rates and the
L-temporal exact-D1 values as artifact-verified claims, and flag the
D2 discrepancy openly.

**Possible causes (unverified hypotheses, not asserted as fact):** a
configuration/default drift between the diagnosis commits and the
final commit (the `use_recent_credit` default flip happened in
between); different code paths in the diagnosis harness vs
`benchmark_v12.py`; or a normalization/aggregation difference. Requires
a rerun to resolve.

**Recommended fix (experiment owner):** rerun `final_f`/`final_h` from
a pinned state of `metargon_v12.py` with `PYTHONHASHSEED=0`, compare
per-seed exact-D2 counters, and update RESEARCH_V121 §4/§6 cells with
an audit-v0.3-style provenance note in whichever direction the data
falls.

---

## Minor notes (verified; recording for completeness, no action required)

1. **Temporal-probe normalization differs between harnesses.**
   `results/v121/eval2_l.json` records `temporal_probes` as raw
   counters (4475/4250) while `results/v12/summary_l_small.json`
   records 44.75/42.5 — different aggregations of comparable runs;
   cited carefully rather than interchangeably in the annotations.
2. **`results/v121/final_l.json` metadata lists `metargon12-notp`
   twice** in `metadata.policies`; harmless (only affects the one-line
   `v121_show.py` listing), not worth an artifact edit.
3. **Stale "never merged into `main`" branch lines.**
   `RESEARCH_V10.md` (#22), `RESEARCH_V11.md` (#25), `RESEARCH_V12.md`
   (#29), `RESEARCH_V121.md` (#33) — corrected in-place with merge-PR
   annotations following the Audit v0.3 annotation precedent.
4. **The "188-test Python suite passes" statement in RESEARCH_V121 §6
   is reproducible today:** the full `python -m unittest discover -s
   python -v` run on the synced `main` state reports **188 tests, OK**
   (1244 s wall time on this machine).
5. **Rust kernel coverage:** `rust/src` contains `v04…v10` modules
   only — the pre-existing ARCHITECTURE_OVERVIEW claim of a `v11.rs`
   mirror was stale and is corrected in this sync (no v11/v12 Rust
   mirror exists).

---

## What this sync changed (scope statement)

- Rewrote the public-facing `README.md` to the merged v0.12.1 state
  (results sourced from committed artifacts; open conflicts flagged).
- Updated `docs/RESEARCH_INDEX.md`, `docs/GETTING_STARTED.md`,
  `docs/BENCHMARK_GUIDE.md`, `docs/ARCHITECTURE_OVERVIEW.md`,
  `docs/README.md` (commands verified by execution where marked).
- Added annotations to `docs/MEMORY_V05.md`, `docs/RESEARCH_V12.md`,
  `docs/RESEARCH_V121.md` (and branch-line fixes in V10/V11).
- No report values were changed; no code or artifact was touched.
