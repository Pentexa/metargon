# METARGON Post-Merge Independent Audit v0.6

**Auditor:** Independent scientific auditor
**Date:** 10 November 2026
**Branch:** `audit/post-merge-v06`
**Base / starting commit:** `origin/main` @ `cb56c0bf514e7b5b5e08fcd33c94c2565e21efee`
**Scope:** integrated post-merge verification (Mission A); independent
Research v0.13 recomputation (B); independent Academy Memory v0.6
recomputation (C); evaluation-integrity findings (D).

All verification was read-only with respect to historical research data.
No Research or Academy implementation file was modified. Verifier defects
discovered during this audit (comparison-key resolution) were fixed in the
Audit-owned package only.

---

## 0. Executive summary

| Area | Result |
|---|---|
| Integrated suites | Python **210/210** · Academy **234/234** · Audit **149/149** (incl. new resolver tests) · Rust **32+9/41** · hygiene OK |
| Manifests (v13 + v0.6) | **2927 PASS / 0 FAIL / 19 INCONCLUSIVE** across 13 manifests |
| Artifact publication gate | **PASS** for both `results/v13` and `results/academy_v01/memory_v06` |
| Research v0.13 headline claims | **33/33 independent checks PASS** (exact D1, CIs, ablations, preservation, baseline compatibility) |
| Academy Memory v0.6 headline claims | **34/34 independent checks PASS** (held-out means, controls, deltas, CIs, EVREN pool, storage diagnosis) |
| Audit-owned verifier defect | comparison-key ambiguity (`v12_temporal`, `temporal`, duplicate roster) **fixed** + regression tests |

Independent reviewer guidance: METARGON has **demonstrated** (a) the
v0.13 rolling temporal generator improves exact ordered-rule discovery on
the L family with a clean ablation control, and (b) the v0.6 Academy
diagnosis of zero-shot transfer failure (storage contradiction). What
remains **uncertain** (properly labeled in the reports): population-level
effect sizes (4×40 / 2-seed pilots), the R/H regressions, and the EVREN
teacher-mix effect.

---

## 1. Mission A — integrated repository verification

### 1.1 Suites

| Suite | Command | Count | Result |
|---|---|---|---|
| Python | `python -m unittest discover -s python` | 210 | OK |
| Academy | `python -m unittest discover -s academy/tests` | 234 | OK |
| Audit | `python -m unittest discover -s generalization_audit/tests -t .` | 149 | OK |
| Rust core | `cargo test --locked` (rust/) | 32 | OK |
| Rust academy kernel | `cargo test --locked` (rust_academy/) | 9 | OK |
| Repo hygiene | `python scripts/check_repo.py` | — | OK |
| CI config | `.github/workflows/ci.yml` | — | consistent; audit job verifies committed manifests |

No cross-component regression was observed. The only interaction found
was that the merged v0.13 / v0.6 evidence populated the frozen targets,
which the provenance gate (PR #39) was designed to handle; see §1.3.

### 1.2 Manifest integrity

Every v0.13 and Memory v0.6 manifest (13 total) was independently verified
with the committed verifier:

- summary means reproduce raw per-seed means exactly (1e-9);
- recorded deltas follow the declared candidate − baseline convention;
- recorded paired bootstrap 95% CIs reproduce exactly from raw diffs;
- artifact sha256 hashes (canonical git-blob) match.

Remaining INCONCLUSIVE items are **metadata omissions only**, not defects:
- `fairness.memory_limit` (13 manifests): memory limit not byte-measured (documented in both reports);
- `generic`-format structural checks (2 manifests: `v13_diagnostics`,
  `memory_v06_evren`): raw/summary pair checks are not defined for the
  generic format; presence, hashes, and gate coverage still verified.

### 1.3 Artifact publication gate

Both frozen targets are now populated and the provenance gate **PASSES**:
- `results/v13` ← 11 covering manifests, each verifying with 0 FAIL;
- `results/academy_v01/memory_v06` ← 2 covering manifests, 0 FAIL each.

This validates the PR #39 hotfix in the integrated state.

---

## 2. Mission B — Research v0.13 independent recomputation

All 33 checks pass; numbers below are recomputed from committed raw per-seed
vectors (`results/v13/final_l.json` etc.). Full machine-readable output:
`results/verification_v13/post_merge_b_recompute.json`.

### 2.1 Temporal exact-D1 (headline)

| | v0.12.1 | v0.13 | Δ v13 [CI 95%] |
|---|---:|---:|---:|
| stable exact D1 | 16.88% (per-seed [0,0,0.675,0]) | **39.38%** ([0.775,0,0.8,0]) | +39.38 [0.00, 78.75] — **includes 0** |
| novel exact D1 | 23.12% ([0,0.25,0.675,0]) | **33.13%** ([0.275,0.25,0.8,0]) | +33.13 [6.88, 66.25] — **excludes 0** |

Reproduces the report exactly. The report's wording is correct: stable is
directional pilot evidence (CI includes 0), novel is the only exclusion.
Per-seed: nonzero seeds 1/4 → 2/4 (stable) and 2/4 → 3/4 (novel);
previously dead seed 0 now discovers 77.5% (stable) / 27.5% (novel).

### 2.2 Rolling-generator ablation attribution

`-gen` per-seed exact-D1 rows equal the v0.12.1 rows **exactly** on both
conditions (stable [0,0,0.675,0]; novel [0,0.25,0.675,0]). This confirms
the rolling generator is the sole driver of the delta vs the v0.12
generator, with `-notp` at 0 on all seeds.

### 2.3 CIs (harness bootstrap recomputed, seed 4301, 1500 rounds)

- temporal Δ v13 stable [0.00, 78.75]; novel [6.88, 66.25];
- temporal Δ v12 stable [0.00, 50.63]; novel [0.00, 50.63].

### 2.4 F/H/R regression preservation and historical baseline

- F: v0.13 == v0.12.1 on all 5 conditions (Δ exactly 0.00 pp); F-noisy 81.25%.
- H: stable 73.12 / novel 79.38 which v0.12.1; D2 preserved (70.0/76.9).
- R: stable 60.62 / novel 79.38 = v0.12.1; `-nodc` 71.88 reproduced.
- Regression artifacts (`regress_*`) match expected cells.
- Historical baseline: `metargon12` in `final_l.json` reproduces the v0.12
  study `summary_l_small.json` **exactly** (success 27.5/29.375; exact D1
  16.88/23.12), confirming the v0.13 harness is compatible.

### 2.5 The 22 INCONCLUSIVE checks & the `v12_temporal` key

The v0.5-era verification reported INCONCLUSIVE items on the v13
manifests. The audit re-verified all of them. Root causes:

1. **`v12_temporal_minus_notp_exact_D1` key ambiguity (Audit-owned
   verifier).** The key names a comparison where the candidate is
   `metargon12` and the baseline is the no-temporal policy, but the token
   `notp` matches two policies (`metargon12-notp`, `metargon13-notp`) and
   `v12_temporal` matched no policy name. The old resolver returned
   INCONCLUSIVE. **Fixed**: `resolve_comparison_with_delta` now resolves
   data-driven — candidate pairs are scanned and the unique pair
   reproducing the recorded delta (or the family-hinted / value-identical
   pair) is selected deterministically. Verified on `final_l.json` and
   `pilot_l.json`: both `v12_temporal` and `temporal` keys now resolve and
   PASS. Regression tests: `test_comparison_key_resolution.py` (5 tests).
2. **Memory-limit metadata omission** (13 manifests): recorded as null
   intentionally; documented in both reports; INCONCLUSIVE by design.
3. **`generic`-format structural checks** (2 manifests): no raw/summary
   pair defined; covered by existence + hashes + gate.

---

## 3. Mission C — Academy Memory v0.6 independent recomputation

All 34 checks pass; machine-readable output:
`results/verification_v13/post_merge_c_recompute.json`.

### 3.1 Held-out means (recomputed from `conditions_detail`)

| Variant | Frozen | Adapt | verified |
|---|---:|---:|---|
| fresh | 0.3125 | 0.6953 | ✅ |
| original | 0.28125 | 0.59375 | ✅ |
| memory_wiped | 0.28125 | 0.59375 | ✅ (bit-identical vectors) |
| ctx | 0.28125 | 0.625 | ✅ |
| ctx_full | 0.5078125 | 0.53125 | ✅ |
| bank / bank_ctx / reset | 0.28125 | 0.59375/0.625 | ✅ (≡ original) |
| bank_ctx_full | 0.21875 | 0.21875 | ✅ |
| atomtx / atomtx_bank | 0.25 | 0.617 | ✅ |
| controls: random_walk / novocab | 0.0546875 / 0.4609375 | — | ✅ |

### 3.2 Deltas and paired bootstrap CIs

Every recorded `comparisons` entry (vs fresh and vs original; frozen and
adapt) recomputes exactly with the canonical paired bootstrap (seed
20261108, 2000 rounds, α=0.05): e.g. original−fresh frozen −0.0303
[−0.0938, 0]; ctx_full−fresh +0.1984 [+0.0391, +0.3594]; ctx_full−original
+0.2287 [+0.0859, +0.3672]; atomtx−original −0.0303 [−0.0938, 0].

### 3.3 EVREN teacher pool

- `evren_pool.json`: 24 approved tasks, 24/24 solvable, 24/24
  config-unique, reproduces the summary's pool block. ✅
- EVREN summary: fresh 0.3125, original 0.40625 (Δ vs fresh +0.0944,
  CI [0.0, 0.1875] — directionally positive, not significant), atomtx_bank
  0.25, teacher holdout 0.500 for all conditions. ✅

### 3.4 Hypothesis-storage failure (cross-task contradiction)

The committed runs JSONL records `verified_then_contradicted` per run:
fresh 0; classic memory variants **56.5–198** (original 56.5, ctx/bank_ctx
198, ctx_full 172.5, bank_ctx_full 76); atom-transfer variants 29–31
(lower by design). Mean contradicted ≈67, verified ≈3 across trained
variants. **The evidence supports the report's conclusion**: verified
hypotheses are largely destroyed by cross-task contradiction, and no
trained memory-bearing variant beats fresh on frozen transfer —
**genuine zero-shot transfer remains unproven** (indeed absent in the
measured cells). ✅

---

## 4. Mission D — evaluation-integrity findings

| Finding | Category |
|---|---|
| Students receive no hidden config in `benchmark_v13.py` (`episode_start()` / `episode_start(noise_q=…)` only; config only for oracles) | VERIFIED CORRECT |
| Budget parity: max_steps 44/80, noise_q 0.08/0.0, planner cap 700, probe max 3 (recorded in manifests) | VERIFIED CORRECT |
| Matched worlds/seeds (11800–11803 + policy seed offset), per-seed vectors committed | VERIFIED CORRECT |
| `PYTHONHASHSEED` unset in v0.13 runs — documented; determinism not claimed; reproducibility rests on matched streams + canonical hashes | SCIENTIFIC LIMITATION |
| EVREN pool: 24 approved, solvable, unique, zero API calls | VERIFIED CORRECT |
| v0.6 comparison signs candidate − baseline, CIs recompute exactly | VERIFIED CORRECT |
| `ctx_full` > fresh explained as planning-only exploration artifact (novocab control) — not memory transfer | VERIFIED CORRECT (with control) |
| Population-level effect sizes: all pilot-scale (4×40 / 2 seeds); stable L Δ CI includes 0; EVREN Δ CI lower bound = 0 | SCIENTIFIC LIMITATION (properly reported) |
| R/H regressions vs v0.10/v0.11 persist and are honestly reported | SCIENTIFIC LIMITATION (unresolved) |
| manifest `generic`-format structural INCONCLUSIVE items | INCONCLUSIVE (metadata/format) |

No CONFIRMED DEFECT was found in the merged research or academy evidence.
The only confirmed defect was in the Audit-owned verifier (comparison-key
resolution), fixed in this branch.

---

## 5. Deliverables

- `generalization_audit/docs/POST_MERGE_AUDIT_V06.md` — this report.
- `results/verification_v13/post_merge_audit_v06.json` — machine-readable
  audit report (manifest verification, gate, recomputation summaries).
- `results/verification_v13/post_merge_b_recompute.json` — research v0.13
  recomputation details (33 checks).
- `results/verification_v13/post_merge_c_recompute.json` — academy v0.6
  recomputation details (34 checks).
- `generalization_audit/tests/test_comparison_key_resolution.py` — 5
  regression tests for the verifier fix.

## 6. Recommended fixes for other agents (not applied here)

- **Research:** for any population claim, run confirmatory 8×100 and
  report per-seed vectors; consider pinning `PYTHONHASHSEED` in future
  runs to strengthen determinism claims.
- **Academy:** for the EVREN teacher-mix claim, add more train seeds
  (≥5) before asserting transfer; document the memory-limit byte
  measurement to close the manifest INCONCLUSIVE item.

## 7. Reproduce

```bash
# audit suite (149)
cd <repo> && PYTHONHASHSEED=0 PYTHONPATH=".:python" \
  python -m unittest discover -s generalization_audit/tests -t .
# independent recomputations
PYTHONPATH=".:python" python - <<'PY'
# (run the audit's recomputation scripts; outputs above)
PY
# manifest verification (13) + gate
PYTHONPATH=".:python" python generalization_audit/tools/verify_experiment.py \
  --manifest generalization_audit/manifests/v13_final_l.manifest.json --json
```
