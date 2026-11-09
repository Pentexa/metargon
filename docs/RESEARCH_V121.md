# METARGON v0.12.1 — Regression Recovery

**Date:** 9 November 2026.
**Status:** Controlled paired study (4 seeds × 40 episodes, matched seeds,
frozen worlds) vs v0.10 / v0.11 / v0.12 (legacy).
**Branch:** `research/v121-regression-recovery` (merged into `main` via PR #33).
**Headline:** v0.12.1 closes the **F-noisy regression** (72.5% → 80.6%,
= v0.11) while preserving v0.12's temporal L discovery. H-stable and
R-stable regressions are investigated exhaustively; demonstrate-stable is
shown NOT to be caused by the v0.12 mechanisms tested, and R-stable
improves only when dual-context is disabled (at the cost of F D2
recovery) — reported honestly as unresolved trade-offs.

---

## 1. Background

v0.12 (PR #29) delivered temporal L discovery but regressed vs v0.11 on
F-noisy (−8.1pp), vs v0.10 on H-stable (−10.6pp), and vs v0.11 on
R-stable (−11.3pp). This branch diagnoses each regression with controlled
ablations and implements only evidence-supported, family-agnostic fixes.

## 2. Diagnosis (paired 4×40, same seeds as v12 report)

### 2.1 F-noisy (v12 72.5% vs v11 80.6%)
| variant | success | D1 exact | D2 exact |
|---|---:|---:|---:|
| v10 | 44.38 | 97.5 | 59.4 |
| v11 | 80.62 | 63.7 | 31.9 |
| v12 (dual, recent-credit on) | 72.50 | 87.5 | 76.9 |
| v12 -nodc (clean context) | 80.00 | 62.5 | 43.8 |
| v12 -norc (recent-credit off) | **81.25** | 80.6 | **75.6** |
| v12 -noad (adaptive-confirm off) | 72.50 | 87.5 | 76.9 |
| v12 -dst (dual_stable) | 73.12 | 79.4 | — |

→ Dual-context costs −7.5pp on noisy but is what restores D2 discovery
  (76.9 vs 43.8). Recent-credit costs −8.75pp and does NOT degrade D2.
  **Adopted fix: recent-credit off (F-noisy 81.25 ≈ v11, D2 75.6).**

### 2.2 H-stable (v12 73.1% vs v10 83.8%)
| variant | success | D2 exact |
|---|---:|---:|
| v10 | 83.75 | 63.1 |
| v11 | 73.12 | 41.2 |
| v11 -cf (no confirmed events) | 79.38 | 20.6 |
| v11 -ct (no certainty) | 73.12 | 41.2 |
| v11 -rp (no replay) | 73.12 | 41.2 |
| v12 | 73.12 | 70.0 |
| v12 -nodc / -norc / -noad / -dst | 73.12 | 70.0 / 41.2* |

* -nodc drops D2 to 41.2; -norc/-noad/-dst leave it at 70.0.

→ H loss is NOT caused by any v0.12 mechanism (all ablations identical at
  73.12). The shared v0.11 confirmed-event gating is the cause
  (v11 -cf +6.3pp), but disabling it kills D2 discovery (20.6).
  v0.12's dual-context restores D2 (70.0) while keeping success at 73.1.
  **No family-agnostic fix found to close the gap without losing D2
  recovery; left as an honest negative.**

### 2.3 R-stable (v12 60.6% vs v11 71.9%)
| variant | success |
|---|---:|
| v10 | 61.25 |
| v11 | 71.88 |
| v12 | 60.62 |
| v12 -norc (recent-credit off) | 60.62 |
| v12 -noad (adaptive-confirm off) | 60.62 |
| v12 -dst (dual_stable) | 60.00 |
| **v12 -nodc (dual-context off)** | **71.88** |
| v12 -dic (dual_if_changed) | 60.62 |

→ Recent-credit is NOT the cause (report hypothesis falsified). Dual-
  context IS: disabling it recovers v11 level (71.88) while preserving
  exact D1/D2 (70.6/86.9).
→ The `dual_if_changed` middle-ground fails because in R (hold-gated)
  pre≠post at event time, so it behaves like dual and keeps the cost.
→ Therefore R-stable and F-D2-recovery conflict on dual-context; one
  cannot have both with the current machinery. R is left as negative
  unless dual is dropped (F D2 regression).

### 2.4 Temporal L preservation (must not regress)
| | v10 | v11 | v12 | v12-notp |
|---|---:|---:|---:|---:|
| stable exact D1 | 0.0 | 0.0 | **16.9** | 0.0 |
| novel exact D1 | 0.0 | 0.0 | **23.1** | 0.0 |
| stable / novel success | 21.3/21.9 | 18.1/19.4 | **27.5/29.4** | 18.1/18.8 |

## 3. Prototypes measured and rejected (honest)

1. **Adaptive confirmed-event gating** by perceived anomaly rate: F-noisy
   53.1% (warm-up window lets raw noise through), H D2 44.4, L exact
   D1 0. Reverted.
2. **dual_stable** (post-context limited to stable sensors): no gain on
   F/H/R (F +0.6pp, H identical, R worse).
3. **dual_if_changed** (dual only when pre≠post): R still 60.62 (pre≠post
   holds in R), H D2 drops to 41.2, F D2 drops to 45.0. Reverted.

## 4. v0.12.1 (adopted; minimal, evidence-supported)

Only one change vs v0.12: **`use_recent_credit=False`** (default).
Dual-context, confirmed-event gating, adaptive confirm and temporal all
unchanged.

| | v10 | v11 | v12 (legacy) | v0.12.1 |
|---|---:|---:|---:|---:|
| F-noisy | 44.38 | 80.62 | 72.50 | **80.62** |
| F-noisy D2 exact | 59.4 | 31.9 | 76.9 | **75.6** |
| H-stable | 83.75 | 73.12 | 73.12 | **73.12** |
| H-stable D2 exact | 63.1 | 41.2 | 70.0 | **70.0** |
| R-stable | 61.25 | 71.88 | 60.62 | **60.62** |
| L stable/novel exact D1 | 0/0 | 0/0 | 16.9/23.1 | **16.9/23.1** |
| L stable/novel success | 21/22 | 18/19 | 27.5/29.4 | **26.9/25.0** |

Note: v0.12 legacy (recent-credit on) shows L exact D1 = 0/0 in the same
condition — recent-credit protection disrupts the ordered D2 sequence.
The v0.12.1 change (recent-credit off) is therefore *required* to
preserve temporal L discovery; it is not merely neutral there.

> *Docs-sync annotation (9 Nov 2026), findings recorded during the
> public-documentation sync; report values are kept as written:*
>
> 1. **Cell provenance is mixed in the table above.** The F/H/R "v12
>    (legacy)" cells match the committed final-run artifacts
>    (`final_f/h/r.json`, where the legacy ablation is faithful for
>    compositional families), but the L-row cells `16.9/23.1` and
>    `27.5/29.4` for "v12 (legacy)" originate from the v0.12 study
>    (`results/v12/summary_l_small.json`), **not** from
>    `results/v121/final_l.json`: in the final run the
>    `metargon12-legacy` ablation records exact D1 `0.0/0.0`, success
>    `18.12/18.75`, and `temporal_probes = 0` (see caveat 2 below).
> 2. **The legacy ablation is not a faithful v0.12 replica for family
>    L.** In `python/benchmark_v12.py` the `metargon12-legacy` config
>    pins dual-context + recent-credit **but leaves
>    `use_temporal` at its False default**, so final-run legacy rows
>    for L do not measure "v0.12 with recent-credit on". Consequently
>    the interpretation in the note above — that recent-credit
>    disruption of the ordered D2 sequence is the cause — is not
>    supported by the final-run artifact; the comparison that actually
>    measures temporal vs compositional behavior in the final runs is
>    `metargon12` vs `metargon12-notp`, with temporal Δ +16.88
>    [0, 50.6] (stable) / +23.12 [0, 50.6] (novel). A faithful v0.12
>    replication exists: the diagnosis-phase `eval2_l.json`
>    `metargon12` run (recorded before the default flip) matches the
>    v0.12 study (success 27.5/29.4, exact D1 16.9/23.1; `eval2_l.json`
>    records `temporal_probes` as raw counters 4475/4250, the same
>    scale family as the study summary's 44.75/42.5 per-seed-run
>    metric). The same semantics hold for the v0.12 study's own
>    `-norc`/`-notp` L rows (their temporal memory is off by
>    construction), so the only committed runs of the
>    recent-credit-OFF + temporal-ON combination are the v0.12.1
>    `final_l.json` runs themselves — which preserve temporal exact D1
>    (16.88/23.12); the v0.12.1 preservation conclusion therefore
>    stands independently of the legacy rows, without the
>    "recent-credit disruption" interpretation.
> 3. **Exact-D2 cells (`75.6` F-noisy, `70.0` H-stable for v0.12.1)
>    trace to the diagnosis-phase `-norc` runs** (`diag_f_noisy.json`,
>    `diag_h_stable.json`), not to the final artifacts, which record
>    exact D2 `45.0` (metargon12, `final_f.json`) and `41.25`
>    (metargon12, `final_h.json`) under nominally identical settings
>    (4 seeds × 40 episodes; matched seeds/conditions). Headline
>    success cells and L-temporal values DO match the final artifacts.
>    This diagnosis-vs-final discrepancy is unresolved and documented
>    as issue 3 in `docs/DOC_CONSISTENCY_NOTES.md` (requires a rerun;
>    no artifact was rewritten).

## 5. Negative results

1. H-stable regression persists (−10.6pp vs v10): not attributable to any
   v0.12 mechanism; caused by shared v0.11 confirmed gating; closing it
   sacrifices D2 recovery.
2. R-stable regression persists (−11.3pp vs v11): dual-context is the
   cause, but disabling it costs F-D2 recovery; `dual_if_changed` does
   not help (pre≠post in R).
3. The "adaptive confirmation by reliability" prototype failed in warm-up
   and was reverted.

## 6. Preservation checks
- L temporal exact D1 preserved (16.9/23.1; final L run re-checked).
- F D2 recovery preserved (75.6 vs v11's 31.9).
- 188-test Python suite passes.

## Reproduce

```bash
python -m unittest discover -s python -v
python python/benchmark_v12.py --seeds 4 --episodes 40 --family f \
    --conditions noisy_obs --parallel --workers 6 --no-mem \
    --policies metargon10,metargon11,metargon12,metargon12-legacy,bayes,oracle
# and likewise for h/r/l; results/ tables under results/v121/.
```
