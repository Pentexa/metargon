# METARGON v0.13 Confirmatory Generalization & Regression Diagnosis

**Branch:** `research/v13-confirmatory` · **Base:** `origin/main` @ `a617365` (includes PR #40 post-merge audit)

## Summary

Preregistered confirmatory study (8 independent seeds × 100 episodes per
cell, seed base 21800, `PYTHONHASHSEED=0`) testing whether the v0.13
rolling temporal generator's L-family exact-D1 gains generalize beyond
the 4×40 pilot. Diagnosis of causal-discovery failures (Mission C) and
H/R regressions (Mission D) included.

## Primary findings (confirmatory, preregistered endpoints)

| endpoint | Δ pp (13−12) | 95% CI | nonzero seeds |
|---|---:|---:|---:|
| L stable exact-D1 | **+36.88** | **[13.13, 69.38]** | 6/8 |
| L novel exact-D1 | **+17.88** | **[6.38, 31.63]** | 6/8 |
| L stable exact-D1 (rolling ablation 13−gen) | +36.88 | [13.13, 69.38] | 6/8 |
| L novel exact-D1 (rolling ablation 13−gen) | +17.88 | [6.38, 31.63] | 6/8 |
| L stable success 13−12 | +0.50 | [0.13, 0.88] | 4/8 |
| L novel success 13−12 | −9.62 | [−24.13, 0.25] | 2/8 |

**Verdict:** all three preregistered primary L endpoints exclude 0 at
8×100; the effect is not carried by a single seed; the `-gen` ablation
(rolling generator off) exactly reproduces v0.12.1 per-seed vectors →
the mechanism attribution holds. The pilot's stable CI [0, 78.75]
tightens to [13.13, 69.38]. Novel success shows no measured effect (CI
includes 0; point estimate negative) — discovery ≠ success, reported
honestly.

## Preservation (Δ 13−12 at 8×100)

| cell | success | exact D1 | exact D2 |
|---|---:|---:|---:|
| F noisy_obs | 62.12 = 62.12 | 66.25 = 66.25 | 55.00 = 55.00 |
| H stable | 74.25 = 74.25 | 98.25 = 98.25 | 73.00 = 73.00 |
| R stable | 68.38 = 68.38 | 86.50 = 86.50 | 80.12 = 80.12 |

All rows identical per seed (Δ exactly 0.00). No new regression from
v0.13 at confirmatory scale.

## Mission C — failure classification (confirmatory L, 8×40 instrumented)

| class | stable v12 | stable v13 | novel v12 | novel v13 |
|---|---:|---:|---:|---:|
| success/discovered-and-used | 32.2% | 59.4% | 27.5% | 44.1% |
| C1 sequence never observed | 27.8% | 26.6% | 35.0% | 35.9% |
| C2 candidate not generated | 27.5% | **0.0%** | 15.6% | **0.0%** |
| C3 generated but not verified | 12.5% | 14.1% | 21.9% | 20.0% |
| C4 contradicted/evicted | 0 | 0 | 0 | 0 |
| C5 discovered but unused | 0 | 0 | 0 | 0 |

Rolling generation eliminates C2 entirely (v12 re-enters its pilot
failure on seeds 21802/21803: 37/40 C2 each); remaining failures are C1
(behavioral/opportunity floor) and C3 (verification latency).

## Mission D — H/R regressions

H/R regressions are **execution-limited, not learning-limited** (≥96%
of episodes learn the correct rule; success lags). H's gap inherited
from confirmed-event gating; R's dual-context cost reproduced at 8×100
(`-nodc` recovers 80.38 vs 68.38, exploratory). Neither touched by
v0.13.

## Protocol compliance

- Preregistered before runs (docs/RESEARCH_V13_CONFIRMATORY.md PART I).
- Seeds 21800+ independent (pilot used 11800–11803); smoke used
  throwaway 90000.
- No agent tuning on confirmatory seeds; v0.13 policy untouched.
- `PYTHONHASHSEED=0`; candidate−baseline; paired bootstrap (seed 4301,
  1500 rounds, 95% CI).
- Confirmatory results separate from pilot (`results/v13_confirmatory/`).
- Resource-safe: smoke-tested, checkpointed JSONL, bounded 3–4 workers,
  full 8×100 completed (no partial reduction).

## Artifacts & verification

- Raw per-seed JSON: `results/v13_confirmatory/final_{l,f,h,r}_8x100.json`,
  `final_r_nodc_8x100.json`, diagnostics `diag_*_8x40.json`, analysis
  JSON/MD.
- 11 stamped Quality v0.5 manifests under
  `generalization_audit/manifests/v13_confirmatory_*`.
- Verifier: **0 FAIL across all 11 manifests** (PASS counts 6–332;
  INCONCLUSIVE only metadata omissions: memory limit, generic-format
  structural checks).
- Artifact gate: PASS for `results/v13` and memory_v06.

## Tests

- Full Python suite (incl. new harness/analysis/classification tests).
- Audit suite (149, incl. artifact-gate provenance).

Not merged; ready for review.
