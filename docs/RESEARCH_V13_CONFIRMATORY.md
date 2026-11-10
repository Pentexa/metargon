# METARGON v0.13 Confirmatory Study — Preregistration & Report

**Status:** Preregistered protocol (this document, first section) written
BEFORE any confirmatory run. Results sections filled after data collection.
**Branch:** `research/v13-confirmatory`
**Starting main SHA:** `a617365bf513db0632c837aa14d638bef6cc3cf3` (includes
merged PR #40 post-merge audit).

---

## PART I — PREREGISTERED PROTOCOL (written before runs)

### 1. Background and objective

The v0.13 pilot (4 seeds × 40 episodes; `results/v13/final_*.json`)
showed temporal exact-D1 discovery lifts on family L (ordered-sequence):
stable 16.88% → 39.38%, novel 23.12% → 33.13%, with the `-gen` ablation
(rolling generator off) reproducing v0.12.1 exactly. The independent
post-merge audit (PR #40) recomputed these numbers and confirmed them.
However, the stable paired CI includes 0, both conditions rest on 4
seeds, and H/R regressions remain unresolved.

**Objective:** determine whether the temporal discovery gain generalizes
beyond the pilot, using an independent, preregistered confirmatory sample.
Primary focus: family L. Secondary: F/H/R preservation cells.

### 2. Independent evaluation seeds (previously unused)

Pilot used world-seed base 11800 (seeds 11800–11803).
**Confirmatory seed base: 21800** → seeds 21800–21807 (8 seeds). These
seeds have never been used by any study in this repository; they generate
entirely fresh world/episode streams (world RNG = `Random(seed)`, episode
noise seed = `seed*1000+e`). Agent policies use their own RNG seeded with
`seed + 1000003` (unchanged convention).

### 3. Frozen protocol

| component | specification |
|---|---|
| families / conditions | primary: L stable, L novel. secondary/preservation: F noisy_obs, H stable, R stable |
| seeds × episodes | 8 × 100 per family-condition cell (frozen v02 confirmatory minimum) |
| policies | `metargon12` (v0.12.1 baseline), `metargon13` (candidate), `metargon13-gen` (rolling-generator ablation = v0.12 generator behavior), `metargon13-notp` (temporal off; attribution only, L) |
| world generation | frozen pools read-only (F: `split_pools()`; H: seed 20261107; R/L: audit frozen JSON) |
| action budget | `max_steps=80` per episode |
| planner budget | cap 700 expansions/decision, probe max 3 plans (unchanged harness constants) |
| `PYTHONHASHSEED` | **0** (pinned for all confirmatory runs) |
| comparison direction | candidate − baseline (`metargon13` minus `metargon12`; `metargon13` minus `metargon13-gen`; `metargon13` minus `metargon13-notp`) |
| bootstrap | paired, seed 4301, 1500 rounds, 95% CI (harness convention) |
| measurement | per-seed rows recorded for every policy × cell; success rate, exact D1/D2, covered D1/D2, steps, invalid actions, CPU, temporal diagnostics |

### 4. Endpoints

**Primary endpoints** (population-level claims require CI excluding 0 at
8×100 and per-seed evidence not carried by a single seed):

1. L stable exact-D1: Δ `metargon13` − `metargon12`.
2. L novel exact-D1: Δ `metargon13` − `metargon12`.
3. Rolling-generator attribution: Δ `metargon13` − `metargon13-gen`
   (identical to 1/2 under the null that rolling generation is the only
   change).

**Secondary endpoints** (preservation, no improvement claim expected):

4. L stable/novel success rate Δ vs v0.12.1.
5. F noisy_obs success + exact D1/D2: `metargon13` vs `metargon12`.
6. H stable success + exact D2.
7. R stable success (known regression; also vs `metargon13-nodc` only if
   run is cheap — documented as exploratory).

### 5. Success criteria (predeclared)

- A confirmatory claim of improvement for endpoint k requires: CI of the
  paired bootstrap delta **excludes 0**, the per-seed vector shows the
  effect is not driven by a single seed (≥3 of 8 seeds with nonzero
  contribution for the primary L endpoints), and the ablation control
  (endpoint 3) supports the mechanism.
- If the CI includes 0, the outcome is reported as **"no measured
  confirmatory effect"** regardless of point estimate.
- Preservation endpoints (4–7): compared against v0.12.1; a decline
  beyond the bootstrap noise bound is reported as a regression.

### 6. Analysis plan

- Means and per-seed vectors per policy × cell; population stddev;
  nonzero-seed counts.
- Paired bootstrap 95% CIs (seed 4301, 1500 rounds) for every delta.
- All numbers kept separate from pilot numbers (`results/v13_confirmatory/`,
  not `results/v13/`).
- Diagnostics for failure classification (Mission C) run on the SAME
  confirmatory seeds, instrumented per episode (see PART III).

### 7. Resource plan (Mission B)

- Smoke test first on **throwaway seeds** (base 90000, 1 seed × 6
  episodes per family) to measure per-episode wall time; the confirmatory
  seeds are touched only by the real study (no tuning on them).
- Bounded parallelism: `--workers 4` (4 logical cores), one background
  process per run, explicit per-run timeouts.
- Incremental artifacts: one JSON output per family-condition run; on
  interruption, completed runs are kept and only the missing cell is
  re-run (safe resumption).
- If 8×100 proves infeasible within the measured budget, the study is
  marked **INCOMPLETE** and partial evidence is preserved with that label
  (never silently reduced sample size).

### 8. Predeclared stopping / correctness rules

- The agent code (`metargon_v13.py`) is **not modified** during the
  confirmatory study. Any future agent change is exploratory and gets its
  own artifacts.
- No confirmatory seed is used for debugging or parameter choice; the
  smoke test uses seed base 90000 only.
- Any deviation from this protocol is documented inline in PART II.

---

## PART II — RESULTS (filled after data collection)

### 8. Confirmatory sample

- 8 independent seeds × 100 episodes per cell, seed base 21800
  (seeds 21800–21807; disjoint from pilot 11800–11803).
- `PYTHONHASHSEED=0` for all runs.
- Policies: `metargon12`, `metargon13`, `metargon13-gen`,
  `metargon13-notp` (L); `metargon12`, `metargon13` (F/H/R preservation);
  plus exploratory `metargon13-nodc`/`metargon12-nodc` on R.
- Artifacts: `results/v13_confirmatory/*.json` (+ `.jsonl` checkpoints).

### 9. Primary endpoints

Source: `results/v13_confirmatory/final_l_8x100.json` +
`analysis_l.json` (8 seeds × 100 episodes, seeds 21800–21807,
`PYTHONHASHSEED=0`, paired bootstrap seed 4301 / 1500 rounds).

| endpoint | Δ pp | 95% CI | seeds | +Δ seeds |
|---|---:|---:|---:|---:|
| P1 L stable exact-D1 13−12 | **+36.88** | **[13.13, 69.38]** | 8 | 6 |
| P2 L novel exact-D1 13−12 | **+17.88** | **[6.38, 31.63]** | 8 | 6 |
| P3 L stable exact-D1 13−gen | +36.88 | [13.13, 69.38] | 8 | 6 |
| P3 L novel exact-D1 13−gen | +17.88 | [6.38, 31.63] | 8 | 6 |
| P3b L stable exact-D1 13−notp | +68.75 | [45.63, 91.75] | 8 | 6 |
| P3b L novel exact-D1 13−notp | +60.38 | [47.25, 79.12] | 8 | 8 |
| P4 L stable success 13−12 | +0.50 | [0.13, 0.88] | 8 | 4 |
| P4 L novel success 13−12 | −9.62 | [−24.13, 0.25] | 8 | 2 |

Per-seed exact-D1 vectors (stable / novel):
- `metargon13`: stable [92, 0, 91, 91, 92, 0, 92, 92] (nonzero 6/8);
  novel [42, 43, 91, 41, 92, 40, 42, 92] (nonzero 8/8).
- `metargon12`: stable [85, 0, 0, 0, 85, 0, 0, 85] (nonzero 3/8);
  novel [35, 43, 50, 0, 85, 42, 0, 85] (nonzero 6/8).
- `metargon13-gen` identically equals `metargon12` on every seed
  (mechanism attribution: the rolling generator is the only change).
- `metargon13-notp` / `metargon12-notp`: all-zero on all seeds.

**Confirmatory verdict for the primary endpoints (predeclared criteria):**
all three primary L endpoints **exclude 0** at 8×100 (stable +36.88
[13.13, 69.38]; novel +17.88 [6.38, 31.63]); the effect is not driven by
a single seed (≥3/8 nonzero seeds on both conditions); and the
rolling-generator ablation reproduces v0.12.1 exactly. The pilot's
directional L temporal-discovery gain **generalizes** to an independent
8×100 sample, with stable now excluding 0 (pilot CI [0.00, 78.75] → [13.13, 69.38]).

Honest secondary nuance: P4 novel success 13−12 has a negative point
estimate (−9.62 pp) with CI including 0 — higher exact discovery on novel
did not translate to higher success at confirmatory scale (v13 takes more
steps, 58.0 vs 53.9). Reported as no measured success effect on novel;
discovery and success remain distinct quantities.

### 10. Secondary / preservation endpoints

Source: `results/v13_confirmatory/final_f_8x100.json`,
`final_h_8x100.json`, `final_r_8x100.json`, `final_r_nodc_8x100.json`
(8 seeds × 100 episodes each, seed base 21800, `PYTHONHASHSEED=0`).

| cell | metric | `metargon12` | `metargon13` | Δ 13−12 [95% CI] |
|---|---:|---:|---:|---:|
| F noisy_obs | success | 62.12 | 62.12 | +0.00 [0.00, 0.00] |
| F noisy_obs | exact D1 | 66.25 | 66.25 | +0.00 [0.00, 0.00] |
| F noisy_obs | exact D2 | 55.00 | 55.00 | +0.00 [0.00, 0.00] |
| H stable | success | 74.25 | 74.25 | +0.00 [0.00, 0.00] |
| H stable | exact D1 | 98.25 | 98.25 | +0.00 [0.00, 0.00] |
| H stable | exact D2 | 73.00 | 73.00 | +0.00 [0.00, 0.00] |
| R stable | success | 68.38 | 68.38 | +0.00 [0.00, 0.00] |
| R stable | exact D1 | 86.50 | 86.50 | +0.00 [0.00, 0.00] |
| R stable | exact D2 | 80.12 | 80.12 | +0.00 [0.00, 0.00] |
| R stable −nodc (exploratory) | success | 80.38 (`12-nodc`) | 80.38 (`13-nodc`) | +0.00 [0.00, 0.00] |

F noisy_obs & H stable notes: `metargon13` and `metargon12` rows are
**identical on every seed** (Δ exactly 0.00 pp on success, exact D1,
exact D2) — exact preservation at 8×100; the rolling-generator change
does not affect F or H. The H-stable regression vs v0.10 (83.75 →
73.12 pilot; 74.25 here) is therefore inherited from earlier versions
(confirmed-event gating), not caused by v0.13 — confirmed at 8×100.

R stable notes: v0.13 == v0.12.1 exactly (68.38 success; Δ=0.00). The
R-stable regression vs v0.11 persists at 8×100, and the exploratory
`-nodc` dual-context-off variant recovers 80.38 (Δ +12.00 pp vs dual,
both policies identical) — reproducing the pilot's direction at
confirmatory scale. This trade-off (F-D2 recovery vs R) is documented,
not resolved by v0.13.

H/R regression localization (instrumented 8×40, v13, `hr_diag_*.md`):

| class | H stable | R stable |
|---|---:|---:|
| learned+success | 73.1% | 67.8% |
| learned but episode failed | 22.5% | 29.4% |
| failed without exact learning | 4.4% | 2.8% |

Both regressions are **execution-limited, not learning-limited**: the
correct rules are learned in ≥96% of episodes (H ≤4.4%, R ≤2.8%
non-learning), yet success lags. H's gap is inherited from confirmed-event
gating; R's from dual-context + sustained-hold execution (the `-nodc`
recovery at +12 pp comes from cleaner learning contexts, not from a
representational fix). Neither is touched by v0.13's temporal change.

### 11. Preregistration compliance

- [x] Protocol written before any confirmatory run (documented above).
- [x] Seeds 21800+ previously unused (pilot used 11800–11803).
- [x] No agent tuning on confirmatory seeds (policy code untouched;
      smoke test used throwaway seed base 90000).
- [x] Confirmatory results kept separate from pilot (`results/v13_confirmatory/`).
- [x] `PYTHONHASHSEED=0` pinned.
- [x] Comparison direction candidate − baseline; paired bootstrap CIs.
- [x] Partial-evidence rule: no silent sample reduction.


---

## PART III — CAUSAL DISCOVERY DIAGNOSIS (Mission C)

Failure taxonomy per episode/seed (instrumented in `diagnose_v13.py`, run
on the confirmatory L seeds):

| class | meaning | instrumentation |
|---|---|---|
| C1 sequence never observed | correct ordered pair never appears in the belief-flip stream | `true_pair_observed` = false |
| C2 candidate not generated | sequence observed but correct temporal hypothesis never created | `correct_created` = 0 while floor true |
| C3 generated but not verified | correct candidate created but never reached `verified` | `correct_verified_max` = 0 with `correct_created` > 0 |
| C4 contradicted / evicted | correct candidate's evidence destroyed | `correct_contradicted` > 0 or `correct_evicted` > 0 |
| C5 discovered but unused | verified correct rule exists but the planner never executes it | verified rule present with 0 temporal probes for that door (or probes blocked by protection) |

**Results (confirmatory seeds 21800–21807, 8×40 instrumented):**

| class | L stable v12 | L stable v13 | L novel v12 | L novel v13 |
|---|---:|---:|---:|---:|
| success/discovered-and-used | 32.2% | 59.4% | 27.5% | 44.1% |
| C1 sequence never observed | 27.8% | 26.6% | 35.0% | 35.9% |
| C2 candidate not generated | 27.5% | **0.0%** | 15.6% | **0.0%** |
| C3 generated but not verified | 12.5% | 14.1% | 21.9% | 20.0% |
| C4 contradicted/evicted | 0.0% | 0.0% | 0.0% | 0.0% |
| C5 discovered but unused | 0.0% | 0.0% | 0.0% | 0.0% |

v12 C2 concentration: stable seeds 21802/21803 show 37/40 C2 episodes
each (floor true, 0 candidates created) — the v0.12 single-slot
generator re-enters its pilot failure mode on these fresh seeds; v13 on
the same seeds creates 66/63 candidates and verifies 31/31 episodes.

Interpretation:
- **Rolling generation eliminated C2 entirely** (the pilot's dominant
  failure mode: v0.12 seed 0 had 37/40 C2 episodes). Candidate
  generation is no longer a bottleneck.
- Remaining failures are (a) C1 — the agent rarely arranges the ordered
  sequence in those episodes (behavioral/opportunity limit: the payoff
  stream never shows the true pair), concentrated in seeds 21801/21805
  (stable) and spread across most seeds (novel); and (b) C3 — the
  candidate is created but does not accumulate 3.0 weighted support
  within the episode's evidence window (verification latency).
- No contradiction/eviction or discovered-but-unused failure was
  observed in the confirmatory diagnostics: evidence is never destroyed
  and every verified rule is used by the planner.
- Per-seed detail in `results/v13_confirmatory/classification_*.md`.

Method note: classification uses instrumented 8×40 runs on the same
confirmatory seeds; the count of "success_or_used" episodes reflects the
diagnostic policy behavior (v13), consistent with the 8×100 benchmark
success rates.

---

## PART III.5 — Resources and profiling (Mission B)

- Smoke/profiling runs used **throwaway seeds 90000–90010** (6–30
  episodes); confirmatory seeds 21800+ were touched only by the study.
- Measured per-episode wall cost under load ≈ 1.8–3.5 s (L) and
  ≈ 0.5–1.0 s (F/H/R), i.e. a full 8×100 cell ≈ 25–35 min single-policy
  wall at 3–4 workers.
- Execution: checkpointed JSONL per cell (safe resumption), bounded
  3–4 workers, explicit per-run timeouts. All cells completed at the
  full preregistered 8×100 — no partial/INCOMPLETE reduction needed.
- Total wall for the full study (L primary + F/H/R + diagnostics)
  ≈ 3.5 h on this 4-core machine.

## PART IV — H/R REGRESSION DIAGNOSIS (Mission D)

Exploratory, separate from frozen policy: H-stable and R-stable 8×100
cells (endpoints 6–7) plus the documented `R -nodc` trade-off check.
Any proposed fix is evaluated in separate artifacts and never changes the
frozen v0.13 confirmatory policy.
