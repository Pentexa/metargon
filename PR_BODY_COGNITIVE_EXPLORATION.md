# METARGON Academy — Cognitive Exploration v0.1

**Branch:** `research/cognitive-exploration-v01` · **Base:** `origin/main` @ `6531b6b`

## What this is

A research branch on the Academy student's **exploration, uncertainty
handling and action planning**: deciding *which experiment to run* when
current beliefs are insufficient. No LLM planner, no fabricated learning,
no predetermined action sequences, no per-environment hard-coding.

## Diagnosis (from `metargon_agent.py` / `planner.py`)

1. Probe deadlines are stamped at *selection* — long plans expire mid-plan,
   marking correct probes (and hypotheses) failed before completion.
2. Any door event erases **all** doors' tried sets → repeated identical probes.
3. Component-feasible but search-infeasible candidates are re-planned every
   step (full object-BFS) → wasted action budget.
4. No stagnation/loop detection — the student can repeat invalid actions
   (belief/reality mismatch on hidden doors) indefinitely.
5. Candidate ordering is fixed-priority only; no information assessment.
6. Conjunctive probes are all-or-nothing — feasible subsets are discarded.
7. No per-act() compute budget.

## Mechanisms (bounded, independent, all opt-in, default OFF)

- knowledge-gap coverage (`cog_knowledge_gaps`) + info-value scoring
  (`cog_info_gain`);
- stagnation detection + anti-loop recovery (`cog_recovery`);
- conjunction decomposition (`cog_decompose`) via new planner helper
  `maximal_feasible_subset`;
- hold-based probe attribution (`cog_probe_hold`) — wait through delayed
  openings, no false contradictions on interrupted probes;
- per-act() planner budget, tiered expansion caps, search-failed marking,
  cheap component pre-filter for goal planning.

Baseline (all flags off) is unchanged; all shared-pruning changes are
outcome-preserving (only skip guaranteed-failing BFS).

## Experiments

Harness: `academy/tests/test_cognitive_exploration_experiment.py`.
Conditions: `classic_test`, `ood`, `noisy_test`, `tight_test`, `conflict`
(switch_half). Endpoints: success (paired bootstrap), action efficiency,
first-open latency, repeated experiments (`probes_new/repeat`),
loop/fallback shares, post-hoc ground-truth useful/exact evidence.

## Results (main; 8 seeds × 60 episodes, paired bootstrap 95% CI)

| cell | base succ% | prop succ% | Δ pp [CI] |
|---|---:|---:|---:|
| ood (unseen causal mechanisms) | 37.5 | 75.8 | **+37.8 [+13.3, +66.2]** |
| classic_test (unseen conjunctive) | 40.4 | 55.6 | +15.6 [−2.7, +41.7] |
| tight_test (44-step budget) | 40.4 | 53.1 | +12.6 [−4.0, +39.0] |
| conflict (mid-run rule flip) | 50.4 | 59.6 | +8.9 [−17.3, +41.5] |
| noisy_test (noise_q=0.08) | 67.1 | 56.0 | **−10.8 [−26.9, −1.0]** |

Secondary metrics (base → proposed): useful/exact ground-truth evidence
rises to ~100% on classic/tight/conflict; repeated probes collapse in the
noisy cell (4.14 → 0.27); successful-episode steps are flat
(±0.5–0.7). Full tables in
`results/academy_v01/cognitive_exploration/main_summary.json`.

## Ablations (6 seeds × 40 episodes; Δ vs proposed)

| ablation | classic Δ pp [CI] | ood Δ pp [CI] |
|---|---:|---:|
| `proposed_norecov` (recovery off) | −5.6 [−14.0, +0.0] | **−38.5 [−67.5, −13.3]** |
| `proposed_nogap` (knowledge-gap off) | +0.8 [+0.0, +2.1] | 0.0 |
| `proposed_noinfo` (info scoring off) | +0.4 [+0.0, +1.2] | 0.0 |
| `proposed_nohold` (hold off) | 0.0 | 0.0 |
| `proposed_nodecomp` (decomposition off) | 0.0 | 0.0 |

## Positive findings

- **Recovery is the load-bearing mechanism.** Disabling it collapses OOD
  success by ~38 pp and classic by ~6 pp, while probe counts explode
  (~600 vs ~50 per episode) — the student without recovery thrashes
  instead of exploring.
- **Strongest gain on unseen causal mechanisms (OOD)**: +37.8 pp with a
  95% CI excluding 0; useful-evidence rate 87.5% → 100%.
- Directionally positive on classic, tight-budget and conflicting-rule
  cells (CIs include 0 at 8 seeds — underpowered, not proof of effect).
- Useful/exact ground-truth evidence improves or matches baseline
  everywhere except noisy (exact 99.4% → 97.7%).

## Negative / open findings (reported honestly)

- **Noisy observations: proposed is worse (−10.8 pp, CI excludes 0).**
  Recovery/held probes burn budget under unreliable readings, and the
  binary door-event extraction from corrupted observations injects false
  events into memory. No denoising was implemented (documented as future
  work); this cell needs a dedicated treatment.
- Hold, knowledge-gap and info-scoring mechanisms show **no measurable
  success contribution** on these cells (0 or ~0 Δ); their value is
  currently limited to secondary metrics (e.g. fewer repeated probes in
  conflict: 0.50 → 0.15).
- Classic cells are not fully powered: 8 seeds × 60 eps leaves wide CIs;
  the effect sizes on classic/tight/conflict should be treated as
  directional until a confirmatory run.

## Files

- `academy/agents/metargon_agent.py` — mechanisms + flags + instrumentation;
- `academy/agents/planner.py` — `maximal_feasible_subset`;
- `academy/tests/test_cognitive_exploration.py` — mechanism unit tests;
- `academy/tests/test_cognitive_exploration_experiment.py` — experiment harness + CI smoke;
- `docs/RESEARCH_COGNITIVE_EXPLORATION.md` — protocol, diagnosis, proposed memory-interface changes;
- `results/academy_v01/cognitive_exploration/*.json` — raw summaries.

## Verification

- Academy suite + new tests pass;
- `scripts/check_repo.py` clean;
- Leakage audit (forbidden hidden-simulator tokens) still clean.

Not merged — ready for review.
