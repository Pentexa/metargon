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

## Results

_fill from `results/academy_v01/cognitive_exploration/main_summary.json`
and `ablations_summary.json`_

## Ablations

_fill_

## Positive findings (preliminary)

→ fill after runs

## Negative / open findings

→ fill after runs

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
