# METARGON Academy — Cognitive Exploration v0.1

**Branch:** `research/cognitive-exploration-v01`
**Run ID:** `cognitive-exploration-v01-main` / `cognitive-exploration-v01-ablations`
**Status:** experimental research branch — not merged, not production default.

This document describes the diagnosis, the bounded mechanisms, the
experiment protocol, and the findings. All mechanisms are opt-in and
**off by default**; the baseline student is byte-for-byte unchanged when
no cognitive flag is set.

---

## 1. Mission

Improve the Academy student's independent exploration, uncertainty
handling and action planning — i.e. better decisions about *which
experiment to perform* when current beliefs are insufficient. Everything
must stay inside `MetargonAcademyAgent.act()`; no LLM planner, no
fabricated learning, no predetermined action sequences, no hard-coded
per-environment solutions.

## 2. Diagnosis (what was wrong)

Inspecting `academy/agents/metargon_agent.py` and `academy/agents/planner.py`:

1. **Probe deadlines start at selection, not at completion.** `pending_probe`
   deadlines (`probe_timeout`, 10 steps) are stamped when a probe is *chosen*,
   not when its plan finishes. Plans to load plates / traverse the map can
   take 15–30 steps, so the deadline routinely fires mid-plan: the antecedent
   is marked tried, and if it came from a hypothesis the hypothesis is
   penalized (`memory_incorrect`) before the experiment even completed.
2. **Every door event erases every door's tried set.** `learn()` resets all
   tried sets on *any* open/close event, so a D1 opening re-opens the full
   D2 probe list — identical antecedents get re-executed (repeated probes).
3. **Search-failed candidates are re-planned every step.** Feasibility is a
   cheap component filter, but *planning* is a BFS over the full public state
   (objects included). Candidates that are component-feasible but impossible
   to plan within the expansion cap (e.g. plate-loading with objects) were
   re-searched from scratch on every act() call, wasting the action budget.
4. **No progress / loop detection.** The student has no notion of being
   stuck: it will repeat invalid actions (belief/reality divergence on hidden
   doors), bounce between cells, or re-try the same plan — no recovery.
5. **No information assessment.** All probe candidates were rank-ordered by
   a fixed priority class; there was no notion of which experiment actually
   *discriminates* the open hypotheses or *closes a knowledge gap*.
6. **Conjunctive probes are all-or-nothing.** If a conjunction was not
   fully feasible (e.g. an object behind a closed door), it was discarded —
   even when a feasible subset would have produced evidence.
7. **No bounded per-act() compute budget.** Act() cost was bounded per plan
   but not per decision; pathological states could trigger many full BFS
   searches in a single step.

## 3. Mechanisms (all bounded, independent, opt-in)

New flags on `MetargonAcademyAgent` (default `False`; `cognitive_exploration=True`
turns all on; each sub-flag can be switched independently for ablations):

| flag | mechanism |
|---|---|
| `cognitive_exploration` | master switch |
| `cog_knowledge_gaps` | per-door local atom evidence profile → coverage (`_uncovered_atoms`) used as a novelty signal and for gap-guided candidate pairs after stalls |
| `cog_info_gain` | deterministic information-value score per candidate: novelty (uncovered atoms) + weighted overlap with live hypotheses − attempt penalty − plan cost |
| `cog_recovery` | stagnation detection (invalid-action streak, no state change, state revisits) + directed anti-loop movement (`_recovery_plan`: least-recently-visited valid action), capped per episode |
| `cog_decompose` | `maximal_feasible_subset()` (planner) turns an infeasible conjunction into its largest feasible subset and probes that |
| `cog_probe_hold` | hold-based probe attribution: model-backed probes are *held* (condition preserved via strict preserving actions) until the door opens, the hold budget expires (genuine failure), or the condition breaks (inconclusive — no penalty) |

Additional hard bounds:

* per-act() planner budget (`planner_budget_per_act`, default 4000 expansions);
* low-tier probes (vocabulary/gap) plan with a reduced expansion cap (700)
  while model-backed probes keep the full cap (1500);
* cheap component pre-filter for goal planning (goal in a different connected
  component ⇒ skip BFS entirely);
* search-failed candidates are marked tried once (no per-step re-search);
* episode-size recovery budget (`recovery_budget`=12) and trigger cap (4);
* hold budget = `hold_budget` (10 steps).

`planner.py` adds one pure helper: `maximal_feasible_subset(state, target, geom)`.

## 4. Measurement

The experiment harness lives in
`academy/tests/test_cognitive_exploration_experiment.py` (importable, CLI,
and CI smoke):

* **Conditions** (matched episode streams, `make_episodes`):
  * `classic_test` — held-out core split (unseen conjunctive mechanisms), 80 steps;
  * `ood` — unseen causal mechanisms (absence-atom family), 80 steps;
  * `noisy_test` — classic test with `noise_q=0.08`, 80 steps;
  * `tight_test` — classic test, 44 steps (tight budget);
  * `conflict` — training pool with `switch_half=True` (rules flip mid-run).
* **Agents:** baseline (all flags off), proposed (all on), and five
  ablations (`proposed_nogap`, `proposed_noinfo`, `proposed_norecov`,
  `proposed_nohold`, `proposed_nodecomp`).
* **Endpoints:** success rate (per-seed paired bootstrap vs baseline),
  mean steps (successful / all), first-open latency, repeated experiments
  (new vs repeat attempt counters `probes_new`/`probes_repeat`), loop
  fraction, random-fallback share, and post-hoc ground-truth match
  (`useful_evidence_rate`, `exact_evidence_rate` — judged against `EnvConfig`
  only *after* the episode; never passed to the agent).

Run:

```
python -m academy.tests.test_cognitive_exploration_experiment --seeds 8 --episodes 60
python -m academy.tests.test_cognitive_exploration_experiment --ablations --seeds 6 --episodes 40
```

Results are written to `results/academy_v01/cognitive_exploration/`.

## 5. Findings (filled after runs)

> Tables populated from `main_summary.json` / `ablations_summary.json`.

## 6. Proposed memory-interface changes (for later integration)

No changes were made to `academy/agents/memory.py` (owned by the memory
line). The agent-side mechanisms were implemented with the current memory
API only. Recommended future integration:

1. **Agent-side doubt / noise filtering.** `learn()` currently forwards every
   observed door transition to `memory.observe_event`; under noisy
   observations, single-step reading flips pollute evidence. A proposed API:
   `observe_event(..., corroborated: bool)` or an agent-side gate that only
   forwards events confirmed by consecutive observations.
2. **Local knowledge-gap profile.** The per-door local evidence
   (`_local_atom_evidence`) is a bounded agent-side structure; a natural
   home is the memory's atom-transfer profile
   (`transfer_atoms` / `_atom_stats`) extended with per-episode local
   counts, so the naming/evidence semantics live in one place.
3. **Attempt bookkeeping.** `_attempts_per_antecedent` (per-door, bounded)
   could live on `CompositionalMemory` as per-antecedent attempt counters
   exposed through `query()` ranking.

## 7. Honest limitations

* The hold mechanism only applies to model-backed probes (hypothesis/bank);
  vocabulary probes still use the deadline path.
* `noisy_test` is the least controlled condition: both arms derive door
  events from (corrupted) observations, and the harness does not evaluate
  a dedicated denoising mechanism.
* Tight-budget and conflict cells stress recovery more than memory; the
  exact contribution of each sub-mechanism is what the ablations measure.
* Results are on the Academy's two 7×5 topologies; the mechanisms are
  intentionally general but were only validated here.
