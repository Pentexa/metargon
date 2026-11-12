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

## 5. Findings

Runs: main = 8 seeds × 60 episodes per cell; ablations = 6 seeds × 40
episodes per cell; `PYTHONHASHSEED=0`; matched episode streams; paired
bootstrap (95% CI, 2000 rounds). Raw JSON in
`results/academy_v01/cognitive_exploration/`.

### 5.1 Success (main)

| cell | base succ% | prop succ% | Δ pp [95% CI] |
|---|---:|---:|---:|
| `ood` (unseen causal mechanisms) | 37.5 | 75.8 | **+37.8 [+13.3, +66.2]** |
| `classic_test` (unseen conjunctive) | 40.4 | 55.6 | +15.6 [−2.7, +41.7] |
| `tight_test` (44-step budget) | 40.4 | 53.1 | +12.6 [−4.0, +39.0] |
| `conflict` (mid-run rule flip) | 50.4 | 59.6 | +8.9 [−17.3, +41.5] |
| `noisy_test` (noise_q=0.08) | 68.3 | 56.0 | **−12.1 [−28.3, −1.5]** |

Only the OOD gain has a CI excluding 0; the classic/tight/conflict gains
are directional at this sample size. The noisy cell is a **significant
regression** (see 5.4).

### 5.2 Secondary metrics (main; base → proposed)

| cell | steps (ok) | repeat probes | useful evidence% | exact evidence% | fallback% |
|---|---:|---:|---:|---:|---:|
| `classic_test` | 18.3 → 18.9 | 0.00 → 0.01 | 75.0 → 99.8 | 75.0 → 87.5 | 13.6 → 17.9 |
| `conflict` | 23.2 → 21.8 | 0.02 → 0.01 | 93.8 → 100.0 | 93.8 → 100.0 | 22.9 → 22.9 |
| `noisy_test` | 25.5 → 25.4 | 3.87 → 0.27 | 99.8 → 99.8 | 99.4 → 97.7 | 13.5 → 17.9 |
| `ood` | 12.3 → 20.6 | 0.00 → 0.00 | 87.5 → 100.0 | 87.5 → 100.0 | 0.0 → 15.5 |
| `tight_test` | 17.3 → 17.6 | 0.00 → 0.02 | 75.0 → 99.4 | 75.0 → 87.5 | 11.3 → 17.1 |

The proposed student learns the true rule far more often (useful evidence
75% → ~100% on classic/tight/conflict), with success lagging only where
execution is costly. In the noisy cell the baseline's repeated probes
(3.87) collapse to 0.27 — the proposed student stops wasting budget on
noise-confused repeats, but still loses overall success.

### 5.3 Ablations (Δ vs proposed; success)

| ablation | `classic_test` | `ood` |
|---|---:|---:|
| `proposed_norecov` (recovery off) | −5.5 [−16.2, +0.0] | **−47.6 [−80.8, −15.8]** |
| `proposed_nogap` (knowledge-gap off) | +1.2 [+0.0, +2.9] | 0.0 |
| `proposed_noinfo` (info scoring off) | 0.0 | 0.0 |
| `proposed_nohold` (hold off) | 0.0 | 0.0 |
| `proposed_nodecomp` (decomposition off) | 0.0 | 0.0 |

**Attribution:** recovery is the load-bearing mechanism. With recovery
off, per-episode probe count explodes (~19–21 vs ~1.3–1.6 for the
proposed student) and useful evidence drops (OOD 100% → 83.3%; classic
99.6% → 66.7%). The remaining mechanisms (gap scoring, info scoring, hold,
decomposition) show no measurable success contribution on these cells —
their value is currently limited to secondary metrics (e.g. repeated
probes in `conflict` 0.02 → 0.01) and to robustness not captured here.

### 5.4 Negative / open findings

1. **Noisy observations: proposed is worse** (−12.1 pp, CI excludes 0).
   Both arms extract door events from corrupted readings; recovery and
   hold mechanics burn budget under unreliable observations, and noisy
   single-step events still pollute memory. No denoising was implemented;
   this is the main open problem and the natural next iteration.
2. **No measurable success effect for gap/info/hold/decomposition**
   subsets on the tested cells (ablations ±0–0.8 pp). The mechanisms are
   cheap and safe, but only recovery is currently justified by the data.
3. **Sample size:** classic/tight/conflict CIs include 0; a confirmatory
   run at higher seeds is needed before claiming effects there.
4. **Cost overhead:** proposed run wall-time is 2–4× baseline per episode
   (bounded BFS re-planning), acceptable for research but worth
   optimizing before any production integration.

## 5.5 Post-hoc review fixes (PR finalization)

1. **`_plan_cached()` is now cap-aware.** Cache entries record the BFS
   expansion cap they were produced under. Successes are reused across
   caps (a proven plan is cap-independent); a cached *failure* is only
   reused for requests with `max_expand <= cached_cap` (a smaller search
   may have stopped before reaching the target). Legacy pre-v0.1 cache
   entries are normalized on load under the default cap.
2. **Goal-connectivity prefilter is genuinely default-off.** The prefilter
   is gated behind `cognitive_exploration`; with all flags off the agent
   executes the exact original goal-planning path.
3. **Artifact counts reconciled.** The committed artifacts are main
   8 seeds × 60 episodes and ablations 6 seeds × 40 episodes; the harness
   defaults now reproduce these exactly, and a regression test guards the
   match. Comparisons were rerun post-fix with the matched seeds (same
   seed base `20261112`); changed outcomes are reported in §5.1/5.3.

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
