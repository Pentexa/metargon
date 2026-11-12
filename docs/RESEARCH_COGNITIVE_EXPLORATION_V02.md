# METARGON Academy — Cognitive Exploration v0.2: Robust Intelligence

**Branch:** `research/cognitive-exploration-v02`
**Run IDs:** `cognitive-exploration-v02-main` / `cognitive-exploration-v02-ablations`
**Status:** experimental research branch — not merged, not production default.

v0.2 targets the one significant regression reported by v0.1: success
drops on noisy observations (−12.1 pp at q=0.08), while preserving the
verified OOD exploration gains (+37.8 pp).

---

## 1. Diagnosis (why v0.1 regresses under noise)

In the Academy environment only the **switch/plate/door readings** are
corrupted (noise_q per reading per step); the agent's own position and
object positions are exact. At q=0.15 a random walk shows ~1–2 phantom
door-reading flips per door every few steps (`d1 False→True` with *no*
event).

v0.1's `learn()` treats every raw single-step flip as a real door event:

1. **Phantom opens** call `memory.observe_event(before_atoms, door,
   True, …)` → false support for hypotheses, phantom `passed[door]=False`
   resets, and phantom resolution of active holds (`holds_resolved_open`
   inflates; `memory_correct` counts fake successes).
2. **Phantom closes** inject `observe_event(..., False, …)` and
   `observe_nonevent` → true hypotheses get contradicted (false
   contradictions destroy correct rules).
3. **Phantom opens also poison decisions.** `belief_from_obs` copies the
   raw reading into `belief.door_open`; the planner then routes through a
   door that is actually closed → the action is invalid (`public_next`
   returns None) → `_invalid_streak` rises → **false stall detection and
   premature recovery**, burning the action budget.
4. Switch/plate reading noise misattributes the atoms recorded at event
   time (`atoms_from_observation`), further corrupting learning.

## 2. Mechanism (v0.2): bounded measurement confirmation

One opt-in flag, `cog_noise_robust` (default False; **not** implied by
`cognitive_exploration` so v0.1 configs keep exact behavior):

* **Exact switch tracking.** Switch states are maintained exactly from
  the Toggle action semantics (`_toggle_switch_if_standing`) — a Toggle
  on a switch cell flips that switch, independent of noisy readings.
* **Exact plate derivation.** Plate loads are derived from (reliable)
  object positions (`_inferred_atoms`) — no reading needed.
* **Exact transition channel.** Door transitions are taken from the step
  result's `events` tuple when present (the environment's visible change
  reports — already part of the student API and used by the replay
  experience memory). The confirmed door state and memory events follow
  the exact transitions, so phantom reading flips cannot create false
  events, false pass-through attempts, false stalls or phantom hold
  resolution.
* **Causal-link fallback** (`_update_events`) for environments without an
  events channel: a reading flip is accepted only when causally linked to
  an (exact) atom change (same step, previous step for delayed doors,
  exit from the door cell, or the episode bootstrap window — the first
  two steps, needed because initial-state rule satisfaction opens doors
  without prior atom changes).
* **Traversal correction.** Attempting to enter a door cell gives exact
  ground truth: a failed entry ⇒ door actually closed; a successful
  entry ⇒ actually open. Corrections override stale confirmed states
  (protects against transient one-step opens that close before the agent
  acts).
* **Confirmed state drives belief** (`_apply_confirmed_belief`): the
  planner, probes, pass-through logic, stall detection and hold
  resolution see confirmed door states and inferred switch states, not
  raw readings.

All state is bounded (per-door confirmed flags, event tuples), fully
checkpointed, and independent of environment specifics.

## 3. Experiment protocol

Harness: `academy/tests/test_cognitive_exploration_v02_experiment.py`
(CLI + CI smoke; imports shared bootstrap/metrics from the v0.1 harness).

* Agents: baseline (flags off), v01 (`cognitive_exploration=True`),
  v02 (+ `cog_noise_robust=True`); ablations below.
* Conditions: `classic_clean`, `classic_q04`, `classic_q08`,
  `classic_q15`, `ood_clean`, `ood_q08`, `conflict`, `tight`.
* Seeds: independent validation base `20261201` (v0.1 used `20261112`)
  with `PYTHONHASHSEED=0`; matched episode streams per seed.
* Endpoints: success (paired bootstrap 95% CI vs baseline and vs v01),
  useful/exact ground-truth evidence (post-hoc), false contradictions
  (a hypothesis matching a true clause ending contradicted), repeated
  probes, confirmed vs unconfirmed flips, recovery steps, planner cost,
  first-open latency.
* Ablations (6×40): `v02_noiseoff` (= v01), `v02_norecov`,
  `v02_nohold`, `v02_nogap`.

## 4. Findings (filled after runs)

> Tables populated from `results/academy_v01/cognitive_exploration_v02/`.

## 5. Proposed memory-interface changes (for later integration)

Unchanged in scope from v0.1 §6. The confirmation layer is implemented
agent-side; a future memory API could accept a `corroborated` flag per
event so the memory itself can weight confirmed evidence higher.

## 6. Honest limitations

* The exact transition channel assumes the environment reports real door
  transitions (as it does in the Academy API). In environments without
  such a channel the causal-link fallback applies, which is approximate
  under noise.
* The mechanism does not estimate q; the causal-link fallback is a fixed,
  bounded rule.
* The noisy-cell comparison remains the least controlled: the baseline
  also learns from corrupted readings, and the harness measures the
  relative improvement of the confirmation layer, not denoising.
* High-noise (q>0.15) regimes are untested.
* At q=0.08 the v0.2 student remains slightly (and non-significantly)
  behind v0.1 in this 6-seed sample; at q=0.15 it is slightly ahead —
  final verdict at the confirmatory size is in the artifact summary.
