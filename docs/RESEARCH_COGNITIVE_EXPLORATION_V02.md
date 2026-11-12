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

* **Confirmation rule.** A door/switch reading is only treated as a
  *change* after `noise_confirm_n` (default 2) consecutive observations
  agree on the new value. A single flip is ignored (perception noise); a
  persistent change (a real environmental event) is confirmed within one
  step. Probability of a false confirmation at q=0.15 ≈ q² ≈ 2.25%
  per signal per step.
* **Confirmed state drives belief** (`_apply_confirmed_belief`): the
  planner, probes, pass-through logic, stall detection and hold
  resolution see confirmed door/switch states, not raw readings →
  phantom opens can no longer cause invalid-action streaks or false
  recovery.
* **Confirmed transitions drive memory** (`_learn_noise_robust`): memory
  events fire only on confirmed transitions; nonevents/maintain use the
  confirmed state → no false contradictions from phantom flips.
* **Exact plate derivation.** Plate loads are derived from (reliable)
  object positions (`_clean_atoms`); switch atoms use confirmed switch
  states — event-time atom capture is exact up to confirmation latency.

All state is bounded (per-signal streak counters), fully checkpointed,
and independent of environment specifics.

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

* Confirmation adds a one-step latency to event detection (a real event
  is usually confirmed on the following observation); this costs one
  step per real event.
* The mechanism does not estimate q; it is a fixed, bounded confirm rule
  (robust across q=0.04–0.15 by design, not by tuning).
* The noisy-cell comparison remains the least controlled: the baseline
  also learns from corrupted readings, and the harness measures the
  relative improvement of the confirmation layer, not denoising.
* High-noise (q>0.15) regimes are untested.
