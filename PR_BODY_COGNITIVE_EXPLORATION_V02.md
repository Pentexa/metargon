# METARGON Academy — Cognitive Exploration v0.2: Robust Intelligence

**Branch:** `research/cognitive-exploration-v02` · **Base:** `origin/main` @ `0d1a71c`

## What this is

Robust-intelligence follow-up to Cognitive Exploration v0.1 (PR #49):
recover the noisy-condition regression while preserving the verified OOD
gains. One small, opt-in, bounded mechanism — **measurement
confirmation** — supported by a root-cause diagnosis.

## Diagnosis

v0.1's `learn()` treats every raw single-step door-reading flip as a real
event. Under noise this causes: phantom memory events (false support),
phantom pass-through attempts (invalid actions → **false stalls and
premature recovery**), phantom hold resolution, and false contradictions
of true hypotheses. Position/object observations are exact; only
switch/plate/door readings are corrupted.

## Mechanism (v0.2)

`cog_noise_robust` (default off, not implied by `cognitive_exploration`):
- readings must be confirmed by `noise_confirm_n`=2 consecutive
  observations before any change is treated as real (false-confirm
  probability ≈ q²);
- confirmed state drives belief (planning/pass-through/stall/hold);
- memory events fire only on confirmed transitions; plate loads derived
  exactly from object positions; switch atoms use confirmed readings;
- bounded per-signal streak counters, fully checkpointed.

## Experiment

Harness `academy/tests/test_cognitive_exploration_v02_experiment.py`;
independent validation seed base 20261201; matched streams; paired
bootstrap. Agents: baseline, v01, v02. Conditions: classic q=0/.04/.08/
.15, OOD clean + q=.08, conflict, tight. Ablations: v02_noiseoff
(=v01), v02_norecov, v02_nohold, v02_nogap.

## Results

_fill after runs_

## Positive / negative findings

_fill after runs_

## Files

- `academy/agents/metargon_agent.py` — `cog_noise_robust` confirmation
  layer + `_learn_noise_robust`;
- `academy/tests/test_cognitive_exploration.py` — new confirmation unit tests;
- `academy/tests/test_cognitive_exploration_v02_experiment.py` — experiment harness + CI smoke;
- `docs/RESEARCH_COGNITIVE_EXPLORATION_V02.md` — diagnosis + protocol;
- `results/academy_v01/cognitive_exploration_v02/*.json` — raw summaries.

No memory.py / NEST / frozen artifact / generalization_audit changes.
Not merged — ready for review.
