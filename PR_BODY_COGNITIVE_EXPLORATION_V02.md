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

Main 6×40 (all conditions) + confirmatory 8×60 (key cells) are
committed in `results/academy_v01/cognitive_exploration_v02/`.

| cell | base% | v01% | v02% | v02−v01 pp [CI] |
|---|---:|---:|---:|---:|
| classic_clean (8×60) | 77.5 | 79.8 | 79.8 | 0.0 [0.0, 0.0] |
| classic_q04 (6×40) | 87.9 | 73.8 | 72.9 | −0.9 [−3.8, +0.8] |
| classic_q08 (8×60) | 86.5 | 84.4 | 73.8 | **−10.9 [−21.5, −1.5]** |
| classic_q15 (8×60) | 79.2 | 76.2 | 74.4 | −1.8 [−5.4, +2.1] |
| ood_clean (8×60) | 37.5 | 66.7 | 66.7 | 0.0 [0.0, 0.0] |
| conflict (6×40) | 62.1 | 57.9 | 57.9 | 0.0 [0.0, 0.0] |
| tight (6×40) | 67.5 | 70.4 | 70.4 | 0.0 [0.0, 0.0] |

Ablations (6×40): `v02_noiseoff` (=v01) q08 76.2 (v02 69.2), q15
65.4 (v02 66.7); `v02_norecov` q08 −9.4, q15 −14.2, ood −33.8;
`v02_nohold` ood_clean +19.6 (hold hurts on clean OOD); `v02_nogap`
q08 −5.8, q15 +1.7.

## Positive / negative findings

Positive:
- Clean, OOD, conflict and tight cells are **exact parity** with v0.1
  (verified per-seed) — the noise mechanism costs nothing in clean
  environments and preserves the v0.1 OOD gain (+29.2 pp vs baseline at
  8×60).
- False contradictions are eliminated on all cells (the exact
  transition channel cannot book phantom events; 0.00 across).
- The mechanism is smallest-change and fully opt-in / default-off.

Negative (preserved honestly):
- **The success criterion for noisy conditions is not met.** At the
  confirmatory scale q=0.08 is a significant regression (−10.9 pp, CI
  excludes 0) and q=0.15 is neutral (−1.8 pp, CI includes 0). v0.1's
  raw-reading approach occasionally benefits from noisy opens that
  deliver transient passage; v0.2 only books events confirmed by the
  exact transition channel or traversal, so it under-exploits them.
- Useful-evidence rate drops on noisy cells (v02 ~83–86% vs v01 100%).
- Recovery remains the load-bearing mechanism; hold is neutral-to-
  harmful on clean OOD (ablated +19.6 pp) — follow-up candidate.
- Recommended next iteration: exploit traversal-confirmed transient
  opens immediately (belief already updated by the exact channel) and
  re-evaluate the hold mechanism on OOD under noise.

## Files

- `academy/agents/metargon_agent.py` — `cog_noise_robust` confirmation
  layer + `_learn_noise_robust`;
- `academy/tests/test_cognitive_exploration.py` — new confirmation unit tests;
- `academy/tests/test_cognitive_exploration_v02_experiment.py` — experiment harness + CI smoke;
- `docs/RESEARCH_COGNITIVE_EXPLORATION_V02.md` — diagnosis + protocol;
- `results/academy_v01/cognitive_exploration_v02/*.json` — raw summaries.

No memory.py / NEST / frozen artifact / generalization_audit changes.
Not merged — ready for review.
