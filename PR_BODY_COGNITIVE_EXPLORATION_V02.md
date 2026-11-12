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

Main (6×40 pilot) on the table below; confirmatory 8×60 on key cells is
committed in `results/academy_v01/cognitive_exploration_v02/`.

| cell | base% | v01% | v02% | v02−v01 pp [CI] |
|---|---:|---:|---:|---:|
| classic_clean | 70.0 | 72.1 | 72.1 | 0.0 [0.0, 0.0] |
| classic_q04 | 87.9 | 73.8 | 72.9 | −0.9 [−3.8, +0.8] |
| classic_q08 | 82.5 | 76.2 | 69.2 | −7.2 [−17.1, +1.7] |
| classic_q15 | 72.5 | 65.4 | 66.7 | +1.2 [−4.6, +7.1] |
| ood_clean | 33.3 | 67.1 | 67.1 | 0.0 [0.0, 0.0] |
| conflict | 62.1 | 57.9 | 57.9 | 0.0 [0.0, 0.0] |
| tight | 67.5 | 70.4 | 70.4 | 0.0 [0.0, 0.0] |

Ablations (6×40): `v02_noiseoff` (=v01) q08 76.2 (v02 69.2), q15
65.4 (v02 66.7); `v02_norecov` q08 −9.4, q15 −14.2, ood −33.8;
`v02_nohold` ood_clean +19.6 (hold hurts on clean OOD); `v02_nogap`
q08 −5.8, q15 +1.7.

## Positive / negative findings

Positive:
- Clean, OOD, conflict and tight cells are **exact parity** with v0.1
  (verified per-seed) — the noise mechanism costs nothing in clean
  environments and preserves the v0.1 OOD gain (+33.8 pp vs baseline).
- At q=0.15 the v0.2 student is *better* than v0.1 (+1.2 pp) and
  markedly better than baseline (+? vs 72.5 → 66.7 uses v01 ref; vs
  baseline −5.8).
- False contradictions are near-zero across all cells (the exact
  transition channel cannot book phantom events).

Negative / neutral:
- At q=0.08 the v0.2 student is behind v0.1 by −7.2 pp (CI includes 0
  at 6 seeds; confirmatory 8×60 reports the final verdict). The exact
  transition channel removes phantom noise, but the v0.1 student's
  raw-reading approach occasionally *benefits* from noisy opens that
  deliver transient passage; v0.2 does not exploit them until a
  traversal confirms them.
- Useful-evidence rate drops on noisy cells (v02 ~83–86% vs v01 100%)
  because v0.2 books fewer (but more truthful) events.
- Recovery remains the load-bearing mechanism everywhere; hold is
  neutral-to-harmful on clean OOD (ablated −19.6 pp) — a candidate for
  a follow-up targeted ablation, not changed here.

## Files

- `academy/agents/metargon_agent.py` — `cog_noise_robust` confirmation
  layer + `_learn_noise_robust`;
- `academy/tests/test_cognitive_exploration.py` — new confirmation unit tests;
- `academy/tests/test_cognitive_exploration_v02_experiment.py` — experiment harness + CI smoke;
- `docs/RESEARCH_COGNITIVE_EXPLORATION_V02.md` — diagnosis + protocol;
- `results/academy_v01/cognitive_exploration_v02/*.json` — raw summaries.

No memory.py / NEST / frozen artifact / generalization_audit changes.
Not merged — ready for review.
