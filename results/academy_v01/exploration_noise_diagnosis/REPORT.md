# Exploration Failure Diagnosis v0.1 — noisy-observation regression

Branch `research/exploration-noise-diagnosis-v01`, run ID
`exploration-noise-diagnosis-v01`, diagnostic seed base **20261118**
(disjoint from v0.1 20261112, v0.2 20261201, and the independent auditor's
20261121; auditor confirmatory seeds unused). `PYTHONHASHSEED=0`.

Reproduce:

```
$env:PYTHONHASHSEED="0"
python -m academy.tests.test_noise_diagnosis_experiment
```

## 1. Context

- PR #49 (Cognitive Exploration v0.1, merged): noisy_test (q=0.08)
  **−12.1 pp** [−28.3, −1.5] vs baseline; OOD +37.8 pp.
- PR #51 (v0.2, closed): noise-robust confirmation did not recover noisy
  performance (q=0.08 v0.2−v0.1 = −10.9 pp [−21.5, −1.5]).
- PR #52 (independent auditor): replication — explore Δ −0.21 [−0.29,
  −0.17] on noisy; explore+epoch −0.33 [−0.50, −0.13].

## 2. Method

Observation-only harness (`academy/tests/test_noise_diagnosis_experiment.py`):
no new mechanism. Per-step counters compare what the student's `learn()`
derives from raw readings against the environment's exact event stream and
exact hidden state (post-hoc scoring only; never fed to the agent).

Agents: baseline; v01 (`cognitive_exploration=True`); ablations
v01_norecov, v01_nohold, v01_nogap, v01_noinfo. Conditions: clean,
noisy (q=0.08), OOD. 6 seeds × 40 episodes per cell (matched streams).

## 3. Headline numbers

| cell | baseline succ | v01 succ | Δ v01−baseline [95% CI] |
|---|---|---|---|
| clean | 0.667 | 1.000 | +0.333 [+0.000, +0.667] |
| noisy (q=0.08) | 0.942 | 0.942 | −0.001 [−0.050, +0.050] |
| OOD | 0.500 | 0.887 | +0.383 [+0.054, +0.721] |

**Important sample caveat:** on these diagnostic seeds the noisy baseline is
94% (ceiling); PR #49's baseline was 68% on its seeds. The replication of
the *negative* RDD (v01 worse than baseline under noise) is **seed-
dependent** — see §6. The mechanism-level evidence below is arm-internal
(matched streams) and robust.

## 4. Root-cause evidence (per-step / per-episode means, noisy)

| metric | baseline | v01 | v01_nohold | v01_norecov | v01_nogap | v01_noinfo |
|---|---|---|---|---|---|---|
| success | 0.942 | 0.942 | **0.762** | 0.833 | 0.954 | 0.938 |
| phantom_open / step | 0.131 | 0.133 | 0.135 | 0.130 | 0.132 | 0.132 |
| phantom_close / step | 0.134 | 0.133 | 0.135 | 0.131 | 0.132 | 0.132 |
| missed_true_open / step | 0.013 | 0.014 | 0.012 | 0.012 | 0.013 | 0.013 |
| belief_reality_divergence / step | 0.151 | 0.151 | 0.152 | 0.148 | 0.149 | 0.150 |
| invalid_action / step | 0.001 | 0.001 | 0.002 | 0.005 | 0.000 | 0.001 |
| stalls / episode | 0.00 | 0.31 | 1.05 | 0.00 | 0.25 | 0.35 |
| recovery_steps / episode | 0.00 | 2.90 | 10.67 | 0.00 | 2.32 | 3.22 |
| holds_started / episode | 0.00 | 2.20 | 0.00 | 3.04 | 2.15 | 2.40 |
| holds_resolved_open / episode | 0.00 | 1.94 | 0.00 | 2.75 | 1.89 | 2.12 |
| holds_interrupted / episode | 0.00 | 0.26 | 0.00 | 0.27 | 0.26 | 0.27 |
| hold_preserve_steps / episode | 0.00 | 1.11 | 0.00 | 1.31 | 1.10 | 1.18 |
| random_fallback / episode | 2.01 | 1.86 | 4.83 | 0.50 | 1.30 | 1.88 |
| memory_correct / episode | 3.27 | 1.89 | 2.80 | 2.66 | 1.85 | 2.08 |
| memory_incorrect / episode | 0.01 | 0.01 | **0.05** | 0.01 | 0.00 | 0.01 |

Clean and OOD cells have zero phantom/divergence (q=0) by construction; the
v01 success gains there (+0.33 clean, +0.38 OOD) replicate PR #49.

## 5. Ranked root causes

### R1 (primary). Perception: phantom events are pervasive and untreated.
At q=0.08 every arm sees ~0.13 phantom open **and** ~0.13 phantom close per
step (readings flip with no real event; the environment's exact
`result.events` tuple is ignored by `learn()`, which derives events from raw
reading diffs). Consequences:
- phantom open → false memory support, `passed[door]=False` resets, phantom
  hold resolution, phantom `pending_probe` completion;
- phantom close → `observe_event(close)` and `observe_nonevent` corrupt
  true hypotheses (false contradictions — the v0.2 diagnosis agrees);
- `belief_reality_divergence ≈ 0.15` per step: the agent plans against raw
  readings (via `belief_from_obs`), so phantom opens make it attempt
  actually-closed doors. This is the direct feed into R2.

### R2. Premature recovery / stagnation detection fires under noise.
Stalls and recovery appear **only** in the noisy cell for v01 (0.31 stalls,
2.9 recovery steps/episode; clean/OOD: 0). With hold disabled, recovery
explodes (10.67 steps/episode). The invalid-action streak detector is the
trigger: phantom-open belief → invalid plan → `_invalid_streak` → stall →
directed recovery burns the action budget. The cost is real but bounded; it
is a symptom of R1, not an independent failure.

### R3 (rejected). Hold-based probing does NOT waste the budget — it protects it.
Removing hold under noise is the single worst ablation: **−0.18 vs v01**
(CI [−0.39, −0.01]) with memory_incorrect rising 0.01 → 0.05 and fallback
1.9 → 4.8. Deadline-based probe expiry (10 steps) routinely fires while the
true (delayed) door is still opening under noisy readings, so hypotheses
are penalized for correct but slow effects. Holds are the compensation; the
cost measured (1.1 preserving steps/episode) is small. PR #49's clean/OOD
ablations showed no hold contribution; the noisy contribution is protective.

### R4. Transient true openings are rare; the transient phenomenon is perceptual.
Ground truth over 120 random-walk episodes: 11 opens, 0 closes — doors stay
open once the rule holds; closures require the agent to break the rule or
stand on the door cell. Missed true opens are ~0.013/step and come from
corrupted readings hiding real events (they share R1's cause). Door-
exploitation failures (open but not passed) are not the dominant channel.

### R5. Gap-guided probing is an additional noise-triggered failure path.
Gap candidates are generated after stalls (which noise inflates), and the
latent `KeyError: door_open_after_gap` (diagnosed and fixed in this PR —
logging only) crashed `learn()` whenever a gap-sourced probe opened a door;
this path executes frequently only under noise. Success-wise, disabling gap
(nogap) is neutral (+0.013); its risk is the crash + added planning, not
success loss.

## 6. Honest limitations

- **Seed-dependence of the headline Δ.** PR #49 / auditor #52's noisy
  regression did not reproduce on diagnostic seeds where the noisy baseline
  sat at a 94% ceiling. Matched-arm comparisons remain valid; absolute
  deltas vs other runs are not directly comparable across seed bases.
- The diagnosis does not propose or evaluate a fix (out of scope). The
  natural next step suggested by R1 is to consume the environment's exact
  `result.events` channel and/or filter raw-reading flips before belief/
  memory updates — noting that PR #51's confirmation attempt (2-observation
  consensus) did not by itself recover noisy success, which points to the
  planning-side (R2) and hold-protection (R3) interactions being equally
  important.
- Sample: 6 seeds × 40 episodes per cell; CIs are paired per-seed
  bootstraps.

## 7. Tests

`academy/tests/test_noise_diagnosis.py` (9 tests): phantom/missed event
classification (clean real open; corrupted-reading phantom; missed true
open), belief divergence, trace-key coverage (regression for the fixed
KeyError), arm validity, harness smoke. Full suite: 123 passed.
