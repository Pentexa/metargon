# METARGON Academy v0.1 — Research Report

**Date:** 7 November 2026
**System:** METARGON Academy v0.1 (Python 3.11, stdlib-only; Rust kernel
via `cargo +1.88.0`)
**Machine:** Windows 11 (win32), CPU-only.

This report documents the first Academy research protocol: a reproducible
continual-learning and curriculum-generation system. It records what was
implemented, what was measured, and what failed — including negative
results, in line with the project's falsifiability protocol.

---

## 1. Pre-registered protocol (summary)

- Task pools: 147 train / 49 val / 49 test / 25 OOD configurations,
  immutable for the run (split seed `pool_split=20260901`).
- Per-policy training budget: 480 episodes from a curriculum-calibrated,
  deterministic schedule (matched across all policies).
- Evaluation: frozen val/test/OOD pools, paired seeds, bootstrap 95% CI;
  a 30-seed paired comparison on the test pool.
- Policies: random, episodic memory, tabular Q-learning, model-based Bayes
  (correct core model class), METARGON without replay, METARGON full,
  plus a privileged oracle (diagnostic only).

## 2. Calibration (measured difficulty)

The curriculum is not a fixed schedule; difficulty is measured with a
reference learner before training:

| bin | measured success | std | n |
|---|---:|---:|---:|
| easy | 0.600 | 0.490 | 40 |
| medium | 0.375 | 0.484 | 40 |
| hard | 0.275 | 0.447 | 40 |

The schedule focuses training where measured success is low and promotes a
bin only when measured success crosses the target.

## 3. Training results

Training: 480 episodes per policy on the identical calibrated stream
(matched exposure). All policies trained for the same number of episodes
on the same sequence of tasks.

| policy | success (train) | mean steps | wall (s) |
|---|---:|---:|---:|
| bayes (model-based) | 0.327 | 68.9 | 817.9 |
| metargon_full | 0.263 | 62.3 | 58.4 |
| metargon_noreplay | 0.263 | 62.3 | 56.0 |
| qlearning | 0.106 | 73.3 | 3.3 |
| random | 0.037 | 79.0 | 4.0 |
| episodic | 0.035 | 79.1 | 3.3 |
| oracle (privileged) | 1.000 | 14.9 | 20.0 |

Block-wise success (training, 120-episode blocks):

| policy | 0–119 | 120–239 | 240–359 | 360–479 |
|---|---:|---:|---:|---:|
| metargon_full | 0.267 | 0.242 | 0.217 | 0.325 |
| bayes | 0.275 | 0.317 | 0.358 | 0.358 |
| qlearning | 0.125 | 0.075 | 0.117 | 0.108 |
| random | 0.025 | 0.042 | 0.017 | 0.067 |
| episodic | 0.025 | 0.033 | 0.017 | 0.067 |

METARGON's trajectory is non-monotonic (an interference dip around block
2–3 followed by recovery), while the Bayesian planner improves steadily —
an early hint that the compositional learner trades sample efficiency for
flexibility.

## 4. Evaluation (held-out & OOD)

All numbers are per-seed means over paired streams (6 seeds × 40
episodes = 240 episodes per policy per pool). 95% bootstrap CIs shown.

### 4.1 Held-out test pool

| policy | success | 95% CI | mean steps |
|---|---:|---:|---:|
| oracle (privileged) | 1.000 | (1.000, 1.000) | 16.5 |
| metargon_full | 0.500 | (0.438, 0.562) | 46.5 |
| metargon_noreplay | 0.500 | (0.438, 0.562) | 46.5 |
| bayes (model-based) | 0.275 | (0.217, 0.329) | 69.4 |
| qlearning | 0.233 | (0.179, 0.283) | 64.8 |
| episodic | 0.037 | (0.017, 0.062) | 78.6 |
| random | 0.033 | (0.013, 0.054) | 78.9 |

### 4.2 Validation pool

| policy | success | 95% CI | mean steps |
|---|---:|---:|---:|
| oracle | 1.000 | (1.000, 1.000) | 12.5 |
| metargon_full | 0.667 | (0.604, 0.729) | 33.7 |
| metargon_noreplay | 0.667 | (0.604, 0.729) | 33.7 |
| bayes | 0.504 | (0.442, 0.571) | 61.6 |
| qlearning | 0.350 | (0.292, 0.412) | 57.2 |
| random | 0.054 | (0.025, 0.083) | 78.5 |
| episodic | 0.054 | (0.025, 0.083) | 78.5 |

### 4.3 OOD (unseen mechanisms, TP2 + absence atoms)

| policy | success | 95% CI | mean steps |
|---|---:|---:|---:|
| oracle | 1.000 | (1.000, 1.000) | 16.5 |
| metargon_full | 0.500 | (0.442, 0.562) | 47.5 |
| metargon_noreplay | 0.500 | (0.442, 0.562) | 47.5 |
| bayes | 0.208 | (0.158, 0.263) | 74.5 |
| random | 0.013 | (0.000, 0.029) | 79.9 |
| episodic | 0.013 | (0.000, 0.029) | 79.9 |
| qlearning | 0.000 | (0.000, 0.000) | 80.0 |

The METARGON agent does not degrade on OOD relative to its held-out test
performance, while the Bayes baseline — whose model class is correct on
core tasks but structurally mismatched on OOD — drops from 0.275 (test)
to 0.208 (OOD) despite stronger core performance. This is the expected
flexibility-vs-bias trade-off.

### 4.4 30-seed paired comparison (test pool)

30 seeds × 30 episodes = 900 episodes per policy; matched streams.

| policy | success | 95% CI |
|---|---:|---:|
| metargon_full | 0.433 | (0.401, 0.464) |
| random | 0.038 | (0.026, 0.051) |

METARGON outperforms the random policy by ≈ +39.5 percentage points
(paired seeds).

## 5. Continual-learning measurements

- **Learning progress:** per-episode success curves per policy
  (`learning_curves.csv`).
- **Retention/forgetting:** early vs late blocks on the same family (see
  curves; recomputed per family in future runs).
- **Adaptation:** episodes to first success after rule change.
- **Zero-shot / few-shot transfer:** test/OOD pools are unseen; OOD
  measures transfer to genuinely novel mechanism combinations.
- **Memory utilization:** hypothesis counts/status per episode
  (`diagnostics_trace`).
- **Computational cost:** wall-clock, planner operations (`planner_ops`),
  plan calls.

## 6. Ablations

- METARGON full vs no-replay (replay enabled/disabled) on identical
  streams. Measured result: **replay is neutral on this benchmark**
  (identical training success 0.263 and identical held-out/val/OOD
  success 0.500/0.667/0.500 in both variants). Replay does change the
  internal memory state (offline adjudication of later-created
  hypotheses), but the change does not yet translate into better task
  performance. This is a documented negative/neutral result that
  motivates better replay curricula (e.g., priority replay of door-event
  transitions rather than uniform sampling).

## 7. Efficiency

- Wall-clock (training, 480 episodes): metargon_full 58.4 s,
  metargon_noreplay 56.0 s, random 4.0 s, episodic 3.3 s, qlearning
  3.3 s, bayes 817.9 s, oracle 20.0 s.
- Model size (JSON state): metargon_full 146 KB, bayes 872 KB
  (particle+plan cache), qlearning 152 KB, episodic 18 KB, random/oracle
  ~10 KB.
- Planner operations: tracked per episode (`planner_ops`, `plan_calls`)
  in training traces.
- Peak RAM: not instrumented in v0.1 (stdlib-only design); simulated
  operation counts are reported separately from any energy claim — **no
  energy claims are made** (no hardware power meter was used).

## 8. Limitations & negative results

1. Single-agent curriculum calibration precedes training; the schedule is
   fixed for all policies to keep exposure matched. Future work: make the
   curriculum itself per-policy while controlling exposure statistically.
2. The hypothesis language is conjunctive singletons/pairs over the public
   atoms; OOD absence-role tasks are learnable in principle but require
   overriding a monotone "load plates helps" bias. Held-out OOD success
   (0.500) is driven mainly by switch-first tasks; absence-atom tasks
   remain hard — documented, not a bug.
3. **Replay is currently neutral** (measured): offline adjudication
   changes memory but not behavior on this benchmark.
4. METARGON does not beat the competence Bayes baseline on core held-out
   tasks (0.500 vs 0.275 is METARGON-favorable on *test*, but on val the
   Bayes baseline reaches 0.504 vs 0.667 — results are pool-dependent; the
   reliable statement is that METARGON is competitive on core held-out and
   more robust on OOD).
5. No energy/hardware measurements; no GPU/cloud/paid APIs.
6. The 30-seed paired comparison uses 30 episodes/seed; the full 240
   episode/seed protocol is available via the documented CLI overrides.

## 9. Next research step

**Single highest-value improvement:** replace uniform replay with
*priority replay of door-event transitions* (resample experiences that
contain door open/close events and goal-reaching trajectories, weighted by
recent surprise), then re-run the ablation. It directly targets the one
measured neutral result (replay) and the non-monotonic training curve, and
keeps the same matched-exposure protocol.
