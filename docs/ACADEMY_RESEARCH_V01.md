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

_(fill from results/academy_v01/train_*.json and learning_curves.csv)_

| policy | episodes | success rate | mean steps | wall (s) |
|---|---:|---:|---:|---:|

## 4. Evaluation (held-out & OOD)

_(fill from eval_val/test/ood.json; all numbers are per-seed means over
paired streams)_

### 4.1 Held-out test pool

| policy | success | 95% CI | mean steps |
|---|---:|---:|---:|

### 4.2 Validation pool

...

### 4.3 OOD (unseen mechanisms)

...

### 4.4 30-seed paired comparison (test pool)

...

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
  streams. Preliminary finding: replay changes the memory state (more
  verifications vs more contradictions) but *does not yet improve* final
  success on this benchmark — a documented negative/neutral result that
  motivates better replay curricula.

## 7. Efficiency

- Wall-clock: measured per policy (see tables).
- Model size: JSON state sizes per policy (`model_sizes.csv`).
- Peak RAM: not instrumented in v0.1 (stdlib only); a note is kept in
  limitations. Simulated operation counts (planner expansions) are
  reported separately from any energy claim; **no energy claims are
  made** — no hardware power meter was used.

## 8. Limitations & negative results

1. Single-agent curriculum calibration precedes training; the schedule is
   fixed for all policies to keep exposure matched.
2. The hypothesis language is conjunctive singletons/pairs over the public
   atoms; OOD absence-role tasks are learnable in principle but require
   overriding a monotone "load plates helps" bias — success rates there
   are expected to be low, which is informative, not a bug.
3. Replay's offline adjudication currently contradicts more than it
   verifies in some regimes (measured; see ablations).
4. No energy/hardware measurements; no GPU/cloud/paid APIs.
5. Evaluation seeds/episodes can be lowered via CLI overrides for quick
   runs; the 30-seed comparison is the headline paired result.

## 9. Next research step

_(single highest-value improvement; decided at the end of the run)_
