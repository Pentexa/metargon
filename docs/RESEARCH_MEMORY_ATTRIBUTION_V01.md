# METARGON Academy — Persistent Learning Diagnosis v0.1

**Branch:** `research/memory-attribution-v01`
**Run IDs:** `memory-diagnosis-attribution-v01-main` / `-rlearn`
**Status:** diagnosis report — no memory mechanism is changed.

This branch is a *measurement and root-cause* study. It does not modify
the planner, the exploration agent, the memory, or any audit framework.
It uses the shipped agent and the existing Memory v0.7 mechanisms
(`memory_episode_scoped`, `probe_reset_per_episode`), plus controls, to
answer five questions about persistent learning.

---

## 1. Context: why this study exists

Prior evidence (closed PR #50 cognitive memory v0.1; closed PR #53
cognitive memory v0.2; Memory v0.7 report) established:

* Memory v0.6-era: most stored hypotheses become contradicted after
  mixed-task training; cross-task negative evidence destroys verified
  hypotheses; the exploration-only (`novocab`) baseline can match the
  trained agent → retained memory often does not help.
* Memory v0.7 (`memory_episode_scoped`): per-epoch evidence scoping +
  demotion instead of destruction + cross-epoch uncertainty-aware
  retrieval → measurable frozen transfer gain vs fresh (0.594 vs 0.312)
  and vs shuffled (Δ +0.315, CI [0.094, 0.531]); `epoch_reset` (adds
  per-episode probe reset) is the best shipped configuration.
* PR #50: a reliability-ranking retrieval layer was **negative** — it
  made hypothesis probes more relevant but starved exact novel-single
  discovery, and repeated learning A/B/A collapsed (A2 − A1 = −0.500).
* PR #53: a protect-exploration gate (trusted-transfer tier + cap)
  fixed the exploration starvation and recovered A/B/A to epoch_reset
  parity, but showed no measured improvement over `epoch_reset` itself.

Open questions this branch addresses with a fresh diagnostic harness:
1. Why do verified hypotheses later become unreliable?
2. Why does retained memory sometimes reduce success?
3. Do successful decisions actually depend on previous knowledge?
4. Is A/B/A recovery memory retention or rediscovery?
5. When does `epoch_reset` protect learning, and when does it fail?

## 2. Protocol

Harness: `academy/tools/memory_diagnosis_attribution_v01.py`
Tests: `academy/tests/test_memory_diagnosis_attribution.py`

### 2.1 Variants (matched kwargs; controls)

| variant | memory | controls |
|---|---|---|
| `fresh` | untrained floor | — |
| `original` | default memory (global counts) | trained |
| `epoch` | `memory_episode_scoped=True`, min_verified=2 | trained |
| `epoch_reset` | epoch + `probe_reset_per_episode=True` | trained (primary baseline) |
| `muted` | original, memory wiped before eval | **memory-muted control** (same policy, no retained knowledge) |
| `shuffled` | original, antecedent→consequence associations shuffled before eval | **content-broken control** |

### 2.2 Task families and regimes

* classic — held-out test pool (compatible mechanisms);
* conflict — flipped-polarity pool (same observable contexts, different
  hidden rules) built by `build_conflict_pool` and isolated from
  train/val/test/ood;
* noisy — classic configs, observation noise q=0.15;
* ood — unseen causal mechanism family.
* Regimes per family: **frozen** zero-shot transfer (policy frozen;
  frozen integrity asserted), and **adaptive** transfer (learning
  enabled during eval).

### 2.3 Diagnostic seeds

Independent base `20270201` (distinct from all published/accredited runs:
20260901, 20261107/08, 20261112, 20261201). `PYTHONHASHSEED=0`,
canonical config ordering (hash-seed independent), paired per-config
bootstrap for CIs, frozen-integrity checks in all frozen cells.

### 2.4 Measured quantities

* success — config-mean success (frozen and adaptive);
* retained knowledge — per-episode memory snapshot classified against
  ground truth: exact/subset/incorrect × status; false contradictions
  (true rules contradicted); usable exact hypotheses;
* reuse — hypothesis-sourced plans during eval, split into plans whose
  hypothesis predates the eval phase (`created_step < eval_start_step`,
  retained knowledge) vs created during eval; `memory_correct` /
  `memory_incorrect` attribution counters;
* negative transfer — `original`/`epoch_reset` vs `muted` and
  `shuffled` controls on the same eval stream;
* exact-rule discovery — novel-single probe exact-hit rates;
* A/B/A — repeated-learning protocol with per-phase hypothesis
  bookkeeping: A1 train → frozen eval A′ → B (conflict) train → frozen
  eval B → A2 train → frozen eval A″; attribution splits A2 success
  between *retained* hypotheses (created in A1, surviving B) and
  *rediscovered* ones (created during B/A2); A1→B status transitions.

### 2.5 Reproduce

```
$env:PYTHONHASHSEED="0"
python academy/tools/memory_diagnosis_attribution_v01.py --seeds 20270201 20270202 --train-episodes 120
python academy/tools/memory_diagnosis_attribution_v01.py --rlearn
python academy/tools/memory_diagnosis_attribution_v01.py --report
```

Artifacts: `results/academy_v01/memory_diagnosis_attribution_v01/`
(main_summary.json, rlearn_summary.json, REPORT.md).

## 3. Findings

> Filled from committed artifacts (see REPORT.md and JSON summaries).

## 4. Constraints honored

* No memory/planner/exploration-agent code modified — harness only.
* No tuning against the independent auditor's final evaluation seeds
  (diagnostic seeds are new and disjoint).
* Anti-cheat: frozen cells assert no learning leakage; eval pools
  isolated from train; hash-seed independent canonical ordering.
