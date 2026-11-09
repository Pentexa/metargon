# Academy Memory v0.4 — Context-Aware Evidence and Continual Learning

**Branch:** `academy/memory-v04`
**Base:** latest `origin/main`
**Status:** implementation complete; measured results below.
**Date:** 2026-11-08

---

## 1. Summary

Memory v0.3 established that training the Academy student on the classic
pool *degrades* held-out performance: fresh 0.695 vs trained 0.562 on
the same frozen grid (−13.25 pp). ~70 hypotheses end up contradicted;
true hypotheses are generated and accumulate support, but cross-task
evidence (same public antecedent, different hidden rule) later
contradicts them.

This work implements **context-aware evidence handling**: the memory
now learns *in which task-context* evidence was collected (the context
signature is derived purely from public observations), so that:

- correct hypotheses survive contradictions observed in **other**
  task contexts (reversible confidence instead of irreversible
  destruction);
- task switches do not count as evidence against previous-task
  hypotheses (consecutive-nonevent windows reset on context change);
- new contexts do not borrow knowledge that has no evidence there
  (conservative transfer gating).

All mechanisms are independent, off by default, and preserve v0.3
behavior exactly when disabled.

---

## 2. Failure-mode recap (from v0.3 diagnosis)

| Observed | Cause |
|---|---|
| ~70/128 hypotheses contradicted | cross-task nonevents contradict correct hypotheses from other tasks |
| 128-cap never reached | status flipping, not capacity pressure |
| true clauses generated early, high support | generation works; attribution is the failure |
| probes rarely hit true conjunctive rules | single-atom probe vocabulary |

---

## 3. Mechanisms (v0.4, all off by default)

### 3.1 Context signature (public observations only)
`belief_from_obs` at episode start yields: agent position, switch
states, plate loads, object positions, door-open states. This tuple is
the task-context signature — no hidden configuration is accessed, and
no environment-family knowledge is hardcoded.

### 3.2 `context_scoped` — per-context evidence
Each hypothesis stores support/contradiction counts **per context**
(`sup_by_ctx`, `con_by_ctx`). Status is computed from the *active*
context's counts only. A hypothesis correct in context A is not
destroyed by contradictions in context B; switching back to A restores
verified status (reversible confidence).

### 3.3 `task_change_reset` — task-switch nonevent reset
When the context signature changes, consecutive-nonevent windows
(`_continuous_true`) are cleared, so a task switch cannot contribute to
contradiction thresholds.

### 3.4 `context_gate_transfer` — conservative transfer gating
`query()` only returns hypotheses that have support in the active
context. New contexts start with no borrowed knowledge until evidence
arrives.

---

## 4. Continual-learning benchmark

`academy/tools/memory_experiment_v04.py`:

- **train seeds** {20261107, 20261108}: each seed defines one episode
  stream; every variant trains on the identical stream for that seed
  (matched exposure). 160 episodes each.
- **learning curves**: at every block boundary (40 episodes), the
  variant is evaluated on a small frozen grid (8 configs × 2 seeds × 2
  episodes) — performance throughout training, not only at the end.
- **final evaluation**: 16-config classic grid (×2 seeds ×4 episodes),
  12-config OOD probe, shared 8-config substrate retention probes.
- **controls**: `fresh` (untrained), `memory_wiped` (train like
  original, then discard hypothesis memory before evaluation — answers
  "does retaining memory help or hurt?"), `original`, `probe_local`
  (best v0.3), and the v0.4 ablations (`ctx`, `ctx_reset`,
  `ctx_gate`, `ctx_full`).
- Paired bootstrap CIs vs `original` on per-config success vectors.

---

## 5. Results

*Pinned run: `PYTHONHASHSEED=0`, 2 train seeds × 160 episodes, frozen
16-config classic grid, 12-config OOD, 8-config substrate retention.*

### 5.1 Final held-out numbers (mean over 2 train seeds)

| Variant | Train | Classic | OOD | Memory status (mean) |
|---|---|---|---|---|
| fresh | — | **0.695** | 0.417 | — |
| original | 0.341 | 0.594 | 0.333 | 2.5 verified / 75 contradicted |
| probe_local | 0.422 | **0.625** | 0.375 | 0.5 v / 77.5 c |
| ctx | 0.369 | **0.625** | 0.417 | 2.5 v / 76.5 c |
| ctx_reset | 0.369 | **0.625** | 0.417 | 2.5 v / 76.5 c |
| ctx_gate | 0.366 | **0.625** | 0.417 | 2.5 v / 76.5 c |
| ctx_full | 0.497 | 0.562 | 0.417 | 75.5 contradicted |
| memory_wiped | 0.341 | 0.594 | 0.333 | (memory discarded) |

### 5.2 Paired deltas vs original (95% CI, per-config vectors)

| Variant | Classic Δ | Train Δ |
|---|---|---|
| fresh | −0.100 ([−0.211, 0.000]) | −0.341 |
| probe_local | −0.032 ([−0.094, 0.000]) | +0.081 |
| ctx | −0.030 ([−0.094, 0.000]) | +0.028 |
| ctx_reset | −0.030 ([−0.094, 0.000]) | +0.028 |
| ctx_gate | −0.030 ([−0.094, 0.000]) | +0.025 |
| ctx_full | +0.030 ([0.000, +0.094]) | +0.156 |
| memory_wiped | 0.000 ([0.000, 0.000]) | 0.000 |

### 5.3 Train-vs-transfer curves (mid-grid success by episode block)

| Variant | ep40 | ep80 | ep120 | ep160 |
|---|---|---|---|---|
| original | 0.438 | 0.438 | 0.500 | 0.500 |
| probe_local | 0.500 | 0.438 | 0.500 | 0.500 |
| ctx | 0.500 | 0.500 | 0.500 | 0.500 |
| ctx_full | 0.500 | 0.500 | 0.500 | 0.438 |
| memory_wiped | 0.438 | 0.438 | 0.500 | 0.500 |

---

## 6. Interpretation

- **Negative transfer decreased but did not close.** Fresh 0.695 vs
  best trained 0.625 (−7 pp) vs original 0.594 (−10 pp). `probe_local`
  and the `ctx` family each recover ~3 pp of the transfer gap; CI
  upper bounds touch 0 (directionally positive, not significant at this
  grid size).
- **No variant beats the untrained control.** The gap is reduced but
  the fresh baseline remains above every trained variant.
- **Attribution changes alone did not rescue transfer.** `ctx`,
  `ctx_reset`, `ctx_gate` all behave identically on the classic grid
  (0.625) — the per-context evidence bookkeeping changes memory state
  (fewer contradicted spills) but not enough to move generalization.
  `ctx_reset`/`ctx_gate` add nothing over `ctx` on this benchmark.
- **More verified hypotheses ≠ success.** `ctx_full` reaches the best
  training success (0.497) but the *worst* trained classic score
  (0.562) — combining adaptive probing with full context gating is
  harmful for transfer (CI [0, +0.094] is directionally negative for
  original, i.e. ctx_full < original).
- **Retaining vs discarding memory does not change final performance.**
  `memory_wiped` (train, then wipe) is bit-identical to `original`
  (0.594). Because evaluation episodes themselves let the agent learn
  (within-eval learning), the persistent memory does not determine
  held-out success; the task is largely solved from scratch inside each
  evaluation.
- **Retention is flat for every variant** (0.000 delta on the shared
  substrate) — no measured forgetting in either direction.
- **Conclusion (honest):** v0.4 context-aware evidence marginally
  reduces negative transfer (≈ −7 pp vs −10 pp) but does not beat the
  untrained control; the dominant mechanism appears to be within-eval
  learning, and none of the interventions yet convert accumulated
  experience into held-out gains.

## 7. Files

| Path | Role |
|---|---|
| `academy/agents/memory.py` | v0.4 context-aware evidence (3 flags) |
| `academy/agents/metargon_agent.py` | context signature derivation; wipe hook |
| `academy/tools/memory_experiment_v04.py` | continual-learning benchmark |
| `academy/tools/memory_merge_v04.py` | JSONL → summary + paired CIs |
| `academy/tests/test_memory_v04.py` | 8 tests (context mechanisms + compatibility) |
| `results/academy_v01/memory_v04/` | summary, jsonl, mid-curves |
| `docs/MEMORY_V04.md` | this report |

---

## 8. How to run

```bash
# full benchmark (offline; pin hash seed)
PYTHONHASHSEED=0 python academy/tools/memory_experiment_v04.py

# per-variant subprocess runner (robust; resumes)
PYTHONHASHSEED=0 python x/run_chunks_v04.py

# merge JSONL into final summary + paired CIs
python academy/tools/memory_merge_v04.py

# tests
python -m unittest academy.tests.test_memory_v04
```
