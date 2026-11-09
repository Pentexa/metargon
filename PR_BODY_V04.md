## Academy Memory v0.4 — Context-Aware Evidence and Continual Learning

### Summary

v0.3 showed training *degrades* held-out performance (fresh 0.695 vs
trained 0.562) with ~70 contradicted hypotheses. This PR implements
**context-aware evidence** so that correct hypotheses survive
contradictions observed in *other task contexts*, task switches do not
count as evidence, and new contexts borrow no unproven knowledge.

All mechanisms are independent, off by default, and preserve v0.3
behavior exactly when disabled (regression-tested).

### Mechanisms (public-observation-derived context signatures)

| Flag | Effect |
|---|---|
| `context_scoped` | per-context support/contradiction counts; reversible confidence (status from active context only) |
| `task_change_reset` | reset consecutive-nonevent windows on context change |
| `context_gate_transfer` | conservative gate: query only returns hypotheses with support in the active context |

### Benchmark

Multi-seed (2 × 160 episodes), matched streams, mid-training frozen
grids (learning curves), final 16-config classic grid + OOD + shared
substrate retention, `fresh` and `memory_wiped` controls, paired CI
comparisons vs `original`.

<!-- RESULTS_TABLE -->

### Measured (2 seeds × 160 episodes, frozen grids, pinned hash)

| Variant | Train | Classic | OOD |
|---|---|---|---|
| fresh | — | **0.695** | 0.417 |
| original | 0.341 | 0.594 | 0.333 |
| probe_local | 0.422 | 0.625 | 0.375 |
| ctx | 0.369 | 0.625 | 0.417 |
| ctx_reset | 0.369 | 0.625 | 0.417 |
| ctx_gate | 0.366 | 0.625 | 0.417 |
| ctx_full | 0.497 | 0.562 | 0.417 |
| memory_wiped | 0.341 | 0.594 | 0.333 |

Honest conclusions:

- Negative transfer **decreased** (−7 pp vs −10 pp) but **no variant
  beats the untrained control**.
- `probe_local` and `ctx` share the best trained classic (0.625),
  directionally positive vs original (CI upper bound touches 0).
- `ctx_full` trains best (0.497) but transfers worst among trained
  (0.562) — combining mechanisms is harmful.
- `memory_wiped` ≡ `original`: discarding memory does **not** change
  final performance (within-eval learning dominates).

### Deliverables

- Implementation: `academy/agents/memory.py`, `metargon_agent.py`
- Benchmark: `academy/tools/memory_experiment_v04.py`, `memory_merge_v04.py`
- Tests: `academy/tests/test_memory_v04.py` (10 tests)
- Report: `docs/MEMORY_V04.md`
- Artifacts: `results/academy_v01/memory_v04/`
