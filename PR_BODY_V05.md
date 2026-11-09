## Academy Memory v0.5 — Measurable Transfer

### Summary

v0.4 showed `memory_wiped ≡ original`: within-evaluation learning
obscures memory effects. v0.5 introduces an **evaluation-safe
3-regime protocol** and a compact **evidence-bank** intervention.

### Protocol (new)

| Regime | Learning during eval | Measures |
|---|---|---|
| Frozen (zero-shot) | disabled (`frozen_policy`) | retained-memory transfer alone |
| Adapt (within-episode) | enabled | v0.4-style result |
| Continual | post-training on eval configs | retention of new knowledge |

### Evidence bank (new, bounded, interpretable, off by default)

Verified hypotheses are banked as `(antecedent, consequence)` pairs
(FIFO cap); bank entries become high-priority probe candidates in new
tasks. Retrievals counted.

### Controls & ablations

fresh, original, memory_wiped, ctx, ctx_full, bank, bank_ctx,
bank_ctx_full — 2 seeds × 160 episodes matched streams, frozen grids,
paired CIs.

<!-- RESULTS -->

### Measured (2 seeds × 160 episodes, frozen grids, pinned hash)

| Variant | Frozen (zero-shot) | Adapt (within-eval) | Train |
|---|---|---|---|
| fresh | **0.312** | **0.695** | — |
| original | 0.281 | 0.594 | 0.341 |
| memory_wiped | 0.281 | 0.594 | 0.341 |
| ctx | 0.281 | 0.625 | 0.369 |
| ctx_full | 0.250 | 0.562 | 0.497 |
| bank | 0.281 | 0.594 | 0.341 |
| bank_ctx | 0.281 | 0.625 | 0.369 |
| bank_ctx_full | 0.273 | 0.633 | 0.231 |

### Honest conclusions

- **Zero-shot transfer is absent**: every trained variant scores below
  fresh on the frozen grid (0.250–0.281 vs 0.312).
- **memory_wiped ≡ original** (bit-identical, both regimes): deleting
  memory does not change held-out performance — persistent memory does
  not determine generalization.
- **All apparent transfer is within-episode adaptation** (adapt −
  frozen ≈ 0.31–0.38 everywhere).
- **Evidence bank does not help**: `bank`/`bank_ctx` ≡
  `original`/`ctx`; `bank_ctx_full` ran 79k retrievals yet adapt is
  directionally negative.
- **No variant beats the untrained control.** Positive transfer was
  NOT achieved — reported honestly.

### Expected outcome / honest reporting

Report whether *frozen* (zero-shot) transfer improves with retained
memory, quantify the adaptation gap (adapt − frozen), and state
plainly whether any variant beats the untrained control.

### Deliverables

- `academy/agents/metargon_agent.py`, `memory.py`
- `academy/tools/memory_experiment_v05.py`, `memory_merge_v05.py`
- `academy/tests/test_memory_v05.py` (12 tests)
- `docs/MEMORY_V05.md`; `results/academy_v01/memory_v05/`
