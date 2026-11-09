# Academy Memory v0.5 — Measurable Transfer

**Branch:** `academy/memory-v05`
**Base:** latest `origin/main`
**Status:** implementation complete; measured results below.
**Date:** 2026-11-09

---

## 1. Summary

v0.4 found `memory_wiped ≡ original` — within-evaluation learning
obscures the effect of persistent memory. v0.5 therefore **separates
transfer into three regimes** with an evaluation-safe protocol:

1. **Frozen (zero-shot)** — learning disabled during evaluation.
   Measures what retained memory alone transfers.
2. **Adapt (within-episode)** — learning enabled (v0.4 protocol).
3. **Continual (post-training)** — agent keeps learning on eval configs
   and is probed on disjoint subsets (retention of new knowledge).

New intervention: an **evidence bank** — a compact, bounded list of
verified (antecedent, consequence) entries that are used as
high-priority probe candidates in new tasks (useful evidence reuse,
public-observation-only).

---

## 2. Implementation

### 2.1 `frozen_policy` (evaluation safety)
`agent.frozen_policy = True` blocks `learn()` and
`learn_from_experience()` — evaluation runs purely on the retained
policy. Checkpointed; off by default.

### 2.2 Evidence bank (`memory_evidence_bank_cap`)
- Verified hypotheses are appended to a bounded bank (FIFO, cap
  configurable, default 0 = off).
- Entries are `(antecedent, consequence)` pairs — interpretable and
  compact — and are saved in checkpoints.
- In `_adaptive_probe_candidates`, bank entries for a door become
  priority-0 probe candidates (before singles, alongside hypothesis-
  backed probes); retrievals are counted (`bank_retrievals`).

### 2.3 Memory tracking
Per variant, the experiment records: useful (verified non-
contradicted), contradicted, context-specific (ctx counts > 0), bank
size, and retrievals.

---

## 3. Experiment

`academy/tools/memory_experiment_v05.py`, 2 train seeds × 160 episodes
(matched streams), frozen 16-config classic grid, 12-config OOD probe,
6-config continual probe, shared substrate retention. Variants:

- controls: `fresh`, `original`, `memory_wiped`, `ctx`, `ctx_full`
- bank ablations: `bank`, `bank_ctx`, `bank_ctx_full`

All offline; `PYTHONHASHSEED=0` pinned and recorded; paired bootstrap
CIs vs `original` on per-config vectors (frozen and adapt separately).

---

## 4. Results

<!-- filled from memory_merge_v05.py at commit time -->

### 4.1 Held-out numbers (mean over 2 seeds)

| Variant | Train | Frozen classic | Adapt classic | OOD frozen | Continual OOD |
|---|---|---|---|---|---|
| fresh | — | | | | |
| original | | | | | |
| memory_wiped | | | | | |
| ctx | | | | | |
| ctx_full | | | | | |
| bank | | | | | |
| bank_ctx | | | | | |
| bank_ctx_full | | | | | |

### 4.2 Paired deltas vs original (95% CI)

| Variant | Frozen Δ | Adapt Δ | Train Δ |
|---|---|---|---|
| ... | | | |

### 4.3 Memory tracking (mean)

| Variant | verified | contradicted | ctx-specific | bank size | retrievals |
|---|---|---|---|---|---|
| ... | | | | | |

---

## 5. Interpretation

<!-- filled after final run -->

---

## 6. Files

| Path | Role |
|---|---|
| `academy/agents/metargon_agent.py` | `frozen_policy`; evidence-bank probing |
| `academy/agents/memory.py` | evidence bank (bounded, checkpointed) |
| `academy/tools/memory_experiment_v05.py` | 3-regime benchmark |
| `academy/tools/memory_merge_v05.py` | JSONL → summary + paired CIs |
| `academy/tests/test_memory_v05.py` | 12 tests |
| `results/academy_v01/memory_v05/` | summary, jsonl |
| `docs/MEMORY_V05.md` | this report |

---

## 7. How to run

```bash
PYTHONHASHSEED=0 python academy/tools/memory_experiment_v05.py
PYTHONHASHSEED=0 python x/run_chunks_v05.py
python academy/tools/memory_merge_v05.py
python -m unittest academy.tests.test_memory_v05
```
