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

*Pinned run: `PYTHONHASHSEED=0`, 2 train seeds × 160 episodes, frozen
16-config classic grid, 12-config OOD, shared substrate retention.*

### 4.1 Held-out numbers (mean over 2 seeds)

| Variant | Train | Frozen classic | Adapt classic | OOD frozen | Continual OOD |
|---|---|---|---|---|---|
| fresh | — | **0.312** | **0.695** | 0.333 | 0.333 |
| original | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| memory_wiped | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| ctx | 0.369 | 0.281 | 0.625 | 0.333 | 0.500 |
| ctx_full | 0.497 | 0.250 | 0.562 | 0.333 | 0.500 |
| bank | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| bank_ctx | 0.369 | 0.281 | 0.625 | 0.333 | 0.500 |
| bank_ctx_full | 0.231 | 0.273 | 0.633 | 0.333 | 0.472 |

### 4.2 Paired deltas vs original (95% CI)

| Variant | Frozen Δ | Adapt Δ | Train Δ |
|---|---|---|---|
| fresh | −0.030 ([−0.094, 0.000]) | −0.100 ([−0.211, 0.000]) | −0.341 |
| memory_wiped | +0.000 ([0, 0]) | +0.000 ([0, 0]) | 0.000 |
| ctx | +0.000 ([0, 0]) | −0.030 ([−0.094, 0.000]) | +0.028 |
| ctx_full | +0.030 ([0, +0.094]) | +0.030 ([0, +0.094]) | +0.156 |
| bank | +0.000 ([0, 0]) | +0.000 ([0, 0]) | 0.000 |
| bank_ctx | +0.000 ([0, 0]) | −0.030 ([−0.094, 0.000]) | +0.028 |
| bank_ctx_full | +0.007 ([−0.070, +0.094]) | −0.037 ([−0.133, +0.039]) | −0.109 |

### 4.3 Memory tracking (mean)

| Variant | verified | contradicted | ctx-specific | bank size | retrievals |
|---|---|---|---|---|---|
| original | 2.5 | 75.0 | 0 | 0 | 0 |
| ctx | 2.5 | 76.5 | 79 | 0 | 0 |
| bank | 2.5 | 75.0 | 0 | 16 | 0 |
| bank_ctx_full | 6.0 | 51.0 | 57 | 16 | 79,345 |

---

## 5. Interpretation

- **Zero-shot (frozen) transfer is absent.** Every trained variant
  scores *below* the fresh control on the frozen grid (0.250–0.281 vs
  0.312). Retained memory alone transfers nothing — and remains mildly
  negative.
- **memory_wiped ≡ original, bit-identical in both regimes.** Deleting
  all learned memory does not change frozen or adaptive scores. The
  persistent memory does not determine held-out performance.
- **All apparent transfer is within-episode adaptation.** adapt −
  frozen ≈ 0.31–0.38 for every variant. The agent solves each unseen
  configuration from scratch inside the evaluation episode.
- **The evidence bank does not help.** `bank` / `bank_ctx` are
  bit-identical to `original` / `ctx`. `bank_ctx_full` performed 79k
  retrievals (the mechanism demonstrably runs) yet adaptation is
  directionally negative (−0.037) vs original.
- **No variant beats the untrained control** in either regime.
- **Retention is flat** (+0.000 for all variants); continual-OOD probe
  is ~0.5 for all trained variants.
- **Conclusion (honest):** v0.5's evaluation-safe protocol confirms
  that the Academy student's persistent memory currently produces no
  measurable zero-shot transfer; all measured generalization is
  within-evaluation learning. The evidence bank, context-scoped
  evidence, and their combinations do not convert retained experience
  into held-out gains. Positive transfer was not achieved.

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
