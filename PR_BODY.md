## Academy Memory v0.3 — Hypothesis-Memory Saturation & Negative Transfer

### Özet / Summary

The v0.2 curriculum research identified the student's hypothesis-memory
saturation (66–73 contradicted hypotheses, ~0 verified, no evictions)
as the binding constraint on transfer. This PR audits the memory,
separates failure modes, and adds **minimal, independently ablatable**
improvements — all off by default, preserving v0.2 behavior exactly.

### Diagnosis (docs + `academy/tools/memory_diagnosis.py`)

- True rules ARE generated early (D1@ep0, D2@ep1) and reach hundreds of
  supports → generation is not the bottleneck.
- Every hypothesis eventually flips to CONTRADICTED via cross-task
  nonevents (same public antecedent, different hidden rule).
- Memory plateaus at ~70 (< the 128 cap) — eviction is irrelevant.
- Probing under-targets conjunctive rules (4/457 attempts).

### Improvements (each independent, off by default)

| Flag | Effect |
|---|---|
| `recycle_contradicted` | recycle contradicted hyps + bounded dead-antecedent set |
| `contradiction_ratio` | gate contradiction transitions by support ratio |
| `evidence_window` | recompute status counts over recent evidence only |
| `probe_adaptive` | composite/priority probe candidates |
| `probe_episode_local` / `protect_episode_local` | gate decisions to current-episode evidence |

Checkpoint compatibility, 128-cap, determinism, and
public-observation-only learning are preserved.

### Results (matched 160-episode streams, pinned hash seed, frozen grids)

| Variant | Train | Classic (16 cfg) | OOD |
|---|---|---|---|
| **fresh (control)** | — | **0.695** | **0.417** |
| original | 0.362 | 0.562 | 0.333 |
| recycle | 0.362 | 0.562 | 0.333 |
| ratio | 0.394 | 0.344 | 0.417 |
| window | 0.381 | 0.312 | 0.306 |
| probe | 0.506 | 0.562 | 0.333 |
| protect_local | 0.369 | 0.562 | 0.333 |
| probe_local | 0.469 | 0.562 | 0.333 |
| combined | 0.188 | 0.375 | 0.417 |

- **probe / probe_local**: training +0.144 / +0.106 with classic
  transfer bit-identical (+0.0000, zero-width CI) — positive
  training-side result, no transfer lift.
- **Fresh-agent control**: untrained student scores 0.695 classic vs
  0.562 for all trained variants — training costs ~0.13 on unseen
  configs (negative transfer), so improvements on this pool are a
  *recovery* of the fresh baseline.
- **ratio / window**: retain 57 verified hyps but classic numerically
  lower (0.344/0.312 vs 0.562); paired CIs include 0 — directionally
  negative.
- **combined** is strictly worse (train −0.175).
- No variant lifts held-out transfer above original on this grid —
  negative result recorded honestly.

### Tests & docs

- `academy/tests/test_memory_v03.py` (14 tests) — full suite: **187 OK**
- `docs/MEMORY_V03.md` — full report
- Tools: `memory_diagnosis.py`, `memory_experiment.py`,
  `memory_merge.py`, `run_chunks.py`
- Artifacts: `results/academy_v01/memory_v03/` (summary, jsonl, policies)
