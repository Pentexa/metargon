# Academy Memory v0.3 — Hypothesis-Memory Saturation and Negative Transfer

**Branch:** `academy/memory-v03`
**Base:** latest `origin/main`
**Status:** implementation complete; measured results below.
**Date:** 2026-11-08

---

## 1. Summary

The v0.2 curriculum research (docs/EVREN_CURRICULUM_V02.md) identified
the student agent's **hypothesis-memory saturation** as the binding
constraint on transfer: trained students carry 66–73 contradicted
hypotheses of 128 slots and almost zero verified hypotheses, and no
curriculum reliably changes that. This work audits the memory, separates
the failure modes, implements minimal ablatable improvements, and
measures them under matched budgets with paired confidence intervals.

**Headline measured result:** adaptive probing with episode-local
decision gates (`probe_local`) raises training-phase success
(+0.175 over original at 160 episodes) while preserving held-out
transfer; the attribution changes (ratio gating, evidence windowing)
retain more verified hypotheses but **degrade** transfer and are
reported as negative results.

---

## 2. Audit and failure-mode separation

`academy/tools/memory_diagnosis.py` trains the original student on a
fixed 80-episode stream and records, per door: whether the true rule
clause is ever generated, its peak support, its status trajectory, the
eviction history, and probe-target statistics.

Measured facts (seed 20261107, 80 episodes):

| Metric | D1 | D2 |
|---|---|---|
| True clause ever generated | yes (ep 0) | yes (ep 1) |
| Peak support of true clause | 389 | 307 |
| Final memory status | 70/70 contradicted | — |
| Evictions | 0 (cap 128 never reached) | — |
| Probe attempts targeting true antecedent | 4 / 457 | 3 / 99 |

Conclusions:

1. **Generation is not the bottleneck.** True clauses enter the
   hypothesis pool early and accumulate hundreds of supports.
2. **Attribution flips them.** Eventually every hypothesis — including
   heavily supported true clauses — becomes CONTRADICTED. The cause is
   cross-task interference: the same public antecedent can hold in many
   tasks, but the hidden door rule differs per task; a nonevent in a
   different task contradicts a hypothesis that was correct in its own
   task. With `max_contradict=2` and a 3-step nonevent window,
   coincidental non-openings kill hypotheses regardless of accumulated
   support.
3. **Eviction is irrelevant here.** Memory plateaus at ~70 (below the
   128 cap); no eviction ever fires. The "saturation" is status
   flipping, not capacity pressure.
4. **Probing under-targets.** Conjunctive true rules are rarely hit by
   the single-atom probe vocabulary (4/457 attempts).

---

## 3. Improvements (each independent, all off by default)

### 3.1 Contradiction-aware recycling (`recycle_contradicted`)
When a hypothesis crosses the contradiction threshold it is removed
immediately and its antecedent is recorded in a bounded per-door
"dead" set (FIFO, cap 96) so it is not regenerated next event. Frees
slots and stops death-churn; the dead set bounds regeneration.

### 3.2 Ratio-gated contradictions (`contradiction_ratio=r`)
A hypothesis may only *transition* to contradicted when its windowed
contradiction count satisfies `con >= r * sup` (r=0 disables, i.e.
original behavior). Protects heavily-supported hypotheses from
coincidental non-openings in other tasks.

### 3.3 Evidence windowing (`evidence_window=w`)
Support/contradict counts used for status are recomputed over the last
`w` evidence events (sliding recency) instead of being lifetime
cumulative. Stale evidence stops driving status.

### 3.4 Adaptive probing (`probe_adaptive`)
Candidate probes are built from: hypothesis-backed antecedents (as
before), conjunctions of the most-supported atoms ("composite
targets"), then singles ordered by support-weighted popularity. Same
per-episode planning budget; deterministic.

### 3.5 Episode-local gates (`probe_episode_local`, `protect_episode_local`)
At decision time, hypotheses evidenced only in earlier episodes no
longer steer probing priority or door-protection (they remain in memory
for statistics). This prevents stale truths from distorting the current
task's behavior.

All features preserve: the 128-hypothesis cap, checkpoint JSON
compatibility (defaults on load), determinism (seeded), and
public-observation-only learning. v0.2 behavior is bit-identical when
all flags are off (regression suite).

---

## 4. Experiment

`academy/tools/memory_experiment.py` trains each variant on the same
160-episode stream (seed 20261107), then evaluates on frozen pools with
independent per-config grids:

- classic test: 16 configs × 2 seeds × 4 episodes;
- OOD probe: 12 configs × 1 seed × 3 episodes;
- retention suite: 8-config shared substrate, probed twice;
- fresh-agent control (no training) for reference.

Paired bootstrap CIs compare each variant vs `original`. Offline by
construction; `PYTHONHASHSEED=0` pinned and recorded.

### 4.1 Results

| Variant | Train | Classic | OOD | Memory status (size) |
|---|---|---|---|---|
| original | 0.375 | 0.667 | 0.375 | 70 contradicted |
| recycle | 0.375 | 0.667 | 0.375 | 0 (recycled) |
| ratio (r=2) | 0.312 | 0.250 | 0.375 | 47 verified / 24 contradicted |
| window (w=24) | 0.312 | 0.250 | 0.375 | 71 verified |
| probe | 0.400 | 0.667 | 0.458 | 83 contradicted / 2 verified |
| protect_local | 0.388 | 0.667 | 0.375 | 70 contradicted |
| probe_local | **0.550** | 0.667 | 0.375 | 75 contradicted / 1 verified |
| combined | 0.188 | 0.333 | 0.375 | 30 verified / 15 predicted |

Paired deltas vs original (bootstrap 95% CI):

- `probe_local` classic: +0.0000 (CI [0,0]) — identical vector, no loss;
  train +0.175 (statistically above original large-margin in matched run).
- `ratio`/`window` classic: −0.4178 (CI [−0.667, −0.167]) — **significant
  degradation**.
- `combined` classic: −0.3393 (CI [−0.667, 0]).
- All other conditions: ±0.0000 on the classic grid.

---

## 5. Interpretation (measured, unexaggerated)

- **Keeping more verified hypotheses does not help transfer — it hurts.**
  Ratio gating and windowing protect stale truths, and those stale
  verified hypotheses then steer probing priority and door-protection,
  inflating protection rejections 4–5× (instrumented: 1657 → 7297)
  and degrading both training and held-out success. The memory's
  function is not a static knowledge base; it is a task-relative
  evidence system, and "knowledge" that is not tied to the current task
  actively misleads.
- **Adaptive probing changes training dynamics without hurting transfer.**
  `probe_local` improves training success (0.550 vs 0.375) and OOD
  parity (0.375; `probe` alone 0.458) with bit-identical classic
  transfer. The gain is sample-efficiency on the training distribution;
  it does not (yet) lift held-out generalization.
- **The classic-test grid is at a ceiling artifact?** 16-config grid
  yields identical success vectors for all conditions that do not lose
  transfer; paired CI width 0 is a consequence of identical per-config
  results, not hidden power. To detect small improvements on the
  classic pool we would need a more sensitive pool (larger n or
  harder configs) — noted as future work.
- **Negative results are recorded.** ratio, window, and combined
  variants are *worse*; recycle is neutral; probe-only is mildly
  positive on OOD only.

---

## 6. Files

| Path | Role |
|---|---|
| `academy/agents/memory.py` | v0.3 flags (recycle, ratio gate, window) |
| `academy/agents/metargon_agent.py` | adaptive probing + episode-local gates |
| `academy/tools/memory_diagnosis.py` | failure-mode separation harness |
| `academy/tools/memory_experiment.py` | matched-budget comparison |
| `academy/tests/test_memory_v03.py` | 14 tests |
| `results/academy_v01/memory_v03/` | summary + policy states |
| `docs/MEMORY_V03.md` | this report |

---

## 7. How to run

```bash
# diagnosis
python academy/tools/memory_diagnosis.py

# comparison (offline; pin hash seed for cross-process reproducibility)
PYTHONHASHSEED=0 python academy/tools/memory_experiment.py

# tests
python -m unittest academy.tests.test_memory_v03
```
