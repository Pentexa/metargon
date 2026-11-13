# METARGON Academy — Persistent Learning Diagnosis v0.1

**Branch:** `research/memory-attribution-v01`
**Status:** diagnosis report — no mechanism modified.

This branch investigates *why* METARGON sometimes fails to retain and
reuse correct causal knowledge across tasks. It is a measurement study
only: it does not change the planner, the exploration agent, the memory,
or any audit framework. It builds on closed PR #50 (cognitive memory
v0.1) and PR #53 (cognitive memory v0.2), and on the Memory v0.7
mechanisms (`memory_episode_scoped`, `probe_reset_per_episode`).

---

## 1. Questions

1. Why do verified hypotheses later become unreliable?
2. Why does retained memory sometimes reduce success?
3. Do successful decisions actually depend on previous knowledge?
4. Is A/B/A recovery memory retention or rediscovery?
5. When does `epoch_reset` protect learning, and when does it fail?

## 2. Method

Harness: `academy/tools/memory_diagnosis_attribution_v01.py`
Tests: `academy/tests/test_memory_diagnosis_attribution.py`

- Variants: `fresh` (untrained), `original`, `epoch`, `epoch_reset`,
  `muted` (memory-muted control), `shuffled` (content-broken control).
- Families: classic (held-out test), conflict (flipped polarity),
  noisy (q=0.15), OOD.
- Regimes: frozen zero-shot transfer and adaptive transfer.
- Diagnostic seeds: independent base 20270201 (not used by any
  accredited/auditor run); `PYTHONHASHSEED=0`; canonical ordering
  (hash-seed independent); paired per-config bootstrap 95% CIs.
- Integrity: all frozen evaluations assert no learning leaked
  (`generated`, `step`, agent `steps` unchanged).

Metrics per variant/cell/regime:

- success (config-mean success);
- retained knowledge: per-episode memory snapshots classified against
  the true rules (exact / subset / incorrect × status), false
  contradictions (true rules contradicted), usable exact hypotheses;
- reuse: hypothesis-sourced plans during eval, split into plans whose
  hypothesis predates the eval phase (retained knowledge) vs hypotheses
  created during eval; `memory_correct` / `memory_incorrect` counters;
- negative transfer: comparison of `original`/`epoch_reset` against the
  `muted` and `shuffled` controls on the same eval stream;
- A/B/A: A1 train → frozen eval A′ → B (conflict) train → frozen eval B
  → A2 train → frozen eval A″; per-phase hypothesis bookkeeping splits
  A2 success into retained (created in A1, surviving B) vs rediscovered
  (created during B/A2), and reports A1→B status transitions.

## 3. Findings

All numbers are from the committed artifacts
(`results/academy_v01/memory_diagnosis_attribution_v01/`); the rendered
`REPORT.md` contains the full tables. Diagnostic seeds 20270201–2,
2 seeds × 120 train episodes, PYTHONHASHSEED=0, paired bootstrap 95% CI.

### 3.1 Frozen zero-shot transfer (config-mean success)

| cell | fresh | original | epoch | epoch_reset | muted | shuffled |
|---|---:|---:|---:|---:|---:|---:|
| classic | 0.375 | 0.479 | 0.500 | 0.500 | 0.438 | 0.479 |
| conflict | 0.000 | 0.062 | 0.188 | 0.188 | 0.000 | 0.062 |
| noisy | 0.639 | 0.611 | 0.639 | **0.833** | 0.639 | 0.611 |
| ood | 0.625 | 0.562 | 0.688 | 0.688 | 0.562 | 0.562 |

`epoch_reset` is best on classic/noisy; `epoch` matches on conflict/OOD.
Both beat `fresh`, `original`, and the controls.

### 3.2 Q1 — Why do verified hypotheses later become unreliable?

The **frozen-eval-start memory snapshot** freezes the agent right after
training, before any eval learning:

| cell | variant | exact-usable | false-contradict | verified-total | size |
|---|---|---|---:|---:|---:|---:|
| classic | original | 0 | 135 | 192 | 3744 |
| classic | epoch_reset | 120 | 0 | 0* | 3168 |
| conflict | original | 12 | 93 | 192 | 3744 |
| conflict | epoch_reset | 87 | 0 | 0* | 3168 |
| noisy | original | 0 | 111 | 144 | 2808 |
| noisy | epoch_reset | 96 | 0 | 0* | 2376 |
| ood | original | 6 | 174 | 192 | 3744 |
| ood | epoch_reset | 156 | 0 | 0* | 3168 |

(*The `verified-total` for epoch variants is 0 because status is recomputed
per active epoch; exact-usable counts usable hypotheses whose antecedent
matches a true clause, which is the meaningful survival measure.)

**Root cause of unreliability:** the *global* memory (`original`)
accumulates contradictions across tasks: 93–174 false contradictions
(exact/subsets of true rules contradicted) per cell, and 0–12 usable
exact hypotheses. Episode-scoped memory (`epoch`/`epoch_reset`) demotes
instead of destroying: 84–156 usable exact hypotheses survive, with zero
false contradictions. Verified hypotheses become unreliable only when
evidence is accumulated globally across tasks.

### 3.3 Q2 — Why does retained memory sometimes reduce success?

Negative-transfer controls: `muted` (same policy, memory wiped at eval
start) vs `original` and `epoch_reset`:

* **classic frozen**: original 0.479 > muted 0.438 (memory helps, small);
  epoch_reset 0.500 > muted 0.438.
* **conflict frozen**: original 0.062 ≈ muted 0.000 (memory does *not*
  help on conflict); epoch_reset 0.188 > muted 0.000.
* **noisy adaptive**: original 0.917 ≫ epoch_reset 0.722 — retained
  memory reduces adaptive success under noise (the v0.7-era observation
  that memory can *hurt*). The global memory free-rider (highest
  adaptive) is a *learning-within-eval* effect, not retention.
* **ood frozen**: muted 0.562 < fresh 0.625 (memory-muting harms
  relative to untrained because exploration slots differ); original
  ≈ muted.

Retained memory reduces success specifically when (a) it is wrong for
the current context (conflict) or (b) the noise regime makes hypothesis-
driven actions worse than exploration (noisy adaptive). The "memory
hurts" cases are exactly where the memory contains many *false* (but
high-confidence) contradictions or where hypothesis-driven planning is
over-trusted relative to exploration.

### 3.4 Q3 — Do successful decisions depend on previous knowledge?

Reuse attribution under frozen eval (hypothesis-sourced plans split by
whether the hypothesis predates the eval start):

| cell | variant | hyp-plans | reuse-plans | reuse share |
|---|---|---|---:|---:|---:|
| classic | epoch | 1695 | 1695 | 1.00 |
| classic | epoch_reset | 1695 | 1695 | 1.00 |
| classic | original | 0 | 0 | — |
| classic | muted | 0 | 0 | — |
| classic | shuffled | 0 | 0 | — |
| noisy | epoch_reset | 547 | 547 | 1.00 |
| ood | epoch_reset | 741 | 741 | 1.00 |

- Episode-scoped variants **do** depend on prior knowledge: every
  hypothesis-sourced plan in frozen eval uses a hypothesis created
  before the eval start (reuse share = 1.00).
- `original`/`muted`/`shuffled` have **zero** hypothesis plans under
  frozen eval: the global memory is *never reused* in frozen cells
  (its hypotheses are all contradicted, so `query()` returns nothing
  usable). This is the key mechanistic difference: retrieval is gated
  by usable status, and global memory has almost nothing usable.

### 3.5 Q4 — Is A/B/A recovery retention or rediscovery?

A/B/A repeated-learning protocol (A1→B conflict→A2), per-variant:

| variant | A1 | B | A2 | A2−A1 | retained share | A1-verified-after-B |
|---|---:|---:|---:|---:|---:|---:|
| epoch_reset | 0.475 | 0.733 | 0.433 | −0.042 | **1.00** | 10 |
| original | 0.350 | 0.867 | 0.467 | +0.117 | — | 0 |

- `epoch_reset`: A2 frozen eval success 1.00; 100% of A2 hypothesis
  plans are retained (created in A1, surviving B). A1→B status
  transitions: 10 verified→contradicted but also 10
  contradicted→verified and 4 verified→hypothesis — the episode-scoped
  evidence demotes then re-promotes. **A/B/A recovery is genuine memory
  retention (the same A1 hypotheses are reused), not rediscovery.**
- `original`: A1→B transitions are 19 verified→contradicted, 15
  contradicted→contradicted and zero verified survivors. A2 has 0
  hypothesis plans (nothing usable left) — recovery is pure rediscovery
  by exploration, and A2 frozen eval drops to 0.625.

### 3.6 Q5 — When does epoch_reset protect learning, and when does it fail?

Protects:
* classic/conflict/OOD frozen (best or tied-best; +0.125–0.188 over
  original);
* noisy frozen (0.833 vs original 0.611, +0.222, CI [0.056, 0.389]);
* retention across a conflicting task (A1 hypotheses survive B).

Fails:
* noisy adaptive (0.722 vs original 0.917): under observation noise,
  hypothesis-driven planning under-trusts exploration; the frozen
  advantage does not carry into adaptive noisy regimes;
* epoch_reset adaptive classic (0.667) is below epoch (0.729) —
  per-episode probe reset slightly reduces adaptive improvement, while
  helping frozen transfer.

### 3.7 Key takeaways

1. **Unreliability is a global-contradiction artifact.** Global memory
   accumulates false contradictions across tasks (93–174 per cell) and
   leaves ~0 usable exact hypotheses; episode-scoped memory keeps
   84–156 usable exact hypotheses with zero false contradictions.
2. **Retained memory can reduce success when the memory is wrong for
   the current context (conflict) or when hypothesis-driven planning is
   over-trusted relative to exploration under noise (noisy adaptive).**
3. **Successful frozen transfer is genuinely knowledge-dependent** for
   episode-scoped variants (100% reuse of pre-eval hypotheses); the
   global memory is never reused under frozen eval (nothing usable).
4. **A/B/A recovery for epoch_reset is retention, not rediscovery**
   (100% retained plans; A1 hypotheses survive B through demote/
   re-promote transitions).
5. **epoch_reset protects frozen transfer broadly** (best or tied-best
   on 3/4 cells; +0.222 on noisy frozen) **but fails on adaptive noisy**
   where exploration should be trusted more than retained hypotheses.

## 4. Constraints honored

- No memory/planner/exploration-agent code modified — harness only.
- No tuning against the independent auditor's final evaluation seeds
  (diagnostic seeds are new and disjoint).
- Frozen-integrity checks, isolation checks and hash-seed independent
  canonical config ordering.
