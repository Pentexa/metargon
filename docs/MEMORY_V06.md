# Academy Memory v0.6 — Genuine Knowledge Transfer

**Branch:** `academy/memory-v06`
**Base:** latest `origin/main`
**Status:** implementation complete; measured results below.
**Date:** 2026-11-09

---

## 1. Summary

v0.5 produced a critical negative result: fresh frozen 0.312 vs trained
frozen 0.281, `memory_wiped ≡ original` bit-identical, and the evidence
bank showed no transfer benefit. v0.6 (a) diagnoses why long-term memory
fails to improve future decisions, and (b) introduces minimal,
interpretable interventions that make retained knowledge change future
decisions in a measured, reproducible way.

**Headline findings — confirmed (final evidence)**

1. **Storage failure: cross-task contradictions destroy verified
   hypotheses.** After 160 training episodes the hypothesis memory is
   almost entirely contradicted (~75 of ~77 hypotheses; ≈2.5 verified
   survive on average). `verified_then_contradicted` transitions run
   56–198 per training stream. Persistent memory therefore provides
   little usable evidence for transfer — retrieval mechanics work, but
   there is almost nothing usable to retrieve.
2. **`memory_wiped ≡ original` (bit-identical) and every `tried`-set
   reset variant ≡ `original`.** The `tried` probe set is cleared by
   the agent's own door events during training and is essentially empty
   at checkpoint time; per-episode probe bookkeeping changes nothing.
   **The early (bookkeeping) hypothesis is REJECTED** — it is not the
   primary cause of the memory failure.
3. **Verbatim evidence-bank reuse transfers irrelevantly.** In frozen
   evaluation 91% of bank-sourced probes are irrelevant to the true
   held-out rule (exact 20/7900), and `bank_ctx_full` frozen 0.219 <
   fresh 0.312.
4. **Context gating blocks transfer.** `ctx_full` uses
   `context_gate_transfer`; in a new evaluation context `query()`
   returns nothing, so the adaptive candidate list is empty and the
   agent behaves like the goal-planning-only `novocab` control (0.508
   vs 0.461) — i.e. **goal-directed planning without vocabulary
   probing, not pure random walking**. Its frozen success increase over
   fresh is an exploration artifact, not retained-memory transfer.
5. **The probe vocabulary itself underperforms planning.** The
   `novocab` control (planning intact, vocabulary probing disabled)
   reaches 0.461 frozen vs fresh 0.312; pure `random_walk` (no
   planning) is 0.055. Planning, not memory, carries the frozen score.
6. **Evaluation integrity fixed.** v0.5's comparison-sign defect is
   corrected (all deltas now use candidate − baseline), evaluation
   cells recreate the full variant configuration (v0.5 dropped
   agent-level flags in eval cells), and frozen cells assert that no
   learning or memory update occurs.

> **Honest claim:** No trained variant demonstrates positive zero-shot
> transfer. Every memory-bearing trained frozen score (0.219–0.281;
> `ctx_full` 0.508 ≈ `novocab` exploration artifact) is at or below
> fresh 0.312, with paired deltas vs fresh ≤ 0 (CIs including 0 or
> negative). Retained knowledge does not produce a measurable zero-shot
> improvement. The primary deliverable is the diagnosis (§2): storage
> failure via cross-task contradiction, verbatim-reuse irrelevance, and
> the probe-vocabulary/planning gap (§3.1, §4.4).

---

## 2. Phase 1 — Diagnosis (measured)

The full chain `observation → hypothesis → verification → storage →
retrieval → planning → action` is instrumented per run (see
`academy/tools/memory_experiment_v06.py`; trace counters on the agent).

### 2.1 Are verified memories actually retrieved?

**Yes, mechanically — but there is almost nothing usable to retrieve.**
Frozen evaluation trace: `query_calls` happen every planning pass and
hypothesis candidates are generated in quantity (original: 6816), but
**zero plans are ever sourced from hypotheses** (`source_hypothesis =
0` across every variant). Why: after 160 training episodes, **~75 of
~77 hypotheses are contradicted** (only ~2.5 verified survive on
average), so `query()` has almost nothing usable to return. The failure
is at *storage survival*, not retrieval mechanics.

### 2.2 Does retrieved knowledge change selected actions?

When a usable hypothesis exists, it becomes a priority-0 probe candidate
and is executed (the mechanism runs — e.g. `bank_ctx_full` registers
79k bank retrievals). But because almost no hypotheses survive to
verification (2.1), the fraction of actions sourced from memory is
≈0 for original/ctx/bank/reset. The evidence-bank reuse *does* change
actions (all probes become bank-sourced) but **reduces** frozen success
(0.219 vs fresh 0.312) — see 2.3.

### 2.3 Is reused knowledge relevant to the new task?

Probe relevance is measured against the true hidden rule of each eval
config (exact clause match / partial atom overlap / irrelevant):

| Frozen eval probes | n | exact | partial | irrelevant |
|---|---|---|---|---|
| fresh (novel singles) | 6960 | 1352 | 1240 | 4368 |
| original (novel singles) | 7260 | 1352 | 1392 | 4516 |
| bank_ctx_full (bank entries) | 7900 | 20 | 664 | **7216** |

Verified bank entries are overwhelmingly **irrelevant** to held-out
tasks (91% of bank probes). Verbatim antecedent reuse transfers poorly:
rules differ across tasks even when they share atoms.

### 2.4 Do cross-task contradictions destroy useful knowledge?

**Yes — this is the central storage failure.** Training logs show
`verified_then_contradicted` 56–198 per run: hypotheses that reached
VERIFIED are later contradicted by nonevents in other tasks (the agent
holds the antecedent atoms while the door stays closed, which the
3-step nonevent window counts as contradiction). With 160 mixed-task
episodes, nearly the whole memory converges to CONTRADICTED. Context-
scoped evidence records the same counts per context but does not
prevent the *global-status* flip (ctx verified still 2.5).

### 2.5 Does the planning system ignore retained evidence?

No — the planner consumes every candidate whose antecedent is feasible
(`skipped_infeasible` is small). The `skipped_tried` counter is large
(≈6.8k) but this is mostly intra-episode re-scanning: the `tried` set
is cleared by the agent itself at every door event during training, so
at checkpoint time it is essentially empty and does **not** block
reuse. The "bookkeeping leak" hypothesis is therefore **rejected**
(`reset` ≡ `original` bit-identical in every measured regime).

### 2.6 Do context signatures prevent legitimate transfer?

**Yes — by design.** `ctx_full` uses `context_gate_transfer`: in a new
evaluation context no hypothesis has local support, so `query()`
returns nothing and the adaptive candidate list is empty. The agent
then behaves like the `novocab` control — **goal-directed planning
without vocabulary probing** (0.508 vs 0.461), *not* pure random
walking (which scores 0.055). Its frozen success above fresh is
therefore an exploration artifact, not retained-memory transfer. Note
also the context signature is the *start-state* signature, identical
for every config on a topology, so "context" as implemented is
effectively topology-scoped.

---

## 3. Phase 2 — Interventions

All are small, interpretable, bounded, and off by default:

| Mechanism | Flag | Why |
|---|---|---|
| Atom-level causal transfer | `memory_atom_transfer` | reuse recurring *parts* of verified causes (atoms survive contradiction better than whole hypotheses) |
| Event-level atom evidence | (part of `memory_atom_transfer`) | atoms present *before* door-opening events are directly supported — dense signal not dependent on hypothesis verification (which rarely happens) |
| Pair co-occurrence transfer | (part of `memory_atom_transfer`) | conjunctive structure: pairs that recur within verified hypotheses across tasks |
| Context-conditioned confidence gate | `probe_min_effective_confidence` | don't trust verbatim hypotheses in unseen contexts |
| Adaptive bank forgetting | `memory_bank_refresh_on_contradiction` | bank keeps only currently verified evidence |
| Per-episode probe bookkeeping | `probe_reset_per_episode` | sanity/exploration control (diagnosed as not the binding constraint) |

Variants: `reset` (bookkeeping only), `atomtx` (+ atom transfer +
confidence gate), `atomtx_bank` (+ bank refresh).

### 3.1 Necessary evaluation controls

Two non-memory controls are computed in `memory_merge_v06.py` (frozen,
same grid, no training):

- **random_walk** — pure random valid actions, no door/goal planning
  (0.055 frozen, pinned hash);
- **novocab** — the full agent with vocabulary probing disabled but
  door-pass and goal planning intact (0.461 frozen, pinned hash).

`novocab` exposes that the probe vocabulary itself *reduces* frozen
performance: the agent's goal-planning alone already solves the
initially-open configurations. Any transfer claim must therefore be
read against these exploration baselines, not only against fresh.
(`ctx_full` 0.508 ≈ `novocab` 0.461: context gating degenerates the
agent to the planning-only explorer — not to random walking.)

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
| ctx_full | 0.497 | 0.508 | 0.531 | 0.542 | 0.611 |
| bank | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| bank_ctx | 0.369 | 0.281 | 0.625 | 0.333 | 0.500 |
| bank_ctx_full | 0.231 | 0.219 | 0.219 | 0.250 | 0.333 |
| reset | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| atomtx | 0.209 | 0.250 | 0.617 | 0.333 | 0.500 |
| atomtx_bank | 0.231 | 0.250 | 0.617 | 0.333 | 0.500 |

Controls (computed in `memory_merge_v06.py`, frozen, same grid):

| Control | Frozen |
|---|---|
| random_walk (pure random, no planning) | 0.055 |
| novocab (goal/door planning, no vocabulary probing) | 0.461 |

### 4.2 Paired deltas vs fresh (candidate − fresh; positive = better)

| Variant | Δ frozen | 95% CI |
|---|---|---|
| original | −0.030 | [−0.094, 0.000] |
| memory_wiped | −0.030 | [−0.094, 0.000] |
| ctx | −0.030 | [−0.094, 0.000] |
| ctx_full | +0.198 | [+0.039, +0.359] |
| bank | −0.030 | [−0.094, 0.000] |
| bank_ctx | −0.030 | [−0.094, 0.000] |
| bank_ctx_full | −0.090 | [−0.281, +0.063] |
| reset | −0.030 | [−0.094, 0.000] |
| atomtx | −0.061 | [−0.188, 0.000] |
| atomtx_bank | −0.061 | [−0.188, 0.000] |

### 4.3 Paired deltas vs original (candidate − original)

| Variant | Δ frozen | 95% CI | Δ adapt | 95% CI |
|---|---|---|---|---|
| fresh | +0.030 | [0, +0.094] | +0.100 | [0, +0.211] |
| ctx_full | +0.229 | [+0.086, +0.367] | −0.064 | [−0.227, +0.094] |
| bank_ctx_full | −0.059 | [−0.219, +0.063] | −0.376 | [−0.625, −0.125] |
| atomtx | −0.030 | [−0.094, 0] | +0.023 | [0, +0.070] |
| atomtx_bank | −0.030 | [−0.094, 0] | +0.023 | [0, +0.070] |
| memory_wiped / bank / bank_ctx / reset | 0.000 | [0, 0] | 0.000 | [0, 0] |

### 4.4 Interpretation

- **Zero-shot transfer is absent for every memory-bearing variant.**
  fresh 0.312 is the upper envelope; all trained variants are at or
  below it. `memory_wiped ≡ original` remains bit-identical.
- **The probe vocabulary itself underperforms planning.** `novocab`
  (goal/door planning with vocabulary probing disabled) reaches 0.461
  frozen — above fresh (0.312). The agent's vocabulary probing
  *reduces* frozen performance; memory cannot compensate within the
  same architecture. Pure `random_walk` (0.055) shows that planning is
  what carries the frozen score, not memory.
- **`ctx_full` (0.508) ≈ `novocab` (0.461):** context gating empties
  the candidate list, so the agent effectively becomes the
  goal-planning-only explorer (not a random walker). Its superiority
  over fresh is an exploration artifact, not memory (its trace shows
  zero hypothesis/bank/atom-sourced plans).
- **Adaptation gap is unchanged** (adapt − frozen ≈ 0.31–0.37): all
  generalization remains within-episode learning.
- **Retention is flat** (+0.000 for all variants).
- **Phase 4 (EVREN):** teacher-mixed training nudges the vanilla
  `original` frozen transfer to 0.406 (paired Δ vs fresh +0.094, CI
  [0.0, +0.1875]) — directionally positive, not significant at 95%;
  dense atom-transfer does not capitalize (0.250). Teacher-holdout
  transfer is flat at 0.500 for fresh and trained conditions.

## 5. Phase 3 — Evaluation integrity

- Three regimes retained: frozen zero-shot, within-episode adaptation,
  continual retention.
- Mandatory controls: `fresh`, `original`, `memory_wiped`, and all
  v0.5 controls (`ctx`, `ctx_full`, `bank`, `bank_ctx`,
  `bank_ctx_full`).
- Evaluation cells are constructed with the variant's full kwargs and
  load the trained checkpoint; frozen cells additionally assert the
  memory state (generated count, step counter) does not move.
- **Comparison-sign defect corrected**: v0.5 reported
  baseline − candidate (it called `paired_difference(orig, candidate)`).
  Every v0.6 comparison is candidate − baseline; positive Δ means the
  candidate improved. The historical v0.5 raw evidence is preserved
  unchanged (`results/academy_v01/memory_v05/`); this correction is
  documented separately here and in `memory_merge_v06.py`. Under the
  corrected convention the v0.5 headline deltas become: fresh frozen
  **+0.031** (CI [0, +0.094]), fresh adapt **+0.100** (CI [0, +0.211]),
  ctx_full frozen **−0.030** (CI [−0.094, 0]), bank_ctx_full adapt
  **+0.037** (CI [−0.039, +0.133]) — no v0.5 variant beat fresh; the
  substantive conclusions are unchanged.
- Per-seed reporting, matched streams, `PYTHONHASHSEED=0`, paired
  bootstrap CIs vs fresh and vs original on per-config vectors.

---

## 6. Phase 4 — EVREN teacher experiences

The EVREN teacher workflow (proposal → strict validation → approved
registry) is used only to generate validated training tasks. This run
uses the deterministic `OfflineTaskGenerator` — the same validation
pipeline as the teacher — so it is reproducible with zero API calls.
Approved task artifact: `results/academy_v01/memory_v06/evren_pool.json`
(validated by `TaskValidator` — schema, leakage, solvability,
evaluation contamination, duplicates; 24 approved, all solvable,
config-unique, zero API calls).
Held-out teacher tasks measure transfer to novel validated tasks.
**All EVREN-benchmark runs must pin `PYTHONHASHSEED=0`** (the main
benchmark chunk runner does this; run the EVREN script under the pinned
hash so its controls are comparable).

Measured (pinned hash, 2 seeds × 160 episodes, 30% teacher-mixed
training, 16-config classic grid, frozen evaluation):

| Condition (teacher-mixed training) | Train | Frozen classic | Adapt classic | OOD frozen | Teacher holdout frozen |
|---|---|---|---|---|---|
| fresh (no training) | — | 0.312 | 0.695 | 0.333 | 0.500 |
| original | 0.375 | **0.406** | 0.625 | 0.333 | 0.500 |
| atomtx_bank | 0.191 | 0.250 | 0.617 | 0.333 | 0.500 |

Paired frozen deltas vs fresh (candidate − fresh): original **+0.094**
(CI [0.0, +0.1875]); atomtx_bank **−0.061** (CI [−0.1875, 0.0]).

Interpretation:

- Mixing validated teacher tasks into training improved the vanilla
  agent's frozen zero-shot transfer on the classic held-out grid
  (0.406 vs fresh 0.312; vs classic-only original 0.281). The paired
  CI lower bound is exactly 0.0, so this is directionally positive but
  **not significant at the 95% level**; it is reported honestly.
- The dense atom-transfer variant does not capitalize on the teacher
  mix (0.250); the teacher-mixed training disfavours it (train drops to
  0.19).
- Teacher holdout frozen is flat (0.500 for fresh and both trained
  conditions): the validated pool's tasks are not harder to solve by
  exploration, and retention adds nothing measurable there.
- EVREN contributes diverse validated training experiences; the effect
  on controlled transfer is directionally positive for the vanilla
  agent and statistically inconclusive. No positive-transfer claim is
  made.

---

## 7. Experiment manifest & evidence preservation (Quality v0.5)

Machine-readable manifests following **Quality v0.5**
(`generalization_audit/verification/manifest.py`; schema
`experiment-manifest-v01`) are prepared and verified for both the main
benchmark and the EVREN phase:

| Manifest | Format | Verifier result |
|---|---|---|
| `memory_v06.manifest.json` | `academy_memory` | 77 PASS / 0 FAIL / 1 INCONCLUSIVE (memory-limit not recorded) |
| `memory_v06_evren.manifest.json` | `generic` | 9 PASS / 0 FAIL / 4 INCONCLUSIVE (generic-format checks; memory-limit) |

Both manifests record: `PYTHONHASHSEED=0`, python 3.11.0, train seeds
[20261107, 20261108], 160 episodes per seed, 16-config classic grid
(4 episodes × 2 seeds per config), 12-config OOD probe, 6-config
continual probe, 8-config retention substrate, paired bootstrap
(seed 20261108, 2000 rounds, α=0.05), candidate − baseline convention,
per-variant kwargs, and the artifact hashes (canonical, content-based).

The raw artifacts (summary, runs JSONL, fresh control, EVREN
pool/summary) are **published in-repository** at
`results/academy_v01/memory_v06/`, covered by the manifests above at
`generalization_audit/manifests/`. The audit artifact-gate hotfix
(PR #39, `audit/artifact-gate-fix`) has merged, replacing the permanent
artifact prohibition with the provenance-aware publication gate; the
gate verifies PASS for both frozen targets with these manifests. Full
regeneration commands are in §8.

## 8. Run instructions

```bash
# unit tests
python -m unittest academy.tests.test_memory_v06
# quick development run
python academy/tools/memory_experiment_v06.py --quick
# full benchmark (chunked, resumable; pins PYTHONHASHSEED=0 per chunk)
python x/run_chunks_v06.py
# merge JSONL -> summary (candidate - baseline convention)
python academy/tools/memory_merge_v06.py
# EVREN teacher-experience benchmark (offline, zero API calls)
python academy/tools/memory_experiment_v06_evren.py
```

Direct runs should pin the hash seed: `PYTHONHASHSEED=0 python …`
(Windows PowerShell: `$env:PYTHONHASHSEED=0; python …`).

**Artifact gate:** the audit artifact-gate hotfix (PR #39) has merged:
the frozen criteria (v02) remain immutable, but the permanent artifact
prohibition is replaced by the provenance-aware publication gate. Raw
`results/academy_v01/memory_v06/` evidence is published with covering
manifests (`generalization_audit/manifests/memory_v06.manifest.json`,
`memory_v06_evren.manifest.json`); both verify with 0 FAIL and the gate
reports PASS for the frozen target (§7). Every number in this report
remains reproducible from the committed artifacts.

---

## 9. Files

| Path | Role |
|---|---|
| `academy/agents/memory.py` | atom-transfer profile, effective confidence, bank refresh |
| `academy/agents/metargon_agent.py` | per-episode reset, atom-transfer candidates, trace |
| `academy/tools/memory_experiment_v06.py` | 3-regime benchmark + diagnosis |
| `academy/tools/memory_merge_v06.py` | JSONL → summary (corrected sign) |
| `academy/tools/memory_experiment_v06_evren.py` | Phase-4 EVREN benchmark |
| `academy/tests/test_memory_v06.py` | 19 tests |
| `results/academy_v01/memory_v06/` | summary, runs, diagnosis, evren pool |
| `docs/MEMORY_V06.md` | this report |
