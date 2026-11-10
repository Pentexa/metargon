## Academy Memory v0.6 — Genuine Knowledge Transfer

### Summary

v0.5's negative result (fresh frozen 0.312 > all trained frozen
0.250–0.281; `memory_wiped ≡ original`; evidence bank no help) is
diagnosed and addressed:

- **Diagnosis (Phase 1)** — instrumented the full memory chain. Final
  evidence distinguishes rejected hypotheses from confirmed findings:
  - **REJECTED — probe-bookkeeping block.** The early hypothesis that
    the checkpointed `tried` set blocks reuse is not supported:
    `tried` is cleared by the agent's own door events during training
    and is essentially empty at checkpoint time; per-episode reset
    (`reset`) is bit-identical to `original` in every regime.
  - **CONFIRMED — cross-task contradictions destroy verified
    hypotheses.** ~75 of ~77 hypotheses are contradicted at checkpoint
    (56–198 verified→contradicted transitions per run); ~2.5 verified
    survive on average. Retrieval mechanics work, but persistent memory
    provides little usable evidence for transfer (`source_hypothesis`
    = 0 executed plans in frozen eval, despite thousands of generated
    candidates).
  - **CONFIRMED — verbatim bank reuse transfers irrelevantly** (91% of
    bank-sourced frozen probes irrelevant; exact 20/7900), and
    `bank_ctx_full` frozen 0.219 < fresh 0.312.
  - **CONFIRMED — context gating blocks transfer:** `ctx_full`
    degenerates to the goal-planning-only `novocab` explorer (0.508 ≈
    0.461), not to random walking (0.055); its frozen gain is an
    exploration artifact.
  - **CONFIRMED — the probe vocabulary itself underperforms planning**
    (`novocab` 0.461 > fresh 0.312).
- **Interventions (Phase 2):** minimal, interpretable, off by default:
  - `memory_atom_transfer` — bounded per-door atom evidence profile
    (event-level evidence with standing-state lift filtering + verified
    hypothesis structure) → reuse recurring *parts* of causes across
    tasks;
  - `probe_min_effective_confidence` — context-conditioned confidence
    gate for verbatim reuse;
  - `memory_bank_refresh_on_contradiction` — adaptive forgetting: the
    bank keeps only currently verified entries;
  - `probe_reset_per_episode` — kept as a diagnostic/exploration
    control (documented as not the binding constraint).
- **Evaluation integrity (Phase 3):** three regimes retained; fresh /
  original / memory-wiped / all v0.5 controls mandatory; eval cells
  recreate full variant configs (v0.5 dropped agent-level flags in eval
  cells); frozen cells assert zero memory movement; per-seed reporting
  and paired bootstrap CIs. **Comparison-sign defect corrected** (all
  deltas now candidate − baseline; historical v0.5 raw files untouched,
  correction documented in `docs/MEMORY_V06.md`).
- **EVREN (Phase 4):** teacher workflow contributes validated training
  tasks (offline generator, same validation pipeline, zero API calls,
  quota policy documented; artifact `results/academy_v01/memory_v06/
  evren_pool.json`).

### Measured results (2 seeds × 160 episodes, frozen grids, pinned hash)

| Variant | Frozen | Adapt | OOD frozen | Continual OOD | Train |
|---|---|---|---|---|---|
| fresh | **0.312** | **0.695** | 0.333 | 0.333 | — |
| original | 0.281 | 0.594 | 0.333 | 0.500 | 0.341 |
| memory_wiped | 0.281 | 0.594 | 0.333 | 0.500 | 0.341 |
| ctx | 0.281 | 0.625 | 0.333 | 0.500 | 0.369 |
| ctx_full | 0.508 | 0.531 | 0.542 | 0.611 | 0.497 |
| bank | 0.281 | 0.594 | 0.333 | 0.500 | 0.341 |
| bank_ctx | 0.281 | 0.625 | 0.333 | 0.500 | 0.369 |
| bank_ctx_full | 0.219 | 0.219 | 0.250 | 0.333 | 0.231 |
| reset | 0.281 | 0.594 | 0.333 | 0.500 | 0.341 |
| atomtx | 0.250 | 0.617 | 0.333 | 0.500 | 0.209 |
| atomtx_bank | 0.250 | 0.617 | 0.333 | 0.500 | 0.231 |

Controls (frozen, same grid, pinned hash): random_walk 0.055;
novocab (planning, no vocabulary probing) 0.461.

Paired frozen deltas vs fresh (candidate − fresh): all trained variants
≤ 0 (original −0.030 [−0.094, 0]; atomtx −0.061 [−0.188, 0]; ctx_full
+0.198 [+0.039, +0.359] — an exploration artifact, see below).

### Honest statement

**No variant demonstrates positive zero-shot transfer.** Every
memory-bearing trained variant is at or below the fresh control on the
frozen grid, with CIs at or below zero. The diagnosis is the
deliverable:

- **Storage failure:** cross-task nonevents destroy verified hypotheses
  (56–198 verified→contradicted transitions per run; ~75 of ~77
  hypotheses contradicted at checkpoint), so retrieval has nothing
  usable to return (`hypothesis_candidates` 6816, executed hypothesis-
  sourced plans 0).
- **Verbatim bank reuse is irrelevant/harmful:** 91% of bank probes in
  frozen eval are irrelevant to the true rule (exact 20/7900);
  bank_ctx_full frozen 0.219.
- **The probe vocabulary itself underperforms planning:** novocab
  0.461 > fresh 0.312; ctx_full (0.508) ≈ novocab — context gating
  empties candidates and the agent becomes the goal-planning-only
  explorer (not a random walker; random_walk = 0.055). Its positive
  delta vs fresh is an exploration artifact, not memory.
- **EVREN (Phase 4):** teacher-mixed training nudges vanilla `original`
  frozen to 0.406 (+0.094 vs fresh, CI [0.0, +0.1875]) — directionally
  positive, not significant at 95%; dense atom-transfer does not
  capitalize; teacher-holdout transfer flat at 0.500.

Positive transfer was NOT achieved; a convincing diagnosis was.

### Diagnosis highlights (Phase 1)

- Frozen-eval trace: hypothesis candidates are generated in quantity
  (original 6816) but **zero plans are sourced from hypotheses**
  (`source_hypothesis = 0`) — of ~77 hypotheses at checkpoint, ~75 are
  contradicted, so storage survival, not retrieval, is the failure
  point.
- `verified_then_contradicted` = 56–198 per run: cross-task nonevents
  destroy verified hypotheses — the central negative mechanism
  (CONFIRMED).
- Evidence-bank verbatim reuse transfers irrelevantly: 91% of bank
  probes in frozen eval are irrelevant to the held-out rule (exact
  20/7900), and `bank_ctx_full` frozen 0.219 < fresh 0.312
  (CONFIRMED).
- The `tried`-set block hypothesis is REJECTED (`reset` ≡ `original`
  bit-identical): the tried set is cleared by the agent's own door
  events during training, so at checkpoint it is empty.
- Context gating (`ctx_full`) filters all hypotheses in a new context →
  zero probes → goal-planning-only exploration (`novocab` 0.461), NOT
  random walking (0.055). Its frozen 0.508 is above fresh 0.312 but is
  an exploration artifact — increased task success without knowledge
  transfer.
- EVREN teacher pool (Phase 4): 24 validated, solvable, eval-disjoint
  tasks (offline generator, same validation pipeline; zero API calls)
  → held-out teacher-task transfer measured.

### Testing

- `python -m unittest academy.tests.test_memory_v06` → 25 tests OK
- `python -m unittest …` (memory + agents + eval + EVREN suites) →
  167 tests OK
- Runs: `PYTHONHASHSEED=0 python x/run_chunks_v06.py` (main benchmark);
  `PYTHONHASHSEED=0 python academy/tools/memory_experiment_v06_evren.py`
  (Phase 4); `python academy/tools/memory_merge_v06.py` (merge +
  controls + diagnosis).

### Artifacts & publication gate

The audit artifact-gate hotfix (**PR #39**, `audit/artifact-gate-fix`)
has merged: the frozen criteria (v02) remain immutable, but the permanent
artifact prohibition is replaced by the provenance-aware publication
gate. The raw evidence is therefore **published in this PR** at
`results/academy_v01/memory_v06/` (summary, runs JSONL, fresh control,
EVREN pool/summary) with covering manifests at
`generalization_audit/manifests/` (`memory_v06.manifest.json`,
`memory_v06_evren.manifest.json`).

Verification (Quality v0.5 tooling, from updated main):

- Verifier: main manifest **77 PASS / 0 FAIL / 1 INCONCLUSIVE**
  (memory-limit not recorded — same as committed v0.5/v121 examples);
  EVREN manifest 9 PASS / 0 FAIL / 4 INCONCLUSIVE (generic-format
  checks; memory-limit).
- Artifact gate (`check_frozen_target_artifacts`): **PASS** for
  `results/academy_v01/memory_v06` (manifest coverage, zero FAIL);
  `results/v13` gate open (no artifacts).
- Raw→summary means, comparison directions and paired-bootstrap CIs
  recomputed from the committed vectors by the verifier.
- EVREN pool metadata: 24 approved, all solvable, config-unique, zero
  API calls, quota policy recorded.
- Frozen-evaluation integrity: per-cell assertions (learning disabled,
  memory `generated`/`step`/agent `steps` unchanged) enforced during
  the runs; fresh per-seed traces recorded in the runs JSONL.

See `docs/MEMORY_V06_EXPERIMENT_MANIFEST.md` for the full inventory and
hashes.

### Branch ancestry

PR #38 was originally based on a branch containing one Research v0.13
commit (`4e61903`, the subject of PR #37). **PR #38 must not be merged
before PR #37.** After #37 merges, this branch will be rebased onto the
updated `origin/main` (keeping only Academy changes) and CI re-run; no
research commits are deleted or overwritten.

### Deliverables

- `academy/agents/memory.py`, `metargon_agent.py`
- `academy/tools/memory_experiment_v06.py`, `memory_merge_v06.py`,
  `memory_experiment_v06_evren.py`, `render_v06_results.py`
- `academy/tests/test_memory_v06.py`
- `x/run_chunks_v06.py`
- `docs/MEMORY_V06.md`, `docs/MEMORY_V06_EXPERIMENT_MANIFEST.md`
- Raw artifacts (published, in-repo): `results/academy_v01/memory_v06/`
  (summary, runs JSONL, fresh control, EVREN pool/summary)
- Manifests: `generalization_audit/manifests/memory_v06.manifest.json`,
  `memory_v06_evren.manifest.json`
