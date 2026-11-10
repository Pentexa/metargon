## Academy Memory v0.7 — Persistent Causal Memory

### Summary

Audit v0.6 confirmed that retained memory does not demonstrate genuine
zero-shot knowledge transfer and that cross-task negative evidence
destroys previously verified hypotheses. v0.7:

- **Pins the root cause (Mission A):** the *global, cross-task
  contradiction accumulator*. The agent's context signature is the
  episode-start state (identical across tasks), and the consecutive-
  nonevent window used to count contradictions is never reset at task
  boundaries. A hypothesis verified in task A is contradicted in task B
  (different hidden mechanism, same observable context) and permanently
  marked unusable — even when the same rule recurs in task C.
- **Introduces a minimal bounded causal memory (Mission B):**
  `memory_episode_scoped` — per-episode support/contradiction records,
  status computed relative to the active epoch only (evidence-based
  promotion/demotion, no global replacement), epoch-boundary reset of
  the nonevent window, and cross-epoch uncertainty-aware retrieval
  (hypotheses verified in ≥2 past epochs, untested in the active epoch).
- **Instruments memory-to-action attribution (Mission C):** per-run
  counters for hypothesis retrieval, cross-epoch retrieval, memory-
  driven actions, correct/incorrect memory-driven decisions, and a
  memory-attributable improvement metric.
- **Controlled evaluation (Mission D):** fresh, original (v0.6
  baseline), memory_wiped, atomtx (v0.6 atom-transfer baseline),
  shuffled-memory negative control, and epoch (v0.7). Evaluation
  configuration is that of v0.6 (eval seed 20261108, 16-config classic
  grid, 12-config OOD, 8-config retention substrate; no new eval seeds).
  **Registration note:** the Quality v0.7 transfer protocol
  (`transfer-eval-protocol-v01`) was frozen by the audit after these
  runs; this experiment is therefore reported as **preliminary
  evidence** (protocol level_1_preliminary), not as protocol-
  preregistered or confirmatory.

### Measured results (2 seeds × 160 episodes, frozen 16-config grid, pinned hash)

| Variant | Frozen | Adapt | OOD frozen | Continual OOD |
|---|---|---|---|---|
| fresh | 0.312 | 0.695 | 0.333 | 0.333 |
| original | 0.281 | 0.594 | 0.333 | 0.500 |
| memory_wiped | 0.281 | 0.594 | 0.333 | 0.500 |
| atomtx | 0.250 | 0.617 | 0.333 | 0.500 |
| shuffled (negative control) | 0.281 | 0.594 | 0.333 | 0.500 |
| **epoch** | **0.594** | 0.703 | 0.458 | 0.333 |

Paired frozen deltas (candidate − baseline):

| Comparison | Δ | 95% CI |
|---|---|---|
| epoch vs fresh | +0.285 | [0.063, 0.531] |
| epoch vs original | +0.315 | [0.094, 0.531] |
| epoch vs shuffled | +0.315 | [0.094, 0.531] |
| epoch vs novocab | +0.134 | [−0.047, 0.320] |
| shuffled vs original | 0.000 | [0, 0] |

Controls: random_walk 0.050, novocab 0.463 (same frozen grid).

### Honest statement (preliminary)

v0.7 offers **preliminary memory-dependent transfer evidence**, not yet
independent confirmatory proof:

- frozen 0.594 vs fresh 0.312 (Δ +0.285, CI [0.063, 0.531]) — 2 train
  seeds (pilot-grade per protocol v01), CI excludes 0 for this primary
  delta;
- shuffled-memory negative control (same training, broken associations)
  loses the gain entirely (0.281; Δ vs shuffled +0.315, CI
  [0.094, 0.531]) — the memory-attributable component;
- against the exploration-only `novocab` control the delta is positive
  (+0.134) but its CI includes 0 — a planning-only contribution cannot
  be fully excluded at 95%; the claim is therefore preliminary, not
  confirmatory (confirmatory would require ≥5 train seeds and
  independent audit recomputation);
- negative results preserved: all v0.6 baselines ≤ fresh; atomtx and
  shuffled do not transfer.

### Memory diagnosis (confirmed)

- Cross-task contradictions previously destroyed verified hypotheses
  (~75/77 contradicted at checkpoint); epoch reduces contradicted to
  ~15 and raises verified to ~10, with 47 hypotheses verified in ≥2
  epochs (transferable).
- Attribution: 23/32 frozen cells show memory-driven actions (3568
  actions). Frozen cells disable learn(), so door-outcome counters are
  0 by construction there; action-side attribution and success splits
  are measurable, and the shuffled ablation is the decisive control
  (see docs/MEMORY_V07.md §4.1 limitation).

### Quality v0.7 checks

- Verifier (incl. --transfer): **52 PASS / 0 FAIL / 4 INCONCLUSIVE**
  (memory-limit; format-level c_missing_duplicate_records;
  transfer.unseen.separation — resting on declaration; transfer.
  attribution.consistency — no merged frozen_trace_mean block).
- Protocol checks PASS: protocol registered, roles resolve
  (fresh/original/memory_wiped/shuffled/novocab), frozen immutability
  attested + trace artifact, ablation paired, budgets.

### Testing

- `python -m unittest academy.tests.test_memory_v07` → 12 tests OK
- memory suites incl. v0.6/v0.5 → 55+ OK
- Full Academy suite → 246/246 OK
- Audit suite (incl. Quality v0.7 transfer-protocol tests) → 170/170 OK

### Deliverables

- `academy/agents/memory.py`, `metargon_agent.py`
- `academy/tools/memory_experiment_v07.py`, `memory_merge_v07.py`
- `academy/tests/test_memory_v07.py`; `x/run_chunks_v07.py`
- `docs/MEMORY_V07.md`
- `results/academy_v01/memory_v07/` (raw artifacts: summary, runs JSONL,
  fresh control)
- `generalization_audit/manifests/memory_v07.manifest.json` (Quality v0.5)
