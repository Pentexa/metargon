# METARGON — Architecture Overview

This document explains how the implemented system actually works — the student agent, its memory and planner, the environments, and the Academy/EVREN teacher layers. It reflects the merged code on `main` (up to METARGON v0.11, Academy v0.1, EVREN curriculum v0.2, Academy Memory v0.3, and the independent generalization audit).

## 1. The research loop

Every METARGON agent implements the same closed loop:

```
Observe → Learn → Hypothesize → Plan → Act → Feedback → (repeat)
```

| Stage | What happens | Where |
|---|---|---|
| Observe | A fixed-size public observation: agent position, objects, switch/plate readings, door readings. The hidden rule wiring is never part of the observation. | `academy/envs/gridworld.py`, `python/lab_env.py` |
| Learn | Door events and nonevents update a compositional hypothesis memory: candidate conjunctive rules over public atoms are generated, verified, contradicted, and evicted. | `python/compositional_memory.py`, `python/compositional_memory_v10.py`, `academy/agents/memory.py` |
| Hypothesize | Verified hypotheses steer probing and door protection; contradicted hypotheses are recycled or replaced under a hard cap (128 slots). | same as above |
| Plan | A bounded BFS planner converts beliefs and hypotheses into actions, under strict expansion and evaluation budgets. | `python/metargon_v09.py`, `python/metargon_v10.py`, `python/metargon_v11.py`, `academy/agents/planner.py` |
| Act | Actions are executed in the environment; outcomes feed back into belief and memory. | `academy/learning/loop.py` |

## 2. Environments

### The Mechanism Lab (research benchmarks, v0.9–v0.11)

A 7×5 grid with two movable objects, two pressure plates, two switches, two door barriers, optional delayed effects, and a noisy observation channel. Each door is governed by a hidden conjunctive rule over public atoms (e.g. `switch S1 on` AND `plate P1 loaded`). All configurations are verified solvable; training/validation/held-out test pools are disjoint.

- Conditions: `stable`, `shift` (mechanism changes mid-run), `rapid_shift`, `iid` (fresh config every episode), `novel` (held-out configs), `noisy_obs` (q = 0.08 corrupted readings), `tight_budget`.
- Family **F** is the declared generative family; family **H** is a structurally novel family (different conjunctive shapes and polarities) that the Bayesian baseline's declared model class **cannot express** (model misspecification).
- Budgets: memory ≤ 128 hypotheses, BFS ≤ 700 expansions per plan, ≤ 3(+2) plan evaluations per decision, identical action budgets, identical world seeds.

### Academy MechanismGrid (Academy)

A 7×5 grid world (two public topologies) with the same atom vocabulary (`sw`, `load`, `loadany`, `empty`) and hidden rules. Splits are immutable: 147 train / 49 val / 49 test / 25 OOD configurations. The evaluator refuses to run if any evaluation configuration overlaps the training pool.

## 3. Belief state

The agent maintains a reconciled belief from public observations:

- **Positions/objects are trusted**; switch/plate readings are adopted only after 2 consecutive agreeing observations (debounce) in v0.10+.
- **The agent's own deterministic actions explain readings immediately** (a toggle flips the switch the agent stands on; a push deposits/clears plates).
- **Movement is physical proof**: a failed move into a door cell adopts `closed`; a successful move through it adopts `open`.
- **Doors are adopted immediately** (not debounced), because a stale door belief self-corrects cheaply via movement.

v0.11 splits the *learning* stream from the *planning* belief: the learning stream accepts a door event only when confirmed (two consecutive agreeing raw readings, or movement proof, with a same-direction dedupe window). Planning still adopts readings immediately so it never lags.

## 4. Hypothesis memory

The compositional memory is the agent's causal model. Each hypothesis is a candidate rule (singleton or pair of public atoms) with a lifecycle:

```
generated → predicted → verified → contradicted → evicted/recycled
```

- **Bounded:** hard cap of 128 hypotheses; deterministic eviction when the cap is reached (in practice, Academy students plateau at ~70, so eviction rarely fires).
- **Support/contradiction:** hypotheses accumulate support from door events and contradiction from nonevents. Thresholds (`verify_support=3`, `max_contradict=2`) drive status transitions.
- **Evidence decay (v0.10+):** support/contradiction are float weights; every episode end decays all evidence by 0.70 so stale rules lose influence after a mechanism shift.
- **Prioritized replay (v0.10):** informative causal events are stored in a bounded (64) replay buffer, priority `1/(1+support)`; the top older entries are replayed after each event. A measured negative result: this mechanism is functionally validated but empirically neutral-to-negative in the Mechanism Lab (see `docs/RESEARCH_V10.md`).
- **Noise-aware learning (v0.11):** each supporting update is weighted by `certainty` — the fraction of context atoms whose sensors were stable at event time (floor 0.25). Replay modes: `filtered` (default; only certainty ≥ 0.5 events replayed), `weighted`, `on` (legacy), `off`.
- **Nonevent gating (v0.11):** contradiction signals count only while the door is stable, so spurious flips cannot destroy correct hypotheses.

Academy Memory v0.3 flags (all off by default) add `recycle_contradicted`, `contradiction_ratio`, `evidence_window`, `probe_adaptive`, and episode-local gates; see `docs/MEMORY_V03.md`.

## 5. Planner

The planner is bounded BFS over public physics:

- **Probe selection** is grouped by priority: hypothesis-backed antecedents > novel/informative changes > already-true antecedents.
- **Probe cooldowns (v0.10):** failed probes are recorded with a timestamp and become eligible again after `PROBE_COOLDOWN=40` steps (fixing the v0.9 "discovery freeze" where a failed probe was permanently marked tried).
- **Information-gathering fallback (v0.10):** when no plan/probe is executable, the agent cycles interaction targets (plates, switches) — guaranteeing continuous information gathering even with an empty or contradicted hypothesis store.
- **Door protection:** verified hypotheses are protected while a door is open and not yet passed; v0.11 changed protection to hypothesis-satisfying atoms + the current load-bearing open-door state (fixing the v0.10 "recent causal credit" stall).
- **Determinism:** same seed → identical traces (covered by dedicated tests). Academy runs are bit-reproducible under a fixed `PYTHONHASHSEED`.

## 6. Academy layer

The Academy (`academy/`) is a reproducible continual-learning laboratory. Strict interfaces separate environment, agent, curriculum, experience memory, evaluator, and checkpoints:

| Interface | Responsibility |
|---|---|
| `Environment` | reset / observe / step; never exposes the hidden config |
| `TaskGenerator` | samples task configurations; emits `EpisodeSpec` |
| `StudentAgent` | observe → decide → learn; save/restore state |
| `CurriculumManager` | selects tasks from measured progress |
| `ExperienceMemory` | bounded storage + replay sampling |
| `Evaluator` | frozen pools, matched seeds, leak enforcement |
| `CheckpointManager` | atomic JSON snapshots with config fingerprints |

- All policies consume the **identical episodic stream** (matched task exposure, seeds, noise, budgets).
- Checkpoints are atomic (temp file + `os.replace`), fingerprinted, and restore-deterministic (a test confirms the next episode is bit-identical to a continuation).
- A source-level audit test scans student modules for forbidden identifiers (hidden-state access).

## 7. EVREN teacher layer

EVREN (`academy/evren/`) is an **optional** teacher that proposes *tasks*, never answers:

1. **Proposal** — strict JSON from an LLM provider (quota-guarded, retried, token-capped) or the deterministic offline generator.
2. **Validation chain** — leakage scan → strict schema (unknown fields rejected) → atom vocabulary/arity → canonical rule normalization → independent BFS solvability → evaluation-contamination check (disjoint from val/test/ood) → duplicate check.
3. **Registry** — approved tasks carry full provenance (model, params, seed, tokens, source); rejected candidates are persisted with reasons.
4. **Training** — approved tasks can enter the student's training pool; the student never sees raw proposals.

The curriculum v0.2 experiment (`evren-curriculum`) is offline by construction (zero API calls). The online commands (`evren-smoke`, `evren-pilot`, `evren-train`, `evren-status`) require `EVREN_API_KEY` and are call-budgeted (default 10 calls/run; unlimited requires explicit opt-in).

## 8. Evaluation methodology

- **Paired evaluation:** identical worlds per seed across policies; per-seed deltas with bootstrap 95% CIs.
- **Frozen pools:** test configs are frozen before benchmarking; pools are disjoint from training.
- **Comparators:** model-free baselines (random, episodic, Q-learning), a **privileged** model-based Bayesian particle-filter planner (receives the declared generative family — deliberately strong, misspecified on H/R/L), and a privileged oracle (diagnostic only, excluded from all rankings).
- **Shortcut checks:** every frozen config is re-verified solvable at run time; runtime scans assert no student policy holds a hidden-config reference.

## 9. Rust kernels

`rust/` (crate `metargon-core`) mirrors the research agent's memory/planner logic (v0.9, v0.10, v0.11) and `rust_academy/` mirrors the Academy memory core. Both are independent implementation artifacts (not FFI bindings) with `#![forbid(unsafe_code)]` and their own unit suites. The research version is pinned to the 1.88.0 toolchain in CI.

## 10. Where the code lives

```text
python/lab_env.py                  Mechanism Lab environment
python/robust_belief.py            reconciled belief (debounce, movement proof)
python/compositional_memory_v10.py hypothesis memory (evidence decay, replay, certainty)
python/metargon_v09/v10/v11.py     student agents
python/benchmark_v09/v10/v11.py    paired benchmarks
python/diagnose_v09.py             v0.9 failure-mode diagnostics
python/diagnose_noise_v10.py       v0.11 noise diagnostics
python/v09/v10/v11_postprocess.py  table rendering
academy/envs/gridworld.py          Academy environment
academy/agents/memory.py           Academy hypothesis memory (incl. v0.3 flags)
academy/agents/metargon_agent.py   Academy student agent
academy/agents/planner.py          Academy bounded planner
academy/learning/loop.py           training loop
academy/eval/evaluator.py          frozen-pool evaluation
academy/evren/                     teacher pipeline (validator, registry, selection, retention)
academy/cli/main.py                CLI entry point
generalization_audit/              independent audit (families R/L)
rust/src/v09.rs, v10.rs, v11.rs   Rust kernels
```
