# METARGON Academy v0.1 — Architecture

This document describes the structure, interfaces and scientific controls of
the Academy: a reproducible continual-learning and curriculum-generation
laboratory for the METARGON research programme.

## 1. Goals

Build a *real learning system* (not a scripted demonstration) in which a
resource-constrained agent:

1. repeatedly interacts with procedurally generated environments,
2. learns from outcomes and preserves useful knowledge across episodes,
3. is evaluated on unseen challenges with strict train/test isolation,
4. supports checkpointing, resumption, and reproducible runs on CPU only.

## 2. Repository layout

```
academy/
  core/interfaces.py        # Environment/TaskGenerator/StudentAgent/
                            # CurriculumManager/ExperienceMemory/Evaluator
  envs/gridworld.py         # MechanismGrid world + hidden rule dynamics
  envs/tasks.py             # task families, immutable splits, episode streams
  agents/common.py          # perception helpers (observations -> atoms)
  agents/memory.py          # compositional hypothesis memory (persistent)
  agents/planner.py         # bounded planner over public physics
  agents/metargon_agent.py  # METARGON Academy student agent
  agents/baselines.py       # random / episodic / qlearning / bayes / oracle
  learning/curriculum.py    # measured-progress curriculum manager
  learning/experience_memory.py  # bounded replay buffer
  learning/loop.py          # online learning loop + episode runner
  eval/metrics.py           # bootstrap CI, curves, forgetting, adaptation
  eval/evaluator.py         # frozen-pool evaluation with leak enforcement
  ckpt/checkpoint.py        # atomic, crash-safe snapshots
  cli/main.py               # smoke|splits|calibrate|train|eval|ood|...
  tests/                    # unit + integration tests
configs/academy_v01.toml    # pre-registered configuration (fingerprinted)
rust_academy/               # bounded Rust kernel (#![forbid(unsafe_code)])
docs/ACADEMY_ARCHITECTURE.md
docs/ACADEMY_RESEARCH_V01.md
results/academy_v01/        # splits, calibration, training curves, eval JSON
checkpoints/academy_v01/    # reproducible model/training-state snapshots
```

## 3. Environment: MechanismGrid

A 7×5 grid world (two public topologies, `TP1` corridor / `TP2` switchback)
with:

- two movable objects (A, B) pushed Sokoban-style,
- two pressure plates (P1, P2), two switches (S1, S2),
- two barrier doors (D1, D2) whose opening is governed by *hidden rules*,
- optional one-step delay and optional noisy readings.

**Atom vocabulary (public):** `("sw", switch, value)`,
`("load", plate, object)`, `("loadany", plate)`, `("empty", plate)`.

**Hidden state:** each door's rule (conjunctive frozenset of atoms, or a
disjunctive clause list), delay flags, and the per-step noise stream.
None of this is exposed through `Observation` (a fixed-size NamedTuple with
13 public fields only).

**Public physics** (known to every agent and planner): movement, pushing,
switch toggling, and the fact that closed doors block cells. The world spec
(geometry) is public; only the wiring is hidden.

## 4. Core interfaces (strict separation)

| Interface | Responsibility |
|---|---|
| `Environment` | reset / observe / step; never exposes the config |
| `TaskGenerator` | samples task configurations; emits `EpisodeSpec` |
| `StudentAgent` | observe → decide → learn; save/restore state |
| `CurriculumManager` | selects tasks from *measured* progress |
| `ExperienceMemory` | bounded storage + replay sampling |
| `Evaluator` | frozen pools, matched seeds, leak enforcement |
| `CheckpointManager` | atomic JSON snapshots with config fingerprints |

Policies never receive hidden rules, private simulator state, correct
answers, or evaluation labels. A source-level audit test scans all student
agent modules for forbidden identifiers (`transition_with_hidden`, rule
evaluation internals, `EnvConfig`, …). The model-based **Bayes baseline is
exempt** by design: it plans under its *own* hypothesized configurations.

## 5. Learning loop (Phase 3)

```
Observe → Decide → Act → Receive Feedback → Update Memory/Model → Continue
```

- One deterministic episode stream is built once per run (matched task
  exposure): every policy consumes the identical sequence of tasks,
  seeds, noise and budgets.
- `run_one_episode` drives each policy through env steps, collects
  experiences into the bounded replay buffer, and records per-episode
  diagnostics.
- `learn()` updates the persistent hypothesis memory from door events and
  nonevents (with a *maintain* bias for persistent causes).
- Replay (`learn_from_experience`) adjudicates hypotheses created after a
  transition was recorded against old stored transitions (retroactive
  validation), giving replay genuine offline value.

## 6. The METARGON student agent

Combines:

- **belief state** rebuilt from public observations,
- **compositional hypothesis memory** (bounded, evidence-driven,
  checkpointable): candidate conjunctive rules → door outcomes, with
  deterministic eviction and verification thresholds,
- **bounded planner** over public physics (BFS) with a connected-component
  feasibility filter and a per-episode plan cache,
- **probe selection** grouped by priority: hypothesis-backed antecedents >
  changes (novel information) > already-true antecedents,
- **door-pass tracking** by physically standing on a door's far cell,
- **protection** of verified hypotheses while a door is open and not yet
  passed (prevents self-inflicted door toggling loops).

## 7. Task families and immutable splits

- Core family: conjunctive rules over both topologies. The full distinct
  configuration space is enumerated, filtered by ground-truth solvability,
  and split 60/20/20 → **147 train / 49 val / 49 test** configurations
  (one deterministic run: train≥100, val≥25, test≥25).
- OOD family: *unseen combinations of mechanisms* — only `TP2`, plus
  rules using absence atoms (`("empty", …)`) in a causal role never seen
  in training, acting as correlated distractors for monotone
  "load plates helps" policies. 25 held-out configurations.
- Splits are a pure function of `pool_split` and are persisted as JSON;
  the evaluator refuses to run if any evaluation configuration overlaps the
  training pool.

## 8. Evaluation design

- Frozen val/test/OOD pools; per-seed streams are built once and reused
  across policies (paired evaluation).
- Metrics: success rate, mean steps, door events, bootstrap 95% CI
  (paired per-seed differences), learning curves, adaptation latency,
  forgetting, memory utilization, planner operations, wall-clock CPU time.
- Baselines: random, episodic memory, tabular Q-learning, model-based
  Bayesian planner (given the correct core model class — a strong
  comparator, deliberately mismatched on OOD), and a privileged oracle
  (diagnostic only, excluded from rankings).
- Ablations: METARGON with replay disabled vs full configuration.

## 9. Checkpointing and safety

- Deterministic seeds everywhere; the config file is fingerprinted into
  every snapshot.
- `CheckpointManager` writes atomically (temp file + `os.replace`), keeps
  a bounded number of snapshots, and verifies fingerprints on load.
- Restoring a snapshot reconstructs the agent's RNG, memory, probe sets,
  and counters — a test confirms the *next episode* is bit-identical to a
  continuation from the original agent (crash-safe resumption).
- All training loops respect bounded memory and configurable episode/
  evaluation limits. No uncontrolled infinite loops; no external
  deployments; no remote pushes.

## 10. Rust kernel

`rust_academy/` implements the bounded memory core, atom vocabulary,
candidate generation, and conjunctive/disjunctive rule evaluation as a
standalone crate with `#![forbid(unsafe_code)]`. It mirrors the Python
semantics for sampled scenarios (events, maintain-verification, nonevent
contradiction, eviction, scoring) and passes its own unit tests. It is an
independent implementation artefact, not an FFI binding.

## 11. EVREN extension points

The Academy is deliberately agnostic to any external teacher/API. Task
proposals from external models (e.g., EVREN) are treated as *candidate
tasks* that must pass environment-side validation (schema, solvability,
train/eval disjointness) before entering a pool; they never act as
ground-truth verification and are never required for training or
evaluation. All Academy operations run offline with the standard library
plus a local Rust toolchain.
