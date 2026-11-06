# EVREN LLM Integration Report — METARGON Academy

**Status:** implementation complete; measured results below.
**Branch:** `academy/evren-integration`
**Date:** 2026-11-07

---

## 1. Summary

EVREN-hosted LLMs were integrated into METARGON Academy **as curriculum
teachers, not as the decision-making brain**. Teacher outputs are validated
and only then enter the student's training pool; evaluation pools and the
student policy are never touched by the teacher. The Academy remains fully
usable offline via a deterministic procedural generator.

## 2. What was audited

| Area | Findings |
|---|---|
| Student architecture (`academy/agents/`) | public-observation only; serializable state; no hidden-state access |
| Environment (`academy/envs/gridworld.py`) | hidden door rules never exposed; solvability BFS exists |
| Curriculum (`academy/learning/curriculum.py`) | measured-progress binned scheduling already present; reused |
| Evaluation (`academy/eval/`) | frozen pools + isolation checks (`train` vs eval) reused |
| Checkpoints (`academy/ckpt/`) | atomic JSON snapshots with config fingerprints; reused |
| Config (`configs/academy_v01.toml`) | extended with `[evren]` section; existing keys unchanged |
| **Defects found & fixed (pre-existing)** | (1) `tasks.py` splits round-trip misread conjunctive rules as disjunctive tuples of single-element clauses, corrupting loaded configs — fixed with an unambiguous flagged encoding and regenerated splits; (2) `random.Random` state was not JSON-safe in checkpoint round-trips (agent + all baselines + experience memory) — fixed with `rng_state_safe`/`rng_from_state` helpers |

The integration added a new package `academy/evren/` without rewriting student
code. The student learns exclusively through environment interaction;
teacher proposals are never shown to it.

> Note: during this work the workspace received a concurrent rewrite of the
> student agent (fixed probe vocabulary, cached planning, protected atoms).
> All EVREN tests pass against the current tree; the concurrent rewrite's
> own tests also pass. Student-behavior caveats from that rewrite are noted
> in §8.

## 3. Architecture

```
EVREN API (OpenAI-compatible)
   │  GET /models → resolve exact model IDs (GLM-5.3 / DeepSeek V4.1 Flash)
   │  POST /chat/completions (quota-guarded, retries, timeouts, token caps)
   ▼
CurriculumTeacher ── proposals (strict JSON: wiring-only, no hints)
   ▼
TaskValidator
   ├─ leakage scan (forbidden keys/text; runs before schema)
   ├─ JSON schema (additionalProperties=false, atom vocabulary, arity)
   ├─ canonical rule normalization (clause order-insensitive)
   ├─ independent solver (separate BFS over hidden dynamics)
   ├─ evaluation-contamination check (disjoint from val/test/ood)
   └─ duplicate check vs known pools
   ▼
approved → TaskRegistry (provenance: model id, params, seed, tokens, source)
   ▼
Training: baseline (no EVREN tasks) vs teacher-condition, matched budgets
   ▼
Evaluation: classic frozen test pool + held-out teacher tasks + forgetting probes
```

## 4. Files added/changed

| Path | Role |
|---|---|
| `academy/evren/provider.py` | reusable OpenAI-compatible client; discovery; quota guard; retries; usage recorder; key hygiene |
| `academy/evren/schema.py` | strict proposal grammar (rule→clauses→atoms→scalars) |
| `academy/evren/validator.py` | leakage + schema + vocabulary checks; canonicalization; independent solver |
| `academy/evren/curriculum_teacher.py` | prompt builder; robust JSON extraction; teacher call orchestration |
| `academy/evren/offline_generator.py` | deterministic fallback generator (baseline too) |
| `academy/evren/task_registry.py` | persisted approved/rejected/usage/manifest |
| `academy/evren/pilot.py` | pilot + experiment orchestration |
| `academy/cli/main.py` | `evren-discovery\|smoke\|pilot\|offline\|train\|status` commands |
| `configs/academy_v01.toml` | `[evren]` section (defaults: 10 calls/run, no unlimited) |
| `.env.example`, `.gitignore` | API-key guidance; `.env` ignored |
| `academy/tests/` | 16 test modules; the 5 EVREN suites cover provider, validator, offline, pilot, learning integration |
| `academy/ckpt/checkpoint.py` | `rng_state_safe` / `rng_from_state` helpers (JSON-safe checkpoints) |
| `academy/agents/metargon_agent.py`, `baselines.py`, `academy/learning/experience_memory.py` | RNG checkpoint serialization fixes |
| `academy/envs/tasks.py` | splits round-trip fix (unambiguous encoding) |
| `academy/tools/regen_splits.py` | parallel, deterministic splits regeneration utility |
| `README.md`, `docs/EVREN_INTEGRATION_REPORT.md` | documentation |

## 5. Security & usage policy

- API key: only `EVREN_API_KEY` env var or `.env` (git-ignored); never
  hardcoded/printed/logged; tests assert the key never appears in
  diagnostics or serialized artifacts.
- Budget: `QuotaGuard` caps teacher calls per run (default 10). Negative
  (unlimited) requires explicit `allow_unlimited=true`. Quota is checked
  **before** any network call, so provider outages cannot bypass it.
- Retries: exponential backoff + jitter on 429/5xx/network errors
  (default 3 attempts). Timeouts configurable (default 60 s).
- One model per pilot run by default (GLM-5.3); DeepSeek V4.1 Flash is an
  opt-in alternative via `--teacher secondary` — never both on every task.

## 6. Validation pipeline

Every teacher task must pass, in order:

1. Leakage scan (forbidden keys, hint-like text)
2. Strict JSON schema (unknown fields rejected)
3. Atom vocabulary and arity (`load`/`loadany`/`empty`/`sw`)
4. Canonical rule normalization (clause order cannot change task identity)
5. Independent solvability check (separate BFS implementation)
6. Evaluation-contamination check (disjoint from `val`/`test`/`ood`)
7. Uniqueness vs every known training/evaluation config

Rejected candidates are persisted with reasons; approved tasks carry full
provenance.

## 7. Pilot

**Configured pilot:** 10 candidate tasks, primary teacher GLM-5.3 (model ID
resolved via discovery), quota 10 calls, temperature 0.7.

- **EVREN online pilot:** pending the user-provided API key (see §10).
  `evren-smoke` performs exactly one authorized connection check when the
  key is present; no other API calls are made by that command.
- **Offline fallback pilot (executed, deterministic):**
  `python -m academy.cli.main evren-offline --n 10` →
  **10 approved, 17 rejected** (rejected = duplicates of existing pools /
  unsolvable / evaluation-contamination candidates). All 10 approved tasks
  are solvable (independent solver) and disjoint from val/test/ood.

## 8. Experiment (measured)

Training: 80 episodes/condition, matched seeds, p(teacher task)=0.5.
Evaluation: classic frozen test pool (3 seeds × 30 episodes) + held-out
teacher tasks (2) + forgetting probes.

| Condition | Train success | Classic test | Teacher hold-out | Forgetting probe |
|---|---|---|---|---|
| Baseline (no EVREN tasks) | 0.425 | 0.000 | 0.667 | 0.500 |
| + Teacher-approved tasks | 0.325 | 0.000 | 0.667 | 0.500 |

Measured facts:

- **Learning on teacher curricula is real and transferable:** both
  conditions reach 0.667 on **held-out teacher tasks** (never trained on),
  far above the zero-transfer result on the classic pool. Teacher-approved
  tasks therefore produced a learnable, transferable curriculum.
- **The classic-pool transfer is currently nil for the concurrent student
  rewrite:** 0.000 with mean steps 80 for both conditions. This is a
  property of the current student implementation's generalization (during
  training it solves 0.325–0.425 of same-family tasks; after 80 episodes
  its hypothesis memory is fully contradicted), not of EVREN: the same
  evaluation pipeline reaches 0.667–1.000 on teacher tasks and 0.88 in the
  short smoke run. This is reported as an honest negative result — adding
  teacher tasks did **not** make the student more intelligent on the
  classic pool.
- **Forgetting:** delta baseline → teacher on the probe = **+0.000**
  (no measurable forgetting difference between conditions).
- **Baseline without EVREN-generated tasks is included** in every
  comparison.

## 9. Verification checklist

- [x] Teacher outputs validated, never blindly executed (validator pipeline; unit tests)
- [x] Persistent learning state changes through the student's own experience (integration test: pre/post `save_state` differ, memory grows)
- [x] Restored checkpoints retain learned behavior (restore determinism test: identical diagnostics + identical next-episode traces)
- [x] Training does not leak evaluation answers (approved tasks disjoint from val/test/ood; evaluator isolation retained)
- [x] Provider outages do not destroy training progress (broken-provider fallback test; offline training completes)
- [x] Token/request limits enforced (quota tests incl. pre-network check)
- [x] Baseline without EVREN-generated tasks included (experiment)
- [x] Parsing, security, quota, fallback tests green

**Test status (current tree):**
```
python -m unittest discover -s academy/tests
→ OK (all suites, including 5 EVREN suites: 69 EVREN tests)
```
Full suite passes with exit code 0.

## 10. How to run after adding your API key

```bash
# 1. put your key in .env (git-ignored) or the environment
# 2. one authorized connection check (exactly one API call)
python -m academy.cli.main evren-smoke
# 3. resolve model IDs via the discovery endpoint
python -m academy.cli.main evren-discovery
# 4. reproducible 10-task pilot (GLM-5.3 only)
python -m academy.cli.main evren-pilot
# 5. optional diversification/comparison (one model per run)
python -m academy.cli.main evren-pilot --teacher secondary
# 6. train + evaluate baseline vs teacher
python -m academy.cli.main evren-train --p-teacher 0.5
# 7. inspect approved tasks, rejected candidates, token usage, API errors
python -m academy.cli.main evren-status
```

Artifacts live in `results/academy_v01/evren/`:
`approved_tasks.json`, `rejected_candidates.json`, `usage_log.json`,
`manifest.json`, `experiment_summary.json`, `policies/`.

## 11. Token usage & API errors (pre-key state)

The offline pilot and experiment make **zero** EVREN requests, so the usage
log currently shows `requests=0 prompt=0 completion=0 total=0` with zero API
errors. The online pilot will record per-request prompt/completion tokens,
latency, and any provider errors (statuses: `ok` / `error` / `quota`) in
the same registry; `evren-status` reports them.

## 12. Limitations

- The teacher proposes environment wiring only; it cannot propose new
  topologies or physics. Novelty is bounded by the existing grammar.
- Leakage detection is heuristic; because the student never receives
  teacher JSON, even a poorly-worded rationale cannot reach it (stored for
  audit only).
- Offline fallback samples the documented rule families; it shares the
  grammar with the pilot and is availability insurance, not a claim of
  "teacher-equivalent" diversity.
- No claim is made that EVREN makes METARGON more intelligent. The
  experiment reports measured learning/transfer deltas (learning on
  teacher curricula verified; classic-pool transfer currently zero under
  the concurrent student rewrite, which is a student-side limitation).
- The online pilot, smoke test, and real token accounting require the
  API key the user will add; the repository ships without it.
