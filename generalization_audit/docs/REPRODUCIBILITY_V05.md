# METARGON Reproducible Experiment Infrastructure — v0.5

**Author:** Independent scientific-validation and experiment-infrastructure engineer
**Date:** 9 November 2026
**Branch:** `audit/reproducibility-v05`
**Base:** `origin/main` (including Audit v0.4, PR #35)

This deliverable makes METARGON experiments **independently inspectable and
validatable without relying on an agent's narrative report**. It is a
systematic layer on top of the fixes delivered by audits v0.2–v0.4
(sign conventions, artifact integrity, frozen criteria) rather than another
one-by-one correction.

---

## 1. Design principles

1. **Provenance first.** Every experiment is described by a machine-readable
   manifest (schema `experiment-manifest-v01`) that records code state,
   protocol, seeds, budgets, metric definitions, and artifact hashes.
2. **Verify, don't trust.** The verifier recomputes aggregates from raw
   vectors, checks comparison directions against the declared convention,
   recomputes bootstrap CIs, and validates resource fairness.
3. **Never modify history.** All checks are read-only; reports are printed
   or written to paths chosen by the caller. Missing metadata yields
   **INCONCLUSIVE**, never invented values.
4. **Standard library only.** No new heavyweight dependencies; no external
   API calls; no multi-hour benchmarks in CI.
5. **Independent of implementations.** The infrastructure is general-purpose
   and does not inspect CASUAL's v0.13 or EVREN's v0.6 work. It speaks only
   through manifests and committed artifacts.

---

## 2. Experiment manifest (`experiment-manifest-v01`)

Located at `generalization_audit/schemas/experiment_manifest_v01.json`.
Every manifest contains:

| Section | Records |
|---|---|
| `schema_version` / `protocol_version` | schema + evaluation-protocol versions |
| `experiment` | name, description, format (`benchmark_v12`, `academy_memory`, `generic`), agent version, harness, `commit_sha`, working-tree status (optional), Python version, platform, `PYTHONHASHSEED`, family, conditions |
| `experiment.seeds` | evaluation seed vector (per-seed rows must match its length) |
| `experiment.episodes_per_seed` | episode budget per seed |
| `experiment.action_budget_max_steps` | per-decision step budget |
| `experiment.policies` | roster as evaluated |
| `experiment.ablation_flags` / `parameters` | experimental parameters and ablation settings |
| `experiment.resources` | planner expansion cap, probe plans, memory limits, whether per-policy CPU is required |
| `experiment.evaluation` | comparison convention (`candidate_minus_baseline` / `baseline_minus_candidate`) and paired-bootstrap parameters (seed, rounds, alpha); metric definitions |
| `artifacts` | each artifact: path, role (`raw`/`summary`/`log`/`table`/`manifest`), sha256, size, generator, per-seed flag |

Paths are relative to the manifest's directory (so manifests and artifacts
travel together). A stamp utility fills hashes/sizes and validates the
result:

```
python -m generalization_audit.tools.verify_experiment \
    --stamp cfg.json -o experiment.manifest.json
```

---

## 3. Verifier checks

Implemented in `generalization_audit/verification/checks.py`; every check
returns PASS, FAIL or INCONCLUSIVE with an explanation:

| Check | Verifies |
|---|---|
| `artifact.exists` / `artifact.hash` / `artifact.size` | declared artifacts exist and match manifest hashes |
| `artifact.records.unique` | no duplicate artifact records |
| `raw.seeds` | per-seed row count == manifest seeds |
| `raw.unique.rows` | no duplicate per-seed rows; no duplicate policy keys inside a row |
| `raw.cover` | summary policies covered by raw rows |
| `summary.vs.raw` | summary means equal raw per-seed means (1e-9) |
| `comparison.direction` | recorded delta == recomputed candidate−baseline (or declared reversed convention) |
| `bootstrap.ci` | paired-bootstrap CI/mean recomputed exactly from raw vectors (canonical or harness algorithm per format) |
| `discovery.*` | discovery metrics within [0,1], exact fractions of episodes, per-seed vectors present |
| `per_seed.variability` | per-seed discovery vectors recorded; n, positive-seed count, stdev reported; n<8 flagged as pilot-grade |
| `fairness.*` | identical seed streams, episode budgets, planner caps, memory-limit declaration, per-policy CPU measurement |

`academy_memory` format additionally validates `conditions_detail` vectors
against the `comparisons` block, per-seed block consistency, and JSONL log
uniqueness.

---

## 4. CLI

```
python -m generalization_audit.tools.verify_experiment --manifest <path>
python -m generalization_audit.tools.verify_experiment --manifest A --against B
python -m generalization_audit.tools.verify_experiment --manifest A \
    --replay-command "<bounded command>"
python -m generalization_audit.tools.verify_experiment --stamp cfg.json -o out.json
```

Exit codes: `0` PASS, `1` FAIL, `2` INCONCLUSIVE. `--json` prints the full
report; `-o` writes it to a file. `--tolerance` (default 0.01) controls the
statistical-reproducibility threshold for contrasts.

### Reproductibility classification (contrast / replay)

When a reference manifest is compared with a second manifest (or with the
output of a bounded replay command):

- **EXACT** — artifact content identical (sha256 match).
- **STATISTICAL** — content differs but aggregate metric rows agree within
  tolerance.
- **INCONCLUSIVE** — structural comparison impossible (layout differences,
  artifact present in only one manifest).
- **NON-REPRODUCIBLE** — aggregates diverge beyond tolerance.

Replay runs the user-supplied command in a fresh temporary workdir, hashes
whatever it produces, and contrasts it with the reference; nothing is
written into the repository. Replays with reduced seeds/episodes cannot
claim exact reproducibility by construction (hash mismatch) and will be
reported accordingly. The infrastructure does **not** claim deterministic
execution for inherently nondeterministic benchmarks; determinism must be
established by the experiment itself (pinned `PYTHONHASHSEED`, seeded
worlds/policies), and the manifest records whether it was pinned.

---

## 5. Example manifests (committed experiments)

| Manifest | Experiment | Outcome |
|---|---|---|
| `generalization_audit/manifests/memory_v05.manifest.json` | Academy Memory v0.5 (summary + JSONL) | **0 FAIL**; comparisons self-consistent under the declared historical `baseline_minus_candidate` convention; bootstrap CIs reproduce exactly; `fairness.memory_limit` INCONCLUSIVE (byte limit not recorded) |
| `generalization_audit/manifests/v121_final_l.manifest.json` | v0.12.1 L-final run (raw+summary) | **0 FAIL**; raw→summary means, success and temporal deltas, and harness bootstrap CIs reproduce exactly; only `fairness.memory_limit` INCONCLUSIVE |

Note: the Memory v0.5 manifest declares the *historical* reversed
convention. The verifier's `comparison.direction` and `bootstrap.ci`
checks honor the declared convention, so historical evidence verifies as
self-consistent while the project-wide `candidate_minus_baseline`
requirement (frozen criteria v01/v02) remains the rule for new experiments.

---

## 6. Integration

- New tests: `test_manifest_schema.py`, `test_verifier_checks.py`,
  `test_verify_cli.py`, `test_example_manifests.py` (33 tests) — all fast,
  no network, no heavy benchmarks.
- CI (`generalization-audit` job): after the unit suite, the two committed
  example manifests are verified and asserted to contain no FAIL.

---

## 7. Limitations (honest)

1. **Hash verification is content-only.** A manifest proves the listed
   files are the listed files; it cannot prove the experiment was actually
   run as declared (provenance is still ultimately a trust claim).
2. **Statistical reproducibility tolerance is a judgement call.** The
   default tolerance (0.01 absolute on metric rows) is documented and
   configurable; it is not a substitute for proper effect-size analysis.
3. **Format coverage is explicit.** `benchmark_v12` and `academy_memory`
   formats get deep checks; `generic` manifests get structural checks only.
   New harness formats need a small format adapter.
4. **Replay is user-invoked and bounded.** The CLI does not run large
   benchmarks itself; it executes the command the caller supplies. A replay
   with fewer seeds is correctly reported as non-exact.
5. **Missing metadata ⇒ INCONCLUSIVE.** Manifests that omit e.g. memory
   limits or bootstrap parameters produce INCONCLUSIVE checks, not fabrications.
6. **Not a security boundary.** An adversary who can alter both artifacts
   and manifests would defeat hash-based verification; this layer is
   designed for honest-error catching and independent inspection, not
   adversarial tamper-proofing.
7. **CI does not run n-hour benchmarks.** The verification job is
   lightweight by design; deep reruns of the benchmark harness remain a
   research-side, on-demand activity.

---

## 8. Verification report (this branch)

- Audit suite: **137/137 OK** (includes 33 new infrastructure tests).
- Example-manifest verification: memory v0.5 → 56 PASS / 0 FAIL / 1
  INCONCLUSIVE; v121 final_l → 374 PASS / 0 FAIL / 1 INCONCLUSIVE.
- Academy suite: **209/209 OK**.
- Python suite: recorded on this branch when run completes (long wall under
  loaded machine).
- Repo hygiene and Rust suites: unchanged and green.
