# Artifact Gate — Provenance-Aware Publication (Quality v0.5 hotfix)

**Date:** 10 November 2026
**Branch:** `audit/artifact-gate-fix`
**Base:** `origin/main` (after merging Quality PR #36)

## Problem

The frozen evaluation criteria for future experiments
(`configs/frozen_eval_criteria_v02.json`, audit v0.4) were — correctly —
defined **before** any v0.13 / Memory v0.6 results existed. The original
regression test enforced that timed relation with a *permanent* assertion
that `results/v13` and `results/academy_v01/memory_v06` never exist. That
prohibition would block Research v0.13 and Academy Memory v0.6 from ever
committing their raw evidence, even when fully documented and reproducible.

## Fix

- The **frozen criteria are unchanged and immutable** (same file, same
  requirements; the test suite still asserts their integrity).
- The permanent prohibition is replaced by a **provenance-aware
  publication gate** (`generalization_audit/verification/artifact_gate.py`):

| Situation | Verdict |
|---|---|
| Target directory absent | **PASS** — gate open, criteria still frozen |
| Raw evidence present, covered by valid manifest(s) that verify with **0 FAIL** | **PASS** — evidence published with provenance |
| Raw evidence present with no covering manifest | **FAIL** — provenance missing |
| Covering manifest invalid or verification has FAIL checks | **FAIL** — with messages |
| Artifact files not declared by any manifest (orphans) | **FAIL** — incomplete provenance |

- Missing metadata inside a manifest (e.g. memory limit) yields
  **INCONCLUSIVE** from the verifier, which the gate treats as non-FAIL:
  it reports it without blocking publication. Nothing is ever invented.
- The gate reuses the existing verifier (`verification/checks.py`) and
  manifest schema — no new machinery, no new dependencies.

## Publication workflow (for Research v0.13 / Academy Memory v0.6)

1. **Run** the experiment with pinned `PYTHONHASHSEED`, recorded seeds,
   episode counts, policies, and parameters (per the frozen criteria:
   pilot vs confirmatory sizes, per-seed reporting, candidate − baseline,
   paired bootstrap CIs).
2. **Commit** the raw per-seed artifacts plus summary files under
   `results/v13/...` or `results/academy_v01/memory_v06/...`.
3. **Create a manifest** under `generalization_audit/manifests/`:
   - copy the nearest example (`manifests/v121_final_l.cfg.json` for the
     mechanism lab, `manifests/memory_v05.cfg.json` for Academy memory),
   - set the experiment block (name, format, agent/harness versions,
     commit SHA, seeds, episodes, policies, params, resources, evaluation
     convention + bootstrap params, metric definitions),
   - list every artifact with its role.
4. **Stamp** hashes/sizes and validate:
   ```
   python -m generalization_audit.tools.verify_experiment \
       --stamp generalization_audit/manifests/<exp>.cfg.json \
       -o generalization_audit/manifests/<exp>.manifest.json
   ```
5. **Verify** the manifest; fix anything that FAILs (do not remove raw
   evidence to make it pass):
   ```
   python -m generalization_audit.tools.verify_experiment \
       --manifest generalization_audit/manifests/<exp>.manifest.json
   ```
6. The **CI gate** (`test_artifact_gate_provenance.py`, part of the
   audit suite) automatically allows the PR once every target artifact is
   covered by a manifest that verifies with zero FAIL.

Backward compatibility: existing manifests (`memory_v05`, `v121_final_l`)
are untouched and still verify; the gate is a superset check that only
activates when the frozen target directories exist.

## Tests

- `tests/test_artifact_gate_provenance.py` (7 tests): simulated absent /
  unmanifested / tampered / valid / orphan scenarios + real-repo
  integration (current main state → gate open).
- `tests/test_future_eval_criteria_v02.py`: prohibition test removed;
  criteria immutability/audit checks retained (sections, protocol meta,
  implementation-independence).

## Independence

No Research or Academy implementation file was modified. The gate is
general-purpose and speaks only through manifests and committed artifacts.

## Verification (this branch)

- Audit suite: **144/144 OK**.
- Repo hygiene: OK.
- Gate on current main: both frozen targets PASS (gate open).
