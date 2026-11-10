# METARGON Independent Transfer Evaluation — v0.7

**Auditor:** Independent transfer-evaluation & scientific-integrity engineer
**Date:** 10 November 2026
**Branch:** `audit/transfer-evaluation-v07`
**Base / starting commit:** `origin/main` @ `a617365bf513db0632c837aa14d638bef6cc3cf3` (includes PR #40)
**Status:** protocol pre-registered; audit-owned checks implemented; no
Memory v0.7 results interpreted.

---

## 1. Purpose

Post-merge audit v0.6 confirmed that the committed evidence is
reproducible (0 FAIL manifests; v0.13 33/33; v0.6 34/34 recomputed) but
that **genuine zero-shot transfer remains unproven**. The next question —
*does the system transfer learned causal knowledge, or does it merely
achieve higher benchmark scores through exploration, memorization, or
leakage?* — cannot be answered by the existing frozen criteria. Mission
v0.7 defines an implementation-neutral evaluation protocol and the audit
machinery to enforce it, pre-registered before any Memory v0.7 results
exist.

---

## 2. The versioned protocol (`transfer-eval-protocol-v01`)

File: `generalization_audit/configs/transfer_eval_protocol_v01.json`
(frozen at registration on 2026-11-10; separate from the historical frozen
criteria v01/v02, which remain unchanged and still govern v0.12.1/v0.5
retrospective checks).

### 2.1 Quantities the protocol distinguishes

| # | Quantity | Operational definition |
|---|---|---|
| 1 | Retained-memory benefit | performance delta caused by knowledge retained from training, isolated from exploration and within-episode learning |
| 2 | Within-episode learning | gain from learning during evaluation episodes (adapt − frozen) |
| 3 | Goal-directed exploration | performance from planning/exploration alone (planning-only control) |
| 4 | Repeated-task memorization | performance on configurations seen in training (must be excluded from transfer claims) |
| 5 | True transfer to unseen causal structures | performance on configurations disjoint from training, with registered split |
| 6 | Negative transfer | measured reduction caused by retained memory relative to the appropriate control |

### 2.2 Mandatory counterfactual control arms

| Control | Purpose |
|---|---|
| fresh | untrained same-architecture policy; establishes the exploration baseline |
| trained with memory | the candidate (retained knowledge) |
| trained, memory wiped | same trained policy, memory erased at eval time; isolates memory from training effects |
| shuffled memory | trained policy with stored memory scrambled; falsifies accidental association reuse |
| planning-only | same agent with vocabulary/candidate probing disabled; isolates goal-directed exploration |
| matched budgets | identical per-decision compute budgets, episode streams, max steps; per-policy CPU reported |

Rules: hidden rules may be used only for independent evaluation labels; agent
decisions consume only observable information; frozen arms must assert
learning/memory immutability; no protocol tuning to any implementation.

### 2.3 Statistics

Paired seeds across arms; candidate − baseline; paired bootstrap 95% CIs
(seed, rounds, alpha recorded); per-seed vectors with stddev and
nonzero-seed counts — all mandatory for any transfer claim.

---

## 3. Mission C — minimum evidence for a genuine zero-shot transfer claim

Claim levels (defined in the protocol file):

- **Level 0 — no claim:** descriptive/inconclusive results only.
- **Level 1 — preliminary (pilot):** all of
  1. improvement vs **fresh** and vs **planning-only** (candidate − baseline positive);
  2. measured retained-memory influence on selected actions (attribution trace shows hypothesis/bank/atom-sourced actions);
  3. reduced or eliminated benefit in **memory-wiped** and **shuffled-memory** controls;
  4. evaluation on unseen configurations (registered disjoint split);
  5. per-seed vectors + paired bootstrap 95% CIs; CI including 0 explicitly labeled "not a measured population effect";
  6. no hidden-rule access during execution (harness-level check).
- **Level 2 — confirmatory:** all level-1 evidence at confirmatory sample
  sizes (≥8 seeds × 100 episodes mechanism lab; ≥5 train seeds Academy),
  primary-delta CI excluding 0, and one independent audit recomputation.

**Failure and inconclusive reporting** (mandatory):
- Negative transfer is reported as a result (cells + CIs), not suppressed.
- INCONCLUSIVE is the verdict whenever required metadata is missing (e.g.,
  unrecorded budget/seed/hash, missing arm, no trace); it is **never**
  converted to PASS by fabricating values.
- Protocol failure (uninterpretable evaluation: missing arms, overlap,
  hidden access) makes the claim not assessable and is reported as such.

---

## 4. Mission B — audit-owned reproducibility & fairness checks

New module: `generalization_audit/verification/transfer_checks.py`
(exposed via `--transfer` on `verify_experiment.py`).

| Check | Semantics |
|---|---|
| `transfer.protocol.registered` | FAIL when the manifest's protocol_version does not match the registered protocol; PASS otherwise |
| `transfer.roles.declared` / `transfer.roles.resolve` | FAIL when required arms (fresh, trained_with_memory, trained_memory_wiped, shuffled_memory, planning_only) are missing/duplicate/unresolved; INCONCLUSIVE when no roles declared |
| `transfer.unseen.separation` | PASS when registered disjoint split and (when pools are exposed in artifacts) zero overlap; FAIL on detected overlap; INCONCLUSIVE when separation is declared but pools not exposed |
| `transfer.frozen.immutability` | PASS when frozen mode has immutability attestation or a frozen trace artifact; INCONCLUSIVE when frozen mode is declared without attestation |
| `transfer.ablation.paired` | PASS when memory/wiped arms co-occur with paired vectors; FAIL on length mismatch; INCONCLUSIVE when arms absent |
| `transfer.attribution.consistency` | PASS when sum(source_*) == plans_used (or probes_evaluated) across trace records; FAIL on mismatch; INCONCLUSIVE without trace data |
| `transfer.budgets.*` | PASS/FAIL on recorded caps, CPU measurement, action budget; INCONCLUSIVE on missing metadata |

Missing-metadata policy: we only turn INCONCLUSIVE into PASS when the
quantity can be obtained reliably from committed artifacts (e.g., CPU from
raw rows). We do **not** fabricate memory-limit bytes or seeds.

---

## 5. Mission D — frozen protocol integrity

- The protocol file is a separate versioned artifact; historical frozen
  criteria (`frozen_eval_criteria_v01/v02.json`) are untouched.
- **Seed registration:** transfer-evaluation seeds are sampled once at
  registration via SHA-256(protocol_id ‖ registration_date ‖ counter),
  truncated and frozen in the protocol file (16 seeds, used in order).
  No seeds are added after development runs begin; Academy runs keep the
  Academy eval-grid seeds (frozen).
- No retroactive edits to v0.13 / Memory v0.6 requirements; no tuning to
  unfinished EVREN results.

---

## 6. Mission E — lightweight automation

- Regression tests: `generalization_audit/tests/test_transfer_protocol.py`
  (15 tests) exercise PASS/FAIL/INCONCLUSIVE semantics on synthetic
  manifests + the protocol file invariants. Runtime ≈ 2 s.
- CI: the `generalization-audit` job already runs the whole audit suite;
  these tests are included (fast). No multi-hour benchmarks are run in CI.
- Compatibility: `verify_experiment` unchanged by default; `--transfer`
  is opt-in. Existing manifests keep their previous verification results.

---

## 7. Verification report (this branch)

| Item | Result |
|---|---|
| Audit suite (incl. 15 new transfer tests) | **164/164 OK** |
| `--transfer` on historical `memory_v06` manifest | protocol mismatch → FAIL (correct: pre-dates protocol); undeclared roles/separation/immutability → INCONCLUSIVE; attribution sums PASS; budget checks PASS |
| Repo hygiene | OK |

Data note: attribution-sums invariant validated against committed Memory
v0.6 trace records (e.g., original: sum(source_*) = 7260 = plans_used).

---

## 8. Deliverables

- `generalization_audit/configs/transfer_eval_protocol_v01.json` — versioned protocol.
- `generalization_audit/verification/transfer_checks.py` — audit-owned checks.
- `generalization_audit/tests/test_transfer_protocol.py` — regression tests (15).
- `generalization_audit/docs/TRANSFER_EVALUATION_V07.md` — this document.
- CLI: `--transfer` flag on `verify_experiment.py`.

## 9. Success criterion

An independent reviewer can, for any future Memory v0.7 claim, map it to
the protocol's quantities, check the required controls, and decide whether
the result demonstrates retained-causal-knowledge transfer or merely a
higher score — without relying on the agent's narrative.
