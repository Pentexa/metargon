# METARGON Generalization Audit

Independent evaluation of METARGON v0.10 causal learning, transfer, and
adaptation on **structurally novel hidden-rule families**.

The audit is fully standalone: it does not modify any shared policy,
training algorithm, environment, or evaluation definition under `python/`
or `academy/`. It defines new environments, new baselines (oracles) for
the audited families, a compute-budget governor, offline metrics, frozen
configurations, tests, results, and this documentation under this
`generalization_audit/` tree.

## What is audited

| Subject | Status |
|---|---|
| Family F (v0.9/v0.10 declared family) | reused, unchanged |
| Family H (structured-OOD) | reused, unchanged |
| Family R (**new**, sustained-hold gates) | this audit |
| Family L (**new**, ordered-sequence gates) | this audit |
| v0.10 student (`MetargonV10`) | unchanged |
| Baselines: random, episodic, Q-learning, family-F Bayes | unchanged |
| Academy v0.1/v0.2 + EVREN | audited (report), not re-run |

## New task families

* **Family R — sustained-hold.** A door opens only after its condition
  has held continuously for `HOLD_K=3` consecutive post-states; it stays
  open while the condition holds and closes as soon as it is lost. This
  introduces a temporal-threshold axis absent from F and H (which open
  instantaneously on satisfaction).
* **Family L — ordered-sequence.** A door opens only when two condition
  atoms become true in a specific order within a sliding window (first
  while second is false, then second). The reverse order never opens it.
  This introduces an order/serial dependency absent from the unordered
  conjunctive families F and H.

Both families are solvable-by-construction (independent ground-truth
search), expressed in the same public vocabulary (switches, plates,
objects), and learnable with the unchanged v0.10 agent.

## Frozen configurations

Pools are generated and **frozen before benchmarking**:

```bash
python generalization_audit/freeze.py --family r
python generalization_audit/freeze.py --family l
```

- `generalization_audit/configs/frozen_r.json` — 19 solvable configs
  (train 11 / val 3 / test 5).
- `generalization_audit/configs/frozen_l.json` — 9 solvable configs
  (train 5 / val 1 / test 3).

The benchmark loads these files at runtime and re-verifies that every
frozen configuration is solvable under the family's own dynamics before
running.

## Running the benchmark

```bash
python generalization_audit/benchmark_generalization.py --family r \
    --seeds 6 --episodes 60 --conditions stable,shift,novel,iid,noisy_obs,cross_family \
    --parallel --workers 5 --json generalization_audit/results/summary_r.json

python generalization_audit/benchmark_generalization.py --family l \
    --seeds 6 --episodes 60 --conditions stable,shift,novel,iid,noisy_obs,cross_family \
    --parallel --workers 5 --json generalization_audit/results/summary_l.json
```

Policies (identical worlds per policy, matched seeds/action budgets/noise
streams):

| Policy | Description |
|---|---|
| `metargon10` | unmodified v0.10 student |
| `random` / `episodic` / `qlearning` | model-free baselines |
| `bayes` | family-F Bayesian particle-filter planner (misspecified on R/L) |
| `metargon10-cap` / `bayes-cap` / `qlearning-cap` | matched-compute arm with per-decision expansion cap 120 |
| `oracle` | privileged perfect-knowledge planner for the audited family (diagnostic; excluded from rankings) |

## Tests

```bash
python -m unittest discover -s generalization_audit/tests -v
```

Coverage: family semantics (hold threshold, strict order, window reset),
interface compatibility of all policies on both families, determinism,
frozen-config solvability, disjoint splits, round-trip freeze, no-hidden-
access runtime checks, noise parity, and metric unit tests.

## Report

See `docs/AUDIT_REPORT.md` for the full protocol audit and benchmark
findings (positive and negative).
