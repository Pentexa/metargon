# METARGON v0.9 — Compositional world models, causal adaptation, and OOD transfer

**Date:** 7–8 November 2026.
**Status:** Exploratory synthetic study.
**Main finding:** METARGON v0.9 learns genuine reusable object–action–outcome
relationships from interaction (success 31–54% across conditions vs ~0.3–0.8%
for random/episodic/Q-learning), but **does not outperform a strong model-based
Bayesian planner that is given the declarative family** (Bayes 65–81%).
The compositional approach is *measurably better than model-free baselines*
and *measurably worse than the privileged-family Bayes baseline* — a negative
result for the hypothesis that hypothesis-composition alone closes the gap to a
family-knowing model, and a positive result for transfer of learned mechanism
knowledge to unseen configurations (novel condition: 31.25% vs 0.6% random).

Main comparison (8 seeds × 100 episodes × 7 conditions; full numbers in
`results/v09/summary.json` and `results/v09/benchmark_table.md`):

| Condition | METARGON | Bayes | Δ (M−B) | 95% CI (pp) | Oracle |
|---|---:|---:|---:|---:|---:|
| stable | 35.12% | 77.50% | −42.38 | [−68.12, −17.00] | 100% |
| shift | 46.38% | 80.62% | −34.25 | [−51.50, −15.87] | 100% |
| rapid_shift | 40.75% | 76.88% | −36.12 | [−55.75, −16.12] | 91.25% |
| iid | 53.62% | 79.00% | −25.37 | [−31.12, −19.62] | 78.88% |
| novel | 31.25% | 78.50% | −47.25 | [−68.50, −23.62] | 93.75% |
| noisy_obs | 33.75% | 73.12% | −39.38 | [−52.12, −28.25] | 98.38% |
| tight_budget | 45.00% | 64.88% | −19.88 | [−39.12, +1.38] | 100% |

## Research question

Can a resource-constrained agent learn **reusable relationships between objects, actions, and outcomes** — moving object A onto plate P1 opens door D1 — and then apply those relationships in previously unseen environments, without relying solely on memorized successful actions?

## Summary of phases

| Phase | Deliverable | File(s) |
|---|---|---|
| A — repository audit | baseline audit + v0.8 benchmark reproduction (bit-exact) | `docs/V09_BASELINE_AUDIT.md`, `results/repro_v08_30seeds.json` |
| B — independent environment | "Mechanism Lab" simulator (7×5 grid, 2 objects, 2 plates, 2 switches, 2 doors, delayed effects, noisy observations, procedural hidden rules) | `python/lab_env.py` |
| C — compositional causal memory | bounded hypothesis store with hypothesis/predicted/verified/contradicted statuses, deterministic eviction | `python/compositional_memory.py`, `rust/src/v09.rs` |
| D — adaptive planning | bounded planner: retrieve hypotheses, probe interventions, follow plans, protect load-bearing atoms, update on prediction failure | `python/metargon_v09.py` |
| E — comparisons | random, episodic memory, tabular Q-learning, model-based Bayes (particle filter over the declared family), privileged oracle | `python/baselines_v09.py` |
| F — resource efficiency | CPU time, peak memory (tracemalloc), planner ops/decisions, memory budget, model size | `python/benchmark_v09.py` |
| G — verification | test suite covering determinism, leakage, fairness, budgets, hypothesis updates, contradictions, rule changes, OOD | `python/test_*.py` (v0.9) |

## Environment (Phase B)

The "Mechanism Lab" is **structurally independent** of the v0.8 five-position workshop:

- 7×5 grid with two movable objects (A, B) pushed Sokoban-style; two pressure plates (P1, P2); two switches (S1, S2); two doors (D1, D2) blocking movement until hidden conjunctive conditions hold; optional one-step delayed effects per door; optional noisy observation channel (q = 0.08).
- Hidden rules are sampled from a **declared generative family**: D1 conditions over left-room components (P1/S1), D2 conditions over any component (S2 sits in the middle room, reachable after D1 opens). Every configuration in the family was verified **ground-truth solvable** (216/216 after the solvability redesign; earlier unsolvable interlock configs were excluded by construction, not by tuning).
- Training / validation / held-out test pools are disjoint config draws (129 / 43 / 44). The `novel` condition switches to the held-out pool at the midpoint.
- The agent receives only the public observation stream. No hidden rule identifier, no latent probabilities, no diagnostics. The oracle comparator is privileged and excluded from deployable rankings.

## Compositional causal memory (Phase C)

- Hypotheses are conjunctive antecedents (grounded atoms over plates/switches) mapped to a door outcome. They are generated from co-occurrence contexts (singletons + pairs, bounded per event).
- Statuses: `hypothesis` → `predicted` (support ≥ 1) → `verified` (support ≥ 3); contradicting evidence (antecedent continuously true for 3 steps without the predicted effect) increments `contradiction`; ≥ 2 contradictions → `contradicted` (unusable).
- Memory is hard-capped (128). Eviction is deterministic: lowest score (confidence + status bonus − age penalty); ties broken by older `last_update`/id.
- The Rust kernel (`rust/src/v09.rs`, `#![forbid(unsafe_code)]`, no steady-state allocation) reimplements this memory independently, with 4 unit tests.

## Adaptive planning (Phase D)

Each decision, METARGON:

1. syncs belief from the observation; protects atoms that currently satisfy a usable hypothesis for any door (plus short causal-credit windows after door events);
2. if a door is open and not yet passed, plans through it (static far-region test);
3. otherwise attempts a goal plan (bounded BFS, complete within expansion budget);
4. otherwise evaluates candidate probes: hypothesis-backed antecedents (by confidence), then untried single interventions, then conjunctive pairs — cheapest feasible plan first, respecting protected atoms;
5. executes one step of the chosen plan and repeats until the plan is exhausted or invalidated by a door event (which clears probe memory).

Notable failure modes found and fixed during development (documented for scientific transparency):

- **State aliasing bug** in the shared transition model (`NamedTuple` inner-dict references): planner/particle forks mutated shared door state. Fixed with deep copies (`clone_ws`).
- **Cross-door interlock deadlocks** in the config space: excluded by the solvability-guaranteed family design.
- **D1 condition loss**: the planner's shortest path destroyed the very pre-conditions keeping a door open (pushing the object off the plate). Fixed with protected-atom planning (including recent-event causal credit, since the cause often appears only *after* a transition).
- **Cold-start discovery blindness**: without hypotheses, distance-ordered probes explored far mechanisms before near ones; hypotheses now gate probe selection once learned.

Computational budgets (pre-registered): BFS expansions per plan ≤ 700, plans per decision ≤ 3 (+2), memory ≤ 128 hypotheses, plan cache bounded.

## Comparisons (Phase E)

| Policy | Description |
|---|---|
| random | uniform valid actions |
| episodic | memorize successful trajectories by state signature, replay (bounded FIFO) |
| qlearning | tabular Q-learning, ε-greedy (α=0.3, γ=0.95, ε0=0.35 decaying) |
| bayes | particle filter (k=48) over the **declared family**; posterior sampling; ground-truth planning under the sampled hypothesis — a strong model-based baseline |
| metargon | v0.9 (above) |
| oracle | privileged: knows the true config; diagnostic only |

Fairness: identical pre-generated worlds (same per-episode noise streams), identical action budgets, identical episode counts per condition, same seeds for all policies, paired bootstrap CIs for differences. No policy is tuned on the held-out test set.

## Resource efficiency (Phase F)

Measured per policy run: wall CPU time, peak Python allocations (`tracemalloc`), and for METARGON: planner BFS expansions, probe-plan calls, memory size/evictions at end. Energy was **not** measured (no appropriate hardware); operation counts are not equated with energy. Model size: METARGON's persistent memory is ≤ 128 hypotheses; the Rust kernel has no steady-state heap allocation.

## Verification (Phase G)

New tests (all passing): environment topology/mechanics (25), memory lifecycle (9), METARGON agent budgets/learning (9), baselines (9), benchmark harness determinism/fairness (5). Historical v0.1–v0.8 tests still pass (82/82). Rust crate tests pass (26/26).

## Known limitations

## Resource usage (Phase F, measured)

| Policy | CPU (stable, 100 ep / seed) | Peak Python memory (stable, 20 ep) |
|---|---:|---:|
| random | 2.04 s | 0.01 MB |
| episodic | 2.04 s | 0.02 MB |
| qlearning | 2.26 s | 0.01 MB |
| bayes | 27.71 s | 0.72 MB |
| metargon | 32.54 s | 0.17 MB |
| oracle | 0.31 s | 0.11 MB |

METARGON planner cost (per run): a few hundred to ~10⁵ BFS expansions per seed
run (recorded in `per_seed_results.json` as `metargon.planner_ops` /
`metargon.plan_calls`); memory bounded at 128 hypotheses; the Rust kernel has
no steady-state heap allocation. Energy was not measured; operation counts are
not equated with energy.

## Known limitations

- The generative family is known to the Bayesian baseline (a correct model class in standard conditions); METARGON is not given it. The `novel` condition is a held-out draw from the *same* family — the structured-OOD diagnostic (novel family) is implemented in `lab_env.H*_TEMPLATES` but not yet benchmarked.
- **The headline comparison uses 8 paired seeds × 100 episodes** (not 30 × 160): the full 30-seed protocol was attempted twice but the runtime (2–4 h with memory tracing; ~1.5–2.5 h without, on this machine) exceeded practical limits, so the reported run is a smaller but honest subsample. Bootstrap intervals reflect this smaller sample.
- The `iid` oracle at 78.88% shows even the privileged planner fails on some configurations within the 80-step budget (plan-search cap or genuinely long optimal plans) — the oracle is an upper *bound*, not a guarantee.
- METARGON's `noisy_obs` performance (33.75%) degrades most relative to Bayes; noisy readings actively corrupt both event detection and protected-atom inference.
- Success on the generous 80-step budget is not an information-gathering stress test; the tight-budget condition (44 steps) is more discriminative (METARGON 45.00% vs Bayes 64.88%).
- Python reference implementation is single-threaded per seed; the benchmark parallelizes across seeds.
- Energy, latency on edge hardware, and device memory footprints are unmeasured.

## Reproduce

```bash
# Python (project root)
python -m unittest discover -s python -v        # full suite
python python/benchmark_v09.py --seeds 8 --episodes 100 \
    --parallel --workers 6 --no-mem --json results/v09/summary.json
python python/v09_postprocess.py results/v09/summary.json

# Rust
cargo +1.88.0 test      # from rust/
```
