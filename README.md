# METARGON v0.9 — Compositional World Models & Out-of-Distribution Transfer

> *Intelligence is not just predicting a result. It is adapting when the world changes — and reusing what it learned in worlds it has never seen.*

**METARGON** is a crow-inspired, resource-conscious intelligence architecture **research project**. It was developed as **CORVID** from experimental versions v0.1–v0.7 and renamed at v0.8. The earlier source files, experiment reports and raw benchmark records are preserved under their original identifiers to retain scientific provenance.

**Status:** Exploratory *synthetic* research. No crow-brain replication, new neural architecture, physical-energy reduction, artificial general intelligence, or state-of-the-art claim has been demonstrated. v0.9 tests whether a bounded agent can learn reusable object–action–outcome relationships and apply them in unseen configurations. Full comparison, including any negative results, is in `docs/RESEARCH_V09.md` and `results/v09/`.

## What v0.9 adds

- **Independently written "Mechanism Lab" environment** (`python/lab_env.py`): 7×5 grid, two movable objects, two pressure plates, two switches, two door barriers, optional delayed effects and a noisy observation channel. Hidden wiring is sampled from a declared generative family (all configs verified solvable); training/validation/held-out test pools are disjoint.
- **Compositional causal memory** (`python/compositional_memory.py`, `rust/src/v09.rs`): bounded hypothesis store with explicit hypothesis → predicted → verified → contradicted lifecycle and deterministic eviction.
- **Adaptive bounded planner** (`python/metargon_v09.py`): retrieves hypotheses, probes interventions, protects load-bearing atoms, and follows plans until invalidated by a door event.
- **Strong comparisons** (`python/baselines_v09.py`): random, episodic memory, tabular Q-learning, a model-based Bayesian particle-filter planner, and a labeled privileged oracle (excluded from deployable rankings).
- **Measured resource usage**: CPU time, peak memory, planner operations per decision, memory budget. Energy is unmeasured.

## Core benchmark (v0.9, 8 paired seeds × 100 episodes × 7 conditions)

| Condition | METARGON | Bayes | Random | Oracle |
|---|---:|---:|---:|---:|
| stable | 35.1% | 77.5% | 0.3% | 100.0% |
| shift | 46.4% | 80.6% | 0.6% | 100.0% |
| rapid_shift | 40.8% | 76.9% | 0.5% | 91.3% |
| iid | 53.6% | 79.0% | 0.4% | 78.9% |
| novel (held-out) | 31.3% | 78.5% | 0.6% | 93.8% |
| noisy_obs | 33.8% | 73.1% | 0.3% | 98.4% |
| tight_budget | 45.0% | 64.9% | 0.0% | 100.0% |

METARGON clearly beats model-free baselines (random/episodic/Q-learning ~0–1%)
and transfers mechanism knowledge to unseen configurations, **but does not
outperform the strong model-based Bayesian planner** (given the declarative
family). This is a negative result for closing the gap to a family-knowing
model via hypothesis composition alone. Full table, bootstrap intervals and
resource measurements: `results/v09/benchmark_table.md`, `docs/RESEARCH_V09.md`.

## Run now — no model download or GPU

Requires **Python 3.10+** and the Python standard library only. Run commands from the project root:

```bash
python3 -m unittest discover -s python -v
python3 python/benchmark_v09.py --seeds 30 --episodes 160 --parallel --workers 6 --json results/v09/summary.json
python3 python/v09_postprocess.py results/v09/summary.json
```

Rust research kernel (crate `metargon-core` v0.9.0, `#![forbid(unsafe_code)]`):

```bash
cd rust
cargo +1.88.0 test
```

---

# METARGON v0.8 — Learning a Changing World (historical)

> *Intelligence is not just predicting a result. It is adapting when the world changes.*

The v0.8 release and its benchmark are preserved below and in `docs/RESEARCH_V08.md`. Its headline negative result — the surprise-adaptive controller trailing the Bayesian baseline under tight budgets — motivated the v0.9 compositional approach.

## What v0.8 adds

This release creates an **independently specified, stateful five-position workshop** rather than extending the old binary-sensor generator:

- An agent can move, pick up or drop a key, flip a switch, test a gate and reach a goal.
- The unknown gate mechanism can change without announcement. Gate outcomes are noisy.
- The agent receives only public state and its *own observed gate result*. It is never given the hidden mechanism or simulator noise.
- A bounded context-specific belief model predicts whether an `open` action will work. Surprise-gated forgetting is a **research hypothesis**, not an established improvement.
- A public-dynamics BFS planner constructs sequences of motor actions; it does not magically discover movement physics or learn full transition dynamics.
- A cost-constrained 7-step test makes bad first choices consequential, unlike the generous-budget condition where switching to the alternate mechanism is almost always possible.
- Comparisons include always-key, always-switch, random, win-stay/lose-switch, fixed-hazard Bayesian, METARGON and a deliberately privileged, *not necessarily optimal* mode-knowing comparator.

The **learned** component models one hidden rule. Calling it a general world model would be inaccurate.

## Run now — no model download or GPU

Requires **Python 3.10+** and the Python standard library only. Run commands from the project root:

```bash
python3 python/metargon_demo.py
python3 -m unittest discover -s python -v
python3 python/metargon_world.py --seeds 30 --episodes 240 --json results/stateful_transfer_30seeds.json
```

The quick demo shows a hidden key-only gate changing to switch-only. Its display prints the evaluator's hidden mode **for the human only**; the agent is not passed that value.

## Core benchmark — most discriminative condition

30 matched seeds, 240 episodes per seed, seven actions maximum per episode. Mean success across seeds:

| Agent | Goal reached |
|---|---:|
| Random first mechanism | 49.24% |
| Win-stay / lose-switch | 85.24% |
| **METARGON** (surprise-adaptive) | **85.24%** |
| Bayesian belief (fixed hazard) | **88.79%** |
| Privileged mode-knowing comparison | 92.88% |

The success-rate delta **METARGON − Bayes is −3.56 percentage points** (paired-seed bootstrap 95% interval: **−4.03 to −3.03 pp**). This interval measures variation in this *synthetic generator*, not reliability on real robots.

For the other seven conditions, both models chose the same mechanism in every tested episode, so their headline success rates were identical. **This is a negative/neutral research result, not evidence that the more complex model is better.**

The full comparison includes stable, shift, rapid shift, IID, novel context, noisy gate, tight budget and noninformative/sham-gate conditions. Full data and limitations: [RESEARCH_V08.md](docs/RESEARCH_V08.md).

## Architecture

```text
User-defined goal (reach position 4)
             |
             v
  Public observation (position, key, switch, gate state)
             |
             v
    METARGON belief state  <--- observed gate result only
  P(hidden switch-rule | context, history)
             |
             v
      Choose mechanism
             |
             v
     Public-dynamics BFS
             |
             v
   Stateful simulator performs actions
             |
             +----> observed result feeds next belief update
```

- `python/metargon_world.py`: independent simulator, public planner, policies, matched benchmark, paired bootstrap.
- `python/metargon_demo.py`: readable six-episode rule-change demonstration.
- `python/test_metargon_world.py`: additional stateful-world correctness and leakage tests.
- `rust/src/v08.rs`: `#![forbid(unsafe_code)]` compatible, constant-size gate-belief kernel, deliberately narrower than the Python simulation.
- `docs/RESEARCH_V08.md`: measured results, limitations and follow-up gates.
- `results/stateful_transfer_30seeds.json`: full 30-seed record for all eight conditions and seven policies.
- `results/summary_v08.csv`: flattened performance summary.
- `results/summary_v08.txt`: readable benchmark output.
- `python/` / `rust/` older modules: historical v0.1–v0.7 experiments, retained unchanged where practical.

## Rust research kernel

```bash
cd rust
cargo test
```

The crate is renamed `metargon-core`, import `metargon_core`. The project declares `#![forbid(unsafe_code)]` in its Rust crate root. **Cargo/rustc were not available in this authoring environment, so Rust compilation and unit tests are unverified.** Python automated tests passed: **82/82**. The Rust core is an illustrative bounded inference implementation, not a Python simulation parity guarantee.

## Research integrity and limitations

- The gate has only **two** hidden possible mechanisms and a **known** public action grammar. This makes a simple memory rule unusually competitive.
- The Bayesian and METARGON models assume gate success rates of 0.93/0.07. The noisy held-out generator uses 0.72/0.28 to expose miscalibration; no policy is given those changed probabilities.
- Multiple strategies have equal success under the generous 17-step budget because the alternative recipe can be tried. **Goal success is then a poor discriminating metric.** Mean steps and first-mechanism accuracy matter more.
- Different policies use the same pre-generated worlds and gate random values for fair paired comparison; observed outcomes still differ when actions differ.
- Novel context is a previously unseen *symbol* in the same generator, **not an independently designed environment or real-world transfer proof**.
- The world planner is hand-specified for public movement and object operations; only the uncertain gate transition is predicted.
- No device sensor, GPU, model API, energy meter or physical robot is involved. No Edge AI efficiency claim is justified yet.

**Next research gate:** multiple latent mechanisms, partially observed mechanics, independently written environments, pre-registered hyperparameters, genuine transition-learning, then Edge-device power/latency tests. Do not increase model complexity before it beats a transparent baseline.

License: MIT. Historical project name CORVID appears in archived source and reports on purpose.
