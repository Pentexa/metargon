# METARGON v0.8 — Learning a Changing World

> *Intelligence is not just predicting a result. It is adapting when the world changes.*

**METARGON** is a crow-inspired, resource-conscious intelligence architecture **research project**. It was developed as **CORVID** from experimental versions v0.1–v0.7 and renamed at v0.8. The earlier source files, experiment reports and raw benchmark records are preserved under their original identifiers to retain scientific provenance.

**Status:** Exploratory *synthetic* research. No crow-brain replication, new neural architecture, physical-energy reduction, artificial general intelligence, or state-of-the-art claim has been demonstrated. The v0.8 learned controller does **not** outperform a strong Bayesian baseline.

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

## Academy (v0.1) — student-agent framework

`academy/` is the current research line: a reproducible continual-learning
platform with a student agent, curriculum manager, bounded experience
memory, frozen evaluation pools, and crash-safe checkpoints (stdlib only).

```bash
python -m academy.cli.main splits            # compute/persist immutable task splits
python -m academy.cli.main smoke             # tiny end-to-end smoke experiment
python -m academy.cli.main calibrate         # measured difficulty schedule
python -m academy.cli.main train             # matched training across policies
python -m academy.cli.main eval --kind test  # frozen-pool evaluation
```

## EVREN LLM integration — curriculum teacher (not the decision brain)

EVREN-hosted language models act strictly as **curriculum teachers**: they
propose structured procedural tasks (environment wiring only), which the
Academy validates against strict schemas, tests with an independent solver,
and only then feeds to the student. Teachers never see student internals,
never supply solutions/hints, and are never consulted during evaluation.

- Provider: OpenAI-compatible, base URL `https://evren-llmapi.ssyz.org.tr/v1`,
  API key read from `EVREN_API_KEY` (or git-ignored `.env`). Never hardcoded.
- Models: primary teacher **GLM-5.3**, secondary **DeepSeek V4.1 Flash**
  (exact IDs resolved from the provider's `/models` discovery endpoint).
- Budget: default **10 teacher calls per run**; unlimited usage is
  explicitly disabled unless `allow_unlimited = true` is set.

```bash
# one authorized connection smoke test (exactly one API call)
python -m academy.cli.main evren-smoke

# reproducible 10-task pilot (GLM-5.3), validates and persists everything
python -m academy.cli.main evren-pilot

# same pipeline but purely offline (no EVREN needed)
python -m academy.cli.main evren-offline --n 10

# train baseline vs teacher-condition, evaluate transfer + forgetting
python -m academy.cli.main evren-train

# inspect approved tasks, rejected candidates, token usage, API errors
python -m academy.cli.main evren-status
```

All EVREN artifacts land in `results/academy_v01/evren/` (approved tasks with
provenance, rejected candidates, usage log, experiment summaries). See
`docs/EVREN_INTEGRATION_REPORT.md` for the full method, verification, and
results.

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
