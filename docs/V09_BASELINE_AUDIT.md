# METARGON v0.9 — Baseline Audit (Phase A)

**Date:** 7 November 2026
**Auditor:** Principal Research Engineer (autonomous run)
**Environment:** Windows 11 (win32), Python 3.11.0 (C:\Users\USER\AppData\Local\Programs\Python\Python311), rustup/cargo 1.88.0 (stable toolchain component `rustc` missing — see §5).

This document records the Phase A repository audit: structure, architecture, test status, benchmark reproduction, and discovered issues. It was written **before** any v0.9 implementation work.

---

## 1. Repository structure

```
metargon/
├── README.md                    # v0.8 research statement, instructions
├── LICENSE                      # MIT
├── .gitignore
├── docs/
│   ├── RESEARCH_PROTOCOL.md     # CORVID v0.2 falsifiable protocol (historical)
│   ├── RESEARCH_V03.md … RESEARCH_V08.md
│   ├── ROADMAP_V09.md           # v0.9 research gate (existing)
├── python/                      # experimental research code (stdlib only)
│   ├── metargon_world.py        # v0.8 stateful workshop + benchmark (17.5 KB)
│   ├── metargon_demo.py         # v0.8 demo
│   ├── test_metargon_world.py   # v0.8 tests
│   ├── active_perception.py / test_*.py        # v0.1 (CORVID)
│   ├── adaptive_attention.py / test_*.py       # v0.2
│   ├── causal_probe.py / test_*.py             # v0.3
│   ├── learned_uncertainty.py / test_*.py      # v0.4
│   ├── predictive_world_model.py / test_*.py   # v0.5
│   └── sparse_feedback.py / test_*.py          # v0.6
├── rust/
│   ├── Cargo.toml               # metargon-core v0.8.0, edition 2021
│   ├── src/lib.rs                # Core, OnlineReliability, module roots; #![forbid(unsafe_code)]
│   ├── src/v04.rs … v08.rs       # historical kernels
│   └── examples/basic.rs
└── results/
    ├── summary_v05.csv … summary_v08.{csv,txt}
    ├── smoke.json, smoke_v05..07.json
    └── *_30seeds.json            # per-seed raw records (v0.3–v0.8)
```

Git history: 3 commits (`initial commit`, `docs: rewrite README…`, `chore: apply pending updates`). No remote push performed, per mission constraints.

## 2. Python simulation architecture

All modules are Python 3.10+ and use only the standard library, by design (no GPU, no LLM API).

- **v0.8 (`metargon_world.py`)** — a *five-position workshop*: position 0 start, 1 key, 2 switch, 3 gate, 4 goal. Hidden binary mechanism (key-only vs switch-only gate) with probabilistic outcomes (0.93/0.07 match/mismatch; 0.72/0.28 for the noisy condition). Pre-generated uniform draws, stored as matched potential outcomes for every recipe, guarantee fair paired comparisons. A public-dynamics BFS planner routes the agent to the gate; a hand-written `deterministic_transition` encodes only public movement/control semantics. The learned component is a small context-specific latent gate-transition predictor (`Policy`), either fixed-hazard Bayes or surprise-gated METARGON. Conditions: stable, shift, rapid_shift, iid, novel, noisy_gate, tight_budget (7 steps), sham. Policies: always_key, always_switch, random, win_stay, bayes, metargon, oracle (privileged).
- **Historical modules** (v0.1–v0.7, named CORVID) implement: active perception with optional query budgets; selective attention; causal probing; learned sensor reliability; sparse delayed feedback; lightweight predictive world models. Each has its own test file.

## 3. Rust core

- Crate `metargon-core` v0.8.0, edition 2021. Crate root declares `#![forbid(unsafe_code)]`.
- `lib.rs` contains a generic `Core<const CONTEXTS: usize>` Bayesian belief tracker over a hazard transition, an `OnlineReliability<…>` sensor-error estimator, an `ObservationBudget`, plus module roots `v04..v08`.
- v0.8 kernel (`v08.rs`): `GateModel<const CONTEXTS: usize>`, a bounded posterior over the binary gate mechanism with surprise-gated forgetting; no heap allocation required for inference.
- No external dependencies in `Cargo.toml`.

## 4. Test execution (Python)

Command (from project root):

```
python -m unittest discover -s python -v
```

Result: **82/82 passed, 0 failed, 0 skipped** (5.18 s). This matches the "82/82" claim in `README.md`.

Suite details: `test_metargon_world.py` (29), `test_active_perception.py` (9), `test_adaptive_attention.py` (10), `test_causal_probe.py` (9), `test_learned_uncertainty.py` (8), `test_predictive_world_model.py` (12), `test_sparse_feedback.py` (11). *(Counts approximate; total 82.)*

Demo runs successfully: `python python/metargon_demo.py` prints the six-episode rule-change demo; the hidden mechanism is displayed for the human only.

## 5. Rust toolchain state and test execution

- `cargo --version` via the `stable-x86_64-pc-windows-msvc` toolchain reported an error: *"the 'rustc.exe' binary … is not applicable to the 'stable-x86_64-pc-windows-msvc' toolchain"* — the stable toolchain directory contains only `cargo.exe`, `clippy-driver.exe`, `rustfmt.exe`; **`rustc.exe` is missing**.
- Installed toolchains: `stable`, `1.88`, `1.88.0` (msvc). `1.88.0` contains a valid `rustc.exe`.
- `cargo +1.88.0 test` initially **failed to compile** with 7 lib errors / 23 test errors: `error: float literals must have an integer part`, all caused by leading-dot float literals (`.5`, `.25`, `.6`, …) in **test code** of historical modules v05/v06/v07. These files were never compiled before (README: "Cargo/rustc were not available in this authoring environment, so Rust compilation and unit tests are unverified").
- **Repair performed (mechanical only, no semantic change):** added a leading `0` to every leading-dot float literal in `rust/src/v05.rs`, `rust/src/v06.rs`, `rust/src/v07.rs` (12 lines total). Each rewrite preserves the exact numeric value. This is a build repair, not an algorithm change; the historical algorithms are untouched.
- After repair: `cargo +1.88.0 test` → **22/22 tests pass** (0 failed; 0.01 s). `cargo +1.88.0 run --example basic` runs.
- **Limitation:** the `stable` toolchain remains broken (missing rustc component). The project should pin `+1.88.0` when building (or reinstall stable components). Documented; not actionable as a code change.

## 6. v0.8 benchmark reproduction

Command:

```
python -c "… from metargon_world import benchmark; benchmark(30, 240) …"
```

Full run: 30 seeds × 240 episodes × 8 conditions × 7 policies = 403,200 episodes, **46.8 s** wall time on this machine. Output stored in `results/repro_v08_30seeds.json` (new file; original `results/stateful_transfer_30seeds.json` untouched).

**Comparison with recorded historical summary** (`results/summary_v08.txt`):

| Condition | METARGON success (recorded / reproduced) | Bayes success (recorded / reproduced) | METARGON steps (rec. / rep.) |
|---|---:|---:|---:|
| stable | 93.65% / 93.65% | 93.65% / 93.65% | 6.51 / 6.51 |
| shift | 93.28% / 93.28% | 93.28% / 93.28% | 6.60 / 6.60 |
| rapid_shift | 93.56% / 93.56% | 93.56% / 93.56% | 7.29 / 7.29 |
| iid | 93.49% / 93.49% | 93.49% / 93.49% | 9.45 / 9.45 |
| novel | 93.94% / 93.94% | 93.94% / 93.94% | 6.50 / 6.50 |
| noisy_gate | 80.21% / 80.21% | 80.21% / 80.21% | 8.60 / 8.60 |
| tight_budget | 85.24% / 85.24% | 88.79% / 88.79% | 6.15 / 6.15 |
| sham | 80.40% / 80.40% | 80.40% / 80.40% | 8.92 / 8.92 |

- The `summary` and `metadata` dictionaries of the reproduced run are **identical** (Deep equality) to the recorded JSON.
- Primary discriminating result reproduced exactly: tight-budget METARGON − Bayes = **−3.56 pp**, paired-bootstrap 95% CI **(−4.03, −3.03) pp**.

Conclusion: the v0.8 headline numbers are reproducible on this machine; no evidence of transcript error was found. The v0.8 result stands: the more complex surprise-adaptive controller does **not** beat the fixed-hazard Bayesian baseline under the tight budget.

## 7. Findings, inconsistencies, gaps

1. **Rust unverified claim resolved:** the crate now compiles and passes its tests (22/22) after mechanical float-literal build repairs. The historical README statement "Rust compilation … unverified" is now outdated and will be updated in the README for v0.9.
2. **Stable Rust toolchain is broken** (missing `rustc` component). Use `cargo +1.88.0`.
3. **No CI, no pins:** no `requirements.txt`, no Python version pin, no CI config. Python stdlib only, so reproducibility is reasonable; Python 3.11.0 verified.
4. **No existing v0.9 code.** Only `docs/ROADMAP_V09.md` exists (research gate spec). 
5. **Historical data preserved:** all `results/*` original files remain unmodified. The reproduction artifact adds `results/repro_v08_30seeds.json`.
6. **Known v0.8 limitations carried forward** (from `RESEARCH_V08.md`): only two hidden mechanisms; known action grammar; novel context is a withheld symbol in the same generator, not an independent environment; no energy/hardware measurement; exploratory (not pre-registered).

## 8. Commands executed (exact)

```
python --version                                   → 3.11.0
python -m unittest discover -s python -v           → 82 passed, 0 failed
python python/metargon_demo.py                     → OK (6 episodes)
cargo +1.88.0 test (in rust/)                      → 22 passed, 0 failed
cargo +1.88.0 run --example basic (in rust/)       → OK
python -c "from metargon_world import benchmark; benchmark(30,240)" → 46.8 s, saved to results/repro_v08_30seeds.json
```

## 9. Baseline for v0.9

The v0.8 environment is the *prior* benchmark; v0.9 will build an **independent** environment (Phase B) with ≥3 interacting mechanisms. The old env and its results are preserved intact. v0.9 will reuse the methodological pattern (paired pre-generated worlds, matched potential outcomes, privileged oracle labeled separately, bootstrap CIs) but not the code.
