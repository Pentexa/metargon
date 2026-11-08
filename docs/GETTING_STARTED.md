# METARGON — Getting Started

This guide gets you from a fresh clone to running the test suites, benchmarks, and Academy commands. All commands are verified against the current repository entry points and run from the **repository root** unless stated otherwise.

## Prerequisites

- **Python 3.11+** — the code uses the standard library only; there are no pip dependencies. The historical and v0.9 suites under `python/` still run on Python 3.10+; the Academy layer requires 3.11+ (standard-library TOML loader).
- **Rust toolchain** — for the research kernel (`rust/`, crate `metargon-core`) and the Academy kernel (`rust_academy/`): `cargo +1.88.0` (the CI pins the same toolchain; both crates declare `#![forbid(unsafe_code)]`).
- **Optional** — an `EVREN_API_KEY` (environment variable or git-ignored `.env`, see `.env.example`) for the online EVREN teacher. All EVREN experiments are **offline by construction** and need no key.

Sanity check:

```bash
python --version        # 3.11+
cargo +1.88.0 --version
```

## 1. Run the test suites

```bash
# full Python suite (historical + v0.9–v0.11) — from the repo root
python -m unittest discover -s python -v

# Academy suite
python -m unittest discover -s academy/tests

# Independent generalization audit suite
python -m unittest discover -s generalization_audit/tests -t . -v

# Repository hygiene (merge markers, Markdown links, Python compile)
python scripts/check_repo.py

# Rust research kernel
cd rust
cargo +1.88.0 test
cd ..

# Rust Academy kernel
cd rust_academy
cargo +1.88.0 test
cd ..
```

## 2. Run a benchmark

The quickest full-pipeline example is the v0.11 benchmark (8 seeds × 100 episodes, all seven conditions):

```bash
python python/benchmark_v11.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --json results/v11/summary.json
python python/v11_postprocess.py            # renders results/v11/benchmark_table_v11.md
```

See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for every benchmark (v0.9, v0.10, v0.11, family H, ablations, memory samples, audit families R/L) and its exact command line.

## 3. Academy CLI

The Academy is operated through a single Python module: `python -m academy.cli.main <command>`.

```bash
# compute and persist the immutable task splits
python -m academy.cli.main splits

# measure difficulty bins and build the curriculum schedule
python -m academy.cli.main calibrate

# tiny end-to-end smoke experiment (fast; also run in CI)
python -m academy.cli.main smoke

# train policies on the matched episode stream (all policies by default)
python -m academy.cli.main train

# evaluate trained policies on val/test pools
python -m academy.cli.main eval --kind test
python -m academy.cli.main eval --kind val

# evaluate on the held-out OOD pool
python -m academy.cli.main ood

# list stored checkpoints / verify checkpoint-restore determinism
python -m academy.cli.main checkpoints
python -m academy.cli.main verify-restore
```

Academy runs are deterministic under a fixed hash seed. Pin it for cross-process reproducibility:

```bash
# PowerShell
$env:PYTHONHASHSEED='0'
# bash / POSIX
PYTHONHASHSEED=0
```

## 4. EVREN teacher commands

All EVREN commands are offline by construction (zero API calls) and need no key. The online commands (`evren-smoke`, `evren-pilot`, `evren-train`, `evren-status`) require `EVREN_API_KEY` and are quota-guarded.

```bash
# v0.2 matched-budget adaptive-curriculum comparison (offline)
python -m academy.cli.main evren-curriculum

# quick sanity run (2 conditions, 16 episodes)
python -m academy.cli.main evren-curriculum --quick

# root-cause diagnosis of the v0.1 null result (offline)
python -m academy.cli.main evren-diagnose

# online smoke test — exactly one authorized connection check (needs key)
python -m academy.cli.main evren-smoke

# reproducible 10-task pilot (needs key)
python -m academy.cli.main evren-pilot
```

## 5. Rust kernels

```bash
cd rust
cargo +1.88.0 test                # metargon-core (v0.9–v0.11 kernels)
cargo clippy --locked --all-targets -- -D warnings
cd ..

cd rust_academy
cargo +1.88.0 test                # metargon-academy-kernel
cd ..
```

## 6. What to read next

- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) — how the agent works.
- [RESEARCH_INDEX.md](RESEARCH_INDEX.md) — all versions and findings.
- [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) — full reproduction commands.
- [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) — experimental discipline.
