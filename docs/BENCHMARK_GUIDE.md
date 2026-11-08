# METARGON — Benchmark & Reproduction Guide

This guide documents how to reproduce every benchmark, ablation, evaluation, and test suite in the repository. All commands are verified against the actual CLI implementations and run from the **repository root** unless stated otherwise.

## Conventions

- **Python 3.11+**, standard library only. No virtualenv or pip install needed.
- **Parallelism:** the benchmark scripts accept `--parallel --workers N`; use fewer workers on small machines.
- **Reproducibility:** Academy experiments (and student state-dependent planning, which iterates frozensets) are bit-reproducible only under a fixed hash seed. Pin it before any Academy run:

  ```bash
  # PowerShell
  $env:PYTHONHASHSEED='0'
  # bash / POSIX
  PYTHONHASHSEED=0
  ```

  The summary JSON records whether a run was pinned.
- **Do not run two experiments writing to the same output file.** The research scripts write to `results/vNN/`; Academy runs write to `results/academy_v01/`.

---

## 1. Test suites

```bash
# full Python suite: historical + v0.9–v0.11 (expect ~140+ tests)
python -m unittest discover -s python -v

# Academy suite (expect ~187 tests)
python -m unittest discover -s academy/tests

# Independent generalization audit suite
python -m unittest discover -s generalization_audit/tests -t . -v

# repository hygiene (merge markers, Markdown links, Python compile)
python scripts/check_repo.py
```

Rust:

```bash
cd rust && cargo +1.88.0 test && cd ..
cd rust_academy && cargo +1.88.0 test && cd ..
```

CI (`.github/workflows/ci.yml`) additionally runs `cargo clippy --locked --all-targets -- -D warnings` on both crates.

---

## 2. v0.9 — Mechanism Lab benchmark

```bash
# main paired run (the committed benchmark table used 8 seeds × 100 episodes)
python python/benchmark_v09.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --json results/v09/summary.json

# render the markdown table
python python/v09_postprocess.py results/v09/summary.json
```

Outputs: `results/v09/summary.json`, `results/v09/benchmark_table.md`, `results/v09/per_seed_results.json`.

Original archived run (30 seeds × 240 episodes) and the v0.8 result are preserved as historical artifacts: `results/stateful_transfer_30seeds.json`, `results/v08/`.

---

## 3. v0.10 — robust perception benchmark

```bash
# main paired run, family F, all seven conditions
python python/benchmark_v10.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --json results/v10/summary.json

# structured-OOD family H run (stable,novel)
python python/benchmark_v10.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --family h --conditions stable,novel \
    --json results/v10/summary_h.json

# render tables (reads the default summary; pass a path to render another)
python python/v10_postprocess.py

# offline discovery analysis (exact-rule discovery, false-verified ratios)
python python/discovery_analysis_v10.py

# optional: loop/stall validation run (8 seeds × 20 episodes × 3 conditions)
python python/benchmark_v10.py --seeds 8 --episodes 20 --parallel \
    --workers 4 --conditions stable,noisy_obs,tight_budget \
    --json results/v10/summary_stall.json
```

Ablations (robust belief, cooldowns, fallback, replay) run as policy variants inside the benchmark; the deltas are reported in `results/v10/benchmark_table_v10.md`. The replay ablation used `--json results/v10/summary_rp.json`.

Outputs: `results/v10/summary.json`, `results/v10/benchmark_table_v10.md`, `results/v10/per_seed_results_v10.json`, `results/v10/discovery_analysis.json`, `results/v10/summary_h.json`.

---

## 4. v0.11 — noise-aware learning benchmark

```bash
# main paired run, family F, all seven conditions
python python/benchmark_v11.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --json results/v11/summary.json

# family H run
python python/benchmark_v11.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --family h --conditions stable,novel \
    --json results/v11/summary_h.json

# CPU/memory sample (2 seeds × 20 episodes, stable + noisy)
python python/benchmark_v11.py --seeds 2 --episodes 20 --parallel \
    --workers 4 --conditions stable,noisy_obs \
    --json results/v11/summary_mem.json

# render tables
python python/v11_postprocess.py results/v11/summary.json

# noise diagnostics (confirmed events, false-verified hypotheses, replay counts)
python python/diagnose_noise_v10.py
```

The ablation policies (`-cf`, `-ct`, `-rp`, `-rw`) run inside the benchmark; see `results/v11/benchmark_table_v11.md`.

Outputs: `results/v11/summary.json`, `results/v11/benchmark_table_v11.md`, `results/v11/benchmark_table_v11_h.md`, `results/v11/benchmark_table_v11_mem.md`, `results/v11/diagnostics_noise.json`.

---

## 5. Academy experiments

### 5.1 Core pipeline

```bash
python -m academy.cli.main splits          # persist immutable splits (results/academy_v01/splits.json)
python -m academy.cli.main calibrate       # measured-difficulty calibration + schedule
python -m academy.cli.main smoke           # tiny end-to-end smoke run
python -m academy.cli.main train           # train all policies on the matched stream (480 episodes)
python -m academy.cli.main eval --kind val
python -m academy.cli.main eval --kind test
python -m academy.cli.main ood
python -m academy.cli.main checkpoints
python -m academy.cli.main verify-restore
```

Outputs: `results/academy_v01/train_summary.json`, `eval_test.json`, `eval_val.json`, `eval_ood.json`, `calibration.json`, `learning_curves.csv`, `policies/*.json`.

### 5.2 EVREN curriculum v0.2

```bash
# full adaptive-curriculum comparison (offline, zero API calls)
python -m academy.cli.main evren-curriculum

# quick sanity (2 conditions, 16 episodes)
python -m academy.cli.main evren-curriculum --quick

# null-result diagnosis (uses saved v0.1 policies when present)
python -m academy.cli.main evren-diagnose

# optional: override episode budget / conditions
python -m academy.cli.main evren-curriculum --episodes 200 --conditions uniform,proc_lp
```

Outputs: `results/academy_v01/evren/curriculum_v02_summary.json`, `diagnosis_summary.json`.

### 5.3 Academy Memory v0.3

```bash
# failure-mode separation harness (trains original student on 80 episodes)
python academy/tools/memory_diagnosis.py

# matched-budget variant comparison (offline; pin the hash seed)
PYTHONHASHSEED=0 python academy/tools/memory_experiment.py

# per-variant subprocess runner (robust; resumes from jsonl)
PYTHONHASHSEED=0 python x/run_chunks.py

# merge JSONL into final summary + paired CIs
python academy/tools/memory_merge.py

# tests
python -m unittest academy.tests.test_memory_v03
```

Outputs: `results/academy_v01/memory_v03/memory_v03_summary.json`, `memory_v03_runs.jsonl`, `policy_*.json`.

### 5.4 EVREN online commands (require `EVREN_API_KEY`)

```bash
python -m academy.cli.main evren-smoke       # exactly one authorized connection check
python -m academy.cli.main evren-discovery   # resolve model IDs
python -m academy.cli.main evren-pilot       # reproducible 10-task pilot
python -m academy.cli.main evren-train --p-teacher 0.5
python -m academy.cli.main evren-status      # registry, usage, API errors
```

---

## 6. Independent generalization audit

```bash
# freeze config pools (one-time; committed, re-running regenerates them)
python generalization_audit/freeze.py --family r
python generalization_audit/freeze.py --family l

# unit tests
python -m unittest discover -s generalization_audit/tests -t . -v

# full-scale benchmarks (the committed results used smaller runs; see below)
python generalization_audit/benchmark_generalization.py --family r \
    --seeds 6 --episodes 60 \
    --conditions stable,shift,novel,iid,noisy_obs,cross_family \
    --parallel --workers 5 --json generalization_audit/results/summary_r.json
python generalization_audit/benchmark_generalization.py --family l \
    --seeds 6 --episodes 60 \
    --conditions stable,shift,novel,iid,noisy_obs,cross_family \
    --parallel --workers 5 --json generalization_audit/results/summary_l.json
```

The committed final results used smaller runs (R: 2 seeds × 15 episodes; L: 3 seeds × 20 episodes) because the machine was heavily loaded; bootstrap CIs are wide and findings are directional. Rendered tables: `generalization_audit/results/rendered_tables.md` (regenerate with `python generalization_audit/tools/render_report_tables.py`).

---

## 7. Expected artifacts (all committed)

| Path | Contents |
|---|---|
| `results/v09/benchmark_table.md` | v0.9 full table |
| `results/v10/benchmark_table_v10.md` | v0.10 table + ablations + deltas |
| `results/v11/benchmark_table_v11.md` | v0.11 table + ablations |
| `results/v11/diagnostics_noise.json` | v0.11 noise diagnostics |
| `results/academy_v01/evren/curriculum_v02_summary.json` | curriculum v0.2 pinned run |
| `results/academy_v01/memory_v03/memory_v03_summary.json` | memory v0.3 matched-budget summary |
| `generalization_audit/results/summary_r.json`, `summary_l.json` | audit raw per-seed results |
| `generalization_audit/results/rendered_tables.md` | audit rendered tables |

---

## 8. Troubleshooting

- **Unittest discovery errors:** run from the repository root (as above); the audit suite additionally needs `-t .` for the top-level import root.
- **Hash-seed mismatch:** if two Academy runs differ, both were not pinned to `PYTHONHASHSEED=0` (the summary records the flag).
- **Rust toolchain:** the crates are developed on 1.88.0; `cargo +1.88.0 test` uses that toolchain explicitly. `cargo test --locked` in CI uses the pinned lockfile.
- **Long runs:** benchmarks are CPU-only; reduce `--workers` if the machine is otherwise loaded. Use `--episodes 20 --seeds 2` for quick sanity runs.
