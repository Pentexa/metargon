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
# full Python suite: historical + v0.9–v0.12.x (expect 188 tests)
python -m unittest discover -s python -v

# Academy suite (expect 209 tests)
python -m unittest discover -s academy/tests

# Independent generalization audit suite (expect 57 tests)
python -m unittest discover -s generalization_audit/tests -t . -v

# Memory v0.5 tests (subset of the Academy suite)
python -m unittest academy.tests.test_memory_v05

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

## 4b. v0.12 / v0.12.1 — general-causality & regression-recovery harness

One script covers both versions: `benchmark_v12.py` runs METARGON v0.10/v0.11,
v0.12/v0.12.1 (`metargon12` — the merged v0.12.1 defaults with recent-credit
off), ablations (`metargon12-nodc/noad/norc/notp/dst/dic/legacy`), baselines,
and the oracle on the four audit/research families. **Temporal memory is
enabled only for the `metargon12` policy on family L**; every ablation
(including `metargon12-legacy`, which pins recent-credit **on**) runs with
temporal memory off — check [`DOC_CONSISTENCY_NOTES.md`](DOC_CONSISTENCY_NOTES.md)
before interpreting any legacy/`-norc` row for family L.

**Minimum `--episodes` is 4** (`lab_env.generate_worlds` requires it); use
`--seeds 2 --episodes 4` for sanity runs. The committed studies below used
4 seeds × 40 episodes.

```bash
# v0.12 study (4 seeds × 40 episodes per family)
python python/benchmark_v12.py --seeds 4 --episodes 40 --family f \
    --conditions stable,shift,noisy_obs,novel,tight_budget \
    --parallel --workers 6 --json results/v12/summary_f.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family h \
    --conditions stable,novel --parallel --workers 6 --json results/v12/summary_h.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family r \
    --conditions stable,novel --parallel --workers 6 --json results/v12/summary_r.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family l \
    --conditions stable,novel --parallel --workers 6 --json results/v12/summary_l.json

# render paired tables + comparison deltas
python python/v12_postprocess.py results/v12/summary_f.json

# memory/CPU sample (small)
python python/benchmark_v12.py --seeds 2 --episodes 20 --family f \
    --conditions stable,noisy_obs --json results/v12/summary_mem.json

# final v0.12.1 policy runs (recent-credit off): commit artifacts results/v121/final_*.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family f \
    --conditions noisy_obs --parallel --workers 6 --no-mem \
    --policies metargon10,metargon11,metargon12,metargon12-legacy,bayes,oracle \
    --json results/v121/final_f.json
# family h: --conditions stable --policies ... (final_h.json)
# family r: --conditions stable,novel --policies metargon10,metargon11,metargon12,metargon12-legacy,bayes,oracle,random \
#           --json results/v121/final_r.json
# family l: --conditions stable,novel (adds metargon12-notp and episodic/qlearning) → final_l.json

# quick one-line inspection of a final/eval artifact
python python/v121_show.py results/v121/final_l.json
```

Outputs (committed): `results/v12/summary_{f,h,r,l}_small.json` +
`results/v12/table_{f,h,r,l}.md` (v0.12 study — reproduce to fresh
filenames as in the block above so committed artifacts are never
overwritten), `results/v12/summary_mem.json`/`table_mem.md`, and for
v0.12.1: `results/v121/final_*.json`, `results/v121/diag_*.json`,
`results/v121/eval*.json`, `results/v121/table_*.md`. Caveats about which
tables come from which run live in
[`DOC_CONSISTENCY_NOTES.md`](DOC_CONSISTENCY_NOTES.md).

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

### 5.5 Academy Memory v0.4 / v0.5 (offline tools)

```bash
# v0.5 three-regime transfer protocol: frozen zero-shot / adapt / continual
# (2 train seeds × 160 episodes, matched streams; suites results/academy_v01/memory_v05/)
PYTHONHASHSEED=0 python academy/tools/memory_experiment_v05.py
PYTHONHASHSEED=0 python x/run_chunks_v05.py   # per-variant subprocess runner (resumes from JSONL)
python academy/tools/memory_merge_v05.py      # JSONL → summary + paired CIs

# v0.4 context-aware-evidence variant harness (same shape)
PYTHONHASHSEED=0 python academy/tools/memory_experiment.py

# v0.3 failure-mode + variant comparison (see section 5.3)
python academy/tools/memory_diagnosis.py

# tests
python -m unittest academy.tests.test_memory_v05
python -m unittest academy.tests.test_memory_v04
```

Outputs: `results/academy_v01/memory_v04/`, `results/academy_v01/memory_v05/`
(`memory_v05_summary.json`, `memory_v05_runs.jsonl`). Note: the committed
`memory_v05_summary.json` comparison deltas use the reversed (baseline −
candidate) argument order — see [`DOC_CONSISTENCY_NOTES.md`](DOC_CONSISTENCY_NOTES.md)
before interpreting signs.

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
| `results/v12/summary_{f,h,r,l}_small.json`, `results/v12/table_*.md` | v0.12 study (4 seeds × 40 episodes) raw summaries + rendered tables |
| `results/v12/summary_mem.json`, `table_mem.md` | v0.12 CPU/memory sample (2 seeds × 20 episodes) |
| `results/v121/final_{f,h,r,l}.json` | v0.12.1 final policy runs (recent-credit off) |
| `results/v121/diag_*.json`, `eval*.json`, `table_*.md` | v0.12.1 regression-diagnosis runs (see DOC_CONSISTENCY_NOTES.md) |
| `results/academy_v01/evren/curriculum_v02_summary.json` | curriculum v0.2 pinned run |
| `results/academy_v01/memory_v03/memory_v03_summary.json` | memory v0.3 matched-budget summary (signs corrected by Audit v0.2) |
| `results/academy_v01/memory_v04/` | memory v0.4 raw JSONL + summary (signs corrected by Audit v0.3) |
| `results/academy_v01/memory_v05/memory_v05_summary.json`, `memory_v05_runs.jsonl` | memory v0.5 three-regime summary (legacy sign convention, see DOC_CONSISTENCY_NOTES.md) |
| `generalization_audit/results/summary_r.json`, `summary_l.json` | audit raw per-seed results |
| `generalization_audit/results/rendered_tables.md` | audit rendered tables |

---

## 8. Troubleshooting

- **Unittest discovery errors:** run from the repository root (as above); the audit suite additionally needs `-t .` for the top-level import root.
- **Hash-seed mismatch:** if two Academy runs differ, both were not pinned to `PYTHONHASHSEED=0` (the summary records the flag).
- **Rust toolchain:** the crates are developed on 1.88.0; `cargo +1.88.0 test` uses that toolchain explicitly. `cargo test --locked` in CI uses the pinned lockfile.
- **Long runs:** benchmarks are CPU-only; reduce `--workers` if the machine is otherwise loaded. Use `--seeds 2 --episodes 4` for quick sanity runs of the v0.12 harness (the environment requires **episodes ≥ 4**) or `--episodes 20 --seeds 2` for the v0.9–v0.11 harnesses.
