# METARGON Documentation

This directory contains the full documentation for the METARGON research project. The repository root [README](../README.md) is the landing page; these documents provide the detail.

## Getting started

| Document | Purpose |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Prerequisites, installation, first commands, and a CLI reference for every entry point in the repository. |
| [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | How the agent works: belief, hypothesis memory, planner, learning loop, and the Academy/EVREN layers. |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | Exactly how to reproduce every benchmark, ablation, evaluation, and test suite in the repository. |
| [RESEARCH_INDEX.md](RESEARCH_INDEX.md) | An index of all research reports — milestones, versions, findings (positive and negative), and limitations. |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | The project's research principles: pairing, baselines, pre-registration, and scientific honesty. |

## Research reports (historical → latest)

The project has grown from **CORVID v0.1–v0.7** to **METARGON v0.8–v0.12.1**, plus the Academy program and three independent audits. Each report is preserved verbatim; the index in [RESEARCH_INDEX.md](RESEARCH_INDEX.md) links them all.

- `RESEARCH_V03.md` … `RESEARCH_V07.md` — CORVID-era experiments (historical)
- `RESEARCH_V08.md` — METARGON v0.8 workshop (negative headline vs Bayesian baseline)
- `RESEARCH_V09.md` — METARGON v0.9 Mechanism Lab (compositional memory vs baselines)
- `RESEARCH_V10.md` — METARGON v0.10 (robust perception, evidence decay, structured-OOD transfer)
- `RESEARCH_V11.md` — METARGON v0.11 (noise-aware learning)
- `RESEARCH_V12.md` — METARGON v0.12 (dual-context learning, temporal ordered-sequence hypotheses)
- `RESEARCH_V121.md` — METARGON v0.12.1 (regression recovery; recent-credit off — latest merged version)
- `ACADEMY_ARCHITECTURE.md` — Academy v0.1 system architecture
- `ACADEMY_RESEARCH_V01.md` — Academy v0.1 research report
- `EVREN_INTEGRATION_REPORT.md` — EVREN LLM teacher integration
- `EVREN_CURRICULUM_V02.md` — curriculum v0.2 adaptive-selection experiment
- `MEMORY_V03.md` — Academy Memory v0.3 (hypothesis-memory saturation, negative transfer)
- `MEMORY_V04.md` — Academy Memory v0.4 (context-scoped evidence; `memory_wiped ≡ original`)
- `MEMORY_V05.md` — Academy Memory v0.5 (three-regime transfer protocol; zero-shot transfer absent)
- `V09_BASELINE_AUDIT.md` — v0.9 baseline audit
- `WORKLOG_V11.md` — v0.11 working notes
- `ROADMAP_V09.md` — original v0.9 scientific gate

The **Independent Generalization Audit** lives in [`generalization_audit/`](../generalization_audit/README.md) with its main report (`generalization_audit/docs/AUDIT_REPORT.md`) and follow-up evaluation-integrity audits (`AUDIT_V02.md`, `AUDIT_V03.md`). Open cross-artifact documentation issues found during the documentation sync are tracked in [`DOC_CONSISTENCY_NOTES.md`](DOC_CONSISTENCY_NOTES.md).

## Repository layout

```text
python/                  B experiments, v0.8 workshop, v0.9–v0.12.1 labs and benchmarks
rust/                    Bounded Rust research kernel (metargon-core, #![forbid(unsafe_code)]; CORVID-era modules + v0.9/v0.10 mirrors)
academy/                 Academy student agents, environments, evaluation, CLI, EVREN
rust_academy/            Bounded Rust kernel for Academy
generalization_audit/    Independent audits (protocol audit, families R/L, evaluation integrity v0.2/v0.3, frozen configs)
configs/                 Pre-registered experiment configuration (academy_v01.toml)
docs/                    This documentation
results/                 Recorded experimental artifacts (raw JSON, tables, summaries)
images/                  Branding (banner, logo)
scripts/check_repo.py    Repository hygiene checker (merge markers, links, compile)
.github/workflows/       CI: hygiene, Python suites, Academy, audit, Rust
```
