# Contributing to METARGON

Thanks for your interest in METARGON. This repository is an **exploratory
research project**: it studies how a bounded agent can learn reusable
object–action–outcome relationships and apply them in unseen configurations.
Contributions — code, experiments, documentation, or careful peer review —
are welcome as long as they respect the research integrity rules below.

## Table of contents

- [Reporting issues](#reporting-issues)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style & conventions](#code-style--conventions)
- [Commit message guidelines](#commit-message-guidelines)
- [Branching & pull requests](#branching--pull-requests)
- [Research integrity](#research-integrity)
- [Communication](#communication)

## Reporting issues

### Before opening an issue

- Search existing issues (open and closed) to avoid duplicates.
- State clearly which area the issue concerns: `python/`, `rust/`,
  `academy/`, `docs/`, or `results/`.

### Bug reports

Include:

1. What you did (the exact command or code path).
2. What you expected to happen.
3. What actually happened (full traceback/error output).
4. Environment: OS, Python version (3.11+ required; the `python/` suites
   still run on 3.10+), Rust toolchain
   (cargo `+1.88.0` for the `rust/` crate).
5. Whether the bug reproduces in a fresh clone.

If an experiment produced unexpected numbers, attach or reference the
relevant seed/config and the raw output in `results/` — do not paste only
a summary.

### Feature / experiment requests

This is a research repository, so feature requests are framed as **research
questions**. Explain:

- The hypothesis the feature would test.
- How it fits the roadmap in `docs/ROADMAP_V09.md`.
- A suggested comparison (e.g. against the existing baselines in
  `python/baselines_v09.py`).

## Development setup

### Prerequisites

- **Python 3.11+** — the Python code uses **the standard library only**;
  there are no pip dependencies. The historical and v0.9 suites under
  `python/` still run on Python 3.10+; the Academy layer requires 3.11+
  for its standard-library TOML configuration loader.
- **Rust toolchain** for the research kernel: `cargo +1.88.0` (crate
  `metargon-core`). The crate declares `#![forbid(unsafe_code)]`.

### Getting started

```bash
# fork + clone, then:
cd metargon
git remote add upstream https://github.com/Pentexa/metargon.git
git fetch upstream
git checkout -b my-feature upstream/main
```

### Environment variables

- API keys (e.g. `EVREN_API_KEY`) are read from `EVREN_API_KEY` or the
  git-ignored `.env` file. Copy `.env.example` to `.env` and fill in local
  values. **Never commit `.env` or any real key.**

## Running tests

Run from the project root.

### Python

```bash
python3 -m unittest discover -s python -v
```

Full benchmark (v0.9):

```bash
python3 python/benchmark_v09.py --seeds 30 --episodes 160 --parallel --workers 6 --json results/v09/summary.json
python3 python/v09_postprocess.py results/v09/summary.json
```

Academy CLI:

```bash
python -m academy.cli.main smoke
python -m academy.cli.main eval --kind test
```

### Rust

```bash
cd rust
cargo +1.88.0 test
```

Keep `#![forbid(unsafe_code)]` intact — the crate must compile with unsafe
code forbidden.

**Before opening a PR:** run the full Python suite *and* the Rust tests, and
make sure your branch is up to date with `main`.

**CI:** GitHub Actions (`.github/workflows/ci.yml`) runs the Python suites,
the Rust crate tests with `cargo clippy -D warnings`, and repo hygiene checks
(`python scripts/check_repo.py`: merge markers, Markdown links, Python
compile). Python jobs pin `PYTHONHASHSEED=0`; Academy runs are deterministic
under a fixed hash seed, but internal hypothesis bookkeeping may differ
across hash seeds (summary metrics reproduce).

## Code style & conventions

- **Python**: standard library only. No new third-party dependencies without
  a strong scientific justification.
- **Rust**: keep `#![forbid(unsafe_code)]`; match the existing module layout
  (`rust/src/v08.rs`, `rust/src/v09.rs`, ...) and naming.
- Write clear docstrings/comments explaining *why*, not just *what*.
- Keep generated/raw benchmark data in `results/` with stable, descriptive
  filenames; never overwrite an existing result file silently.
- Do **not** edit or rename archived modules and reports from earlier
  versions (v0.1–v0.8). They are preserved for scientific provenance — see
  the README note on CORVID/METARGON history.

## Commit message guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>
```

- **Type**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`,
  `build`, `ci`.
- **Scope** (optional): `python`, `rust`, `academy`, `docs`, `results`,
  `evren`.
- Write the summary in the imperative mood, lowercase, ≤ 72 characters.
- Use the body to explain *what* and *why*, and reference related issues
  (e.g. `Closes #12`).

Examples:

```
docs: add contributing guidelines
test(python): cover v09 eviction edge cases
feat(rust): implement bounded hypothesis store in v09
```

## Branching & pull requests

### Branch naming

- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `refactor/<short-description>`
- `research/<experiment-name>` for experiment branches

### Workflow

1. Create a branch from the latest `main` (or `upstream/main`).
2. Keep the branch focused: one logical change per PR.
3. Run the tests (see above) before pushing.
4. Open a pull request against `main`. In the description include:
   - **Summary** — what the change does.
   - **Motivation** — the research question or bug it addresses.
   - **Testing** — which commands you ran and the results.
   - **Related issues** — links to issues/PRs.
5. Address review feedback in follow-up commits on the same branch.
6. Never force-push a shared branch; if `main` moved, merge/rebase it in
   instead.

## Research integrity

This project takes scientific honesty seriously. Please observe these rules:

- **No overclaiming.** METARGON is exploratory *synthetic* research. Do not
  claim crow-brain replication, new neural architectures, energy reduction,
  AGI, or state-of-the-art results. These are explicitly out of scope.
- **Report negative results.** A null/negative outcome is a valid, valuable
  contribution — the README documents several. Never hide or massage
  unfavorable comparisons.
- **Fair comparisons.** Use matched seeds and identical generators across
  policies; do not tune hyperparameters after seeing results without
  labeling that as post-hoc.
- **Pre-registration.** Where feasible, pre-register hyperparameters and
  evaluation schemes (see `docs/RESEARCH_PROTOCOL.md`) and document any
  deviations.
- **Preserve provenance.** Never alter raw benchmark records or historical
  files to make the story "cleaner". Add new files rather than editing old
  ones.

## Communication

- **Issues** — bug reports and feature/research requests (see above).
- **Pull requests** — code and documentation contributions.
- **Discussion** — use GitHub Discussions or the issue tracker for open
  research questions and design conversations.
- Significant unmerged topics and planned experiments are tracked in
  `docs/ROADMAP_V09.md`.

Thank you for contributing to a careful, honest research effort.
