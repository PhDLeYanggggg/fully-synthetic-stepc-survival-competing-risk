# GitHub Reproducibility Guide

This repository is designed to make the code and aggregate synthetic results understandable and reproducible when the fully synthetic data package is available locally.

## Install Dependencies

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate slam-synthetic-institutionalisation
```

## Expected Local Data Layout

The modelling scripts expect:

```text
fully_synthetic_stepC_v1/
```

at the repository root. That folder is not tracked in GitHub.

## Run Debug Analyses

```bash
bash scripts/run_stepC2_debug.sh
bash scripts/run_stepC3_debug.sh
```

Debug mode runs only a small number of repetitions and is intended as a smoke test.

## Run Full Analyses

```bash
bash scripts/run_stepC2_full.sh
bash scripts/run_stepC3_full.sh
```

The full run processes all eight scenarios and all 50 repetitions per scenario.

## Run Tests

```bash
python -m pytest -q
```

The tests check import availability and expected source-code paths. They do not run the full analyses.

## Verify Safe Commit Scope

Before committing or pushing:

```bash
git status
find . -type f -size +20M -print
git diff --cached --name-only
git diff --cached --stat
```

Do not use `git add .` unless the file list has been inspected and confirmed safe.
