# GitHub Reproducibility Guide

This repository is designed to make the simulation and analysis implementation
auditable. With the exact fully synthetic package available locally, the
analysis stages can be rerun. The code-only public release does not by itself
reproduce the manuscript's numerical results.

## Install Dependencies

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate synthetic-carehome-competing-risk
```

The conda environment includes R with the `survival` and `cmprsk` packages for
Fine-Gray. If dependencies are installed with pip instead, R and those packages
must be installed separately and `Rscript` must be available on `PATH`.

## Expected Local Data Layout

The modelling scripts expect:

```text
fully_synthetic_stepC_v1/
```

at the repository root. That folder is not tracked in GitHub.

The generator also requires an authorised internal aggregate-summary source
that is not included. Public code therefore supports methodological audit and
re-implementation, while exact numerical reproduction requires the locked
local synthetic package.

When generation is authorised, set `STEPC_INTERNAL_SOURCE_CSV` and
`STEPC_INTERNAL_AUDIT_DIR`. The latter must point outside
`fully_synthetic_stepC_v1/`; the generator refuses to place source-derived
summary audits inside the exportable package.

## Run Debug Analyses

```bash
bash scripts/run_stepC2_debug.sh
bash scripts/run_stepC3_debug.sh
bash scripts/run_stepC4_debug.sh
```

Debug mode runs only a small number of repetitions and is intended as a smoke test.

## Run Full Analyses

```bash
bash scripts/run_stepC2_full.sh
bash scripts/run_stepC3_full.sh
bash scripts/run_stepC4_full.sh
```

The full run processes all eight scenarios and all 50 repetitions per scenario.
`run_stepC4_postQC_sharded.py` is not the default from-scratch runner. It
reproduces the study's QC migration by requiring a legacy C4 table, preserving
four screened classical entries, and rerunning six entries in scenario shards.

## Run DGM and Publication Audits

```bash
bash scripts/run_stepC5A_exact_DGM_coefficients.sh
bash scripts/run_stepC6_publication_analysis.sh
```

C5A accepts either the original local Step C generator notebook or the public
Python generator source, together with fully synthetic Step C metadata tables.
C6 expects completed post-QC C2, C3, and C4 outputs plus the C5A DGM-audit
tables. Generated audit and synthesis outputs are intentionally not tracked by
Git. C6 records SHA-256 manifests for the input package, final model-result and
DGM-audit tables, and final audited analysis-source state so that the internal
evidence version can be verified without publishing those files.

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
