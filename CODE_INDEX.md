# Code Index

This file lists the complete code surface included in this repository. The
repository is code-first: it does not include real SLAM data, Step B internal
data, large fully synthetic scenario datasets, per-person predictions, fitted
model binaries, or unapproved figures.

## Organised Source Tree

| Step | Path | Purpose |
| --- | --- | --- |
| Step C generator | `src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py` | Fully synthetic scenario generator, included for methods transparency. |
| Step C generator audits | `src/stepC_generator/stepC_quick_audits.py` | Synthetic-package missingness, truth-gradient, and monotonicity checks. |
| Step C2 | `src/stepC2_model_comparison/stepC2_fully_synthetic_model_comparison.py` | Cause-specific survival model comparison. |
| Step C3 | `src/stepC3_competing_risk/stepC3_competing_risk_evaluation.py` | Competing-risk 5-year cumulative-incidence evaluation. |
| Step C4 | `src/stepC4_extended_models/stepC4_extended_model_comparison.py` | Extended Fine-Gray, RSF, GBSA, and optional deep-survival comparison. |
| Step C5A | `src/stepC5A_exact_DGM_coefficients/stepC5A_exact_DGM_coefficients.py` | Exact raw DGM coefficient export and bounded true-LP reconstruction audit. |
| Step C6 | `src/stepC6_publication_analysis/stepC6_publication_analysis.py` | Paired publication-oriented statistical synthesis; fits no model. |

Root-level script copies are retained for backward compatibility:

- `stepC2_fully_synthetic_model_comparison.py`
- `stepC3_competing_risk_evaluation.py`
- `stepC4_extended_model_comparison.py`
- `stepC5A_exact_DGM_coefficients.py`
- `stepC6_publication_analysis.py`
- `run_stepC4_postQC_sharded.py`

## Run Scripts

| Command | Description |
| --- | --- |
| `bash scripts/run_stepC2_debug.sh` | Step C2 debug run. |
| `bash scripts/run_stepC2_full.sh` | Step C2 full run. |
| `bash scripts/run_stepC3_debug.sh` | Step C3 debug run. |
| `bash scripts/run_stepC3_full.sh` | Step C3 full run. |
| `bash scripts/run_stepC4_debug.sh` | Step C4 debug run. |
| `bash scripts/run_stepC4_full.sh` | Step C4 full run. |
| `bash scripts/run_stepC5A_exact_DGM_coefficients.sh` | Step C5A exact DGM coefficient export. |
| `bash scripts/run_stepC6_publication_analysis.sh` | Step C6 paired statistical synthesis. |

The modelling scripts expect the local input folder `fully_synthetic_stepC_v1/`.
That folder is intentionally not included in this repository.

## Output Directories Not Tracked

Generated analysis outputs are intentionally ignored:

- `fully_synthetic_stepC2_model_comparison/`
- `fully_synthetic_stepC3_competing_risk_evaluation/`
- `fully_synthetic_stepC4_extended_models/`
- `fully_synthetic_stepC5A_exact_DGM_coefficients/`
- `fully_synthetic_stepC6_publication_analysis/`

No result table or figure is currently committed. Any future result artifact
must be added only after explicit author review and approval.

## Environment Provenance

- `environment.yml` provides the portable conda specification.
- `requirements.txt` provides the pip-oriented Python specification.
- `environment-lock-20260727.txt` records the exact Python and R package
  versions used for the locked post-QC analyses.

## Reporting Audit

- `docs/14_analysis_specification_and_deviations.md` records the original C2
  requirements, later extensions, QC corrections, and non-changes.
- `docs/15_reporting_checklist_ADEMP_TRIPODAI.md` maps the manuscript to ADEMP
  and relevant TRIPOD+AI transparency principles.
