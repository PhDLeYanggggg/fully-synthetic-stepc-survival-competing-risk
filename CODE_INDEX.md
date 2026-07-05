# Code Index

This file lists the complete code surface included in this repository. The
repository is code-first: it does not include real SLAM data, Step B internal
data, large fully synthetic scenario datasets, per-person predictions, fitted
model binaries, or unapproved figures.

## Organised Source Tree

| Step | Path | Purpose |
| --- | --- | --- |
| Step C generator | `src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py` | Fully synthetic scenario generator, included for methods transparency. |
| Step C2 | `src/stepC2_model_comparison/stepC2_fully_synthetic_model_comparison.py` | Cause-specific survival model comparison. |
| Step C3 | `src/stepC3_competing_risk/stepC3_competing_risk_evaluation.py` | Competing-risk 5-year cumulative-incidence evaluation. |
| Step C4 | `src/stepC4_extended_models/stepC4_extended_model_comparison.py` | Extended Fine-Gray, RSF, GBSA, and optional deep-survival comparison. |
| Step C4A | `src/stepC4A_calibration_audit/stepC4A_calibration_audit.py` | Calibration-slope audit for the single C4 oracle sanity flag. |
| Step C5A | `src/stepC5A_exact_DGM_coefficients/stepC5A_exact_DGM_coefficients.py` | Exact raw DGM coefficient export and bounded true-LP reconstruction audit. |

Root-level script copies are retained for backward compatibility:

- `stepC2_fully_synthetic_model_comparison.py`
- `stepC3_competing_risk_evaluation.py`
- `stepC4_extended_model_comparison.py`
- `stepC4A_calibration_audit.py`
- `stepC5A_exact_DGM_coefficients.py`

## Run Scripts

| Command | Description |
| --- | --- |
| `bash scripts/run_stepC2_debug.sh` | Step C2 debug run. |
| `bash scripts/run_stepC2_full.sh` | Step C2 full run. |
| `bash scripts/run_stepC3_debug.sh` | Step C3 debug run. |
| `bash scripts/run_stepC3_full.sh` | Step C3 full run. |
| `bash scripts/run_stepC4_debug.sh` | Step C4 debug run. |
| `bash scripts/run_stepC4_full.sh` | Step C4 full run. |
| `bash scripts/run_stepC4A_calibration_audit.sh` | Step C4A calibration audit. |
| `bash scripts/run_stepC5A_exact_DGM_coefficients.sh` | Step C5A exact DGM coefficient export. |

The modelling scripts expect the local input folder `fully_synthetic_stepC_v1/`.
That folder is intentionally not included in this repository.

## Output Directories Not Tracked

Generated analysis outputs are intentionally ignored:

- `fully_synthetic_stepC2_model_comparison/`
- `fully_synthetic_stepC3_competing_risk_evaluation/`
- `fully_synthetic_stepC4_extended_models/`
- `fully_synthetic_stepC4_calibration_audit/`
- `fully_synthetic_stepC5A_exact_DGM_coefficients/`

Only small, deliberately curated aggregate summaries may be committed under
`results_summary/`. Figures and additional tables should be added only after
explicit human review.
