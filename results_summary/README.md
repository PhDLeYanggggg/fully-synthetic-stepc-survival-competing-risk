# Results Summary Tables

This directory contains small aggregate summary and audit tables from a completed local full run on fully synthetic data.

It does not contain scenario-level synthetic datasets, per-person prediction rows, fitted model objects, raw data, or internal semi-synthetic data.

## Included C2 tables

- `C2_summary/scenario_model_summary_mean_sd_ci.csv`
- `C2_summary/model_difference_vs_oracle_by_scenario.csv`
- `C2_summary/best_non_oracle_model_by_scenario.csv`
- `C2_summary/xgb_oracle_sanity_audit_full.csv`
- `C2_summary/full_run_sanity_checks.csv`

## Included C3 tables

- `C3_summary/scenario_competing_risk_summary_mean_sd_ci.csv`
- `C3_summary/model_difference_vs_oracle_C3_by_scenario.csv`
- `C3_summary/best_C3_model_by_scenario.csv`
- `C3_summary/oracle_sanity_audit_C3.csv`
- `C3_summary/full_run_sanity_checks_C3.csv`
- `C3_summary/finegray_optional_status.csv`

## Included C4 debug tables

`C4_debug_summary/` contains small aggregate and replicate-model debug outputs from Step C4. These are model-level or replicate-level summaries only. They do not include scenario datasets, full per-person predictions, or figures.

Key C4 files include:

- `C4_debug_summary/README_stepC4_extended_model_comparison.md`
- `C4_debug_summary/replicate_extended_model_performance.csv`
- `C4_debug_summary/scenario_extended_model_summary_mean_sd_ci.csv`
- `C4_debug_summary/best_absolute_risk_model_by_scenario_C4.csv`
- `C4_debug_summary/best_ranking_model_by_scenario_C4.csv`
- `C4_debug_summary/oracle_sanity_audit_C4.csv`
- `C4_debug_summary/full_run_sanity_checks_C4.csv`

The C4 full run has also been completed locally. The larger full-run C4 outputs and all figure files remain local unless explicitly reviewed and approved.
