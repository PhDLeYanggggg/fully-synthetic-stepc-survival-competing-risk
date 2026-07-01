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

The larger replicate-level output tables remain local and are intentionally not tracked.
