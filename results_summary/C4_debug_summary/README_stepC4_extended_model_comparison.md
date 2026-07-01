# Step C4 Extended Model Comparison

## Aim

C4 extends the completed C2/C3 analyses by adding Fine-Gray, Random Survival Forest, and Gradient Boosting Survival models. It does not regenerate Step C data and does not rerun the original C2/C3 models.

## Run Status

- Mode: debug
- Started: 2026-07-01T17:23:18
- Finished: 2026-07-01T17:25:32
- Scenarios observed: 8
- Replicate rows: 160
- Successful models: cs_gbsa_all_safe_cif, cs_gbsa_dgm_cif, cs_rsf_all_safe_cif, cs_rsf_dgm_cif, finegray_all_safe_reduced_R, finegray_dgm_cif_R
- Skipped models: deephit_competing_risk_all_safe_reduced_optional, deephit_competing_risk_dgm_optional, deepsurv_all_safe_cause_specific_optional, deepsurv_dgm_cause_specific_optional
- Failed models: none

## Data Source

- Uses only `fully_synthetic_stepC_v1/` plus C2/C3 predictor lists and summaries.
- Does not use real SLAM data.
- Does not use Step B semi-synthetic data.
- Does not use raw CSV files, death spreadsheets, WMH spreadsheets, or real identifiers.
- Does not save full per-person predictions.

## Added Models

- Fine-Gray: `finegray_dgm_cif_R`, `finegray_all_safe_reduced_R`.
- Random Survival Forest: `cs_rsf_dgm_cif`, `cs_rsf_all_safe_cif`.
- Gradient Boosting Survival: `cs_gbsa_dgm_cif`, `cs_gbsa_all_safe_cif`.
- Optional deep models: DeepSurv and DeepHit are logged as skipped if dependencies are unavailable.

## Dependency Status

| dependency | language | required_for | available | version_or_status | failure_reason |
| --- | --- | --- | --- | --- | --- |
| numpy | python | core_or_plotting | True | 1.24.4 |  |
| pandas | python | core_or_plotting | True | 2.3.3 |  |
| scipy | python | core_or_plotting | True | 1.9.1 |  |
| sklearn | python | core_or_plotting | True | 1.5.2 |  |
| lifelines | python | core_or_plotting | True | 0.30.0 |  |
| sksurv | python | core_or_plotting | True | 0.23.1 |  |
| xgboost | python | core_or_plotting | True | 2.1.4 |  |
| matplotlib | python | core_or_plotting | True | 3.5.2 |  |
| torch | python | optional_deep | True | 2.0.0 |  |
| torchtuples | python | optional_deep | False |  | ModuleNotFoundError: No module named 'torchtuples' |
| pycox | python | optional_deep | False |  | ModuleNotFoundError: No module named 'pycox' |
| Rscript | R | Fine-Gray | True | available |  |
| survival | R | Fine-Gray | True | available |  |
| cmprsk | R | Fine-Gray | True | available |  |

## Main Results

- Fine-Gray status: completed.
- RSF status: completed.
- GBSA status: completed.
- DeepHit status: skipped/not completed.
- Any oracle sanity flag: True.
- Oracle/audit flag details: calibration_slope_outside_0_5_to_1_5=16.
- S6 combined best non-oracle absolute-risk model: finegray_all_safe_reduced_R.
- S7 combined best non-oracle absolute-risk model: cs_cox_dgm_cif.
- S7 Fine-Gray / competing-risk comparison note: S7 best non-oracle absolute-risk model in combined C3/C4 table: cs_cox_dgm_cif.
- RSF versus C3 Cox in S3/S6/S7: S3 cs_rsf_dgm_cif MAE 0.0634 vs cs_cox_dgm_cif MAE 0.0525; not better; S6 cs_rsf_all_safe_cif MAE 0.0529 vs cs_cox_dgm_cif MAE 0.0471; not better; S7 cs_rsf_dgm_cif MAE 0.0511 vs cs_cox_dgm_cif MAE 0.0367; not better.
- GBSA versus C3 Cox in S3/S6/S7: S3 cs_gbsa_all_safe_cif MAE 0.1010 vs cs_cox_dgm_cif MAE 0.0525; not better; S6 cs_gbsa_all_safe_cif MAE 0.0859 vs cs_cox_dgm_cif MAE 0.0471; not better; S7 cs_gbsa_all_safe_cif MAE 0.0765 vs cs_cox_dgm_cif MAE 0.0367; not better.
- C4 debug does not overturn C2/C3 core conclusions; it establishes that Fine-Gray, RSF, and GBSA can run under the strict synthetic-only leakage guard, while GBSA calibration needs audit.

### C4 Scenario Summary

| scenario_id | model | n_successful_reps | n_failed_reps | n_skipped_reps | risk5_mae_vs_true_risk_mean | brier_5y_naive_mean | auc_5y_observed_event1_mean | cause_specific_cindex_event1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0_linear_PH_inst30 | cs_gbsa_all_safe_cif | 2 | 0 | 0 | 0.0803 | 0.1988 | 0.6596 | 0.6737 |
| S0_linear_PH_inst30 | cs_gbsa_dgm_cif | 2 | 0 | 0 | 0.0822 | 0.1991 | 0.6721 | 0.6838 |
| S0_linear_PH_inst30 | cs_rsf_all_safe_cif | 2 | 0 | 0 | 0.0642 | 0.1949 | 0.6689 | 0.6826 |
| S0_linear_PH_inst30 | cs_rsf_dgm_cif | 2 | 0 | 0 | 0.0503 | 0.1924 | 0.6841 | 0.7005 |
| S0_linear_PH_inst30 | deephit_competing_risk_all_safe_reduced_optional | 0 | 0 | 2 |  |  |  |  |
| S0_linear_PH_inst30 | deephit_competing_risk_dgm_optional | 0 | 0 | 2 |  |  |  |  |
| S0_linear_PH_inst30 | deepsurv_all_safe_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S0_linear_PH_inst30 | deepsurv_dgm_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S0_linear_PH_inst30 | finegray_all_safe_reduced_R | 2 | 0 | 0 | 0.0544 | 0.1917 | 0.6837 | 0.6805 |
| S0_linear_PH_inst30 | finegray_dgm_cif_R | 2 | 0 | 0 | 0.0413 | 0.1889 | 0.6970 | 0.7029 |
| S1_linear_PH_inst15 | cs_gbsa_all_safe_cif | 2 | 0 | 0 | 0.0569 | 0.1257 | 0.6172 | 0.6550 |
| S1_linear_PH_inst15 | cs_gbsa_dgm_cif | 2 | 0 | 0 | 0.0565 | 0.1254 | 0.6243 | 0.6612 |
| S1_linear_PH_inst15 | cs_rsf_all_safe_cif | 2 | 0 | 0 | 0.0438 | 0.1236 | 0.6524 | 0.6714 |
| S1_linear_PH_inst15 | cs_rsf_dgm_cif | 2 | 0 | 0 | 0.0345 | 0.1220 | 0.6670 | 0.6904 |
| S1_linear_PH_inst15 | deephit_competing_risk_all_safe_reduced_optional | 0 | 0 | 2 |  |  |  |  |
| S1_linear_PH_inst15 | deephit_competing_risk_dgm_optional | 0 | 0 | 2 |  |  |  |  |
| S1_linear_PH_inst15 | deepsurv_all_safe_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S1_linear_PH_inst15 | deepsurv_dgm_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S1_linear_PH_inst15 | finegray_all_safe_reduced_R | 2 | 0 | 0 | 0.0321 | 0.1229 | 0.6641 | 0.6709 |
| S1_linear_PH_inst15 | finegray_dgm_cif_R | 2 | 0 | 0 | 0.0280 | 0.1236 | 0.6683 | 0.6867 |
| S2_linear_PH_inst45 | cs_gbsa_all_safe_cif | 2 | 0 | 0 | 0.0837 | 0.2284 | 0.6773 | 0.6711 |
| S2_linear_PH_inst45 | cs_gbsa_dgm_cif | 2 | 0 | 0 | 0.0851 | 0.2290 | 0.6746 | 0.6693 |
| S2_linear_PH_inst45 | cs_rsf_all_safe_cif | 2 | 0 | 0 | 0.0782 | 0.2282 | 0.6678 | 0.6649 |
| S2_linear_PH_inst45 | cs_rsf_dgm_cif | 2 | 0 | 0 | 0.0627 | 0.2243 | 0.6799 | 0.6768 |
| S2_linear_PH_inst45 | deephit_competing_risk_all_safe_reduced_optional | 0 | 0 | 2 |  |  |  |  |
| S2_linear_PH_inst45 | deephit_competing_risk_dgm_optional | 0 | 0 | 2 |  |  |  |  |
| S2_linear_PH_inst45 | deepsurv_all_safe_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S2_linear_PH_inst45 | deepsurv_dgm_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S2_linear_PH_inst45 | finegray_all_safe_reduced_R | 2 | 0 | 0 | 0.0601 | 0.2261 | 0.6723 | 0.6577 |
| S2_linear_PH_inst45 | finegray_dgm_cif_R | 2 | 0 | 0 | 0.0536 | 0.2241 | 0.6841 | 0.6736 |
| S3_nonlinear_interaction_inst30 | cs_gbsa_all_safe_cif | 2 | 0 | 0 | 0.1010 | 0.1896 | 0.7020 | 0.7117 |
| S3_nonlinear_interaction_inst30 | cs_gbsa_dgm_cif | 2 | 0 | 0 | 0.1018 | 0.1893 | 0.7003 | 0.7095 |
| S3_nonlinear_interaction_inst30 | cs_rsf_all_safe_cif | 2 | 0 | 0 | 0.0810 | 0.1850 | 0.7106 | 0.7120 |
| S3_nonlinear_interaction_inst30 | cs_rsf_dgm_cif | 2 | 0 | 0 | 0.0634 | 0.1801 | 0.7254 | 0.7297 |
| S3_nonlinear_interaction_inst30 | deephit_competing_risk_all_safe_reduced_optional | 0 | 0 | 2 |  |  |  |  |
| S3_nonlinear_interaction_inst30 | deephit_competing_risk_dgm_optional | 0 | 0 | 2 |  |  |  |  |
| S3_nonlinear_interaction_inst30 | deepsurv_all_safe_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S3_nonlinear_interaction_inst30 | deepsurv_dgm_cause_specific_optional | 0 | 0 | 2 |  |  |  |  |
| S3_nonlinear_interaction_inst30 | finegray_all_safe_reduced_R | 2 | 0 | 0 | 0.0599 | 0.1812 | 0.7151 | 0.7102 |
| S3_nonlinear_interaction_inst30 | finegray_dgm_cif_R | 2 | 0 | 0 | 0.0522 | 0.1790 | 0.7247 | 0.7267 |

## Limitations

- These are fully synthetic benchmark results, not real clinical conclusions.
- If Fine-Gray failed or was skipped, C4 must not be interpreted as a completed Fine-Gray comparison.
- Optional deep models are not blockers when dependencies are unavailable.
- A naive 5-year classifier is not included as a main survival model.
- Naive Brier is not a censoring-adjusted IPCW Brier; IPCW results are recorded only when available.
- C4 does not alter the Step C data-generating mechanism.
- All-safe reduced feature selection is performed within the training split only.
- Exact DGM coefficients are still not exported, so DGM predictor sets remain fallback lists.

## Key Outputs

- `tables/replicate_extended_model_performance.csv`
- `tables/scenario_extended_model_summary_mean_sd_ci.csv`
- `tables/best_absolute_risk_model_by_scenario_C4.csv`
- `tables/best_ranking_model_by_scenario_C4.csv`
- `tables/oracle_sanity_audit_C4.csv`
- `tables/full_run_sanity_checks_C4.csv`

## Next Steps

- C5 can export exact DGM coefficients.
- C5 can add post-care-home death-hazard sensitivity analyses.
- C5 can add 1/3/5-year horizons.
- The full validation stage should use real care-home outcomes when available in the authorised environment.

## Full-run Sanity Checks

| check_name | expected | observed | passed |
| --- | --- | --- | --- |
| expected_scenarios | 8 | 8 | True |
| observed_scenarios | 8 | 8 | True |
| expected_reps_per_scenario | 2 | 2 | True |
| observed_min_reps_per_scenario | 2 | 2 | True |
| observed_max_reps_per_scenario | 2 | 2 | True |
| attempted_models | core_and_optional | cs_gbsa_all_safe_cif;cs_gbsa_dgm_cif;cs_rsf_all_safe_cif;cs_rsf_dgm_cif;deephit_competing_risk_all_safe_reduced_optional;deephit_competing_risk_dgm_optional;deepsurv_all_safe_cause_specific_optional;deepsurv_dgm_cause_specific_optional;finegray_all_safe_reduced_R;finegray_dgm_cif_R | True |
| successful_models | at_least_one_core | cs_gbsa_all_safe_cif;cs_gbsa_dgm_cif;cs_rsf_all_safe_cif;cs_rsf_dgm_cif;finegray_all_safe_reduced_R;finegray_dgm_cif_R | True |
| skipped_models | logged | deephit_competing_risk_all_safe_reduced_optional;deephit_competing_risk_dgm_optional;deepsurv_all_safe_cause_specific_optional;deepsurv_dgm_cause_specific_optional | True |
| failed_model_fits | 0 preferred | 0 | True |
| export_safety_passed | True | True | True |
| predictor_leakage_audit_passed | True | True | True |
| oracle_sanity_passed | True | False | False |
| finegray_completed | True | True | True |
| rsf_completed | True | True | True |
| gbsa_completed | True | True | True |
| deephit_completed | optional | False | True |
| full_run_passed | True | True | True |
| publication_ready | True | False | False |
