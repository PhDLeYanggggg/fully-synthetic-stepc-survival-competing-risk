# Results Summary

Small aggregate summary tables are included under `results_summary/`. The larger replicate-level tables and per-person outputs remain local and are not tracked.

## Step C2

Step C2 completed:

```text
8 scenarios x 50 repetitions x 4 models = 1,600 rows
failures = 0
full_run_passed = True
```

Best fitted C2 model by mean C-index:

| Scenario | Best fitted model | Mean C-index |
| --- | --- | ---: |
| S0_linear_PH_inst30 | cox_dgm_features | 0.6968 |
| S1_linear_PH_inst15 | cox_dgm_features | 0.6919 |
| S2_linear_PH_inst45 | cox_dgm_features | 0.6933 |
| S3_nonlinear_interaction_inst30 | cox_dgm_features | 0.7329 |
| S4_nonPH_inst30 | cox_dgm_features | 0.6900 |
| S5_MAR_missingness_inst30 | cox_dgm_features | 0.6897 |
| S6_highdim_sparseMRI_inst30 | penalised_cox_all_safe_predictors | 0.7097 |
| S7_strong_death_competing_inst30 | cox_dgm_features | 0.6925 |

## Step C3

Step C3 completed:

```text
8 scenarios x 50 repetitions x 4 mandatory models = 1,600 rows
failures = 0
full_run_passed = True
oracle sanity flags = 0
```

Best fitted C3 model by mean 5-year risk MAE:

| Scenario | Best model | MAE | Brier | AUC | Calibration slope |
| --- | --- | ---: | ---: | ---: | ---: |
| S0_linear_PH_inst30 | cs_cox_dgm_cif | 0.03708 | 0.1912 | 0.6860 | 0.9483 |
| S1_linear_PH_inst15 | cs_cox_dgm_cif | 0.02707 | 0.1217 | 0.6668 | 0.8879 |
| S2_linear_PH_inst45 | cs_cox_dgm_cif | 0.04013 | 0.2200 | 0.6929 | 0.9498 |
| S3_nonlinear_interaction_inst30 | cs_cox_dgm_cif | 0.05246 | 0.1808 | 0.7252 | 0.9118 |
| S4_nonPH_inst30 | cs_cox_dgm_cif | 0.04990 | 0.1950 | 0.6668 | 0.8777 |
| S5_MAR_missingness_inst30 | cs_cox_dgm_cif | 0.04015 | 0.1930 | 0.6770 | 0.9098 |
| S6_highdim_sparseMRI_inst30 | cs_penalised_cox_all_safe_cif | 0.04558 | 0.1834 | 0.7175 | 0.8793 |
| S7_strong_death_competing_inst30 | cs_cox_dgm_cif | 0.03665 | 0.1940 | 0.6711 | 0.9334 |

## Interpretation

The results suggest that well-specified Cox models are robust in most synthetic settings. The high-dimensional sparse MRI scenario is the clearest exception, where the all-safe penalised Cox model is preferred. In the strong competing-death scenario, individualised competing-risk prediction improves substantially on a non-individualised Aalen-Johansen null.
