# Step C3 Competing-risk Evaluation

Step C3 evaluates 5-year absolute risk of care-home entry under competing risk.

## Endpoint

The event of interest is care-home entry or institutionalisation within 5 years. Death before care home is treated as a competing event. Death after care-home entry is not the competing event for this primary endpoint.

## Models

- `oracle_true_risk_not_a_model`: synthetic benchmark using exported true 5-year risk.
- `nonparametric_aj_null`: non-individualised Aalen-Johansen null.
- `cs_cox_dgm_cif`: cause-specific Cox cumulative-incidence prediction using DGM-relevant predictors.
- `cs_penalised_cox_all_safe_cif`: cause-specific Cox cumulative-incidence prediction using strict all-safe predictors.
- `finegray_dgm_optional`: optional Fine-Gray model, not part of the mandatory Python-only run.

## Metrics

The key metrics are MAE and RMSE against the exported true synthetic 5-year risk, naive 5-year Brier score, observed 5-year AUC, calibration slope/intercept, calibration deciles, and cause-specific C-index for continuity with C2.

## Full-run Result

The local full run completed 1,600 mandatory rows with 0 failures and passed oracle sanity checks. Fine-Gray was skipped because `cmprsk` was not available in the local R environment.

`cs_cox_dgm_cif` was the best fitted model in seven scenarios. `cs_penalised_cox_all_safe_cif` was best in S6. In S7, individualised CIF prediction clearly improved on the Aalen-Johansen null baseline.
