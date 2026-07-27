# Step C3 Competing-risk Evaluation

Step C3 evaluates 5-year absolute risk of care-home entry under competing risk.

## Endpoint

The event of interest is care-home entry or institutionalisation within 5 years. Death before care home is treated as a competing event. Death after care-home entry is not the competing event for this primary endpoint.

## Models

- `oracle_true_risk_not_a_model`: synthetic benchmark using the exported
  five-year risk target. This is the closed-form CIF implied by the returned
  calibrated hazards and LPs in the seven PH mechanisms, with the
  finite-repetition calibration dependence described in the generator
  documentation; it is an additional structural proxy in S4.
- `nonparametric_aj_null`: non-individualised Aalen-Johansen null.
- `cs_cox_dgm_cif`: cause-specific Cox cumulative-incidence prediction using DGM-relevant predictors.
- `cs_penalised_cox_all_safe_cif`: cause-specific Cox cumulative-incidence prediction using strict all-safe predictors.
Fine-Gray is outside the prespecified four-entry C3 comparison and is evaluated
in the extended C4 stage.

## Metrics

The key metrics are MAE and RMSE against the exported synthetic five-year
target, fully observed-status five-year Brier score, binary observed-status
five-year AUC, calibration slope/intercept, calibration deciles, and
cause-specific C-index for continuity with C2.

Truth-based S4 metrics are exploratory. S4 observed-outcome Brier score, AUC,
logistic calibration, and cause-specific C-index remain valid because all
individuals have complete administrative five-year outcome status.

## Full-run Result

The local full run completed all 1,600 specified C3 rows with 0 failures and
passed the stage sanity checks. The separate C4 stage completed the specified
Fine-Gray comparisons using `cmprsk`.

`cs_cox_dgm_cif` was the best fitted model in seven scenarios. `cs_penalised_cox_all_safe_cif` was best in S6. In S7, individualised CIF prediction clearly improved on the Aalen-Johansen null baseline.
