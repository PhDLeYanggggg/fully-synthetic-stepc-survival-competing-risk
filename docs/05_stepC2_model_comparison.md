# Step C2 Cause-specific Model Comparison

Step C2 compares discrimination for care-home entry using cause-specific survival modelling on fully synthetic data.

## Outcome Handling

The outcome coding is:

```text
status = 0: event-free or censored
status = 1: care-home entry
status = 2: death before care home
```

For Step C2, `status = 1` is the event of interest. Both `status = 0` and `status = 2` are treated as censored. This gives a cause-specific survival comparison, not a full competing-risk absolute-risk analysis.

## Models

- `oracle_true_lp_not_a_model`: benchmark using the known synthetic linear predictor.
- `cox_dgm_features`: cause-specific Cox model using DGM-relevant features.
- `penalised_cox_all_safe_predictors`: penalised Cox model using strict all-safe baseline predictors.
- `xgb_survival_cox_strict`: XGBoost survival Cox risk-score model using the same strict safe predictors.

## Main Metrics

- Harrell C-index for care-home entry;
- Spearman correlation between model risk score and synthetic truth;
- 5-year risk error metrics for models with calibrated 5-year risk probabilities;
- failure rate and runtime.

## Full-run Result

The full run completed 1,600 replicate-model rows with 0 failures. The best fitted model was `cox_dgm_features` in seven scenarios and `penalised_cox_all_safe_predictors` in S6.

The XGBoost model did not exceed the oracle benchmark implausibly, supporting the leakage guard.
