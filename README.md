# Fully Synthetic Step C Survival and Competing-Risk Evaluation

This repository contains code-only analysis pipelines for model comparison on a fully synthetic dementia care-home institutionalisation benchmark.

No data files, generated tables, figures, per-person predictions, or compressed scenario datasets are included in this repository.

## Repository contents

- `stepC2_fully_synthetic_model_comparison.py`: cause-specific survival model comparison for time to care-home entry.
- `stepC3_competing_risk_evaluation.py`: 5-year competing-risk cumulative-incidence evaluation for care-home entry in the presence of death before care home.
- `requirements.txt`: Python package requirements.

## Data boundary

The scripts expect a local synthetic export folder named:

```text
fully_synthetic_stepC_v1/
```

That folder is not included here. The analyses are designed to use only exportable fully synthetic files from that folder. They do not require, read, or assume access to any real SLAM records, raw clinical files, Step B files, death spreadsheets, WMH spreadsheets, or semi-synthetic data.

The expected local input includes synthetic scenario datasets plus audit and metadata tables, including `export_safety_audit.csv`, `feature_dictionary.csv`, `dgm_definition_table.csv`, `scenario_summary.csv`, and `repetition_summary.csv`.

## Analysis design

The primary endpoint is time to care-home entry or institutionalisation. The outcome coding is:

- `status = 0`: event-free or censored
- `status = 1`: care-home entry, the event of interest
- `status = 2`: death before care home, a competing event

Step C2 uses cause-specific survival modelling, where `status = 1` is treated as the event and `status = 0` or `status = 2` are treated as censored for the care-home endpoint.

Step C3 evaluates 5-year cumulative incidence under competing risk. It compares oracle synthetic risk, an Aalen-Johansen null baseline, and cause-specific Cox cumulative-incidence reconstructions.

Both scripts process one scenario file at a time and one replicate at a time, with debug-first defaults:

```text
DEBUG_MODE = True
MAX_REPS_PER_SCENARIO = 2
```

Run the debug mode first:

```bash
python stepC2_fully_synthetic_model_comparison.py
python stepC3_competing_risk_evaluation.py
```

Run the full synthetic benchmark:

```bash
python stepC2_fully_synthetic_model_comparison.py --full
python stepC3_competing_risk_evaluation.py --full
```

## Models

Step C2 evaluates:

- `oracle_true_lp_not_a_model`: benchmark using the known synthetic data-generating truth.
- `cox_dgm_features`: cause-specific Cox model using DGM-relevant synthetic baseline features.
- `penalised_cox_all_safe_predictors`: penalised Cox model using all strict leakage-guarded safe baseline predictors.
- `xgb_survival_cox_strict`: XGBoost `survival:cox` risk-score model using the same strict safe predictors.

Step C3 evaluates:

- `oracle_true_risk_not_a_model`: benchmark using the exported synthetic 5-year care-home risk.
- `nonparametric_aj_null`: Aalen-Johansen nonparametric null prediction.
- `cs_cox_dgm_cif`: cause-specific Cox cumulative-incidence prediction using DGM-relevant features.
- `cs_penalised_cox_all_safe_cif`: cause-specific Cox cumulative-incidence prediction using strict all-safe predictors.

Fine-Gray evaluation is treated as optional in this version and is not required for the mandatory Python-only pipeline.

## Local full-run summary

The following summary describes one completed local full run on the fully synthetic export. The underlying data, replicate-level predictions, tables, and figures are not included in this repository.

Step C2 completed 1,600 scenario-replicate-model evaluations: 8 scenarios x 50 replicates x 4 models, with 0 model failures. The oracle is interpreted as an upper benchmark under the simulated data-generating mechanism, not as a deployable model.

Best fitted Step C2 model by mean Harrell C-index:

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

Step C3 completed 1,600 scenario-replicate-model evaluations: 8 scenarios x 50 replicates x 4 mandatory models, with 0 model failures. Oracle sanity checks passed, and no forbidden predictors were included. Optional Fine-Gray modelling was skipped because `cmprsk` was unavailable in the local R environment.

Best fitted Step C3 model by mean 5-year risk MAE versus the exported synthetic true risk:

| Scenario | Best fitted model | Mean MAE |
| --- | --- | ---: |
| S0_linear_PH_inst30 | cs_cox_dgm_cif | 0.0371 |
| S1_linear_PH_inst15 | cs_cox_dgm_cif | 0.0271 |
| S2_linear_PH_inst45 | cs_cox_dgm_cif | 0.0401 |
| S3_nonlinear_interaction_inst30 | cs_cox_dgm_cif | 0.0525 |
| S4_nonPH_inst30 | cs_cox_dgm_cif | 0.0499 |
| S5_MAR_missingness_inst30 | cs_cox_dgm_cif | 0.0402 |
| S6_highdim_sparseMRI_inst30 | cs_penalised_cox_all_safe_cif | 0.0456 |
| S7_strong_death_competing_inst30 | cs_cox_dgm_cif | 0.0367 |

Across scenarios, the fitted model family selected by C2 discrimination and C3 absolute-risk accuracy agreed in 8 of 8 scenarios after mapping the cause-specific model families. In S7, which strengthens the death competing-risk mechanism, the best C3 fitted model had mean MAE 0.0367 compared with 0.1088 for the Aalen-Johansen null baseline.

## Interpretation

The results support the pipeline sanity checks expected under the synthetic design:

- DGM-feature Cox models perform close to the oracle benchmark in the linear proportional-hazards scenarios.
- The strict all-safe penalised Cox model is most useful in the high-dimensional sparse MRI scenario.
- XGBoost is restricted to risk-score evaluation in Step C2 and is not assigned fabricated absolute 5-year risk.
- XGBoost does not unrealistically exceed the oracle benchmark, which supports the leakage guard.
- The competing-risk evaluation gives a more appropriate 5-year absolute-risk view when death before care home is common.

These findings are benchmark results on fully synthetic data. They should not be interpreted as clinical performance on real patients.
