# From Internal SLAM Semi-synthetic Data to Fully Synthetic Competing-risk Evaluation

This document describes the full project workflow from the internal SLAM diagnostic modelling data structure to the exportable fully synthetic survival and competing-risk benchmark.

## 1. Project Background

The starting point was an internal SLAM diagnostic modelling dataset for dementia subtype classification. The diagnostic labels in that earlier project were AD, Mixed, and Non-AD. During internal preparation, the raw diagnostic modelling data contained approximately 24,418 rows. After aligning MMSE with scan date and de-duplicating at scan level, the working scan-level dataset contained approximately 4,213 scans. A further baseline-only restriction, keeping the first scan for each person, produced approximately 3,963 baseline individuals.

Those internal counts are included here only to explain the development history. The underlying real patient-level data are not part of this repository.

The internal data structure provided the variable organisation needed for simulation. Baseline predictors included:

- age;
- sex;
- MMSE;
- MRI regional volume percentages;
- MRI composite features;
- NLP-derived pathology annotation features;
- NLP missingness, presence, and log1p feature variants;
- comorbidity variables;
- deprivation indices;
- WMH-related variables.

This predictor structure motivated the later synthetic feature blocks, but the real patient rows themselves were not exported to GitHub.

## 2. Why Simulation Was Necessary

At the time of this methodological work, real care-home entry or institutionalisation dates were not fully available for direct prognostic modelling. Without a real endpoint, it would be inappropriate to claim clinical predictive performance.

Simulation was therefore used to build and stress-test the analysis pipeline. The simulation framework allowed the project to:

- define time-to-care-home outcome coding;
- handle death before care home as a competing event;
- compare model families under controlled data-generating mechanisms;
- check whether complex models outperform simpler models;
- identify data leakage and evaluation artefacts before real-outcome modelling;
- document a reproducible workflow for later validation.

## 3. Step B: Internal Semi-synthetic SLAM Stage

Step B was an internal semi-synthetic stage. Its structure was:

```text
real SLAM baseline predictors
+ simulated care-home entry time
+ internal death-date information for competing-risk reference
```

This stage could not be exported because the predictor matrix still contained real patient-level SLAM baseline rows.

Within the authorised environment, Step B:

- read the internal SLAM baseline data;
- merged updated death-date information;
- restricted the data to baseline-only observations;
- built baseline predictor blocks;
- specified a simulated care-home entry data-generating mechanism;
- constructed risk scores using age, MMSE, MRI, NLP, comorbidity, and deprivation features;
- generated simulated time to care-home entry;
- treated death before care home as a competing event;
- encoded status as 0 for event-free or censored, 1 for care-home entry, and 2 for death before care home.

An important Step B finding was that an initial XGBoost survival result was implausibly high and exceeded the oracle benchmark. That behaviour suggested predictor leakage or an evaluation artefact. After strict predictor filtering removed outcome, time, status, death, true-risk, and simulation-truth variables, the XGBoost results returned to a plausible range. This was a key demonstration of why simulation is useful: it can reveal modelling workflow errors before real-outcome modelling begins.

## 4. Step C: Fully Synthetic Exportable Data Generator

Step C was designed to generate fully synthetic data that could be exported and used outside the internal SLAM environment. It no longer outputs real patient-level rows.

The Step C generator uses the SLAM-inspired variable structure and broad summary information to create synthetic baseline features and simulated outcomes. The generated benchmark contains:

```text
8 scenarios
50 repetitions per scenario
5,000 synthetic individuals per repetition
2,000,000 synthetic rows in total
```

The eight scenarios are:

- `S0_linear_PH_inst30`: baseline linear proportional-hazards scenario with approximately 30% care-home event rate;
- `S1_linear_PH_inst15`: lower event-rate scenario, approximately 15%;
- `S2_linear_PH_inst45`: higher event-rate scenario, approximately 45%;
- `S3_nonlinear_interaction_inst30`: nonlinear and interaction-effect scenario;
- `S4_nonPH_inst30`: non-proportional-hazards scenario;
- `S5_MAR_missingness_inst30`: MAR missingness scenario;
- `S6_highdim_sparseMRI_inst30`: high-dimensional sparse MRI signal scenario;
- `S7_strong_death_competing_inst30`: stronger death competing-risk scenario.

The primary outcome coding is:

```text
status = 0: event-free or censored
status = 1: care-home entry
status = 2: death before care home
```

`death_after_carehome` is a secondary post-care-home outcome. It is not the competing event for the primary endpoint.

The post-care-home death-hazard multiplier should be interpreted cautiously. In the simulation, post-care-home status can be treated as a marker of higher frailty and disease severity. It should not be interpreted as a causal claim that care-home entry increases mortality. Future sensitivity analyses could vary this multiplier, for example 0.8, 1.0, 1.5, and 2.0.

The Step C export audit found that the synthetic export passed safety checks, had no forbidden identifier columns, achieved event-rate calibration, activated the S5 missingness mechanism, activated the S7 competing-death mechanism, and showed a monotonic risk gradient across true linear-predictor quartiles.

## 5. Step C2: Cause-specific Survival Model Comparison

Step C2 compares model ranking and discrimination for care-home entry using cause-specific survival modelling. In C2, `status = 1` is the event, while `status = 0` and `status = 2` are treated as censored. This is not a full competing-risk absolute-risk analysis.

The C2 models are:

- `oracle_true_lp_not_a_model`;
- `cox_dgm_features`;
- `penalised_cox_all_safe_predictors`;
- `xgb_survival_cox_strict`.

The completed full run contained:

```text
8 scenarios x 50 repetitions x 4 models = 1,600 replicate-model rows
failures = 0
```

The main C2 findings were:

- Cox using DGM-relevant features performed close to the oracle in the linear proportional-hazards scenarios.
- In the nonlinear interaction scenario, XGBoost did not dominate; the best fitted model remained `cox_dgm_features`.
- Missingness in S5 had only modest impact on the leading fitted model.
- In S6, the high-dimensional sparse MRI scenario, `penalised_cox_all_safe_predictors` was the best fitted model.
- In S7, the stronger death competing-risk scenario, `cox_dgm_features` remained the best fitted model for cause-specific discrimination.
- XGBoost did not unrealistically exceed the oracle, supporting the strict leakage filtering.

## 6. Step C3: Competing-risk 5-year Absolute-risk Evaluation

Step C3 addresses the main limitation of C2 by evaluating 5-year care-home cumulative incidence under competing risk.

The C3 endpoint interpretation is:

```text
status = 1: care-home entry
status = 2: death before care home, competing event
horizon = 5 years
```

Again, death after care-home entry is not the primary competing event.

C3 uses cause-specific Cox models to reconstruct the care-home cumulative incidence function. The main evaluated models are:

- `oracle_true_risk_not_a_model`;
- `nonparametric_aj_null`;
- `cs_cox_dgm_cif`;
- `cs_penalised_cox_all_safe_cif`.

Fine-Gray evaluation is optional. In the local run, it was skipped because the R package `cmprsk` was unavailable or failed. This did not affect the mandatory C3 pipeline.

The completed C3 full run contained:

```text
8 scenarios x 50 repetitions x 4 main models = 1,600 rows
failures = 0
full_run_passed = True
```

Core C3 metrics included MAE and RMSE versus the exported synthetic 5-year true risk, naive 5-year Brier score, observed 5-year AUC, calibration slope and intercept, calibration deciles, and a cause-specific C-index for continuity with C2.

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

The best non-oracle model was `cs_cox_dgm_cif` in 7 of 8 scenarios. The exception was S6, where `cs_penalised_cox_all_safe_cif` was best. In S7, the best C3 model had mean MAE 0.0367, whereas the Aalen-Johansen null MAE was 0.1088. This supports the value of individualised competing-risk prediction in the stronger competing-death setting.

## 7. Overall Interpretation

This project is a methodological simulation study. It does not make clinical claims about real patients.

The main lessons are:

- simulation helped test the pipeline before real outcome availability;
- correctly structured Cox models were robust across most scenarios;
- more complex models did not generally outperform Cox;
- the high-dimensional sparse MRI scenario was the main setting where an all-safe penalised Cox model was preferable;
- individualised competing-risk CIF prediction improved on a non-individualised Aalen-Johansen null in the strong competing-death scenario;
- strict leakage filtering prevented XGBoost from producing implausibly high results.

## 8. Limitations

The care-home endpoint is simulated, not a real SLAM outcome. Step B semi-synthetic data cannot be exported because it contains real patient-level predictors. Step C synthetic data can be exported, but the large scenario datasets are not committed to GitHub. Step C2 is a cause-specific ranking analysis rather than a competing-risk absolute-risk analysis. Step C3 uses cause-specific Cox CIF reconstruction. The Brier score in C3 is naive rather than IPCW-adjusted. Earlier DGM-feature models used fallback predictor lists; Step C5A now exports exact raw pre-rescaling DGM coefficients, while final effective coefficients still require repetition-level raw-LP moments that were not exported in Step C v1. The post-care-home death-hazard multiplier is a simulation assumption about a high-frailty post-care-home state, not a causal claim.
