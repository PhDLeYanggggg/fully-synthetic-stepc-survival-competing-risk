# From Internal SLAM Semi-synthetic Data to Fully Synthetic Competing-risk Evaluation

This document describes the full project workflow from the internal SLAM diagnostic modelling data structure to the exportable fully synthetic survival and competing-risk benchmark.

## 1. Project Background

The starting point was an internal SLAM diagnostic modelling workflow for
dementia subtype classification. The diagnostic labels in that earlier
project were AD, Mixed, and Non-AD. The internal workflow aligned cognition
with scan date, de-duplicated scans, and retained a baseline observation per
person. Exact internal cohort counts are intentionally not disclosed in this
code-only repository.

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
- `S5_MAR_missingness_inst30`: higher MAR-lite structured missingness scenario;
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

Fine-Gray is outside the specified four-entry C3 comparison. It is evaluated in
Step C4, where the final specified Fine-Gray comparisons completed using the R
package `cmprsk`.

The completed C3 full run contained:

```text
8 scenarios x 50 repetitions x 4 main models = 1,600 rows
failures = 0
full_run_passed = True
```

Core C3 metrics included MAE and RMSE versus the exported five-year DGM target
in the seven proportional-hazards scenarios (with the S4 field treated as a
qualified proxy), five-year Brier score, observed five-year AUC, calibration
slope and intercept, calibration deciles, and a cause-specific C-index for
continuity with C2. All status-0 observations were administratively censored at
five years, so the binary five-year event indicator was observed for every
individual.

Post-QC results are described qualitatively in the public repository.
DGM-informed Cox CIF prediction was generally stable, while the all-safe model
was most useful when additional regional MRI variables carried signal. In the
strong competing-mortality scenario, individualised prediction clearly
improved on the non-individualised Aalen-Johansen baseline. Numerical tables
remain local pending explicit author approval.

## 7. Overall Interpretation

This project is a methodological simulation study. It does not make clinical claims about real patients.

The main lessons are:

- simulation helped test the pipeline before real outcome availability;
- correctly structured Cox models were robust across most scenarios;
- no model family dominated every scenario and metric;
- the high-dimensional sparse MRI scenario was the main setting where an all-safe penalised Cox model was preferable;
- individualised competing-risk CIF prediction improved on a non-individualised Aalen-Johansen null in the strong competing-death scenario;
- strict leakage filtering prevented forbidden reference and truth fields from
  entering fitted models.

## 8. Limitations

The care-home endpoint is simulated, not a real SLaM outcome. Step B
semi-synthetic data cannot be exported because it contains real patient-level
predictors. Step C synthetic data can be exported, but the large scenario
datasets are not committed to GitHub. Step C2 is a cause-specific ranking
analysis rather than a competing-risk absolute-risk analysis. Step C3 uses
cause-specific Cox CIF reconstruction. Step C5A exports exact raw
pre-rescaling DGM coefficients, while final effective coefficients still
require repetition-level raw-LP moments that were not exported in Step C v1.
The post-care-home death-hazard multiplier is a simulation assumption about a
high-frailty post-care-home state, not a causal claim.
