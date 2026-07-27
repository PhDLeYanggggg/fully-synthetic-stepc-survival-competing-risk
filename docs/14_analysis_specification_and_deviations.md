# Step C Analysis Specification and Deviations Record

## Purpose

This record distinguishes the original analysis requirements from later
extensions, implementation corrections, and publication-oriented analyses. It
is an audit trail, not a claim of prospective registration.

## Original Step C2 Specification

The initial Step C2 request fixed:

- eight fully synthetic scenarios;
- 50 repetitions of 5,000 individuals per scenario;
- a 70/30 train/test split within each repetition;
- status 1 as care-home entry and status 2 as death before care home;
- cause-specific treatment of status 2 in Step C2;
- four model entries: oracle true LP, DGM-informed Cox, all-safe penalised Cox,
  and strict XGBoost-Cox;
- held-out-test evaluation;
- C-index, truth correlations, five-year error metrics where valid, runtime,
  failure logging, and scenario-level aggregation;
- no DeepSurv, DeepHit, RSF, Fine-Gray, or neural model in Step C2.

## Later Planned Extensions

Step C3 added competing-risk-specific five-year cumulative-incidence
evaluation. Step C4 added Fine-Gray, Random Survival Forest, Gradient Boosting
Survival, DeepSurv, and DeepHit. These extensions were not part of the original
Step C2 model list and should not be described as if they were.

## QC-Driven Corrections

### Predictor role

`diagnosis_reference_NOT_PREDICTOR` was initially permitted through the
diagnosis-reference block despite a role marking it as not for prediction.
The correction:

- removed `diagnosis_reference` from allowed blocks;
- excluded role strings containing `not_for_prediction` or `not_predictor`;
- made C4 independently re-read feature-dictionary roles when screening
  inherited C2/C3 predictor lists;
- added `not_predictor`, `diagnosis_reference`, `target_diag`, and
  `final_diagnosis` to forbidden-name fragments;
- reran affected C2, C3, and C4 all-safe models.

C2 and C3 were rerun in full after their relevant corrections. The final C4
migration retained four unaffected classical entries from the earlier complete
checkpoint only after auditing their actual fitted-predictor strings:
DGM-informed Fine-Gray, training-reduced all-safe Fine-Gray, DGM-informed RSF,
and DGM-informed GBSA (1,600 scenario-repetition-model evaluations). Strict
all-safe RSF and GBSA and all four canonical neural entries were fitted anew
(2,400 evaluations). The merge required exactly 4,000 unique keys, 400 rows per
model, no failed or skipped fit, and a complete
model-by-scenario-by-repetition grid.

### Export-safety booleans

String values such as `"False"` could be treated as truthy by `astype(bool)`.
Export and merge gates now use explicit accepted tokens:
`{"true", "1", "yes"}`.

The same explicit parser is now used for C2/C3 checkpoint failure flags,
oracle-audit flags, fitted-predictor inclusion flags, failure summaries, and
resume decisions. This was a static hardening change after the verified full
runs; it did not alter any fitted model or result row.

### C3 calibration compatibility

The original unpenalised logistic calibration used
`LogisticRegression(penalty="none")`, which current scikit-learn rejected.
The estimator was changed to `penalty=None`, covered by a deterministic unit
test, and C3 was rerun in full. No model or DGM changed.

### Canonical neural implementations

Early custom NumPy neural-style baselines were removed. The current C4 uses:

- `pycox.models.CoxPH` for two cause-specific DeepSurv models;
- `pycox.models.DeepHit` for a joint two-risk model;
- an internal validation subset for early stopping;
- training-only DeepHit alpha selection;
- network weights estimated on the 80% development portion of the outer
  training set without a post-selection full-training refit;
- full outer-training baseline-hazard estimation for DeepSurv after network
  training;
- an explicit label transform that preserves status 0, 1, and 2 after time
  discretisation.

### Native thread stability

The local native runtime crashed when multiple numerical thread pools competed.
The full sharded run limits OMP, MKL, VecLib, and Torch thread counts. This is
an execution-stability control and does not alter the estimand or DGM.

### Exported S4 truth target

Post-QC code review established that the exported five-year target in S0-S3
and S5-S7 is the closed-form two-hazard CIF implied by the returned
time-constant calibrated hazards and LPs. If those hazards are treated as
fixed, the expression is their algebraic CIF. Because each repetition's
baseline rates were selected using its realised unit-exponential draws, the
quantity is interpreted as an analytic target on the calibrated DGM hazard
surface rather than a population conditional probability fixed independently
before that finite repetition was generated. S4 has an additional structural mismatch: it
generates care-home times from separate early and late LPs,
including an unexported latent-frailty term, while the exported risk and LP use
a single base predictor. Therefore:

- S4 truth-MAE, truth correlation, and oracle comparisons are exploratory;
- observed-outcome Brier score, AUC, calibration, and C-index remain valid;
- S4 oracle exceedance is recorded but does not by itself block publication;
- no DGM or synthetic row was changed or regenerated.

### Missingness terminology

The scenario name contains `MAR`, but the generator's MAR-lite score also uses
unexported latent frailty, vascular and neurodegenerative factors. The
publication therefore describes S5 as a stronger structured informative
missingness stress test, not as strict MAR conditional on the observed export.
No missingness mechanism was changed.

## Publication-Oriented Analyses Added After Model Fitting

Step C6 was added after C2-C4 model development. It fits no model and
regenerates no data. It adds:

- paired differences within the same scenario and repetition;
- Monte Carlo standard errors;
- normal and percentile-bootstrap confidence intervals;
- exploratory Wilcoxon tests with Holm adjustment;
- decile-weighted calibration errors;
- a complete 10-decile and 1,500-test-individual grid check for every
  scenario-repetition-model calibration table;
- model ranks and probability of best;
- scenario-versus-S0 stress-test contrasts;
- DeepHit alpha-selection summaries;
- an administrative-censoring audit;
- a scenario-level truth-target validity audit;
- SHA-256 manifests for the local synthetic package, final C2-C4 model-result
  tables, C5A DGM-audit tables, and final audited analysis-source state.

These analyses should be described as a post-model-fitting statistical
synthesis, with effect sizes and Monte Carlo uncertainty treated as primary.
The original 50-repetition design was a pragmatic computational choice rather
than a prospective Monte Carlo precision calculation; realised MCSEs and
confidence-interval half-widths are reported instead.

## Deliberate Non-Changes

The QC work did not:

- change any DGM coefficient;
- change any scenario target;
- regenerate any synthetic individual;
- add a model to Step C2;
- use real SLaM data;
- use the outer test set for preprocessing, tuning, or early stopping.

## Remaining Non-Computational Decisions

Before submission, the study team must confirm:

- author list, order, and CRediT roles;
- ethics and data-governance wording;
- funding and competing interests;
- target journal and article type;
- whether any aggregate result table or figure may be made public.
