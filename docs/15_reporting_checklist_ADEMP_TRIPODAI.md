# Reporting Checklist for the Step C Simulation Manuscript

## ADEMP

| Domain | Current manuscript evidence | Status |
| --- | --- | --- |
| Aims | Primary and secondary simulation aims are stated in the Introduction. | Complete |
| Data-generating mechanisms | Base care-home and death equations, LP rescaling, event-time generation, missingness, and S3/S4/S6 mechanisms are reported. | Complete |
| Estimands | Cause-specific ranking and five-year care-home cumulative incidence are distinguished. | Complete |
| Methods | Model families, predictor sets, train/test split, internal neural validation, and leakage controls are reported. | Complete |
| Performance measures | C-index, truth correlation, MAE, RMSE, Brier, AUC, calibration, runtime, failure rate, paired differences, and Monte Carlo uncertainty are reported. | Complete |
| Truth-target validity | The closed-form PH CIF target is qualified for finite-sample baseline-hazard calibration and distinguished from the additional approximate S4 non-PH proxy; S4 truth metrics are exploratory. | Complete |

## TRIPOD+AI-Informed Transparency

TRIPOD+AI was written for prediction-model studies rather than purely
synthetic software benchmarks. The following principles are applied where
relevant.

| Item | Current handling | Status |
| --- | --- | --- |
| Source of data | Fully synthetic package and boundary from real/Semi-synthetic data are explicit. | Complete |
| Participants | No real participants; synthetic population mechanism is described. | Complete |
| Outcome | Duration, status codes, horizon, and competing event are explicit. | Complete |
| Predictors | Exact fitted DGM-informed and all-safe lists are exported locally; the DGM-informed list is explicitly identified as an observable proxy set. | Complete |
| Sample size | 5,000 individuals x 50 repetitions x 8 scenarios is explicit; 50 is identified as pragmatic and realised MCSE is reported. | Complete |
| Missing data | Outcome-before-missingness generation is reported; MAR-lite is explicitly qualified as informative because latent drivers are unexported. | Complete |
| Model development | Outer split, preprocessing, hyperparameters, early stopping, DeepHit selection, development-subset weight fitting, and the absence of a post-selection neural refit are reported. | Complete |
| Performance | Discrimination, absolute error, and calibration are jointly reported. | Complete |
| Internal validation | Repeated independent datasets with held-out test sets; no cross-validation claim. | Complete |
| Model specification | Software versions and neural specification are supplied as supplementary evidence. | Complete |
| Limitations | Synthetic-only scope, DGM favourability, fixed horizon/sample size, tuning differences, and event-rate calibration are stated. | Complete |
| Clinical interpretation | Manuscript explicitly avoids claims of clinical utility or transportability. | Complete |
| Ethics and governance | Placeholder requires formal wording and project reference from the study team. | Team confirmation required |
| Funding and conflicts | Placeholders remain. | Team confirmation required |
| Authors and contributions | Le Yang draft role entered; co-author list and CRediT roles remain. | Team confirmation required |
| Data/code availability | Code-only repository, dependence on a non-public authorised aggregate-summary source, non-release of the individual synthetic package, resulting limit on independent numerical reproduction, SHA-256 evidence manifests, and prohibition on unapproved result artifacts are stated. | Complete |
| AI disclosure | Draft disclosure is included and must be adapted to journal policy. | Team confirmation required |
| Target-journal structure | The working manuscript includes a structured abstract, abbreviation list, and complete BMC declaration subheadings. | Complete |

## Submission Gate

The computational evidence is locked: C2 and calibrated C3 each contain 1,600
complete evaluations; strict C4 contains 4,000 evaluations and 40,000
calibration-decile records, with 0 failures, 0 skips, an empty leakage audit,
and all 26 sanity gates passed; and C6 contains 7,200 harmonised
replicate-level records with all automated integrity checks passed. The
manuscript cannot be submitted until the author team completes the
ethics/governance, authorship, funding, and competing-interest fields.
