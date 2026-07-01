# Step B Internal Semi-synthetic Stage

Step B was an internal, non-exportable stage used to develop and stress-test the survival modelling pipeline.

It combined real SLAM baseline predictors with simulated care-home entry times. Internal death-date information was used to define death before care home as a competing event reference. Because the predictor matrix still contained real patient-level rows, Step B data are not included in this repository and should not be exported.

## What Step B Tested

Step B tested whether the modelling workflow could:

- construct baseline-only predictor matrices;
- encode time to care-home entry;
- encode censoring and death before care home;
- compare survival models against a known simulated truth;
- detect leakage from outcome, time, death, or truth variables.

## Leakage Lesson

An early XGBoost survival result was implausibly strong and exceeded the oracle benchmark. This was a warning sign. After strict predictor filtering removed outcome-derived and simulation-truth variables, model behaviour became plausible.

This was one of the central motivations for the strict all-safe predictor audit used later in Step C2 and Step C3.
