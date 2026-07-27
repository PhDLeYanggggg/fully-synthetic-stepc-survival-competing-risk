# Post-QC Results Summary

No numerical result table or figure is committed to this repository. This file
records only the interpretation approved for the code release.

## Completed Analysis Cells

- Step C2: 8 scenarios x 50 repetitions x 4 models.
- Step C3: 8 scenarios x 50 repetitions x 4 models.
- Step C4: 8 scenarios x 50 repetitions x 10 models.
- Every required model cell completed without a failed or skipped fit.
- The strict all-safe predictor set contains 157 baseline predictors.
- No forbidden fitted predictor remained after the post-QC reruns.

## Qualitative Findings

The DGM-informed Cox models recovered much of the intended ordering in
predominantly additive proportional-hazards scenarios. Individualised
competing-risk predictions were more informative than a non-individualised
Aalen-Johansen estimate. Broader predictor sets became most relevant when the
simulation deliberately distributed signal across regional MRI measures.

Fine-Gray, Random Survival Forest, Gradient Boosting Survival, canonical pycox
DeepSurv, and canonical pycox DeepHit were evaluated under the same outer
train/test split. Relative performance varied by scenario and metric; no model
family dominated every stress test.

These findings concern recovery under explicit synthetic mechanisms, using
closed-form PH targets with the stated finite-repetition calibration
qualification and an explicitly qualified S4 proxy. They are not clinical
performance estimates and do not establish utility in a real memory-clinic
population.

## Local Statistical Synthesis

The uncommitted Step C6 output contains paired within-repetition comparisons,
Monte Carlo standard errors, normal and bootstrap confidence intervals,
calibration summaries, model ranks, stress-test contrasts, and DeepHit
training-only alpha-selection frequencies.

Result files remain local until the author team explicitly approves selected
tables or figures for public release.
