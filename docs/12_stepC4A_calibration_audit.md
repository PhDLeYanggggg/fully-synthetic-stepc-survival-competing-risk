# Step C4A Calibration-Slope Audit

Step C4A is a targeted audit of the single C4 oracle sanity flag:

```text
S1_linear_PH_inst15 / cs_gbsa_dgm_cif / calibration_slope_5y_mean = 1.6026
```

The audit reads aggregate C4 outputs only. It does not read real SLAM data,
Step B data, raw spreadsheets, large scenario datasets, fitted models, or
per-person predictions.

## Purpose

C4 completed the full extended-model comparison, but publication readiness was
held pending review of one calibration-slope flag. C4A determines whether that
flag indicates leakage/oracle outperformance or instead reflects a model
calibration issue in a low-event-rate scenario.

## Script

```bash
python src/stepC4A_calibration_audit/stepC4A_calibration_audit.py
```

or:

```bash
bash scripts/run_stepC4A_calibration_audit.sh
```

## Expected Local Inputs

- `fully_synthetic_stepC4_extended_models/tables/replicate_extended_model_performance.csv`
- `fully_synthetic_stepC4_extended_models/tables/scenario_extended_model_summary_mean_sd_ci.csv`
- `fully_synthetic_stepC4_extended_models/tables/oracle_sanity_audit_C4.csv`
- `fully_synthetic_stepC4_extended_models/tables/calibration_deciles_summary_C4.csv`
- optional C3 summaries for Cox/Aalen-Johansen/oracle comparison

These inputs are generated locally and are not included in the repository by
default.

## Local Interpretation

The completed local C4A audit found that the flag is best interpreted as
calibration instability for `cs_gbsa_dgm_cif` in the low-institutionalisation
S1 scenario. It was not accompanied by leakage evidence or oracle
outperformance, and the flagged model was not the preferred S1 model.

Figures and detailed audit tables remain local unless explicitly approved for
upload.
