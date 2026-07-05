# Step C4 Extended Model Comparison

Step C4 extends the completed Step C2 and Step C3 analyses by adding additional survival and competing-risk model families. It does not regenerate Step C data and does not rerun the original C2/C3 models.

## Added Models

The C4 pipeline includes:

- Fine-Gray via R `cmprsk`: `finegray_dgm_cif_R` and `finegray_all_safe_reduced_R`;
- Random Survival Forest via `scikit-survival`: `cs_rsf_dgm_cif` and `cs_rsf_all_safe_cif`;
- Gradient Boosting Survival via `scikit-survival`: `cs_gbsa_dgm_cif` and `cs_gbsa_all_safe_cif`;
- DeepSurv neural Cox baselines: `deepsurv_dgm_cause_specific_optional` and `deepsurv_all_safe_cause_specific_optional`;
- DeepHit-style competing-risk neural PMF baselines: `deephit_competing_risk_dgm_optional` and `deephit_competing_risk_all_safe_reduced_optional`.

## Full Local Outcome

The C4 full run completed all eight synthetic scenarios with 50 repetitions per scenario. It produced 4,000 scenario-replicate-model rows:

```text
8 scenarios x 50 repetitions x 10 model entries = 4,000 rows
Model failures = 0
Model skips = 0
Fine-Gray = completed
RSF = completed
GBSA = completed
DeepSurv = completed
DeepHit = completed
full_run_passed = True
publication_ready = False pending review of oracle-sanity calibration/risk-distribution flags
```

All ten model entries completed successfully in every full-run replicate. There were no failed model fits and no skipped model fits. The local `pycox` and `torchtuples` dependencies were installed, but their training path was unstable in the local native runtime; C4 therefore uses deterministic NumPy neural-network implementations for the DeepSurv and DeepHit-style baselines.

Fine-Gray, RSF, GBSA, DeepSurv, and DeepHit therefore all passed the full execution gate. The updated C4 oracle-sanity audit is not fully clean because several deep-model calibration and risk-distribution screens require review. The earlier Step C4A audit reviewed the original `S1_linear_PH_inst15 / cs_gbsa_dgm_cif / calibration_slope_5y_mean = 1.6026` flag and interpreted it as calibration instability, not leakage or oracle outperformance. After adding DeepSurv and DeepHit, C4A should be rerun before making a publication-ready claim.

## GitHub Scope

Only code and small debug summary/audit tables are included. Full C4 tables and figures were generated locally but are not committed because result-table and figure upload requires explicit user confirmation.

The C4 debug summary files are under:

```text
results_summary/C4_debug_summary/
```

The C4 source code is under:

```text
src/stepC4_extended_models/stepC4_extended_model_comparison.py
```

and retained at the repository root for compatibility.

The C4A calibration-audit source code is under:

```text
src/stepC4A_calibration_audit/stepC4A_calibration_audit.py
```
