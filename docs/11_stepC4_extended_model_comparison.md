# Step C4 Extended Model Comparison

Step C4 extends the completed Step C2 and Step C3 analyses by adding additional survival and competing-risk model families. It does not regenerate Step C data and does not rerun the original C2/C3 models.

## Added Models

The C4 pipeline includes:

- Fine-Gray via R `cmprsk`: `finegray_dgm_cif_R` and `finegray_all_safe_reduced_R`;
- Random Survival Forest via `scikit-survival`: `cs_rsf_dgm_cif` and `cs_rsf_all_safe_cif`;
- Gradient Boosting Survival via `scikit-survival`: `cs_gbsa_dgm_cif` and `cs_gbsa_all_safe_cif`;
- optional DeepSurv and DeepHit placeholders, logged as skipped when `pycox`/`torchtuples` are unavailable.

## Full Local Outcome

The C4 full run completed all eight synthetic scenarios with 50 repetitions per scenario. It produced 4,000 scenario-replicate-model rows:

```text
8 scenarios x 50 repetitions x 10 model entries = 4,000 rows
Core classical model failures = 0
Fine-Gray = completed
RSF = completed
GBSA = completed
DeepSurv / DeepHit = skipped because optional pycox/torchtuples dependencies were unavailable
full_run_passed = True
publication_ready = False pending review of one calibration-slope audit flag
```

The six core classical models completed successfully in every full-run replicate. DeepSurv and DeepHit were skipped because optional deep-learning dependencies were unavailable. There were no failed model fits.

Fine-Gray, RSF, and GBSA therefore all passed the full execution gate. The only C4 oracle sanity flag was `S1_linear_PH_inst15 / cs_gbsa_dgm_cif / calibration_slope_5y_mean = 1.6026`; no fitted C4 model exceeded the oracle MAE, AUC, or C-index screens. This is a calibration-slope review item, so the full run passes technically but is not marked publication-ready until that flag is reviewed.

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
