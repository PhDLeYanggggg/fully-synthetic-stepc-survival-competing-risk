# Step C4 Extended Model Comparison

Step C4 extends the completed Step C2 and Step C3 analyses by adding additional survival and competing-risk model families. It does not regenerate Step C data and does not rerun the original C2/C3 models.

## Added Models

The C4 debug run includes:

- Fine-Gray via R `cmprsk`: `finegray_dgm_cif_R` and `finegray_all_safe_reduced_R`;
- Random Survival Forest via `scikit-survival`: `cs_rsf_dgm_cif` and `cs_rsf_all_safe_cif`;
- Gradient Boosting Survival via `scikit-survival`: `cs_gbsa_dgm_cif` and `cs_gbsa_all_safe_cif`;
- optional DeepSurv and DeepHit placeholders, logged as skipped when `pycox`/`torchtuples` are unavailable.

## Debug Outcome

The C4 debug run completed all eight synthetic scenarios with two repetitions per scenario. It produced 160 replicate-model rows:

```text
8 scenarios x 2 repetitions x 10 model entries = 160 rows
```

The six core classical models completed successfully in every debug replicate. DeepSurv and DeepHit were skipped because optional deep-learning dependencies were unavailable. There were no failed model fits.

Fine-Gray, RSF, and GBSA therefore all passed the debug execution gate. The oracle/audit table flagged GBSA calibration slopes outside the prespecified 0.5 to 1.5 range in the debug run, so GBSA calibration requires further audit before publication-level interpretation.

## GitHub Scope

Only code and small debug summary/audit tables are included. Figures were generated locally but are not committed because figure upload requires explicit user confirmation.

The C4 debug summary files are under:

```text
results_summary/C4_debug_summary/
```

The C4 source code is under:

```text
src/stepC4_extended_models/stepC4_extended_model_comparison.py
```

and retained at the repository root for compatibility.
