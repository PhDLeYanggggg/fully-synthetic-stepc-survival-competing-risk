# Limitations and Next Steps

## Limitations

The care-home entry outcome in this repository is simulated. It is not a real SLAM endpoint.

The project is designed for methodological testing and workflow validation. It should not be interpreted as clinical inference or as a validated real-world prediction model.

Step B internal semi-synthetic data cannot be exported because it contains real patient-level predictors. Step C fully synthetic data are exportable after audit, but the large scenario datasets are not tracked in GitHub.

Step C2 is a cause-specific ranking and discrimination analysis. It is not a full competing-risk absolute-risk analysis.

Step C3 uses cause-specific Cox cumulative-incidence reconstruction. Fine-Gray was optional and did not run in the local full run because `cmprsk` was unavailable.

The C3 Brier score is a naive Brier score, not an IPCW Brier score.

The Cox DGM feature list uses a fallback clinically meaningful predictor list because exact DGM coefficients were not exported.

The post-care-home death-hazard multiplier is a state-risk simulation assumption. It should not be interpreted causally.

## Next Steps

Useful future extensions include:

- real care-home outcome validation when the endpoint becomes available;
- formal Fine-Gray modelling;
- IPCW Brier scoring;
- time-dependent AUC;
- multi-horizon evaluation at 1, 3, and 5 years;
- exact DGM coefficient export;
- sensitivity analyses for the post-care-home death-hazard multiplier;
- packaging the fully synthetic scenario datasets through an approved data-release channel rather than GitHub source tracking.
