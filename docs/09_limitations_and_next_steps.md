# Limitations and Next Steps

## Limitations

The care-home entry outcome in this repository is simulated. It is not a real SLAM endpoint.

The project is designed for methodological testing and workflow validation. It should not be interpreted as clinical inference or as a validated real-world prediction model.

The synthetic baseline does not include functional dependency, caregiver
availability, living arrangements, or access to social-care services. These
factors may be important determinants of real care-home entry.

Step B internal semi-synthetic data cannot be exported because it contains real patient-level predictors. Step C fully synthetic data are exportable after audit, but the large scenario datasets are not tracked in GitHub.

Step C2 is a cause-specific ranking and discrimination analysis. It is not a full competing-risk absolute-risk analysis.

Step C3 deliberately evaluates four specified benchmark entries and uses
cause-specific Cox cumulative-incidence reconstruction for its individualised
fitted models. Fine-Gray is evaluated in Step C4, where all specified
scenario-repetition fits completed.

The C3 five-year Brier score uses the fully observed binary five-year status.
No censoring weights are needed in this synthetic design because every
individual's event-1 status is known at the horizon. This choice would not be
valid without modification in real data censored before five years.

The exported five-year target under the seven time-constant PH mechanisms is
the closed-form CIF implied by the returned calibrated hazards and LPs. If
those hazards are treated as fixed, the expression is their algebraic CIF. The
baseline rates were selected using each finite repetition's realised
unit-exponential draws, so this is an analytic target on the calibrated DGM
hazard surface rather than a population conditional probability fixed
independently before that repetition was generated. In S4,
care-home times use unexported early and late LPs, including
an unexported latent-frailty term. The exported S4 risk and LP are therefore
proxies; S4 truth-MAE and oracle comparisons are exploratory.

The generator label `MAR-lite` is not strict MAR conditional on the exported
observed data because unexported latent frailty, vascular, and
neurodegenerative factors also influence missingness.

The C2/C3/C4 DGM-informed models use a fixed clinically meaningful observable
proxy set; they are not exact algebraic DGM models. Step C5A separately exports
the exact raw pre-rescaling generator coefficients, but final effective
coefficients still require repetition-level raw-LP centering/scaling moments
that were not exported in Step C v1.

The post-care-home death-hazard multiplier is a state-risk simulation assumption. It should not be interpreted causally.

The exported oracle/proxy is a DGM benchmark, not a guaranteed finite-sample
upper bound for every empirical metric. Small apparent exceedances can occur
through held-out sample variation, score-definition differences and the S4
target mismatch. Oracle screens therefore use tolerances and trigger audit
rather than enforcing strict ordering.

## Next Steps

Useful future extensions include:

- real care-home outcome validation when the endpoint becomes available;
- competing-risk-appropriate censoring-weighted Brier scoring for real data;
- time-dependent AUC;
- multi-horizon evaluation at 1, 3, and 5 years;
- if a future data version is generated, export the S4 early/late LPs and exact
  piecewise competing-risk CIF;
- export repetition-level raw-LP moments for final effective DGM coefficient reconstruction;
- sensitivity analyses for the post-care-home death-hazard multiplier;
- packaging the fully synthetic scenario datasets through an approved data-release channel rather than GitHub source tracking.
