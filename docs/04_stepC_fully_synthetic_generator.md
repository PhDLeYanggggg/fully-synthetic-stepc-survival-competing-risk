# Step C Fully Synthetic Generator

Step C generates fully synthetic, exportable datasets inspired by the internal SLAM variable structure. Its purpose is to create a benchmark that can be shared as synthetic data without releasing real patient-level records.

The generator code is included under:

```text
src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py
```

The original notebook source is not included in this repository. The script was converted from the local notebook for source-code transparency.

## Authorised Internal Configuration

The generator is not supplied with an internal summary source. Inside the
authorised environment, configure the source and a separate internal-only audit
directory before running:

```bash
export STEPC_INTERNAL_SOURCE_CSV="/authorised/path/to/source.csv"
export STEPC_INTERNAL_AUDIT_DIR="/authorised/path/to/stepC_internal_audit"
python src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py
```

The internal audit directory must be outside `fully_synthetic_stepC_v1/`.
The generator enforces this boundary and stops if source-derived summaries
would be written inside the exportable package.

## Generated Scenarios

The intended synthetic benchmark contains eight scenarios:

- S0: linear proportional hazards, target institutionalisation rate 30%;
- S1: linear proportional hazards, target rate 15%;
- S2: linear proportional hazards, target rate 45%;
- S3: nonlinear and interaction effects;
- S4: non-proportional hazards;
- S5: stronger MAR-lite structured missingness;
- S6: high-dimensional sparse MRI signal;
- S7: stronger death competing risk.

Each scenario contains 50 repetitions and each repetition contains 5,000 synthetic individuals.

## Exported Truth Targets

For S0-S3 and S5-S7, care-home and death times are generated from independent,
time-constant cause-specific hazards. The exported observable five-year risk is
the closed-form two-hazard CIF implied by the returned calibrated hazards and
LPs. If those hazards are treated as fixed, the expression is their algebraic
CIF. The baseline rates are selected using the realised unit-exponential draws
in each finite repetition, so this is interpreted as an analytic target on the
calibrated DGM hazard surface rather than a population conditional probability
fixed independently before the repetition was generated.

S4 uses separate early and late care-home predictors. The export does not
contain those two final rescaled predictors, and the late predictor includes an
unexported latent-frailty term. The exported S4 risk and care-home LP are
therefore approximate base-LP proxies, not an exact non-PH individual CIF or a
full time-varying oracle score. This is a limitation of the existing export and
cannot be corrected without regenerating data.

## Missingness Qualification

The generator calls its mechanism `MAR-lite`. It is applied after outcomes are
generated and depends on exported age, cognition, deprivation and related
features, but also on unexported latent frailty, vascular and neurodegenerative
factors. Consequently, it is structured informative missingness from the
analyst's observed-data perspective and should not be described as strict
Rubin-MAR. S5 increases the intensity of this same mechanism.

## Export Safety

The generator writes only synthetic export-safety and simulation summary tables
inside the exportable package. Source-derived summaries stay in the separate
internal-only audit directory. After generation,
`src/stepC_generator/stepC_quick_audits.py` checks high missingness,
true-LP outcome gradients, and monotonicity using only the fully synthetic
package. The large scenario datasets themselves are not tracked in this GitHub
repository.

## Important Interpretation

The simulated post-care-home death process is a state-risk assumption. It should be interpreted as a marker of higher frailty or severity after care-home entry, not as a causal statement that care-home entry increases mortality.
