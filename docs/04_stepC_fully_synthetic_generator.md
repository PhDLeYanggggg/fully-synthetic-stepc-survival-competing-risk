# Step C Fully Synthetic Generator

Step C generates fully synthetic, exportable datasets inspired by the internal SLAM variable structure. Its purpose is to create a benchmark that can be shared as synthetic data without releasing real patient-level records.

The generator code is included under:

```text
src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py
```

The original notebook source is not included in this repository. The script was converted from the local notebook for source-code transparency.

## Generated Scenarios

The intended synthetic benchmark contains eight scenarios:

- S0: linear proportional hazards, target institutionalisation rate 30%;
- S1: linear proportional hazards, target rate 15%;
- S2: linear proportional hazards, target rate 45%;
- S3: nonlinear and interaction effects;
- S4: non-proportional hazards;
- S5: MAR missingness;
- S6: high-dimensional sparse MRI signal;
- S7: stronger death competing risk.

Each scenario contains 50 repetitions and each repetition contains 5,000 synthetic individuals.

## Export Safety

The generator writes audit tables designed to check export safety, missingness behaviour, event-rate calibration, outcome gradients, and forbidden identifier names. The large scenario datasets themselves are not tracked in this GitHub repository.

## Important Interpretation

The simulated post-care-home death process is a state-risk assumption. It should be interpreted as a marker of higher frailty or severity after care-home entry, not as a causal statement that care-home entry increases mortality.
