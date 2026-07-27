# Step C5A Exact DGM Coefficients

Step C5A exports the raw data-generating-mechanism coefficients from the fully
synthetic Step C generator. It separates the exact raw simulation terms from
the fixed 36-variable observable DGM-informed proxy set used in C2/C3/C4.
C5A improves generator transparency; it does not retrospectively redefine
those fitted models as exact algebraic DGM models.

## Script

```bash
python src/stepC5A_exact_DGM_coefficients/stepC5A_exact_DGM_coefficients.py
```

or:

```bash
bash scripts/run_stepC5A_exact_DGM_coefficients.sh
```

## Expected Local Inputs

- one generator source:
  `code/stepC_fully_synthetic_exportable_generator_v1.ipynb`, when available
  internally, or
  `src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py`
- `fully_synthetic_stepC_v1/tables/dgm_definition_table.csv`
- `fully_synthetic_stepC_v1/tables/feature_dictionary.csv`
- `fully_synthetic_stepC_v1/tables/scenario_summary.csv`
- `fully_synthetic_stepC_v1/tables/repetition_summary.csv`
- bounded samples from `fully_synthetic_stepC_v1/scenario_datasets/*.csv.gz`

The scenario datasets are not copied into this repository.

## Output

The local script writes:

- exact care-home raw DGM coefficient table
- exact death-before-carehome raw DGM coefficient table
- post-carehome death mechanism table
- scenario-level DGM summary
- bounded true-LP reconstruction audit
- README and documentation update suggestions

## Interpretation

C5A exports exact raw pre-rescaling coefficients from the generator code.
Final effective LP values also depend on repetition-level centering/scaling
moments that were not exported in Step C v1. The nonlinear S3 scenario contains
one latent frailty-by-vascular interaction term that is intentionally not
available as an exportable baseline predictor, so reconstruction is expected to
be incomplete for S3.

The post-carehome death multiplier should be described as a state-risk or
frailty-state assumption, not as a causal claim that care-home entry increases
mortality.
