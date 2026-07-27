# Fully Synthetic Competing-Risk Benchmark for Care-Home Admission

This repository contains the code and methodological documentation for a
repeated simulation study of time to care-home admission in the presence of
death before care-home admission.

The release is intentionally code-only. It contains no real SLaM records, no
semi-synthetic records, no synthetic individual-level datasets, no
individual-level predictions, and no result tables or figures.

## Study Purpose

Exact care-home entry dates were not available when the modelling pipeline was
designed. A fully synthetic benchmark with a closed-form five-year CIF target in seven
proportional-hazards scenarios and an explicitly qualified proxy in the
non-proportional-hazards scenario was therefore created to test:

- outcome coding and train/test separation;
- cause-specific and competing-risk prediction;
- predictor leakage controls;
- discrimination, absolute-risk error, and calibration;
- robustness to nonlinearity, non-proportional hazards, missingness,
  high-dimensional sparse MRI signal, and stronger competing mortality.

The study is a software and methods benchmark. It is not a clinical validation
study and its coefficients are not estimates of effects in real patients.

## Simulation Design

Eight scenarios each contain 50 independently generated repetitions of 5,000
synthetic individuals. Baseline demographic, cognition, MRI, WMH, structured
NLP, comorbidity, and deprivation variables are generated from explicit
probability models. Four shared latent factors induce dependence among
observable variables.

The generator's `MAR-lite` label denotes structured missingness imposed after
outcome generation. Missingness depends partly on exported variables and partly
on unexported latent frailty, vascular, and neurodegenerative factors. It is
therefore informative missingness from the analyst's observed-data perspective,
not a claim of strict Rubin-MAR.

The primary endpoint is time to care-home entry:

- `status = 0`: event-free at five years;
- `status = 1`: care-home entry;
- `status = 2`: death before care-home entry.

Death after care-home entry is a secondary post-care-home state and is not the
competing event for the primary endpoint.

The care-home and death data-generating mechanisms are prespecified simulation
designs. Their coefficients control signal direction and difficulty; they are
not fitted clinical hazard ratios. See
[the generator documentation](docs/04_stepC_fully_synthetic_generator.md) and
[the exact DGM audit](docs/13_stepC5A_exact_DGM_coefficients.md).

## Analysis Stages

```text
Step C   fully synthetic data generation
Step C2  cause-specific ranking baselines
Step C3  five-year competing-risk prediction
Step C4  extended classical and canonical neural models
Step C5A exact DGM coefficient and reconstruction audit
Step C6  paired publication-oriented statistical synthesis
```

Step C2 compares the exported oracle/proxy benchmark, DGM-informed cause-specific Cox,
penalised all-safe Cox, and strict XGBoost-Cox risk scores.

Step C3 compares the oracle five-year risk benchmark, a non-individualised
Aalen-Johansen estimate, and two-cause cumulative-incidence predictions from
DGM-informed and all-safe Cox models.

Step C4 adds Fine-Gray, Random Survival Forest, Gradient Boosting Survival,
canonical `pycox.models.CoxPH` DeepSurv models, and canonical
`pycox.models.DeepHit` competing-risk models. DeepHit tuning is performed only
inside the outer training sample.

Step C6 performs paired comparisons within scenario and repetition, Monte Carlo
standard-error estimation, normal and bootstrap confidence intervals,
calibration summaries, model ranks, and stress-test contrasts. It fits no model
and regenerates no data. It also writes SHA-256 manifests for the local
synthetic package, C2-C4 model-result tables, C5A DGM-audit tables, and final
audited analysis-source state used by the publication synthesis.

The original 50-repetition count was a pragmatic computational choice, not a
prospective precision calculation. Step C6 reports realised Monte Carlo
standard errors and confidence-interval half-widths for transparency.

The original C2 specification preceded its post-QC full run. C3, C4, and C6
were later extensions and were not prospectively registered; C6 hypothesis
tests are exploratory, with paired effect estimates and Monte Carlo uncertainty
treated as primary.

## Truth-Target Qualification

In S0-S3 and S5-S7, conditional event times use two independent, time-constant
cause-specific hazards. The exported two-hazard expression is the closed-form
five-year CIF implied by the returned calibrated hazards and LPs. If those
hazards are treated as fixed, the expression is their algebraic CIF. Because
baseline rates were selected using the realised unit-exponential draws in each
finite repetition, it is interpreted as an analytic target on the calibrated
DGM hazard surface rather than a population conditional probability fixed
independently before that repetition was generated.

S4 generates care-home times from separate early and late predictors, including
an unexported latent-frailty term. Its exported risk and LP fields use a single
base predictor and are approximate non-PH proxies. S4 truth-based error and
oracle comparisons are exploratory. Observed-outcome Brier score, AUC,
calibration, and cause-specific C-index remain valid.

## Post-QC Status

The locked local analyses contain:

- 1,600 Step C2 scenario-repetition-model evaluations;
- 1,600 calibrated Step C3 evaluations;
- 4,000 strict Step C4 evaluations;
- 7,200 harmonised C2-C4 replicate-model records in the Step C6 synthesis;
- 50 complete repetitions in every scenario-model cell;
- no failed or skipped required model fits;
- no forbidden predictor in a fitted post-QC model; and
- independently recomputed SHA-256 manifests matching the locked input tables
  and audited analysis-source state.

The final all-safe set contains 157 baseline predictors. Reference diagnosis,
outcome, post-baseline, identifier, hazard, risk, censoring, and simulation-truth
fields are excluded by block, role, and forbidden-name checks. Export-safety
values are parsed explicitly, so the string `"False"` cannot be interpreted as
safe.

## Qualitative Findings

The post-QC results are described here without publishing unapproved tables or
figures.

- DGM-informed Cox models recovered much of the intended risk ranking in
  predominantly additive proportional-hazards scenarios.
- Individualised cumulative-incidence models improved substantially on the
  non-individualised Aalen-Johansen baseline.
- Relative performance depended on the stress scenario; no model family
  dominated every setting.
- Broader all-safe predictors were most relevant when signal was deliberately
  distributed across regional MRI variables.
- Canonical DeepSurv and DeepHit implementations produced finite probabilities
  without using test data for training or tuning, but the study does not claim
  that neural or classical models are universally superior.
- These findings validate behaviour under the simulated mechanisms and
  qualified truth targets only. Evaluation in authorised real data, followed
  by external validation in an independent population, is required before any
  clinical claim.

## Installation

Using pip:

```bash
python -m pip install -r requirements.txt
```

Using conda:

```bash
conda env create -f environment.yml
conda activate synthetic-carehome-competing-risk
```

The exact package versions used for the locked post-QC analyses are recorded in
[`environment-lock-20260727.txt`](environment-lock-20260727.txt). This
provenance record is not intended to replace the portable installation
specifications above.

The conda environment includes R, `survival`, and `cmprsk`, which are required
for the Fine-Gray models. A pip-only installation does not install R; provide
`Rscript` with those two R packages separately before running complete C4.

The modelling scripts expect a local input directory named
`fully_synthetic_stepC_v1/`. That directory is not distributed.

## Running the Pipeline

```bash
python src/stepC_generator/stepC_quick_audits.py
bash scripts/run_stepC2_debug.sh
bash scripts/run_stepC2_full.sh
bash scripts/run_stepC3_debug.sh
bash scripts/run_stepC3_full.sh
bash scripts/run_stepC4_debug.sh
bash scripts/run_stepC4_full.sh
bash scripts/run_stepC5A_exact_DGM_coefficients.sh
bash scripts/run_stepC6_publication_analysis.sh
```

Full Step C4 execution is computationally intensive. For a new run, use
`scripts/run_stepC4_full.sh`. The following scenario-sharded utility documents
the post-QC migration used for this study: it requires an existing legacy C4
result table, preserves four screened classical model entries, and reruns six
specified entries by scenario.

```bash
python run_stepC4_postQC_sharded.py --max-workers 8
```

## Repository Structure

```text
.
├── docs/
├── scripts/
├── src/
│   ├── stepC_generator/
│   ├── stepC2_model_comparison/
│   ├── stepC3_competing_risk/
│   ├── stepC4_extended_models/
│   ├── stepC5A_exact_DGM_coefficients/
│   └── stepC6_publication_analysis/
├── tests/
├── CODE_INDEX.md
├── CITATION.cff
├── environment-lock-20260727.txt
├── environment.yml
└── requirements.txt
```

Root-level script copies are retained as direct command-line entry points. See
[CODE_INDEX.md](CODE_INDEX.md) for the complete map.

The [analysis specification and deviations record](docs/14_analysis_specification_and_deviations.md)
separates original requirements from later extensions and QC corrections. The
[ADEMP/TRIPOD+AI-informed checklist](docs/15_reporting_checklist_ADEMP_TRIPODAI.md)
tracks reporting completeness without treating a synthetic benchmark as a
real-patient validation study.

## Data and Result Release Policy

Generated outputs are ignored by Git. No result table or figure will be added
without explicit author review and approval. The current
[`results_summary`](results_summary/README.md) directory contains only this
policy statement.

The public generator documents the full probability mechanism but expects an
authorised internal aggregate-summary source that is not included. The
repository therefore supports methodological audit and re-implementation of
the design, but it cannot independently reproduce the reported numerical
results without the locked local synthetic package. SHA-256 manifests provide
an internal evidence lock without distributing those files.

## Limitations

- The benchmark cannot establish clinical transportability, fairness, or
  utility.
- Functional dependency, caregiver availability, living arrangements, and
  access to social-care services were not simulated, although they may be
  important determinants of real care-home entry.
- The generator is informed by broad aggregate characteristics and does not
  reproduce a real empirical covariance matrix.
- The authorised aggregate-summary source and individual synthetic package are
  not public, so the code-only release does not provide independent numerical
  reproduction of the manuscript results.
- Most scenarios retain substantial additive structure.
- Model-development budgets are not identical across every family.
- The design uses one sample size and one five-year horizon.
- Marginal care-home rates are tightly calibrated design quantities.
- S4 does not contain an exact exported individual CIF or full time-varying
  oracle LP; its truth-based metrics are proxy analyses.
- `MAR-lite` is not strict observed-data MAR because unexported latent factors
  also drive missingness.

## License

The code is released under the [MIT License](LICENSE).
