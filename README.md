# SLAM-informed Synthetic Simulation for Dementia Institutionalisation Prediction

This repository contains a code and documentation release for a synthetic simulation framework inspired by the structure of SLAM dementia imaging and clinical data. The project evaluates survival and competing-risk methods for predicting care-home entry or institutionalisation.

The repository is intentionally code-first. It does not contain real SLAM data, internal semi-synthetic SLAM data, large synthetic scenario datasets, raw spreadsheets, identifiers, or per-person prediction files.

## Research Aim

The motivation is methodological. Real care-home entry dates were not yet available for direct prognostic modelling, so the project first builds a controlled simulation framework to test the modelling pipeline, outcome coding, competing-death handling, model comparison workflow, and leakage controls before applying similar methods to real outcomes.

The simulation route is:

```text
Real SLAM diagnostic data structure
    |
Internal semi-synthetic Step B
    |
Fully synthetic exportable Step C
    |
Step C2 cause-specific model comparison
    |
Step C3 competing-risk 5-year risk evaluation
```

## Repository Contents

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── docs/
├── src/
│   ├── stepC_generator/
│   ├── stepC2_model_comparison/
│   └── stepC3_competing_risk/
├── scripts/
├── results_summary/
└── tests/
```

The root-level C2 and C3 scripts are retained for backward compatibility. The organised source copies live under `src/`.

## Data Governance Statement

This GitHub repository does not include:

- real SLAM patient-level data
- Step B internal semi-synthetic SLAM data
- raw CSV or Excel files
- death-date or WMH spreadsheets
- real identifiers or scan-level identifiers
- large fully synthetic scenario datasets
- per-person prediction files
- fitted model binary files

The repository includes only code, documentation, environment files, and small aggregate summary/audit tables from a fully synthetic local run. Real data must remain inside the authorised research environment.

See [Data Governance and Export Safety](docs/08_data_governance_and_export_safety.md) for the detailed rules.

## Quick Start

Create an environment with either pip:

```bash
pip install -r requirements.txt
```

or conda:

```bash
conda env create -f environment.yml
conda activate slam-synthetic-institutionalisation
```

The C2/C3 scripts expect a local data folder named `fully_synthetic_stepC_v1/`. That folder is not included in this repository.

Run Step C2 in debug mode:

```bash
bash scripts/run_stepC2_debug.sh
```

Run Step C2 in full mode:

```bash
bash scripts/run_stepC2_full.sh
```

Run Step C3 in debug mode:

```bash
bash scripts/run_stepC3_debug.sh
```

Run Step C3 in full mode:

```bash
bash scripts/run_stepC3_full.sh
```

If `fully_synthetic_stepC_v1/` is not present, the modelling scripts will not run. The tests do not run the full analyses.

## Methods Summary

The primary endpoint is time to care-home entry or institutionalisation. The outcome coding is:

- `status = 0`: event-free or censored
- `status = 1`: care-home entry, the event of interest
- `status = 2`: death before care home, a competing event

Step C2 performs cause-specific survival model comparison. It treats `status = 1` as the event and treats `status = 0` and `status = 2` as censored.

Step C3 evaluates 5-year care-home cumulative incidence under competing risk. Death before care home is treated as a competing event. Death after care-home entry is a secondary post-care-home variable and is not the competing event for the primary endpoint.

Step C4 extends the comparison with Fine-Gray, Random Survival Forest, and Gradient Boosting Survival models. A full C4 run has been completed locally. The committed C4 output files remain debug-level summaries only until the full tables and figures are explicitly reviewed and approved for upload.

## Local Full-run Results Summary

The following results describe one completed local full run on fully synthetic data. The underlying scenario datasets and per-person predictions are not included here.

Step C2 completed 1,600 scenario-replicate-model evaluations:

```text
8 scenarios x 50 repetitions x 4 models = 1,600 rows
Failures = 0
```

Best fitted Step C2 model by mean Harrell C-index:

| Scenario | Best fitted model | Mean C-index |
| --- | --- | ---: |
| S0_linear_PH_inst30 | cox_dgm_features | 0.6968 |
| S1_linear_PH_inst15 | cox_dgm_features | 0.6919 |
| S2_linear_PH_inst45 | cox_dgm_features | 0.6933 |
| S3_nonlinear_interaction_inst30 | cox_dgm_features | 0.7329 |
| S4_nonPH_inst30 | cox_dgm_features | 0.6900 |
| S5_MAR_missingness_inst30 | cox_dgm_features | 0.6897 |
| S6_highdim_sparseMRI_inst30 | penalised_cox_all_safe_predictors | 0.7097 |
| S7_strong_death_competing_inst30 | cox_dgm_features | 0.6925 |

Step C3 completed 1,600 mandatory competing-risk evaluations:

```text
8 scenarios x 50 repetitions x 4 main models = 1,600 rows
Failures = 0
Oracle sanity flags = 0
```

Best fitted Step C3 model by mean 5-year risk MAE against the exported synthetic true risk:

| Scenario | Best fitted model | MAE | Brier | AUC | Calibration slope |
| --- | --- | ---: | ---: | ---: | ---: |
| S0_linear_PH_inst30 | cs_cox_dgm_cif | 0.0371 | 0.1912 | 0.6860 | 0.9483 |
| S1_linear_PH_inst15 | cs_cox_dgm_cif | 0.0271 | 0.1217 | 0.6668 | 0.8879 |
| S2_linear_PH_inst45 | cs_cox_dgm_cif | 0.0401 | 0.2200 | 0.6929 | 0.9498 |
| S3_nonlinear_interaction_inst30 | cs_cox_dgm_cif | 0.0525 | 0.1808 | 0.7252 | 0.9118 |
| S4_nonPH_inst30 | cs_cox_dgm_cif | 0.0499 | 0.1950 | 0.6668 | 0.8777 |
| S5_MAR_missingness_inst30 | cs_cox_dgm_cif | 0.0402 | 0.1930 | 0.6770 | 0.9098 |
| S6_highdim_sparseMRI_inst30 | cs_penalised_cox_all_safe_cif | 0.0456 | 0.1834 | 0.7175 | 0.8793 |
| S7_strong_death_competing_inst30 | cs_cox_dgm_cif | 0.0367 | 0.1940 | 0.6711 | 0.9334 |

Across scenarios, the fitted-model family selected by C2 discrimination and C3 absolute-risk accuracy agreed in 8 of 8 scenarios after mapping the cause-specific model families. In S7, the stronger death competing-risk scenario, the best C3 fitted model had mean MAE 0.0367 compared with 0.1088 for the Aalen-Johansen null baseline.

Step C4 full run completed all eight scenarios with 50 repetitions per scenario:

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

The C4 full run does not overturn the C2/C3 core interpretation. Fine-Gray, RSF, and GBSA completed under the strict synthetic-only leakage guard. The only C4 oracle sanity flag was `S1_linear_PH_inst15 / cs_gbsa_dgm_cif / calibration_slope_5y_mean = 1.6026`; no fitted C4 model exceeded the oracle MAE, AUC, or C-index screens. Full C4 tables and figures remain local pending explicit review.

## Scientific Interpretation

These are synthetic benchmark results, not real clinical performance estimates. The main interpretation is that a carefully specified Cox model is robust across most simulated settings, while the high-dimensional sparse MRI scenario is the clearest case where the all-safe penalised Cox model performs best. XGBoost did not unrealistically exceed the oracle benchmark, supporting the strict leakage guard.

## Documentation

- [Project overview](docs/00_project_overview.md)
- [Research rationale](docs/01_research_rationale.md)
- [Workflow from SLAM to synthetic evaluation](docs/02_workflow_from_SLAM_to_synthetic.md)
- [Internal Step B semi-synthetic stage](docs/03_stepB_internal_semi_synthetic.md)
- [Step C fully synthetic generator](docs/04_stepC_fully_synthetic_generator.md)
- [Step C2 model comparison](docs/05_stepC2_model_comparison.md)
- [Step C3 competing-risk evaluation](docs/06_stepC3_competing_risk_evaluation.md)
- [Results summary](docs/07_results_summary.md)
- [Data governance and export safety](docs/08_data_governance_and_export_safety.md)
- [Limitations and next steps](docs/09_limitations_and_next_steps.md)
- [Reproducibility guide](docs/10_github_reproducibility_guide.md)
- [Step C4 extended model comparison](docs/11_stepC4_extended_model_comparison.md)

## Future Work

- Apply the pipeline to real care-home outcomes when they become available.
- Add a formal Fine-Gray implementation.
- Add IPCW Brier scores and time-dependent AUC.
- Evaluate multiple horizons, for example 1, 3, and 5 years.
- Export exact DGM coefficients for more precise oracle comparisons.
- Run sensitivity analyses for the post-care-home death-hazard multiplier.

## License

This code is released under the MIT License. See [LICENSE](LICENSE).
