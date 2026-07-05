# Data Governance and Export Safety

This repository is governed by a strict separation between internal data, exportable synthetic data, and GitHub-tracked code/documentation.

## Internal-only Data

The following must remain inside the authorised SLAM research environment:

- real SLAM raw data;
- real patient-level baseline predictors;
- internal Step B semi-synthetic datasets;
- death-date spreadsheets;
- WMH spreadsheets;
- files containing real identifiers or scan-level identifiers;
- any patient-level rows derived from real data.

## Exportable Synthetic Data

Step C produces fully synthetic datasets designed for export. However, the large scenario datasets are not committed to GitHub. They are too large for normal repository tracking and should be managed through a controlled data package, release asset, or other approved storage route if sharing is required.

## GitHub-tracked Content

This repository may contain:

- Python source code;
- documentation;
- requirements and environment files;
- shell scripts;
- tests;
- small aggregate summary CSV files;
- small audit CSV files.

It must not contain:

- raw data;
- real or semi-synthetic patient-level rows;
- large synthetic scenario datasets;
- full per-person predictions;
- fitted binary model files.

## Step B vs Step C

Step B is semi-synthetic because it combines real SLAM baseline predictors with simulated outcomes. It is methodologically useful but not exportable.

Step C is fully synthetic because both predictors and outcomes are synthetic. It can be exported after audit, but the large scenario files are still excluded from GitHub because they are data products rather than source code.

## Predictor and Export-Safety QC Gates

Predictor filters must exclude reference-only diagnosis columns. In particular,
`diagnosis_reference_NOT_PREDICTOR` and similarly named target/final diagnosis
reference columns are forbidden by name and by feature-dictionary role.

Export-safety checks parse `safe_to_export_column_names` explicitly as string
booleans. The string `"False"` is treated as false, not as a truthy non-empty
string.

## Pre-commit Safety Checks

Run these commands before committing:

```bash
git status
find . -type f -size +20M -print
git diff --cached --name-only
git diff --cached --stat
```

If any staged file is a large dataset, raw data file, Excel spreadsheet, scenario dataset, per-person prediction file, or internal data artifact, unstage it immediately:

```bash
git reset
```

Then stage only the intended code, documentation, and approved summary files.
