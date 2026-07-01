# Contributing

This is a research code repository. Contributions should preserve the data governance boundary of the project.

Before opening a pull request or committing changes, check:

```bash
git status
find . -type f -size +20M -print
git diff --cached --name-only
git diff --cached --stat
```

Do not commit real SLAM data, internal semi-synthetic data, raw spreadsheets, large scenario datasets, per-person predictions, fitted binary models, or files containing real identifiers.

Small aggregate summary tables may be committed under `results_summary/` when they contain no patient-level records.
