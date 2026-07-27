# Contributing

This is a research code repository. Contributions should preserve the data governance boundary of the project.

Before opening a pull request or committing changes, check:

```bash
git status
find . -type f -size +20M -print
git diff --cached --name-only
git diff --cached --stat
```

Do not commit real SLAM data, internal semi-synthetic data, raw spreadsheets,
synthetic scenario datasets, per-person predictions, fitted binary models,
result tables, figures, or files containing real identifiers.

An aggregate result table or figure may be added only after the repository
owner has reviewed and explicitly approved that specific artifact. Approval is
not inferred from approval of code, documentation, or a different artifact.
