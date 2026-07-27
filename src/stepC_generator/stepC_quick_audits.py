#!/usr/bin/env python3
"""Run lightweight integrity audits on a generated Step C package.

This script reads only the fully synthetic output package. It does not read
the internal summary source used by the generator and does not fit models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("fully_synthetic_stepC_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def scenario_id_from_path(path: Path) -> str:
    name = path.name
    return name[: -len(".csv.gz")] if name.endswith(".csv.gz") else path.stem


def load_feature_dictionary(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "tables" / "feature_dictionary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "column" not in frame and "feature_name" in frame:
        frame = frame.rename(columns={"feature_name": "column"})
    required = {"column", "block"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Feature dictionary is missing columns: " + ", ".join(missing)
        )
    if "role" not in frame:
        frame["role"] = ""
    return frame[["column", "block", "role"]].copy()


def outcome_gradient(
    frame: pd.DataFrame,
    scenario_id: str,
) -> pd.DataFrame:
    required = {"true_lp_carehome", "duration_years", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{scenario_id} is missing required columns: {', '.join(missing)}"
        )
    status = pd.to_numeric(frame["status"], errors="raise").astype(int)
    work = pd.DataFrame(
        {
            "true_lp_carehome": pd.to_numeric(
                frame["true_lp_carehome"], errors="coerce"
            ),
            "duration_years": pd.to_numeric(
                frame["duration_years"], errors="coerce"
            ),
            "event_carehome": (status == 1).astype(int),
            "event_death_before_carehome": (status == 2).astype(int),
            "event_free_or_censored": (status == 0).astype(int),
        }
    )
    for optional in [
        "death_after_carehome",
        "true_risk_carehome_5y_observable_approx",
        "any_death_5y",
    ]:
        if optional in frame:
            work[optional] = pd.to_numeric(frame[optional], errors="coerce")
    work["lp_quartile"] = pd.qcut(
        work["true_lp_carehome"],
        q=4,
        labels=["Q1_lowest_LP", "Q2", "Q3", "Q4_highest_LP"],
        duplicates="drop",
    )
    aggregations: Dict[str, tuple[str, str]] = {
        "n": ("event_carehome", "size"),
        "mean_true_lp": ("true_lp_carehome", "mean"),
        "carehome_rate": ("event_carehome", "mean"),
        "death_before_carehome_rate": (
            "event_death_before_carehome",
            "mean",
        ),
        "event_free_or_censored_rate": (
            "event_free_or_censored",
            "mean",
        ),
        "median_duration_years": ("duration_years", "median"),
    }
    optional_aggregations = {
        "death_after_carehome": "death_after_carehome_rate",
        "true_risk_carehome_5y_observable_approx": (
            "mean_true_risk_carehome_5y_observable_approx"
        ),
        "any_death_5y": "any_death_5y_rate",
    }
    for source, output in optional_aggregations.items():
        if source in work:
            aggregations[output] = (source, "mean")
    grouped = (
        work.groupby("lp_quartile", observed=False)
        .agg(**aggregations)
        .reset_index()
    )
    grouped.insert(0, "scenario_id", scenario_id)
    return grouped


def run_audits(data_dir: Path) -> Dict[str, Path]:
    scenario_dir = data_dir / "scenario_datasets"
    files = sorted(scenario_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"No scenario .csv.gz files found in {scenario_dir}"
        )
    feature_dictionary = load_feature_dictionary(data_dir)
    high_missing_rows: List[pd.DataFrame] = []
    gradient_rows: List[pd.DataFrame] = []

    for path in files:
        scenario_id = scenario_id_from_path(path)
        frame = pd.read_csv(path, low_memory=False)
        missingness = (
            frame.isna()
            .mean()
            .rename("missing_rate")
            .rename_axis("column")
            .reset_index()
            .merge(feature_dictionary, on="column", how="left")
        )
        missingness.insert(0, "scenario_id", scenario_id)
        high_missing_rows.append(
            missingness.loc[
                missingness["block"].isin(["DGM_truth", "other"])
                & (missingness["missing_rate"] >= 0.50)
            ].copy()
        )
        gradient_rows.append(outcome_gradient(frame, scenario_id))

    high_missing = pd.concat(high_missing_rows, ignore_index=True)
    gradients = pd.concat(gradient_rows, ignore_index=True)
    monotonic_rows = []
    for scenario_id, group in gradients.groupby("scenario_id"):
        ordered = group.sort_values("lp_quartile")
        rates = ordered["carehome_rate"].to_numpy(dtype=float)
        monotonic_rows.append(
            {
                "scenario_id": scenario_id,
                "Q1_carehome_rate": rates[0],
                "Q4_carehome_rate": rates[-1],
                "Q4_minus_Q1": rates[-1] - rates[0],
                "monotonic_non_decreasing": bool(
                    np.all(np.diff(rates) >= -1e-8)
                ),
            }
        )
    monotonic = pd.DataFrame(monotonic_rows)

    audit_dir = data_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "high_missing": (
            audit_dir
            / "quick_audit_high_missing_DGM_truth_other_columns.csv"
        ),
        "gradient": (
            audit_dir
            / "quick_audit_true_lp_quartile_outcome_gradient.csv"
        ),
        "monotonicity": (
            audit_dir / "quick_audit_true_lp_monotonicity_check.csv"
        ),
    }
    high_missing.to_csv(outputs["high_missing"], index=False)
    gradients.to_csv(outputs["gradient"], index=False)
    monotonic.to_csv(outputs["monotonicity"], index=False)
    return outputs


def main() -> None:
    outputs = run_audits(parse_args().data_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
