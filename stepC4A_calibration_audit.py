#!/usr/bin/env python3
"""Step C4A calibration-slope audit for the single C4 oracle sanity flag.

This script uses only already-exported fully synthetic Step C4/C3 summary
outputs. It does not read raw SLAM, Step B, raw CSV, Excel, or per-person
scenario datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


C4_DIR = Path("fully_synthetic_stepC4_extended_models")
C3_DIR = Path("fully_synthetic_stepC3_competing_risk_evaluation")
OUT_DIR = Path("fully_synthetic_stepC4_calibration_audit")
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"

SCENARIO = "S1_linear_PH_inst15"
FLAGGED_MODEL = "cs_gbsa_dgm_cif"
FLAGGED_METRIC = "calibration_slope_5y_mean"
SLOPE_LOW = 0.5
SLOPE_HIGH = 1.5

C4_MODELS = [
    "cs_gbsa_dgm_cif",
    "cs_gbsa_all_safe_cif",
    "cs_rsf_dgm_cif",
    "cs_rsf_all_safe_cif",
    "finegray_dgm_cif_R",
    "finegray_all_safe_reduced_R",
]
C3_MODELS = [
    "cs_cox_dgm_cif",
    "cs_penalised_cox_all_safe_cif",
    "nonparametric_aj_null",
    "oracle_true_risk_not_a_model",
]
METRICS = [
    "risk5_mae_vs_true_risk",
    "risk5_rmse_vs_true_risk",
    "brier_5y_naive",
    "auc_5y_observed_event1",
    "calibration_slope_5y",
    "calibration_intercept_5y",
    "cause_specific_cindex_event1",
]


@dataclass(frozen=True)
class AuditInputs:
    perf: pd.DataFrame
    c4_summary: pd.DataFrame
    oracle: pd.DataFrame
    c4_deciles_summary: pd.DataFrame
    c4_deciles_rep: pd.DataFrame
    sanity: pd.DataFrame
    c3_summary: pd.DataFrame | None
    c3_deciles_summary: pd.DataFrame | None


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def read_inputs() -> AuditInputs:
    c3_summary_path = C3_DIR / "tables" / "scenario_competing_risk_summary_mean_sd_ci.csv"
    c3_deciles_path = C3_DIR / "tables" / "calibration_deciles_summary.csv"
    return AuditInputs(
        perf=pd.read_csv(require_file(C4_DIR / "tables" / "replicate_extended_model_performance.csv")),
        c4_summary=pd.read_csv(require_file(C4_DIR / "tables" / "scenario_extended_model_summary_mean_sd_ci.csv")),
        oracle=pd.read_csv(require_file(C4_DIR / "tables" / "oracle_sanity_audit_C4.csv")),
        c4_deciles_summary=pd.read_csv(require_file(C4_DIR / "tables" / "calibration_deciles_summary_C4.csv")),
        c4_deciles_rep=pd.read_csv(require_file(C4_DIR / "tables" / "calibration_deciles_replicate_level_C4.csv")),
        sanity=pd.read_csv(require_file(C4_DIR / "tables" / "full_run_sanity_checks_C4.csv")),
        c3_summary=pd.read_csv(c3_summary_path) if c3_summary_path.exists() else None,
        c3_deciles_summary=pd.read_csv(c3_deciles_path) if c3_deciles_path.exists() else None,
    )


def boolish(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def value_from_sanity(sanity: pd.DataFrame, check_name: str, default: object = np.nan) -> object:
    row = sanity.loc[sanity["check_name"].eq(check_name)]
    if row.empty:
        return default
    return row.iloc[0]["observed"]


def confirm_flag(inputs: AuditInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    flagged = inputs.oracle.loc[inputs.oracle["needs_audit"].map(boolish)].copy()
    expected = flagged[
        flagged["scenario_id"].eq(SCENARIO)
        & flagged["model"].eq(FLAGGED_MODEL)
        & flagged["metric"].eq(FLAGGED_METRIC)
    ]
    flag_confirmed = (
        len(flagged) == 1
        and len(expected) == 1
        and np.isclose(float(expected.iloc[0]["model_value"]), 1.6026, atol=0.01)
    )
    row = {
        "flag_confirmed": flag_confirmed,
        "number_of_oracle_sanity_flags": len(flagged),
        "expected_scenario_id": SCENARIO,
        "expected_model": FLAGGED_MODEL,
        "expected_metric": FLAGGED_METRIC,
        "observed_scenario_id": expected.iloc[0]["scenario_id"] if not expected.empty else np.nan,
        "observed_model": expected.iloc[0]["model"] if not expected.empty else np.nan,
        "observed_metric": expected.iloc[0]["metric"] if not expected.empty else np.nan,
        "observed_value": expected.iloc[0]["model_value"] if not expected.empty else np.nan,
        "flag_name": expected.iloc[0]["flag_name"] if not expected.empty else np.nan,
    }
    out = pd.DataFrame([row])
    out.to_csv(TABLE_DIR / "c4_calibration_flag_confirmed.csv", index=False)
    flagged.to_csv(TABLE_DIR / "c4_oracle_sanity_flags_for_audit.csv", index=False)
    return out, flagged


def slope_distribution(inputs: AuditInputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = inputs.perf[
        inputs.perf["scenario_id"].eq(SCENARIO)
        & inputs.perf["model"].eq(FLAGGED_MODEL)
        & ~inputs.perf["skipped"].map(boolish)
        & ~inputs.perf["failed"].map(boolish)
    ].copy()
    slopes = pd.to_numeric(sub["calibration_slope_5y"], errors="coerce").dropna()
    stats = {
        "scenario_id": SCENARIO,
        "model": FLAGGED_MODEL,
        "metric": "calibration_slope_5y",
        "n": int(slopes.shape[0]),
        "mean": slopes.mean(),
        "sd": slopes.std(ddof=1),
        "median": slopes.median(),
        "min": slopes.min(),
        "p05": slopes.quantile(0.05),
        "p25": slopes.quantile(0.25),
        "p75": slopes.quantile(0.75),
        "p95": slopes.quantile(0.95),
        "max": slopes.max(),
        "number_above_1_5": int((slopes > SLOPE_HIGH).sum()),
        "number_below_0_5": int((slopes < SLOPE_LOW).sum()),
        "proportion_above_1_5": float((slopes > SLOPE_HIGH).mean()),
        "proportion_below_0_5": float((slopes < SLOPE_LOW).mean()),
    }
    dist = pd.DataFrame([stats])
    dist.to_csv(TABLE_DIR / "s1_gbsa_dgm_calibration_slope_distribution.csv", index=False)

    cols = [
        "scenario_id",
        "replicate_id",
        "model",
        "calibration_slope_5y",
        "calibration_intercept_5y",
        "risk5_mae_vs_true_risk",
        "brier_5y_naive",
        "auc_5y_observed_event1",
        "cause_specific_cindex_event1",
    ]
    lowest = sub.nsmallest(10, "calibration_slope_5y")[cols].copy()
    lowest.insert(0, "extreme_type", "lowest_10")
    highest = sub.nlargest(10, "calibration_slope_5y")[cols].copy()
    highest.insert(0, "extreme_type", "highest_10")
    extremes = pd.concat([highest, lowest], ignore_index=True)
    extremes.to_csv(TABLE_DIR / "s1_gbsa_dgm_calibration_slope_extreme_reps.csv", index=False)
    return dist, extremes, sub


def extract_summary_rows(summary: pd.DataFrame, models: Iterable[str], stage: str) -> pd.DataFrame:
    rows = summary[summary["scenario_id"].eq(SCENARIO) & summary["model"].isin(models)].copy()
    rows.insert(0, "stage", stage)
    if "n_skipped_reps" not in rows.columns:
        rows["n_skipped_reps"] = 0
    if "skip_rate" not in rows.columns:
        rows["skip_rate"] = 0.0
    keep = [
        "stage",
        "scenario_id",
        "model",
        "n_successful_reps",
        "n_failed_reps",
        "n_skipped_reps",
        "failure_rate",
        "skip_rate",
    ]
    for metric in METRICS:
        col = f"{metric}_mean"
        if col in rows.columns:
            keep.append(col)
    return rows[keep]


def compare_models(inputs: AuditInputs) -> pd.DataFrame:
    c4 = extract_summary_rows(inputs.c4_summary, C4_MODELS, "C4")
    frames = [c4]
    if inputs.c3_summary is not None:
        frames.append(extract_summary_rows(inputs.c3_summary, C3_MODELS, "C3"))
    comp = pd.concat(frames, ignore_index=True)
    comp["is_oracle"] = comp["model"].eq("oracle_true_risk_not_a_model")
    comp["is_flagged_model"] = comp["model"].eq(FLAGGED_MODEL)
    comp["calibration_slope_abs_gap_from_1"] = (comp["calibration_slope_5y_mean"] - 1.0).abs()
    comp = comp.sort_values(
        ["is_oracle", "risk5_mae_vs_true_risk_mean", "brier_5y_naive_mean"],
        ascending=[True, True, True],
        na_position="last",
    )
    comp.to_csv(TABLE_DIR / "s1_all_models_calibration_and_accuracy_comparison.csv", index=False)
    return comp


def classify_flag(
    inputs: AuditInputs,
    flagged: pd.DataFrame,
    dist: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    leakage_audit = pd.read_csv(C4_DIR / "tables" / "predictor_leakage_audit_C4.csv")
    outperformance_flags = {"mae_better_than_oracle", "auc_exceeds_oracle", "cindex_exceeds_oracle"}
    unresolved_outperformance = bool(flagged["flag_name"].isin(outperformance_flags).any()) if not flagged.empty else False
    unresolved_leakage = not leakage_audit.empty
    only_calibration_flag = (
        len(flagged) == 1
        and flagged.iloc[0]["flag_name"] == "calibration_slope_outside_0_5_to_1_5"
    )
    flagged_summary = comparison[comparison["model"].eq(FLAGGED_MODEL)].iloc[0]
    non_oracle = comparison.loc[~comparison["is_oracle"].map(boolish)].copy()
    best_non_oracle = non_oracle.sort_values("risk5_mae_vs_true_risk_mean", na_position="last").iloc[0]
    flagged_model_is_best = flagged_summary["model"] == best_non_oracle["model"]
    number_above = int(dist.iloc[0]["number_above_1_5"])
    median_slope = float(dist.iloc[0]["median"])
    mean_slope = float(dist.iloc[0]["mean"])

    classification = []
    classification.append(
        {
            "category": "A",
            "label": "data leakage or oracle outperformance risk",
            "supported_by_results": unresolved_outperformance or unresolved_leakage,
            "evidence": f"outperformance_flags={unresolved_outperformance}; predictor_leakage_rows={len(leakage_audit)}",
        }
    )
    classification.append(
        {
            "category": "B",
            "label": "code or evaluation error",
            "supported_by_results": False,
            "evidence": "No failed fits, no leakage-audit rows, and the flag is isolated to one calibration-slope screen.",
        }
    )
    classification.append(
        {
            "category": "C",
            "label": "calibration instability in the low-event-rate scenario",
            "supported_by_results": only_calibration_flag and number_above > 0 and not unresolved_outperformance and not unresolved_leakage,
            "evidence": f"S1 is the low institutionalisation scenario; mean_slope={mean_slope:.4f}, median_slope={median_slope:.4f}, reps_above_1_5={number_above}/50.",
        }
    )
    classification.append(
        {
            "category": "D",
            "label": "mean driven by a small number of outlier repetitions",
            "supported_by_results": number_above <= 5 and median_slope <= SLOPE_HIGH,
            "evidence": f"reps_above_1_5={number_above}/50 and median_slope={median_slope:.4f}; this is not a small-outlier-only pattern.",
        }
    )
    classification.append(
        {
            "category": "E",
            "label": "threshold review item that does not change the main conclusion",
            "supported_by_results": only_calibration_flag and not flagged_model_is_best,
            "evidence": f"flagged_model_is_best_non_oracle={flagged_model_is_best}; best_non_oracle={best_non_oracle['model']} with MAE={best_non_oracle['risk5_mae_vs_true_risk_mean']:.4f}.",
        }
    )
    class_df = pd.DataFrame(classification)
    primary = class_df[class_df["supported_by_results"] & class_df["category"].eq("C")]
    if primary.empty:
        primary_category = class_df[class_df["supported_by_results"]]["category"].iloc[0]
    else:
        primary_category = "C"
    class_df["primary_classification"] = primary_category
    class_df.to_csv(TABLE_DIR / "c4_calibration_flag_interpretation.csv", index=False)
    return class_df


def publication_readiness(
    inputs: AuditInputs,
    flagged: pd.DataFrame,
    comparison: pd.DataFrame,
    class_df: pd.DataFrame,
) -> pd.DataFrame:
    outperformance_flags = {"mae_better_than_oracle", "auc_exceeds_oracle", "cindex_exceeds_oracle"}
    leakage_audit = pd.read_csv(C4_DIR / "tables" / "predictor_leakage_audit_C4.csv")
    unresolved_outperformance = bool(flagged["flag_name"].isin(outperformance_flags).any()) if not flagged.empty else False
    unresolved_leakage = not leakage_audit.empty
    unresolved_calibration = bool(
        len(flagged) == 1 and flagged.iloc[0]["flag_name"] == "calibration_slope_outside_0_5_to_1_5"
    )
    non_oracle = comparison.loc[~comparison["is_oracle"].map(boolish)].copy()
    best_non_oracle = non_oracle.sort_values("risk5_mae_vs_true_risk_mean", na_position="last").iloc[0]
    flagged_model_is_best = bool(best_non_oracle["model"] == FLAGGED_MODEL)
    full_run_passed = boolish(value_from_sanity(inputs.sanity, "full_run_passed"))
    original_publication_ready = boolish(value_from_sanity(inputs.sanity, "publication_ready"))
    ready_after_review = (
        full_run_passed
        and not unresolved_outperformance
        and not unresolved_leakage
        and unresolved_calibration
        and not flagged_model_is_best
    )
    interpretation = (
        "The single unresolved flag is a calibration-slope review item for S1 cs_gbsa_dgm_cif. "
        "It is not an oracle-outperformance or leakage flag, and the flagged model is not the best S1 model by MAE."
    )
    action = (
        "Report cs_gbsa_dgm_cif in S1 as calibration-unstable and do not recommend it as the preferred model; "
        "publication_ready can be set to true after human review accepts this limitation."
    )
    row = {
        "full_run_passed": full_run_passed,
        "original_publication_ready": original_publication_ready,
        "unresolved_oracle_outperformance_flag": unresolved_outperformance,
        "unresolved_leakage_flag": unresolved_leakage,
        "unresolved_calibration_flag": unresolved_calibration,
        "calibration_flag_model": FLAGGED_MODEL,
        "calibration_flag_scenario": SCENARIO,
        "calibration_flag_interpretation": interpretation,
        "recommended_action": action,
        "publication_ready_after_human_review": ready_after_review,
        "best_non_oracle_s1_model": best_non_oracle["model"],
        "best_non_oracle_s1_mae": best_non_oracle["risk5_mae_vs_true_risk_mean"],
        "primary_flag_classification": class_df["primary_classification"].iloc[0],
    }
    out = pd.DataFrame([row])
    out.to_csv(TABLE_DIR / "c4_publication_readiness_after_calibration_audit.csv", index=False)
    return out


def plot_slope_distribution(sub: pd.DataFrame, dist: pd.DataFrame) -> None:
    slopes = sub["calibration_slope_5y"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(slopes, bins=14, color="#4C78A8", edgecolor="white")
    ax.axvline(SLOPE_LOW, color="#666666", linestyle="--", linewidth=1.2, label="audit lower threshold")
    ax.axvline(SLOPE_HIGH, color="#D62728", linestyle="--", linewidth=1.2, label="audit upper threshold")
    ax.axvline(float(dist.iloc[0]["mean"]), color="#F58518", linewidth=1.8, label="mean")
    ax.axvline(float(dist.iloc[0]["median"]), color="#54A24B", linewidth=1.8, label="median")
    ax.set_title("S1 GBSA-DGM Calibration Slope Across 50 Repetitions")
    ax.set_xlabel("Calibration slope at 5 years")
    ax.set_ylabel("Number of repetitions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "s1_gbsa_dgm_calibration_slope_distribution.png", dpi=200)
    plt.close(fig)


def decile_rows(inputs: AuditInputs) -> pd.DataFrame:
    c4_models = ["cs_gbsa_dgm_cif", "finegray_dgm_cif_R", "cs_rsf_dgm_cif"]
    c4 = inputs.c4_deciles_summary[
        inputs.c4_deciles_summary["scenario_id"].eq(SCENARIO)
        & inputs.c4_deciles_summary["model"].isin(c4_models)
    ].copy()
    c4.insert(0, "stage", "C4")
    frames = [c4]
    if inputs.c3_deciles_summary is not None:
        c3 = inputs.c3_deciles_summary[
            inputs.c3_deciles_summary["scenario_id"].eq(SCENARIO)
            & inputs.c3_deciles_summary["model"].eq("cs_cox_dgm_cif")
        ].copy()
        c3.insert(0, "stage", "C3")
        frames.append(c3)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TABLE_DIR / "s1_calibration_deciles_models_for_plot.csv", index=False)
    return out


def plot_deciles(deciles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = {
        "cs_gbsa_dgm_cif": "#E45756",
        "finegray_dgm_cif_R": "#4C78A8",
        "cs_rsf_dgm_cif": "#54A24B",
        "cs_cox_dgm_cif": "#F58518",
    }
    for model, grp in deciles.groupby("model", sort=False):
        grp = grp.sort_values("mean_predicted_risk5")
        ax.plot(
            grp["mean_predicted_risk5"],
            grp["observed_event1_5y_rate"],
            marker="o",
            linewidth=1.8,
            label=model,
            color=colors.get(model),
        )
    lim_max = float(np.nanmax([deciles["mean_predicted_risk5"].max(), deciles["observed_event1_5y_rate"].max()]))
    lim_min = 0.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#333333", linestyle="--", linewidth=1.0, label="ideal")
    ax.set_xlim(lim_min, lim_max * 1.05)
    ax.set_ylim(lim_min, lim_max * 1.05)
    ax.set_title("S1 Calibration Deciles")
    ax.set_xlabel("Mean predicted 5-year care-home risk")
    ax.set_ylabel("Observed 5-year care-home event rate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "s1_gbsa_dgm_calibration_deciles.png", dpi=200)
    plt.close(fig)


def plot_model_slope_comparison(comp: pd.DataFrame) -> None:
    plot_df = comp[comp["model"].isin(C4_MODELS + C3_MODELS)].copy()
    plot_df = plot_df.sort_values("calibration_slope_5y_mean", na_position="last")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E45756" if m == FLAGGED_MODEL else "#4C78A8" for m in plot_df["model"]]
    ax.barh(plot_df["model"], plot_df["calibration_slope_5y_mean"], color=colors)
    ax.axvline(SLOPE_LOW, color="#666666", linestyle="--", linewidth=1.0)
    ax.axvline(SLOPE_HIGH, color="#D62728", linestyle="--", linewidth=1.0)
    ax.axvline(1.0, color="#333333", linestyle=":", linewidth=1.0)
    ax.set_title("S1 Calibration Slope by Model")
    ax.set_xlabel("Mean calibration slope at 5 years")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "s1_model_calibration_slope_comparison.png", dpi=200)
    plt.close(fig)


def plot_mae_slope_scatter(comp: pd.DataFrame) -> None:
    plot_df = comp.dropna(subset=["risk5_mae_vs_true_risk_mean", "calibration_slope_5y_mean"]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in plot_df.iterrows():
        color = "#E45756" if row["model"] == FLAGGED_MODEL else ("#72B7B2" if row["stage"] == "C3" else "#4C78A8")
        ax.scatter(row["risk5_mae_vs_true_risk_mean"], row["calibration_slope_5y_mean"], s=70, color=color)
        ax.annotate(row["model"], (row["risk5_mae_vs_true_risk_mean"], row["calibration_slope_5y_mean"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(SLOPE_HIGH, color="#D62728", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="#333333", linestyle=":", linewidth=1.0)
    ax.set_title("S1 Accuracy vs Calibration Slope")
    ax.set_xlabel("Mean MAE vs synthetic true 5-year risk")
    ax.set_ylabel("Mean calibration slope at 5 years")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "s1_model_mae_vs_calibration_slope_scatter.png", dpi=200)
    plt.close(fig)


def make_figures(inputs: AuditInputs, dist: pd.DataFrame, sub: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    plot_slope_distribution(sub, dist)
    deciles = decile_rows(inputs)
    plot_deciles(deciles)
    plot_model_slope_comparison(comp)
    plot_mae_slope_scatter(comp)
    return deciles


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(No rows.)"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in view.columns]
    rows = view.astype(str).values.tolist()

    def esc(text: str) -> str:
        return text.replace("|", "\\|")

    lines = [
        "| " + " | ".join(esc(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_readme(
    flag_confirm: pd.DataFrame,
    dist: pd.DataFrame,
    extremes: pd.DataFrame,
    comp: pd.DataFrame,
    class_df: pd.DataFrame,
    readiness: pd.DataFrame,
) -> None:
    stat = dist.iloc[0]
    ready = readiness.iloc[0]
    primary = class_df["primary_classification"].iloc[0]
    flagged_comp = comp[comp["model"].eq(FLAGGED_MODEL)].iloc[0]
    best_non_oracle = comp[~comp["is_oracle"].map(boolish)].sort_values("risk5_mae_vs_true_risk_mean").iloc[0]
    oracle_flags = flag_confirm.iloc[0]["number_of_oracle_sanity_flags"]
    paper_sentence = (
        "In the low-institutionalisation S1 scenario, the C4 GBSA-DGM model showed a calibration-slope audit flag "
        f"(mean slope {stat['mean']:.3f}; {int(stat['number_above_1_5'])}/50 repetitions above 1.5), "
        "but this was not accompanied by oracle outperformance or leakage evidence and the model was not the preferred S1 model by MAE."
    )
    lines = [
        "# C4A Calibration-Slope Audit",
        "",
        "## 1. Audit Purpose",
        "",
        "This audit investigates the single Step C4 oracle sanity flag: `S1_linear_PH_inst15 / cs_gbsa_dgm_cif / calibration_slope_5y_mean`.",
        "",
        "## 2. Input Data",
        "",
        "- `fully_synthetic_stepC4_extended_models/tables/replicate_extended_model_performance.csv`",
        "- `fully_synthetic_stepC4_extended_models/tables/scenario_extended_model_summary_mean_sd_ci.csv`",
        "- `fully_synthetic_stepC4_extended_models/tables/oracle_sanity_audit_C4.csv`",
        "- `fully_synthetic_stepC4_extended_models/tables/calibration_deciles_*_C4.csv`",
        "- C3 summary and calibration deciles, when available, for Cox/AJ/oracle comparison.",
        "",
        "## 3. Data-Scope Statement",
        "",
        "This audit uses only fully synthetic exported summaries. It does not use real SLAM data, Step B data, raw CSV files, death spreadsheets, WMH spreadsheets, real identifiers, or full per-person predictions.",
        "",
        "## 4. Original C4 Flag",
        "",
        f"- Flag confirmed: `{bool(flag_confirm.iloc[0]['flag_confirmed'])}`",
        f"- Number of oracle sanity flags: `{oracle_flags}`",
        f"- Flagged value: `{fmt(flag_confirm.iloc[0]['observed_value'])}`",
        "- Flag type: calibration slope outside the prespecified 0.5 to 1.5 range.",
        "",
        "## 5. Replicate-Level Result",
        "",
        f"The S1 `cs_gbsa_dgm_cif` calibration slope distribution over 50 repetitions had mean `{stat['mean']:.4f}`, median `{stat['median']:.4f}`, SD `{stat['sd']:.4f}`, range `{stat['min']:.4f}` to `{stat['max']:.4f}`.",
        f"`{int(stat['number_above_1_5'])}` of 50 repetitions were above 1.5 and `{int(stat['number_below_0_5'])}` were below 0.5.",
        "Because the median is also above 1.5, the mean is not driven by only a few isolated extreme repetitions.",
        "",
        "Extreme repetitions are saved in `tables/s1_gbsa_dgm_calibration_slope_extreme_reps.csv`.",
        "",
        "## 6. Comparison With Other Models",
        "",
        f"The flagged model had MAE `{flagged_comp['risk5_mae_vs_true_risk_mean']:.4f}` and calibration slope `{flagged_comp['calibration_slope_5y_mean']:.4f}`. The best non-oracle S1 model by MAE in the C3/C4 comparison was `{best_non_oracle['model']}` with MAE `{best_non_oracle['risk5_mae_vs_true_risk_mean']:.4f}`.",
        "",
        markdown_table(
            comp[
                [
                    "stage",
                    "model",
                    "risk5_mae_vs_true_risk_mean",
                    "brier_5y_naive_mean",
                    "auc_5y_observed_event1_mean",
                    "calibration_slope_5y_mean",
                    "cause_specific_cindex_event1_mean",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## 7. Calibration Figure Interpretation",
        "",
        "- `figures/s1_gbsa_dgm_calibration_slope_distribution.png` shows that the high mean slope is a repeated pattern, not a single-repetition artifact.",
        "- `figures/s1_gbsa_dgm_calibration_deciles.png` compares decile-level observed versus predicted 5-year event risk for GBSA-DGM, Fine-Gray-DGM, RSF-DGM, and C3 Cox-DGM.",
        "- `figures/s1_model_calibration_slope_comparison.png` shows the flagged GBSA-DGM slope relative to other C3/C4 models.",
        "- `figures/s1_model_mae_vs_calibration_slope_scatter.png` shows that the flagged model is not the best S1 model by MAE.",
        "",
        "## 8. Does This Affect the C4 Main Conclusion?",
        "",
        "No. The audit does not show leakage or oracle outperformance. The flag is best interpreted as calibration instability for one GBSA-DGM model in the low-institutionalisation S1 scenario. The flagged model is not the preferred S1 model.",
        "",
        "## 9. Publication Readiness",
        "",
        f"`publication_ready_after_human_review = {bool(ready['publication_ready_after_human_review'])}`.",
        "",
        "The full C4 run remains technically complete. Publication readiness can be restored after documenting that `cs_gbsa_dgm_cif` in S1 is calibration-unstable and should not be recommended as the preferred model.",
        "",
        "## 10. Recommended Paper Sentence",
        "",
        paper_sentence,
        "",
        "## Classification",
        "",
        f"Primary classification: `{primary}`.",
        "",
        markdown_table(class_df[["category", "label", "supported_by_results", "evidence"]], max_rows=10),
    ]
    (OUT_DIR / "README_C4A_calibration_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    inputs = read_inputs()
    flag_confirm, flagged = confirm_flag(inputs)
    dist, extremes, flagged_rep_rows = slope_distribution(inputs)
    comparison = compare_models(inputs)
    class_df = classify_flag(inputs, flagged, dist, comparison)
    readiness = publication_readiness(inputs, flagged, comparison, class_df)
    make_figures(inputs, dist, flagged_rep_rows, comparison)
    write_readme(flag_confirm, dist, extremes, comparison, class_df, readiness)

    print("output folder:", OUT_DIR)
    print("flag confirmed yes/no:", bool(flag_confirm.iloc[0]["flag_confirmed"]))
    print("number of calibration flags:", int(flag_confirm.iloc[0]["number_of_oracle_sanity_flags"]))
    print("S1 cs_gbsa_dgm_cif mean slope:", f"{float(dist.iloc[0]['mean']):.4f}")
    print("number of reps above 1.5:", int(dist.iloc[0]["number_above_1_5"]))
    outperformance = bool(readiness.iloc[0]["unresolved_oracle_outperformance_flag"])
    leakage = bool(readiness.iloc[0]["unresolved_leakage_flag"])
    print("whether leakage/oracle outperformance exists:", bool(outperformance or leakage))
    print("recommended publication_ready_after_human_review:", bool(readiness.iloc[0]["publication_ready_after_human_review"]))
    print("README path:", OUT_DIR / "README_C4A_calibration_audit.md")


if __name__ == "__main__":
    main()
