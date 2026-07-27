#!/usr/bin/env python3
"""Publication-grade statistical synthesis of Step C2-C4 results.

This script fits no models and regenerates no data. It reads only completed
fully synthetic replicate-level outputs, verifies their shape and QC status,
then produces paired model comparisons, Monte Carlo precision estimates,
calibration diagnostics, ranks, stress-test contrasts, and manuscript tables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import platform
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_C2_DIR = Path("fully_synthetic_stepC2_model_comparison_postQC")
DEFAULT_C3_DIR = Path(
    "fully_synthetic_stepC3_competing_risk_evaluation_postQC_calibrated"
)
DEFAULT_C4_DIR = Path("fully_synthetic_stepC4_extended_models_postQC")
DEFAULT_C5A_DIR = Path("fully_synthetic_stepC5A_exact_DGM_coefficients")
DEFAULT_OUT_DIR = Path("fully_synthetic_stepC6_publication_analysis")
DEFAULT_DATA_DIR = Path("fully_synthetic_stepC_v1")

SCENARIOS = [
    "S0_linear_PH_inst30",
    "S1_linear_PH_inst15",
    "S2_linear_PH_inst45",
    "S3_nonlinear_interaction_inst30",
    "S4_nonPH_inst30",
    "S5_MAR_missingness_inst30",
    "S6_highdim_sparseMRI_inst30",
    "S7_strong_death_competing_inst30",
]
APPROXIMATE_TRUTH_SCENARIOS = {"S4_nonPH_inst30"}
TRUTH_BASED_METRICS = {
    "risk5_mae_vs_true_risk",
    "risk5_rmse_vs_true_risk",
    "risk5_spearman_with_true_risk",
    "spearman_true_lp",
}

EXPECTED_MODELS = {
    "C2": {
        "oracle_true_lp_not_a_model",
        "cox_dgm_features",
        "penalised_cox_all_safe_predictors",
        "xgb_survival_cox_strict",
    },
    "C3": {
        "oracle_true_risk_not_a_model",
        "nonparametric_aj_null",
        "cs_cox_dgm_cif",
        "cs_penalised_cox_all_safe_cif",
    },
    "C4": {
        "finegray_dgm_cif_R",
        "finegray_all_safe_reduced_R",
        "cs_rsf_dgm_cif",
        "cs_rsf_all_safe_cif",
        "cs_gbsa_dgm_cif",
        "cs_gbsa_all_safe_cif",
        "deepsurv_dgm_cause_specific_pycox",
        "deepsurv_all_safe_cause_specific_pycox",
        "deephit_competing_risk_dgm_pycox",
        "deephit_competing_risk_all_safe_pycox",
    },
}

METRIC_DIRECTIONS = {
    "risk5_mae_vs_true_risk": "lower",
    "risk5_rmse_vs_true_risk": "lower",
    "brier_5y_naive": "lower",
    "absolute_calibration_slope_error": "lower",
    "absolute_calibration_intercept": "lower",
    "cause_specific_cindex_event1": "higher",
    "spearman_true_lp": "higher",
    "risk5_spearman_with_true_risk": "higher",
    "auc_5y_observed_event1": "higher",
}

PRIMARY_ABSOLUTE_MODELS = [
    "cs_cox_dgm_cif",
    "cs_penalised_cox_all_safe_cif",
    "finegray_dgm_cif_R",
    "finegray_all_safe_reduced_R",
    "cs_rsf_dgm_cif",
    "cs_rsf_all_safe_cif",
    "cs_gbsa_dgm_cif",
    "cs_gbsa_all_safe_cif",
    "deepsurv_dgm_cause_specific_pycox",
    "deepsurv_all_safe_cause_specific_pycox",
    "deephit_competing_risk_dgm_pycox",
    "deephit_competing_risk_all_safe_pycox",
]

PREDICTOR_SET_PAIRS = {
    "penalised_cox_all_safe_minus_cox_dgm": (
        "cs_penalised_cox_all_safe_cif",
        "cs_cox_dgm_cif",
    ),
    "finegray_all_safe_reduced_minus_dgm": (
        "finegray_all_safe_reduced_R",
        "finegray_dgm_cif_R",
    ),
    "rsf_all_safe_minus_dgm": (
        "cs_rsf_all_safe_cif",
        "cs_rsf_dgm_cif",
    ),
    "gbsa_all_safe_minus_dgm": (
        "cs_gbsa_all_safe_cif",
        "cs_gbsa_dgm_cif",
    ),
    "deepsurv_all_safe_minus_dgm": (
        "deepsurv_all_safe_cause_specific_pycox",
        "deepsurv_dgm_cause_specific_pycox",
    ),
    "deephit_all_safe_minus_dgm": (
        "deephit_competing_risk_all_safe_pycox",
        "deephit_competing_risk_dgm_pycox",
    ),
}

FORBIDDEN_FRAGMENTS = [
    "not_predictor",
    "diagnosis_reference",
    "target_diag",
    "final_diagnosis",
    "true_lp",
    "true_risk",
    "true_hazard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-dir", type=Path, default=DEFAULT_C2_DIR)
    parser.add_argument("--c3-dir", type=Path, default=DEFAULT_C3_DIR)
    parser.add_argument("--c4-dir", type=Path, default=DEFAULT_C4_DIR)
    parser.add_argument("--c5a-dir", type=Path, default=DEFAULT_C5A_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_integrity_manifest(
    named_roots: Sequence[Tuple[str, Path]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group, root in named_roots:
        if not root.exists():
            raise FileNotFoundError(root)
        files = (
            [root]
            if root.is_file()
            else sorted(
                path
                for path in root.rglob("*")
                if path.is_file()
                and not any(part.startswith(".") for part in path.relative_to(root).parts)
            )
        )
        for path in files:
            relative = path.name if root.is_file() else str(path.relative_to(root))
            rows.append(
                {
                    "group": group,
                    "relative_path": relative,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(
        rows,
        columns=["group", "relative_path", "size_bytes", "sha256"],
    )
    if manifest.empty:
        raise RuntimeError("Integrity manifest contains no files.")
    if not bool(
        manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").fillna(False).all()
    ):
        raise RuntimeError("Integrity manifest contains an invalid SHA-256 digest.")
    return manifest


def analysis_script_integrity_manifest(project_root: Path) -> pd.DataFrame:
    root_candidates = [project_root, *project_root.parents]
    project_root = next(
        (
            root
            for root in root_candidates
            if (root / "stepC2_fully_synthetic_model_comparison.py").exists()
            and (root / "stepC3_competing_risk_evaluation.py").exists()
        ),
        project_root,
    )
    generator_candidates = [
        project_root
        / "code"
        / "stepC_fully_synthetic_exportable_generator_v1.ipynb",
        project_root
        / "src"
        / "stepC_generator"
        / "stepC_fully_synthetic_exportable_generator_v1.py",
    ]
    generator_source = next(
        (path for path in generator_candidates if path.exists()),
        generator_candidates[0],
    )
    script_paths = [
        project_root / "stepC2_fully_synthetic_model_comparison.py",
        project_root / "stepC3_competing_risk_evaluation.py",
        project_root / "stepC4_extended_model_comparison.py",
        project_root / "stepC5A_exact_DGM_coefficients.py",
        project_root / "stepC6_publication_analysis.py",
        project_root / "run_stepC4_postQC_sharded.py",
        generator_source,
    ]
    missing = [path for path in script_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing analysis source files for integrity audit: "
            + ", ".join(str(path) for path in missing)
        )
    rows = []
    for path in script_paths:
        rows.append(
            {
                "group": "analysis_source",
                "relative_path": str(path.relative_to(project_root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def successful_rows(frame: pd.DataFrame) -> pd.Series:
    failed = (
        frame["failed"].map(parse_bool)
        if "failed" in frame
        else pd.Series(False, index=frame.index)
    )
    skipped = (
        frame["skipped"].map(parse_bool)
        if "skipped" in frame
        else pd.Series(False, index=frame.index)
    )
    return ~failed & ~skipped


def verify_stage(
    frame: pd.DataFrame,
    stage: str,
    expected_models: Sequence[str],
) -> None:
    key = ["scenario_id", "replicate_id", "model"]
    if frame.duplicated(key).any():
        raise RuntimeError(f"{stage} contains duplicate replicate-model keys.")
    observed_models = set(frame["model"].astype(str))
    expected_set = set(expected_models)
    if observed_models != expected_set:
        raise RuntimeError(
            f"{stage} model set mismatch. Missing "
            f"{sorted(expected_set - observed_models)}; unexpected "
            f"{sorted(observed_models - expected_set)}."
        )
    observed_scenarios = set(frame["scenario_id"].astype(str))
    if observed_scenarios != set(SCENARIOS):
        raise RuntimeError(
            f"{stage} scenario set mismatch. Missing "
            f"{sorted(set(SCENARIOS) - observed_scenarios)}; unexpected "
            f"{sorted(observed_scenarios - set(SCENARIOS))}."
        )
    expected_rows = len(SCENARIOS) * 50 * len(expected_set)
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{stage} has {len(frame)} rows; expected {expected_rows}."
        )
    counts = frame.groupby(["scenario_id", "model"])["replicate_id"].nunique()
    if not (counts == 50).all():
        raise RuntimeError(f"{stage} does not contain 50 repetitions per cell.")
    expected_repetitions = set(range(1, 51))
    replicate_sets = frame.groupby(["scenario_id", "model"])[
        "replicate_id"
    ].apply(
        lambda values: set(pd.to_numeric(values, errors="raise").astype(int))
    )
    if not replicate_sets.map(lambda values: values == expected_repetitions).all():
        raise RuntimeError(
            f"{stage} does not contain exactly repetition identifiers 1-50 "
            "in every scenario-model cell."
        )
    if not successful_rows(frame).all():
        raise RuntimeError(f"{stage} contains failed or skipped rows.")


def verify_stage_sanity(
    stage_dir: Path,
    filename: str,
    required_checks: Sequence[str],
) -> None:
    sanity = pd.read_csv(require_file(stage_dir / "tables" / filename))
    observed = set(sanity["check_name"].astype(str))
    missing = sorted(set(required_checks) - observed)
    if missing:
        raise RuntimeError(
            f"{filename} is missing checks: {', '.join(missing)}."
        )
    required = sanity.loc[sanity["check_name"].isin(required_checks)]
    passed = required["passed"].map(parse_bool)
    if not passed.all():
        raise RuntimeError(
            f"{filename} contains failed required checks:\n"
            + required.loc[~passed].to_string(index=False)
        )


def verify_calibration_availability(
    c3: pd.DataFrame,
    c4: pd.DataFrame,
) -> None:
    required_c3 = c3["model"].isin(
        [
            "oracle_true_risk_not_a_model",
            "cs_cox_dgm_cif",
            "cs_penalised_cox_all_safe_cif",
        ]
    )
    required_c4 = pd.Series(True, index=c4.index)
    for stage, frame, required in [
        ("C3", c3, required_c3),
        ("C4", c4, required_c4),
    ]:
        subset = frame.loc[
            required,
            [
                "scenario_id",
                "replicate_id",
                "model",
                "calibration_intercept_5y",
                "calibration_slope_5y",
            ],
        ].copy()
        finite = np.isfinite(
            subset[
                ["calibration_intercept_5y", "calibration_slope_5y"]
            ].apply(pd.to_numeric, errors="coerce")
        ).all(axis=1)
        if not finite.all():
            bad = subset.loc[~finite].head(20)
            raise RuntimeError(
                f"{stage} has unavailable calibration metrics for required "
                f"individualised predictions ({int((~finite).sum())} rows). "
                "Refusing publication synthesis.\n"
                + bad.to_string(index=False)
            )


def verify_c4_qc(c4_dir: Path, c4: pd.DataFrame) -> None:
    sanity_path = require_file(
        c4_dir / "tables" / "full_run_sanity_checks_C4.csv"
    )
    sanity = pd.read_csv(sanity_path)
    required_checks = {
        "expected_scenarios",
        "observed_scenarios",
        "observed_min_reps_per_scenario",
        "observed_max_reps_per_scenario",
        "failed_model_fits",
        "export_safety_passed",
        "predictor_leakage_audit_passed",
        "expected_available_model_rows",
        "all_available_core_models_completed",
        "all_available_models_completed",
        "exact_model_scenario_repetition_grid_completed",
        "all_available_deep_models_completed",
        "full_run_passed",
        "publication_ready",
    }
    observed_checks = set(sanity["check_name"].astype(str))
    missing = sorted(required_checks - observed_checks)
    if missing:
        raise RuntimeError("C4 sanity checks are missing: " + ", ".join(missing))
    required = sanity.loc[sanity["check_name"].isin(required_checks)]
    if not required["passed"].map(parse_bool).all():
        raise RuntimeError(
            "C4 has failed publication checks:\n"
            + required.loc[
                ~required["passed"].map(parse_bool)
            ].to_string(index=False)
        )
    invalid_ipcw_columns = [
        "brier_5y_ipcw",
        "integrated_brier_score_1to5",
    ]
    missing_ipcw_columns = [
        column for column in invalid_ipcw_columns if column not in c4.columns
    ]
    if missing_ipcw_columns:
        raise RuntimeError(
            "C4 is missing explicit unavailable-IPCW fields: "
            + ", ".join(missing_ipcw_columns)
        )
    for column in invalid_ipcw_columns:
        if pd.to_numeric(c4[column], errors="coerce").notna().any():
            raise RuntimeError(
                f"C4 {column} must be unavailable: generic single-event IPCW "
                "metrics are not valid for these competing-risk CIF predictions."
            )
    if "ipcw_metric_reason" not in c4.columns or not c4[
        "ipcw_metric_reason"
    ].fillna("").astype(str).str.contains(
        "competing_risk_specific_ipcw",
        regex=False,
    ).all():
        raise RuntimeError(
            "C4 does not document why generic IPCW/IBS metrics are unavailable."
        )
    if "selected_predictors" in c4:
        selected = c4["selected_predictors"].fillna("").astype(str)
        bad = sorted(
            {
                predictor
                for value in selected
                for predictor in value.split(";")
                if predictor
                and any(
                    fragment in predictor.lower()
                    for fragment in FORBIDDEN_FRAGMENTS
                )
            }
        )
        if bad:
            raise RuntimeError(
                "Forbidden predictors remain in C4 fitted rows: "
                + ", ".join(bad)
            )
        dgm = set(
            pd.read_csv(
                require_file(
                    c4_dir / "tables" / "predictor_list_dgm_C4.csv"
                )
            )["predictor"].dropna().astype(str)
        )
        all_safe = set(
            pd.read_csv(
                require_file(
                    c4_dir / "tables" / "predictor_list_all_safe_C4.csv"
                )
            )["predictor"].dropna().astype(str)
        )
        unexpected = []
        for _, row in c4.iterrows():
            selected = {
                value
                for value in str(row["selected_predictors"]).split(";")
                if value and value.lower() != "nan"
            }
            allowed = (
                dgm
                if str(row.get("predictor_set", "")).lower() == "dgm"
                else all_safe
            )
            extras = sorted(selected - allowed)
            if extras:
                unexpected.append(
                    {
                        "scenario_id": row["scenario_id"],
                        "replicate_id": row["replicate_id"],
                        "model": row["model"],
                        "unexpected_predictors": ";".join(extras),
                    }
                )
        if unexpected:
            raise RuntimeError(
                "C4 fitted rows contain predictors outside their locked "
                "predictor set:\n"
                + pd.DataFrame(unexpected).head(20).to_string(index=False)
            )


def administrative_censoring_audit(data_dir: Path) -> pd.DataFrame:
    rows = []
    scenario_dir = data_dir / "scenario_datasets"
    for scenario in SCENARIOS:
        path = require_file(scenario_dir / f"{scenario}.csv.gz")
        frame = pd.read_csv(path, usecols=["duration_years", "status"])
        censored = frame["status"].astype(int) == 0
        censor_times = frame.loc[censored, "duration_years"].astype(float)
        rows.append(
            {
                "scenario_id": scenario,
                "n_rows": len(frame),
                "n_censored": int(censored.sum()),
                "n_censored_before_5y": int(
                    (
                        censored
                        & (
                            frame["duration_years"].astype(float)
                            < 5.0 - 1e-8
                        )
                    ).sum()
                ),
                "minimum_censoring_time": float(censor_times.min()),
                "maximum_censoring_time": float(censor_times.max()),
                "all_censoring_administrative_at_5y": bool(
                    np.allclose(censor_times.to_numpy(), 5.0)
                ),
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["all_censoring_administrative_at_5y"].all():
        raise RuntimeError(
            "At least one scenario has censoring before the five-year horizon."
        )
    return audit


def truth_target_status(scenario_id: str) -> str:
    return (
        "approximate_nonPH_proxy"
        if scenario_id in APPROXIMATE_TRUTH_SCENARIOS
        else "closed_form_calibrated_PH_target"
    )


def truth_target_validity_audit() -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        approximate = scenario in APPROXIMATE_TRUTH_SCENARIOS
        rows.append(
            {
                "scenario_id": scenario,
                "exported_true_risk_status": (
                    "approximate_nonPH_proxy"
                    if approximate
                    else "closed_form_calibrated_PH_target"
                ),
                "exported_true_lp_status": (
                    "base_LP_proxy_not_full_time_varying_oracle"
                    if approximate
                    else "exact_time_constant_carehome_log_hazard"
                ),
                "truth_based_metrics_primary": not approximate,
                "observed_outcome_metrics_primary": True,
                "reason": (
                    "S4 generated care-home times from unexported early and "
                    "late piecewise LPs, including an unexported latent-frailty "
                    "term. The exported risk used a single base LP and is only "
                    "an approximation."
                    if approximate
                    else "Conditional event times used independent exponential "
                    "cause-specific hazards. The exported formula is the "
                    "closed-form five-year CIF implied by the returned "
                    "calibrated hazards and LPs; the baseline rates were "
                    "selected using the finite repetition's realised "
                    "unit-exponential draws."
                ),
            }
        )
    return pd.DataFrame(rows)


def harmonize(
    c2: pd.DataFrame,
    c3: pd.DataFrame,
    c4: pd.DataFrame,
) -> pd.DataFrame:
    c2h = c2.rename(
        columns={
            "cindex": "cause_specific_cindex_event1",
            "spearman_true_risk5": "risk5_spearman_with_true_risk",
            "risk5_naive_brier": "brier_5y_naive",
            "risk5_calibration_slope": "calibration_slope_5y",
        }
    ).copy()
    c2h["stage"] = "C2"
    c2h["skipped"] = False
    c2h["mean_predicted_risk5"] = np.nan
    c2h["mean_true_risk5"] = np.nan
    c2h["calibration_intercept_5y"] = np.nan
    c2h["auc_5y_observed_event1"] = np.nan

    c3h = c3.copy()
    c3h["stage"] = "C3"
    c3h["skipped"] = False

    c4h = c4.copy()
    c4h["stage"] = "C4"

    common = sorted(set(c2h.columns) | set(c3h.columns) | set(c4h.columns))
    combined = pd.concat(
        [
            c2h.reindex(columns=common),
            c3h.reindex(columns=common),
            c4h.reindex(columns=common),
        ],
        ignore_index=True,
        sort=False,
    )
    combined["absolute_calibration_slope_error"] = (
        pd.to_numeric(combined["calibration_slope_5y"], errors="coerce") - 1.0
    ).abs()
    combined["absolute_calibration_intercept"] = pd.to_numeric(
        combined["calibration_intercept_5y"],
        errors="coerce",
    ).abs()
    combined["prediction_bias_vs_true_risk"] = (
        pd.to_numeric(combined["mean_predicted_risk5"], errors="coerce")
        - pd.to_numeric(combined["mean_true_risk5"], errors="coerce")
    )
    combined["successful"] = successful_rows(combined)
    combined["truth_target_status"] = combined["scenario_id"].map(
        truth_target_status
    )
    return combined


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    derived = (int(seed) + zlib.crc32(text)) % (2**32)
    return np.random.default_rng(derived)


def bootstrap_mean_ci(
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    sampled = rng.choice(values, size=(draws, len(values)), replace=True)
    means = sampled.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_delta_summary(
    model_values: pd.DataFrame,
    reference_values: pd.DataFrame,
    metric: str,
    direction: str,
    draws: int,
    seed: int,
    label: str,
) -> Dict[str, object]:
    paired = model_values[["replicate_id", metric]].merge(
        reference_values[["replicate_id", metric]],
        on="replicate_id",
        suffixes=("_model", "_reference"),
    ).dropna()
    raw_delta = (
        paired[f"{metric}_model"].to_numpy(dtype=float)
        - paired[f"{metric}_reference"].to_numpy(dtype=float)
    )
    improvement = -raw_delta if direction == "lower" else raw_delta
    n = len(raw_delta)
    sd = float(np.std(raw_delta, ddof=1)) if n > 1 else np.nan
    se = sd / math.sqrt(n) if n > 1 else np.nan
    mean_delta = float(np.mean(raw_delta)) if n else np.nan
    normal_low = mean_delta - 1.96 * se if np.isfinite(se) else np.nan
    normal_high = mean_delta + 1.96 * se if np.isfinite(se) else np.nan
    rng = stable_rng(seed, label, metric)
    boot_low, boot_high = bootstrap_mean_ci(raw_delta, draws, rng)
    if n and np.any(np.abs(raw_delta) > 0):
        try:
            p_value = float(
                wilcoxon(
                    raw_delta,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )
        except ValueError:
            p_value = np.nan
    else:
        p_value = np.nan
    return {
        "n_paired_reps": n,
        "mean_model_minus_reference": mean_delta,
        "sd_paired_difference": sd,
        "mcse_paired_difference": se,
        "normal_ci_low": normal_low,
        "normal_ci_high": normal_high,
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
        "mean_improvement_oriented": (
            float(np.mean(improvement)) if n else np.nan
        ),
        "proportion_model_better": (
            float(np.mean(improvement > 0)) if n else np.nan
        ),
        "wilcoxon_p_unadjusted": p_value,
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return adjusted
    order = valid.sort_values().index
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(valid.loc[index]))
        running = max(running, candidate)
        adjusted.loc[index] = running
    return adjusted


def paired_vs_reference(
    combined: pd.DataFrame,
    models: Sequence[str],
    reference: str,
    metrics: Sequence[str],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    absolute = combined.loc[
        combined["stage"].isin(["C3", "C4"]) & combined["successful"]
    ].copy()
    for scenario in SCENARIOS:
        scenario_frame = absolute.loc[absolute["scenario_id"] == scenario]
        reference_frame = scenario_frame.loc[
            scenario_frame["model"] == reference
        ]
        if reference_frame.empty:
            raise RuntimeError(
                f"Missing reference {reference} in {scenario}."
            )
        for model in models:
            if model == reference:
                continue
            model_frame = scenario_frame.loc[
                scenario_frame["model"] == model
            ]
            if model_frame.empty:
                raise RuntimeError(f"Missing model {model} in {scenario}.")
            for metric in metrics:
                summary = paired_delta_summary(
                    model_frame,
                    reference_frame,
                    metric,
                    METRIC_DIRECTIONS[metric],
                    draws,
                    seed,
                    f"{scenario}|{model}|{reference}",
                )
                rows.append(
                    {
                        "scenario_id": scenario,
                        "model": model,
                        "reference_model": reference,
                        "metric": metric,
                        "direction_better": METRIC_DIRECTIONS[metric],
                        "truth_target_status": truth_target_status(scenario),
                        "truth_based_metric_primary": not (
                            scenario in APPROXIMATE_TRUTH_SCENARIOS
                            and metric in TRUTH_BASED_METRICS
                        ),
                        **summary,
                    }
                )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm_within_scenario_metric"] = (
        out.groupby(["scenario_id", "metric"], group_keys=False)[
            "wilcoxon_p_unadjusted"
        ].apply(holm_adjust)
    )
    return out


def paired_c2_ranking(
    combined: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    c2 = combined.loc[
        (combined["stage"] == "C2") & combined["successful"]
    ]
    reference_model = "cox_dgm_features"
    comparison_models = [
        "oracle_true_lp_not_a_model",
        "penalised_cox_all_safe_predictors",
        "xgb_survival_cox_strict",
    ]
    metrics = ["cause_specific_cindex_event1", "spearman_true_lp"]
    for scenario in SCENARIOS:
        scenario_frame = c2.loc[c2["scenario_id"] == scenario]
        reference = scenario_frame.loc[
            scenario_frame["model"] == reference_model
        ]
        for model in comparison_models:
            model_frame = scenario_frame.loc[
                scenario_frame["model"] == model
            ]
            for metric in metrics:
                summary = paired_delta_summary(
                    model_frame,
                    reference,
                    metric,
                    METRIC_DIRECTIONS[metric],
                    draws,
                    seed,
                    f"C2|{scenario}|{model}",
                )
                rows.append(
                    {
                        "scenario_id": scenario,
                        "model": model,
                        "reference_model": reference_model,
                        "metric": metric,
                        "direction_better": "higher",
                        "truth_target_status": truth_target_status(scenario),
                        "truth_based_metric_primary": not (
                            scenario in APPROXIMATE_TRUTH_SCENARIOS
                            and metric in TRUTH_BASED_METRICS
                        ),
                        **summary,
                    }
                )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm_within_scenario_metric"] = (
        out.groupby(["scenario_id", "metric"], group_keys=False)[
            "wilcoxon_p_unadjusted"
        ].apply(holm_adjust)
    )
    return out


def paired_predictor_sets(
    combined: pd.DataFrame,
    metrics: Sequence[str],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    absolute = combined.loc[
        combined["stage"].isin(["C3", "C4"]) & combined["successful"]
    ]
    for pair_name, (all_safe_model, dgm_model) in PREDICTOR_SET_PAIRS.items():
        for scenario in SCENARIOS:
            scenario_frame = absolute.loc[
                absolute["scenario_id"] == scenario
            ]
            all_safe = scenario_frame.loc[
                scenario_frame["model"] == all_safe_model
            ]
            dgm = scenario_frame.loc[scenario_frame["model"] == dgm_model]
            for metric in metrics:
                summary = paired_delta_summary(
                    all_safe,
                    dgm,
                    metric,
                    METRIC_DIRECTIONS[metric],
                    draws,
                    seed,
                    f"{pair_name}|{scenario}",
                )
                rows.append(
                    {
                        "comparison": pair_name,
                        "scenario_id": scenario,
                        "all_safe_model": all_safe_model,
                        "dgm_model": dgm_model,
                        "metric": metric,
                        "direction_better": METRIC_DIRECTIONS[metric],
                        "truth_target_status": truth_target_status(scenario),
                        "truth_based_metric_primary": not (
                            scenario in APPROXIMATE_TRUTH_SCENARIOS
                            and metric in TRUTH_BASED_METRICS
                        ),
                        **summary,
                    }
                )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm_within_scenario_metric"] = (
        out.groupby(["scenario_id", "metric"], group_keys=False)[
            "wilcoxon_p_unadjusted"
        ].apply(holm_adjust)
    )
    return out


def monte_carlo_precision(
    combined: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (stage, scenario, model), group in combined.loc[
        combined["successful"]
    ].groupby(["stage", "scenario_id", "model"]):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            if n == 0:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else np.nan
            mcse = sd / math.sqrt(n) if n > 1 else np.nan
            rows.append(
                {
                    "stage": stage,
                    "scenario_id": scenario,
                    "model": model,
                    "metric": metric,
                    "truth_target_status": truth_target_status(scenario),
                    "truth_based_metric_primary": not (
                        scenario in APPROXIMATE_TRUTH_SCENARIOS
                        and metric in TRUTH_BASED_METRICS
                    ),
                    "n_successful_reps": n,
                    "mean": mean,
                    "sd_between_reps": sd,
                    "monte_carlo_se": mcse,
                    "ci_low": mean - 1.96 * mcse,
                    "ci_high": mean + 1.96 * mcse,
                    "ci_half_width": 1.96 * mcse,
                    "relative_mcse_abs": (
                        mcse / abs(mean)
                        if np.isfinite(mcse) and abs(mean) > 1e-12
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cumulative, quantile, side="left")])


def verify_calibration_decile_grid(deciles: pd.DataFrame) -> None:
    key = ["stage", "scenario_id", "replicate_id", "model"]
    required = {
        *key,
        "decile",
        "n",
        "mean_predicted_risk5",
        "mean_true_risk5",
        "observed_event1_5y_rate",
    }
    missing = sorted(required - set(deciles.columns))
    if missing:
        raise RuntimeError(
            "Calibration-decile inputs are missing columns: "
            + ", ".join(missing)
        )
    if deciles.duplicated([*key, "decile"]).any():
        raise RuntimeError(
            "Calibration-decile inputs contain duplicate group-decile keys."
        )

    expected_repetitions = set(range(1, 51))
    expected_deciles = set(range(1, 11))
    for stage in ["C3", "C4"]:
        stage_frame = deciles.loc[deciles["stage"] == stage].copy()
        expected_models = EXPECTED_MODELS[stage]
        if set(stage_frame["scenario_id"].astype(str)) != set(SCENARIOS):
            raise RuntimeError(
                f"{stage} calibration deciles have an unexpected scenario set."
            )
        if set(stage_frame["model"].astype(str)) != expected_models:
            raise RuntimeError(
                f"{stage} calibration deciles have an unexpected model set."
            )
        expected_groups = (
            len(SCENARIOS) * len(expected_repetitions) * len(expected_models)
        )
        grouped = stage_frame.groupby(key, sort=False)
        if grouped.ngroups != expected_groups:
            raise RuntimeError(
                f"{stage} calibration deciles contain {grouped.ngroups} "
                f"scenario-repetition-model groups; expected {expected_groups}."
            )
        repetition_sets = stage_frame.groupby(
            ["scenario_id", "model"]
        )["replicate_id"].apply(
            lambda values: set(
                pd.to_numeric(values, errors="raise").astype(int)
            )
        )
        if not repetition_sets.map(
            lambda values: values == expected_repetitions
        ).all():
            raise RuntimeError(
                f"{stage} calibration deciles do not contain exactly "
                "repetitions 1-50 in every scenario-model cell."
            )
        decile_sets = grouped["decile"].apply(
            lambda values: set(
                pd.to_numeric(values, errors="raise").astype(int)
            )
        )
        if not decile_sets.map(
            lambda values: values == expected_deciles
        ).all():
            raise RuntimeError(
                f"{stage} calibration inputs do not contain deciles 1-10 "
                "for every scenario-repetition-model group."
            )
        test_counts = grouped["n"].sum()
        if not pd.to_numeric(test_counts, errors="coerce").eq(1500).all():
            raise RuntimeError(
                f"{stage} calibration deciles do not sum to 1,500 held-out "
                "individuals in every scenario-repetition-model group."
            )
        numeric = stage_frame[
            [
                "n",
                "mean_predicted_risk5",
                "mean_true_risk5",
                "observed_event1_5y_rate",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric).all(axis=None):
            raise RuntimeError(
                f"{stage} calibration deciles contain non-finite values."
            )


def decile_calibration_by_replicate(
    c3_dir: Path,
    c4_dir: Path,
) -> pd.DataFrame:
    c3 = pd.read_csv(
        require_file(
            c3_dir / "tables" / "calibration_deciles_replicate_level.csv"
        )
    )
    c3["stage"] = "C3"
    c4 = pd.read_csv(
        require_file(
            c4_dir
            / "tables"
            / "calibration_deciles_replicate_level_C4.csv"
        )
    )
    c4["stage"] = "C4"
    deciles = pd.concat([c3, c4], ignore_index=True, sort=False)
    verify_calibration_decile_grid(deciles)
    rows: List[Dict[str, object]] = []
    for (stage, scenario, replicate, model), group in deciles.groupby(
        ["stage", "scenario_id", "replicate_id", "model"]
    ):
        weights = group["n"].to_numpy(dtype=float)
        predicted = group["mean_predicted_risk5"].to_numpy(dtype=float)
        truth = group["mean_true_risk5"].to_numpy(dtype=float)
        observed = group["observed_event1_5y_rate"].to_numpy(dtype=float)
        truth_error = np.abs(predicted - truth)
        observed_error = np.abs(predicted - observed)
        rows.append(
            {
                "stage": stage,
                "scenario_id": scenario,
                "replicate_id": int(replicate),
                "model": model,
                "truth_target_status": truth_target_status(scenario),
                "n_deciles": len(group),
                "decile_weighted_absolute_calibration_error_vs_truth": float(
                    np.average(truth_error, weights=weights)
                ),
                "decile_e50_vs_truth": weighted_quantile(
                    truth_error,
                    weights,
                    0.50,
                ),
                "decile_e90_vs_truth": weighted_quantile(
                    truth_error,
                    weights,
                    0.90,
                ),
                "decile_weighted_absolute_calibration_error_vs_observed": float(
                    np.average(observed_error, weights=weights)
                ),
                "decile_e50_vs_observed": weighted_quantile(
                    observed_error,
                    weights,
                    0.50,
                ),
                "decile_e90_vs_observed": weighted_quantile(
                    observed_error,
                    weights,
                    0.90,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_decile_calibration(replicate: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "decile_weighted_absolute_calibration_error_vs_truth",
        "decile_e50_vs_truth",
        "decile_e90_vs_truth",
        "decile_weighted_absolute_calibration_error_vs_observed",
        "decile_e50_vs_observed",
        "decile_e90_vs_observed",
    ]
    rows: List[Dict[str, object]] = []
    for (stage, scenario, model), group in replicate.groupby(
        ["stage", "scenario_id", "model"]
    ):
        row: Dict[str, object] = {
            "stage": stage,
            "scenario_id": scenario,
            "model": model,
            "truth_target_status": truth_target_status(scenario),
            "n_reps": group["replicate_id"].nunique(),
        }
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_mcse"] = float(
                values.std(ddof=1) / math.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def model_ranks(combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    absolute = combined.loc[
        combined["stage"].isin(["C3", "C4"])
        & combined["model"].isin(PRIMARY_ABSOLUTE_MODELS)
        & combined["successful"]
    ].copy()
    rank_rows = []
    for (scenario, replicate), group in absolute.groupby(
        ["scenario_id", "replicate_id"]
    ):
        for metric, ascending in [
            ("risk5_mae_vs_true_risk", True),
            ("cause_specific_cindex_event1", False),
        ]:
            values = pd.to_numeric(group[metric], errors="coerce")
            ranks = values.rank(method="average", ascending=ascending)
            for index, rank in ranks.items():
                rank_rows.append(
                    {
                        "scenario_id": scenario,
                        "replicate_id": int(replicate),
                        "metric": metric,
                        "model": group.loc[index, "model"],
                        "truth_target_status": truth_target_status(scenario),
                        "truth_based_metric_primary": not (
                            scenario in APPROXIMATE_TRUTH_SCENARIOS
                            and metric in TRUTH_BASED_METRICS
                        ),
                        "rank": float(rank),
                        "is_best": bool(rank == ranks.min()),
                    }
                )
    replicate_ranks = pd.DataFrame(rank_rows)
    summary = (
        replicate_ranks.groupby(["scenario_id", "metric", "model"])
        .agg(
            mean_rank=("rank", "mean"),
            sd_rank=("rank", "std"),
            median_rank=("rank", "median"),
            probability_best=("is_best", "mean"),
            n_reps=("replicate_id", "nunique"),
        )
        .reset_index()
    )
    summary["truth_target_status"] = summary["scenario_id"].map(
        truth_target_status
    )
    summary["truth_based_metric_primary"] = ~(
        summary["scenario_id"].isin(APPROXIMATE_TRUTH_SCENARIOS)
        & summary["metric"].isin(TRUTH_BASED_METRICS)
    )
    return replicate_ranks, summary


def stress_test_contrasts(
    precision: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    data = precision.loc[
        precision["stage"].isin(["C3", "C4"])
        & precision["model"].isin(PRIMARY_ABSOLUTE_MODELS)
        & precision["metric"].isin(metrics)
    ]
    for (model, metric), group in data.groupby(["model", "metric"]):
        baseline = group.loc[
            group["scenario_id"] == "S0_linear_PH_inst30"
        ]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in group.iterrows():
            if row["scenario_id"] == "S0_linear_PH_inst30":
                continue
            delta = float(row["mean"] - base["mean"])
            se = math.sqrt(
                float(row["monte_carlo_se"]) ** 2
                + float(base["monte_carlo_se"]) ** 2
            )
            direction = METRIC_DIRECTIONS[metric]
            deterioration = delta if direction == "lower" else -delta
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "scenario_id": row["scenario_id"],
                    "truth_target_status": truth_target_status(
                        row["scenario_id"]
                    ),
                    "truth_based_metric_primary": not (
                        row["scenario_id"] in APPROXIMATE_TRUTH_SCENARIOS
                        and metric in TRUTH_BASED_METRICS
                    ),
                    "reference_scenario": "S0_linear_PH_inst30",
                    "scenario_minus_S0": delta,
                    "se_independent_scenario_contrast": se,
                    "ci_low": delta - 1.96 * se,
                    "ci_high": delta + 1.96 * se,
                    "deterioration_oriented_positive_worse": deterioration,
                }
            )
    return pd.DataFrame(rows)


def deephit_alpha_selection(c4: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    deep = c4.loc[
        c4["model"].astype(str).str.startswith("deephit_")
    ].copy()
    deep["selected_alpha"] = pd.to_numeric(
        deep["training_details"].astype(str).str.extract(
            r"selected_alpha=([0-9.]+)"
        )[0],
        errors="coerce",
    )
    deep["validation_brier"] = pd.to_numeric(
        deep["training_details"].astype(str).str.extract(
            r"validation_brier=([0-9.]+)"
        )[0],
        errors="coerce",
    )
    if deep["selected_alpha"].isna().any():
        raise RuntimeError("Could not parse selected alpha for every DeepHit fit.")
    counts = (
        deep.groupby(["scenario_id", "model", "selected_alpha"])
        .agg(
            n_selected=("replicate_id", "size"),
            mean_validation_brier=("validation_brier", "mean"),
        )
        .reset_index()
    )
    totals = counts.groupby(["scenario_id", "model"])[
        "n_selected"
    ].transform("sum")
    counts["selection_proportion"] = counts["n_selected"] / totals
    return deep, counts


def primary_results_table(
    combined: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    absolute = combined.loc[
        combined["stage"].isin(["C3", "C4"])
        & combined["model"].isin(PRIMARY_ABSOLUTE_MODELS)
        & combined["successful"]
    ]
    metrics = [
        "cause_specific_cindex_event1",
        "risk5_mae_vs_true_risk",
        "brier_5y_naive",
        "calibration_slope_5y",
    ]
    rows = []
    for (scenario, model), group in absolute.groupby(
        ["scenario_id", "model"]
    ):
        row: Dict[str, object] = {
            "scenario_id": scenario,
            "model": model,
            "truth_target_status": truth_target_status(scenario),
            "truth_based_metrics_primary": (
                scenario not in APPROXIMATE_TRUTH_SCENARIOS
            ),
            "n_successful_reps": group["replicate_id"].nunique(),
            "failure_rate": 0.0,
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            se = sd / math.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = mean - 1.96 * se
            row[f"{metric}_ci_high"] = mean + 1.96 * se
        rows.append(row)
    table = pd.DataFrame(rows)
    cal = calibration[
        [
            "scenario_id",
            "model",
            "decile_weighted_absolute_calibration_error_vs_truth_mean",
            "decile_e90_vs_truth_mean",
            "decile_weighted_absolute_calibration_error_vs_observed_mean",
            "decile_e90_vs_observed_mean",
        ]
    ]
    return table.merge(cal, on=["scenario_id", "model"], how="left")


def verify_primary_results_table(primary: pd.DataFrame) -> None:
    key = ["scenario_id", "model"]
    expected_rows = len(SCENARIOS) * len(PRIMARY_ABSOLUTE_MODELS)
    if len(primary) != expected_rows or primary.duplicated(key).any():
        raise RuntimeError(
            "Publication primary table does not contain the exact unique "
            f"{len(SCENARIOS)} x {len(PRIMARY_ABSOLUTE_MODELS)} grid."
        )
    if set(primary["scenario_id"].astype(str)) != set(SCENARIOS):
        raise RuntimeError(
            "Publication primary table has an unexpected scenario set."
        )
    if set(primary["model"].astype(str)) != set(PRIMARY_ABSOLUTE_MODELS):
        raise RuntimeError(
            "Publication primary table has an unexpected model set."
        )
    if not pd.to_numeric(
        primary["n_successful_reps"], errors="coerce"
    ).eq(50).all():
        raise RuntimeError(
            "Publication primary table does not have 50 successful "
            "repetitions in every cell."
        )
    if not pd.to_numeric(
        primary["failure_rate"], errors="coerce"
    ).eq(0.0).all():
        raise RuntimeError(
            "Publication primary table contains a non-zero failure rate."
        )
    required_finite = [
        "cause_specific_cindex_event1_mean",
        "risk5_mae_vs_true_risk_mean",
        "brier_5y_naive_mean",
        "calibration_slope_5y_mean",
        "decile_weighted_absolute_calibration_error_vs_truth_mean",
        "decile_e90_vs_truth_mean",
        "decile_weighted_absolute_calibration_error_vs_observed_mean",
        "decile_e90_vs_observed_mean",
    ]
    numeric = primary[required_finite].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric).all(axis=None):
        bad = primary.loc[
            ~np.isfinite(numeric).all(axis=1),
            [*key, *required_finite],
        ]
        raise RuntimeError(
            "Publication primary table contains non-finite required metrics:\n"
            + bad.head(20).to_string(index=False)
        )


def software_environment() -> pd.DataFrame:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lifelines",
        "xgboost",
        "scikit-survival",
        "torch",
        "pycox",
        "torchtuples",
        "matplotlib",
    ]
    rows = [
        {
            "component": "Python",
            "version": platform.python_version(),
        },
        {
            "component": "platform",
            "version": platform.platform(),
        },
    ]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"component": package, "version": version})
    rscript = shutil.which("Rscript")
    if rscript is None:
        rows.append({"component": "R", "version": "not installed"})
    else:
        expression = (
            'cat(paste("R", as.character(getRversion()), sep="\\t"), "\\n"); '
            'for (p in c("survival", "cmprsk")) { '
            'v <- if (requireNamespace(p, quietly=TRUE)) '
            'as.character(packageVersion(p)) else "not installed"; '
            'cat(paste(paste("R", p), v, sep="\\t"), "\\n") }'
        )
        result = subprocess.run(
            [rscript, "-e", expression],
            check=False,
            capture_output=True,
            text=True,
        )
        parsed = 0
        for line in result.stdout.splitlines():
            component, separator, version = line.partition("\t")
            component = component.strip()
            version = version.strip()
            if separator and component and version:
                rows.append({"component": component, "version": version})
                parsed += 1
        if parsed == 0:
            rows.append(
                {
                    "component": "R",
                    "version": f"detection failed (exit {result.returncode})",
                }
            )
    return pd.DataFrame(rows)


def make_heatmap(
    frame: pd.DataFrame,
    value: str,
    output: Path,
    title: str,
    colorbar_label: str,
    center_zero: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    pivot = frame.pivot(index="model", columns="scenario_id", values=value)
    pivot = pivot.reindex(columns=SCENARIOS)
    values = pivot.to_numpy(dtype=float)
    if center_zero:
        limit = np.nanmax(np.abs(values))
        vmin, vmax, cmap = -limit, limit, "RdBu_r"
    else:
        vmin, vmax, cmap = np.nanmin(values), np.nanmax(values), "viridis"
    fig_height = max(6.0, 0.42 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [scenario.split("_")[0] for scenario in pivot.columns],
        rotation=0,
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black",
                )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def make_figures(
    tables: Dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    primary = tables["primary"]
    make_heatmap(
        primary,
        "risk5_mae_vs_true_risk_mean",
        figures_dir / "primary_mae_heatmap.png",
        "Mean 5-year MAE versus exported target (S4 is an approximate proxy)",
        "MAE (lower is better)",
    )
    make_heatmap(
        primary,
        "cause_specific_cindex_event1_mean",
        figures_dir / "primary_cindex_heatmap.png",
        "Mean cause-specific C-index across scenarios",
        "C-index (higher is better)",
    )
    calibration = tables["calibration"]
    make_heatmap(
        calibration.rename(
            columns={
                "decile_weighted_absolute_calibration_error_vs_truth_mean": "value"
            }
        ),
        "value",
        figures_dir / "calibration_error_heatmap.png",
        (
            "Decile-weighted calibration error versus exported synthetic "
            "target (S4 is an approximate proxy)"
        ),
        "Absolute error (lower is better)",
    )
    make_heatmap(
        calibration.rename(
            columns={
                "decile_weighted_absolute_calibration_error_vs_observed_mean": "value"
            }
        ),
        "value",
        figures_dir / "calibration_error_observed_heatmap.png",
        "Decile-weighted calibration error versus observed 5-year outcome",
        "Absolute error (lower is better)",
    )
    paired = tables["paired"]
    for metric, filename, title in [
        (
            "risk5_mae_vs_true_risk",
            "paired_mae_improvement_vs_dgm_cox.png",
            "Paired MAE improvement versus DGM Cox CIF",
        ),
        (
            "cause_specific_cindex_event1",
            "paired_cindex_improvement_vs_dgm_cox.png",
            "Paired C-index improvement versus DGM Cox CIF",
        ),
    ]:
        subset = paired.loc[paired["metric"] == metric].copy()
        make_heatmap(
            subset.rename(columns={"mean_improvement_oriented": "value"}),
            "value",
            figures_dir / filename,
            title,
            "Positive values favour the row model",
            center_zero=True,
        )


def write_readme(
    out_dir: Path,
    c2: pd.DataFrame,
    c3: pd.DataFrame,
    c4: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines = [
        "# Step C6 Publication Analysis",
        "",
        "This stage fits no model and regenerates no synthetic data. It performs "
        "publication-oriented statistical synthesis of the completed post-QC "
        "Step C2, C3, and C4 replicate-level outputs.",
        "",
        "## Verified Inputs",
        "",
        f"- C2 rows: {len(c2):,}.",
        f"- C3 rows: {len(c3):,}.",
        f"- C4 rows: {len(c4):,}.",
        "- The C5A DGM reconstruction tables are included in the input "
        "integrity manifest.",
        "- Every scenario-model cell contains 50 successful repetitions.",
        "- C4 publication checks and fitted-predictor leakage checks passed.",
        "",
        "## Statistical Principles",
        "",
        "- Model comparisons use paired differences within the same scenario "
        "and repetition.",
        "- Mean paired differences are reported with Monte Carlo SE, normal "
        "95% CI, and percentile bootstrap 95% CI.",
        "- Wilcoxon p-values are exploratory and Holm-adjusted within each "
        "scenario-metric family; effect estimates and uncertainty remain "
        "primary.",
        "- In the seven PH scenarios, the exported five-year target is the "
        "closed-form two-hazard CIF on the returned calibrated hazard surface. "
        "Because baseline hazards were selected using each repetition's realised "
        "unit-exponential draws, it is not a population conditional probability "
        "fixed independently before generation. In S4, the exported risk and LP are approximate "
        "proxies because the early/late LPs were not exported; S4 truth-based "
        "metrics are exploratory, while observed-outcome metrics remain valid.",
        "- Calibration is summarized against both the exported synthetic target "
        "and observed five-year outcomes. Decile-weighted values are "
        "aggregate calibration diagnostics, not individual-level ICI.",
        "- Calibration inputs are rejected unless every scenario-repetition-"
        "model group contains deciles 1-10 summing to all 1,500 held-out "
        "individuals.",
        "- Stress-test contrasts compare scenario means with S0 and use "
        "independent-scenario Monte Carlo SEs.",
        "",
        "## Primary Reference",
        "",
        "`cs_cox_dgm_cif` is the designated fitted reference for absolute "
        "five-year risk and cause-specific discrimination. The oracle is "
        "retained only as a simulation benchmark and is not ranked as a "
        "deployable model.",
        "",
        "## Outputs",
        "",
        "- `tables/harmonized_replicate_metrics.csv`",
        "- `tables/administrative_censoring_audit.csv`",
        "- `tables/truth_target_validity_audit.csv`",
        "- `tables/paired_comparisons_vs_dgm_cox.csv`",
        "- `tables/paired_C2_ranking_comparisons_vs_dgm_cox.csv`",
        "- `tables/paired_predictor_set_comparisons.csv`",
        "- `tables/monte_carlo_precision.csv`",
        "- `tables/calibration_error_replicate_level.csv`",
        "- `tables/calibration_error_summary.csv`",
        "- `tables/model_rank_summary.csv`",
        "- `tables/scenario_stress_test_contrasts.csv`",
        "- `tables/deephit_alpha_selection_summary.csv`",
        "- `tables/publication_primary_results_table.csv`",
        "- `tables/input_file_sha256_manifest.csv`",
        "- `tables/analysis_script_sha256_manifest.csv`",
        "- `figures/*.png`",
        "",
        f"Paired comparison rows: {len(paired):,}.",
    ]
    (out_dir / "README_stepC6_publication_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    c2 = pd.read_csv(
        require_file(
            args.c2_dir / "tables" / "replicate_model_performance.csv"
        )
    )
    c3 = pd.read_csv(
        require_file(
            args.c3_dir
            / "tables"
            / "replicate_competing_risk_performance.csv"
        )
    )
    c4 = pd.read_csv(
        require_file(
            args.c4_dir
            / "tables"
            / "replicate_extended_model_performance.csv"
        )
    )
    verify_stage(c2, "C2", sorted(EXPECTED_MODELS["C2"]))
    verify_stage(c3, "C3", sorted(EXPECTED_MODELS["C3"]))
    verify_stage(c4, "C4", sorted(EXPECTED_MODELS["C4"]))
    verify_stage_sanity(
        args.c2_dir,
        "full_run_sanity_checks.csv",
        ["full_run_passed", "export_safety_passed"],
    )
    verify_stage_sanity(
        args.c3_dir,
        "full_run_sanity_checks_C3.csv",
        [
            "publication_full_mode",
            "full_run_passed",
            "export_safety_passed",
        ],
    )
    verify_c4_qc(args.c4_dir, c4)
    verify_calibration_availability(c3, c4)
    censoring_audit = administrative_censoring_audit(args.data_dir)
    censoring_audit.to_csv(
        tables_dir / "administrative_censoring_audit.csv",
        index=False,
    )
    truth_validity = truth_target_validity_audit()
    truth_validity.to_csv(
        tables_dir / "truth_target_validity_audit.csv",
        index=False,
    )

    combined = harmonize(c2, c3, c4)
    combined.to_csv(
        tables_dir / "harmonized_replicate_metrics.csv",
        index=False,
    )

    comparison_metrics = [
        "risk5_mae_vs_true_risk",
        "risk5_rmse_vs_true_risk",
        "brier_5y_naive",
        "cause_specific_cindex_event1",
        "risk5_spearman_with_true_risk",
        "auc_5y_observed_event1",
        "absolute_calibration_slope_error",
        "absolute_calibration_intercept",
    ]
    paired = paired_vs_reference(
        combined,
        PRIMARY_ABSOLUTE_MODELS,
        "cs_cox_dgm_cif",
        comparison_metrics,
        args.bootstrap_draws,
        args.seed,
    )
    paired.to_csv(
        tables_dir / "paired_comparisons_vs_dgm_cox.csv",
        index=False,
    )
    paired_c2 = paired_c2_ranking(
        combined,
        args.bootstrap_draws,
        args.seed,
    )
    paired_c2.to_csv(
        tables_dir / "paired_C2_ranking_comparisons_vs_dgm_cox.csv",
        index=False,
    )
    predictor_pairs = paired_predictor_sets(
        combined,
        comparison_metrics,
        args.bootstrap_draws,
        args.seed,
    )
    predictor_pairs.to_csv(
        tables_dir / "paired_predictor_set_comparisons.csv",
        index=False,
    )

    precision = monte_carlo_precision(combined, comparison_metrics)
    precision.to_csv(
        tables_dir / "monte_carlo_precision.csv",
        index=False,
    )

    calibration_replicate = decile_calibration_by_replicate(
        args.c3_dir,
        args.c4_dir,
    )
    calibration_replicate.to_csv(
        tables_dir / "calibration_error_replicate_level.csv",
        index=False,
    )
    calibration_summary = summarize_decile_calibration(
        calibration_replicate
    )
    calibration_summary.to_csv(
        tables_dir / "calibration_error_summary.csv",
        index=False,
    )

    replicate_ranks, rank_summary = model_ranks(combined)
    replicate_ranks.to_csv(
        tables_dir / "model_rank_replicate_level.csv",
        index=False,
    )
    rank_summary.to_csv(
        tables_dir / "model_rank_summary.csv",
        index=False,
    )

    stress = stress_test_contrasts(
        precision,
        [
            "risk5_mae_vs_true_risk",
            "cause_specific_cindex_event1",
        ],
    )
    stress.to_csv(
        tables_dir / "scenario_stress_test_contrasts.csv",
        index=False,
    )

    alpha_replicate, alpha_summary = deephit_alpha_selection(c4)
    alpha_replicate[
        [
            "scenario_id",
            "replicate_id",
            "model",
            "selected_alpha",
            "validation_brier",
        ]
    ].to_csv(
        tables_dir / "deephit_alpha_selection_replicate_level.csv",
        index=False,
    )
    alpha_summary.to_csv(
        tables_dir / "deephit_alpha_selection_summary.csv",
        index=False,
    )

    primary = primary_results_table(combined, calibration_summary)
    verify_primary_results_table(primary)
    primary.to_csv(
        tables_dir / "publication_primary_results_table.csv",
        index=False,
    )
    software_environment().to_csv(
        tables_dir / "software_environment.csv",
        index=False,
    )
    input_manifest = file_integrity_manifest(
        [
            ("fully_synthetic_stepC_v1", args.data_dir),
            ("C2_tables", args.c2_dir / "tables"),
            ("C3_tables", args.c3_dir / "tables"),
            ("C4_tables", args.c4_dir / "tables"),
            ("C5A_tables", args.c5a_dir / "tables"),
        ]
    )
    required_manifest_groups = {
        "fully_synthetic_stepC_v1",
        "C2_tables",
        "C3_tables",
        "C4_tables",
        "C5A_tables",
    }
    if set(input_manifest["group"].astype(str)) != required_manifest_groups:
        raise RuntimeError("Input integrity manifest has an unexpected group set.")
    input_manifest.to_csv(
        tables_dir / "input_file_sha256_manifest.csv",
        index=False,
    )
    analysis_script_integrity_manifest(
        Path(__file__).resolve().parent
    ).to_csv(
        tables_dir / "analysis_script_sha256_manifest.csv",
        index=False,
    )

    make_figures(
        {
            "primary": primary,
            "calibration": calibration_summary.loc[
                calibration_summary["model"].isin(PRIMARY_ABSOLUTE_MODELS)
            ],
            "paired": paired,
        },
        figures_dir,
    )
    write_readme(args.out_dir, c2, c3, c4, paired)
    print("Step C6 publication analysis complete.", flush=True)
    print(f"Output: {args.out_dir}", flush=True)
    print(f"Harmonized rows: {len(combined):,}", flush=True)
    print(f"Paired comparison rows: {len(paired):,}", flush=True)
    print(f"Primary table rows: {len(primary):,}", flush=True)


if __name__ == "__main__":
    main()
