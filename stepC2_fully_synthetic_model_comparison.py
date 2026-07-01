#!/usr/bin/env python3
"""Step C2 fully synthetic survival / competing-risk model comparison.

This script uses only the exportable fully synthetic Step C package under
fully_synthetic_stepC_v1/. It treats status=1 as care-home entry and treats
status=0 and status=2 as censored for cause-specific survival modelling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("fully_synthetic_stepC_v1")
OUT_DIR = Path("fully_synthetic_stepC2_model_comparison")

DEBUG_MODE = True
MAX_REPS_PER_SCENARIO: Optional[int] = 2
TEST_SIZE = 0.30
RANDOM_SEED = 42
EVAL_HORIZON_YEARS = 5.0
DEBUG_PRED_SAMPLE_N = 100

DURATION_COL = "duration_years"
STATUS_COL = "status"
EVENT_STATUS = 1
COMPETING_STATUS = 2

MODEL_ORACLE = "oracle_true_lp_not_a_model"
MODEL_COX_DGM = "cox_dgm_features"
MODEL_COX_SAFE = "penalised_cox_all_safe_predictors"
MODEL_XGB = "xgb_survival_cox_strict"
MODELS = [MODEL_ORACLE, MODEL_COX_DGM, MODEL_COX_SAFE, MODEL_XGB]

ALLOWED_BLOCKS = {
    "demographic",
    "cognition",
    "MRI",
    "NLP",
    "WMH",
    "comorbidity",
    "deprivation",
    "diagnosis_reference",
}

EXCLUDED_BLOCKS = {
    "identifier_synthetic",
    "outcome",
    "DGM_truth",
    "simulation_truth_or_audit",
}

FORBIDDEN_NAME_SUBSTRINGS = [
    "duration",
    "status",
    "event",
    "death",
    "carehome_to_death",
    "t_death",
    "t_carehome",
    "true_lp",
    "true_risk",
    "true_hazard",
    "hazard",
    "risk",
    "censor",
    "scenario_id",
    "replicate_id",
    "synthetic_id",
]

KNOWN_COMORBIDITY_COLUMNS = {
    "falls",
    "hypertension",
    "Cerebrovascular_accident",
    "Epilepsy",
    "Diabetes_mellitus",
    "Chronic_kidney_disease",
    "Parkinsons_disease",
    "Transient_ischemic_attack",
    "Chronic_obstructive_lung_disease",
    "Arthritis",
    "Heart_failure",
    "Ischemic_heart_disease",
    "Atrial_fibrillation",
    "Chronic_liver_disease",
    "Coronary_arteriosclerosis",
}

CLINICAL_DGM_INTENDED_FEATURES = [
    "age",
    "sex_Female",
    "MMSE",
    "MTL_total_pct",
    "Temporal_lateral_total_pct",
    "Medial_parietal_total_pct",
    "Posterior_total_pct",
    "Ventricles_total_pct",
    "wmh_icv_pct",
    "wmh_log_icv",
    "wmh_missing",
    "pathology_ischaemic",
    "pathology_ischaemic__missing",
    "pathology_ischaemic__present",
    "pathology_ischaemic__log1p",
    "pathology_neurodegenerative_dementia",
    "pathology_neurodegenerative_dementia__missing",
    "pathology_neurodegenerative_dementia__present",
    "pathology_neurodegenerative_dementia__log1p",
    "falls",
    "hypertension",
    "Cerebrovascular_accident",
    "Epilepsy",
    "Diabetes_mellitus",
    "Chronic_kidney_disease",
    "Parkinsons_disease",
    "Transient_ischemic_attack",
    "Chronic_obstructive_lung_disease",
    "Arthritis",
    "Heart_failure",
    "Ischemic_heart_disease",
    "Atrial_fibrillation",
    "Chronic_liver_disease",
    "Coronary_arteriosclerosis",
    "IMD_Score_2019_synthetic",
    "IMD_Decile_2019_synthetic",
]

REQUIRED_COLUMNS = [
    "scenario_id",
    "replicate_id",
    DURATION_COL,
    STATUS_COL,
    "true_lp_carehome",
]

TRUTH_RISK_COL = "true_risk_carehome_5y_observable_approx"

EXPECTED_FULL_REPLICATE_MODEL_ROWS = 8 * 50 * 4
EXPECTED_FULL_SCENARIOS = 8
EXPECTED_REPS_PER_SCENARIO = 50

SCENARIO_MEANINGS = {
    "S0_linear_PH_inst30": "baseline linear PH, observed care-home target 30%",
    "S1_linear_PH_inst15": "low-event scenario",
    "S2_linear_PH_inst45": "high-event scenario",
    "S3_nonlinear_interaction_inst30": "nonlinear + interaction scenario",
    "S4_nonPH_inst30": "non-proportional hazards scenario",
    "S5_MAR_missingness_inst30": "MAR missingness scenario",
    "S6_highdim_sparseMRI_inst30": "high-dimensional sparse MRI signal scenario",
    "S7_strong_death_competing_inst30": "stronger death competing risk scenario",
}

SCENARIO_PRIMARY_QUESTIONS = {
    "S0_linear_PH_inst30": "Do fitted baseline models recover the linear proportional-hazards DGM?",
    "S1_linear_PH_inst15": "How stable are fitted models when the event rate is lower?",
    "S2_linear_PH_inst45": "How stable are fitted models when the event rate is higher?",
    "S3_nonlinear_interaction_inst30": "Does the nonlinear/interaction scenario favour the nonlinear XGBoost risk score?",
    "S4_nonPH_inst30": "How much does non-proportional hazards affect proportional Cox baselines?",
    "S5_MAR_missingness_inst30": "Does MAR missingness reduce fitted-model performance or alignment with truth?",
    "S6_highdim_sparseMRI_inst30": "Do high-dimensional safe predictors help in the sparse MRI-signal scenario?",
    "S7_strong_death_competing_inst30": "Does stronger death-before-care-home competing risk change relative model performance?",
}

SCENARIO_EXPECTED_PATTERNS = {
    "S0_linear_PH_inst30": "Cox DGM should stay close to oracle; flexible models should not exceed oracle suspiciously.",
    "S1_linear_PH_inst15": "Lower event counts may reduce precision, but Cox DGM should remain close to oracle.",
    "S2_linear_PH_inst45": "Higher event counts should support stable discrimination; Cox DGM should remain close to oracle.",
    "S3_nonlinear_interaction_inst30": "A nonlinear model may become more competitive, while oracle remains the upper benchmark.",
    "S4_nonPH_inst30": "Proportional Cox models may lose some discrimination under time-varying effects.",
    "S5_MAR_missingness_inst30": "Missingness can reduce fitted-model performance and truth alignment.",
    "S6_highdim_sparseMRI_inst30": "Penalised Cox or XGBoost may become more competitive if sparse MRI signal is captured.",
    "S7_strong_death_competing_inst30": "Cause-specific models may show changed performance because competing deaths censor more observations.",
}


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    out_dir: Path = OUT_DIR
    debug_mode: bool = DEBUG_MODE
    max_reps_per_scenario: Optional[int] = MAX_REPS_PER_SCENARIO
    test_size: float = TEST_SIZE
    random_seed: int = RANDOM_SEED


def ensure_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Import required third-party packages with explicit install guidance."""
    missing: List[str] = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        missing.append("matplotlib")
        plt = None

    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except Exception:
        missing.append("lifelines")
        CoxPHFitter = None
        concordance_index = None

    try:
        import xgboost as xgb
    except Exception:
        missing.append("xgboost")
        xgb = None

    try:
        from scipy.stats import spearmanr
    except Exception:
        missing.append("scipy")
        spearmanr = None

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        missing.append("scikit-learn")
        ColumnTransformer = None
        SimpleImputer = None
        LogisticRegression = None
        train_test_split = None
        Pipeline = None
        OneHotEncoder = None
        StandardScaler = None

    if missing:
        unique_missing = sorted(set(missing))
        print("Missing required Python package(s): " + ", ".join(unique_missing), file=sys.stderr)
        if "lifelines" in unique_missing:
            print("pip install lifelines", file=sys.stderr)
        if "xgboost" in unique_missing:
            print("pip install xgboost", file=sys.stderr)
        other = [p for p in unique_missing if p not in {"lifelines", "xgboost"}]
        if other:
            print("pip install " + " ".join(other), file=sys.stderr)
        sys.exit(1)

    return (
        plt,
        CoxPHFitter,
        concordance_index,
        xgb,
        spearmanr,
        ColumnTransformer,
        SimpleImputer,
        LogisticRegression,
        train_test_split,
        Pipeline,
        OneHotEncoder,
        StandardScaler,
    )


(
    plt,
    CoxPHFitter,
    concordance_index,
    xgb,
    spearmanr,
    ColumnTransformer,
    SimpleImputer,
    LogisticRegression,
    train_test_split,
    Pipeline,
    OneHotEncoder,
    StandardScaler,
) = ensure_dependencies()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--debug", action="store_true", help="Run debug mode.")
    parser.add_argument("--full", action="store_true", help="Run all repetitions.")
    parser.add_argument("--max-reps", type=int, default=None, help="Maximum reps per scenario.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    debug_mode = DEBUG_MODE
    max_reps = MAX_REPS_PER_SCENARIO
    if args.full:
        debug_mode = False
        max_reps = None
    if args.debug:
        debug_mode = True
        max_reps = MAX_REPS_PER_SCENARIO
    if args.max_reps is not None:
        max_reps = args.max_reps
        debug_mode = args.max_reps != 0

    return Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        debug_mode=debug_mode,
        max_reps_per_scenario=max_reps,
        test_size=args.test_size,
        random_seed=args.seed,
    )


def make_output_dirs(config: Config) -> Dict[str, Path]:
    paths = {
        "root": config.out_dir,
        "tables": config.out_dir / "tables",
        "figures": config.out_dir / "figures",
        "predictions": config.out_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def bool_series_all_true(series: pd.Series) -> bool:
    values = series.map(lambda x: str(x).strip().lower() in {"true", "1", "yes"})
    return bool(values.all())


def get_csv_header(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def audit_scenario_file(path: Path) -> Dict[str, Any]:
    columns = get_csv_header(path)
    present_required = [c for c in REQUIRED_COLUMNS if c in columns]
    missing_required = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_required:
        return {
            "file_name": path.name,
            "scenario_id": "",
            "n_rows": np.nan,
            "n_columns": len(columns),
            "columns_json": json.dumps(columns),
            "n_replicate_ids": np.nan,
            "replicate_ids_json": "[]",
            "status_distribution_per_scenario_json": "{}",
            "status_distribution_per_replicate_json": "{}",
            "missing_required_columns_json": json.dumps(missing_required),
        }

    n_rows = 0
    scenario_ids: set[str] = set()
    replicate_ids: set[Any] = set()
    status_counts: Dict[str, int] = {}
    status_by_rep: Dict[str, Dict[str, int]] = {}
    usecols = ["scenario_id", "replicate_id", STATUS_COL]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        n_rows += len(chunk)
        scenario_ids.update(chunk["scenario_id"].dropna().astype(str).unique().tolist())
        replicate_ids.update(chunk["replicate_id"].dropna().unique().tolist())
        counts = chunk[STATUS_COL].value_counts(dropna=False).to_dict()
        for key, value in counts.items():
            status_counts[str(key)] = status_counts.get(str(key), 0) + int(value)
        rep_counts = chunk.groupby(["replicate_id", STATUS_COL], dropna=False).size()
        for (rep, status), value in rep_counts.items():
            rep_key = str(rep)
            status_key = str(status)
            status_by_rep.setdefault(rep_key, {})
            status_by_rep[rep_key][status_key] = status_by_rep[rep_key].get(status_key, 0) + int(value)

    try:
        replicate_ids_sorted = sorted(replicate_ids)
    except TypeError:
        replicate_ids_sorted = sorted([str(x) for x in replicate_ids])

    return {
        "file_name": path.name,
        "scenario_id": ";".join(sorted(scenario_ids)),
        "n_rows": n_rows,
        "n_columns": len(columns),
        "columns_json": json.dumps(columns),
        "n_replicate_ids": len(replicate_ids),
        "replicate_ids_json": json.dumps(list(replicate_ids_sorted), default=str),
        "status_distribution_per_scenario_json": json.dumps(status_counts, sort_keys=True),
        "status_distribution_per_replicate_json": json.dumps(status_by_rep, sort_keys=True),
        "missing_required_columns_json": json.dumps(missing_required),
    }


def load_and_audit_inputs(config: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    data_dir = config.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    safety = read_csv_required(data_dir / "audit" / "export_safety_audit.csv")
    if "safe_to_export_column_names" not in safety.columns:
        raise ValueError("export_safety_audit.csv lacks safe_to_export_column_names")
    if not bool_series_all_true(safety["safe_to_export_column_names"]):
        unsafe = safety.loc[
            ~safety["safe_to_export_column_names"].map(lambda x: str(x).strip().lower() in {"true", "1", "yes"})
        ]
        raise RuntimeError(
            "Export safety audit failed. Unsafe files:\n" + unsafe.to_string(index=False)
        )

    feature_dictionary = read_csv_required(data_dir / "tables" / "feature_dictionary.csv")
    dgm_definition = read_csv_required(data_dir / "tables" / "dgm_definition_table.csv")
    scenario_summary = read_csv_required(data_dir / "tables" / "scenario_summary.csv")
    repetition_summary = read_csv_required(data_dir / "tables" / "repetition_summary.csv")

    scenario_files = sorted((data_dir / "scenario_datasets").glob("*.csv.gz"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found under {data_dir / 'scenario_datasets'}")

    print(f"Found {len(scenario_files)} scenario files.")
    for file_path in scenario_files:
        print(f"  - {file_path.name}")

    audit_rows = []
    for file_path in scenario_files:
        print(f"Auditing {file_path.name} ...")
        audit_rows.append(audit_scenario_file(file_path))
    data_file_audit = pd.DataFrame(audit_rows)
    data_file_audit.to_csv(paths["tables"] / "data_file_audit.csv", index=False)

    if data_file_audit["missing_required_columns_json"].ne("[]").any():
        bad = data_file_audit.loc[
            data_file_audit["missing_required_columns_json"].ne("[]"),
            ["file_name", "missing_required_columns_json"],
        ]
        raise RuntimeError("Missing required columns in scenario files:\n" + bad.to_string(index=False))

    print("Scenario file audit saved to", paths["tables"] / "data_file_audit.csv")
    return {
        "safety": safety,
        "feature_dictionary": feature_dictionary,
        "dgm_definition": dgm_definition,
        "scenario_summary": scenario_summary,
        "repetition_summary": repetition_summary,
        "scenario_files": scenario_files,
        "data_file_audit": data_file_audit,
    }


def contains_forbidden_name(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in FORBIDDEN_NAME_SUBSTRINGS)


def derived_block_for_column(column: str, block: str) -> str:
    if column in KNOWN_COMORBIDITY_COLUMNS:
        return "comorbidity"
    return block


def build_predictor_lists(
    feature_dictionary: pd.DataFrame,
    all_columns: Sequence[str],
    paths: Dict[str, Path],
) -> Dict[str, List[str]]:
    colset = set(all_columns)
    if not {"column", "block", "role"}.issubset(feature_dictionary.columns):
        raise ValueError("feature_dictionary.csv must contain column, block, and role columns")

    fd = feature_dictionary.copy()
    fd["column"] = fd["column"].astype(str)
    all_safe_rows: List[Dict[str, Any]] = []
    safe_predictors: List[str] = []

    for _, row in fd.iterrows():
        col = str(row["column"])
        block = str(row["block"])
        role = str(row["role"])
        derived_block = derived_block_for_column(col, block)
        present = col in colset
        forbidden = contains_forbidden_name(col)
        included = False
        reasons: List[str] = []

        if not present:
            reasons.append("not_present_in_scenario_files")
        if block in EXCLUDED_BLOCKS or role in EXCLUDED_BLOCKS:
            reasons.append("excluded_block_or_role")
        if forbidden:
            reasons.append("forbidden_name_substring")
        if col == "sex" and "sex_Female" in colset:
            reasons.append("excluded_redundant_with_sex_Female")

        if not reasons:
            if derived_block in ALLOWED_BLOCKS:
                included = True
                reasons.append("included_allowed_block")
            else:
                reasons.append("block_not_in_allowed_predictor_blocks")

        if included:
            safe_predictors.append(col)

        all_safe_rows.append(
            {
                "column": col,
                "present_in_scenario_files": present,
                "feature_dictionary_block": block,
                "feature_dictionary_role": role,
                "derived_predictor_block": derived_block,
                "included_in_all_safe_predictors": included,
                "contains_forbidden_name_substring": forbidden,
                "reason": ";".join(reasons),
            }
        )

    unknown_columns = [c for c in all_columns if c not in set(fd["column"])]
    for col in unknown_columns:
        all_safe_rows.append(
            {
                "column": col,
                "present_in_scenario_files": True,
                "feature_dictionary_block": "",
                "feature_dictionary_role": "",
                "derived_predictor_block": "",
                "included_in_all_safe_predictors": False,
                "contains_forbidden_name_substring": contains_forbidden_name(col),
                "reason": "not_listed_in_feature_dictionary",
            }
        )

    predictor_audit_all_safe = pd.DataFrame(all_safe_rows)
    predictor_audit_all_safe.to_csv(paths["tables"] / "predictor_audit_all_safe.csv", index=False)

    suspicious = [c for c in safe_predictors if contains_forbidden_name(c)]
    if suspicious:
        print("Suspicious predictor columns remain after strict filtering:", file=sys.stderr)
        for col in suspicious:
            print("  " + col, file=sys.stderr)
        raise RuntimeError("Forbidden/suspicious predictors remain in all-safe list.")

    missing_intended = []
    dgm_rows: List[Dict[str, Any]] = []
    dgm_predictors: List[str] = []
    safe_set = set(safe_predictors)
    for col in CLINICAL_DGM_INTENDED_FEATURES:
        present = col in colset
        included = present and col in safe_set
        reason = "included_clinical_fallback_dgm_feature" if included else ""
        if not present:
            reason = "missing_from_scenario_files"
            missing_intended.append(col)
        elif col not in safe_set:
            reason = "present_but_not_in_strict_safe_predictor_list"
        if included:
            dgm_predictors.append(col)
        dgm_rows.append(
            {
                "intended_feature": col,
                "present_in_scenario_files": present,
                "included_in_dgm_features": included,
                "source": "clinical_fallback_because_dgm_definition_table_has_no_predictor_columns",
                "reason": reason,
            }
        )

    predictor_audit_dgm = pd.DataFrame(dgm_rows)
    predictor_audit_dgm.to_csv(paths["tables"] / "predictor_audit_dgm_features.csv", index=False)
    pd.DataFrame({"predictor": safe_predictors}).to_csv(
        paths["tables"] / "predictor_list_all_safe.csv", index=False
    )
    pd.DataFrame({"predictor": dgm_predictors}).to_csv(
        paths["tables"] / "predictor_list_dgm_features.csv", index=False
    )

    if not dgm_predictors:
        raise RuntimeError("No DGM predictor features could be built.")
    if not safe_predictors:
        raise RuntimeError("No all-safe predictors could be built.")

    print(f"All-safe predictor count: {len(safe_predictors)}")
    print(f"DGM fallback predictor count: {len(dgm_predictors)}")
    if missing_intended:
        print("Missing intended DGM fallback features:", ", ".join(missing_intended))
    print("Predictor audits saved under", paths["tables"])

    return {
        "all_safe_predictors": safe_predictors,
        "dgm_predictors": dgm_predictors,
        "missing_intended_dgm_features": missing_intended,
    }


def replicate_random_state(base_seed: int, replicate_id: Any) -> int:
    try:
        return int(base_seed + int(replicate_id))
    except Exception:
        return int(base_seed + abs(hash(str(replicate_id))) % 10_000)


def split_train_test(rep_df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    counts = rep_df[STATUS_COL].value_counts(dropna=False)
    if counts.size > 1 and bool((counts >= 2).all()):
        stratify = rep_df[STATUS_COL]

    random_state = replicate_random_state(config.random_seed, rep_df["replicate_id"].iloc[0])
    try:
        train_df, test_df = train_test_split(
            rep_df,
            test_size=config.test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except Exception:
        train_df, test_df = train_test_split(
            rep_df,
            test_size=config.test_size,
            random_state=random_state,
            stratify=None,
        )
    return train_df.copy(), test_df.copy()


def status_event(series: pd.Series) -> np.ndarray:
    return (series.astype(int).to_numpy() == EVENT_STATUS).astype(int)


def harrell_c_index(duration: Sequence[float], event: Sequence[int], risk_score: Sequence[float]) -> float:
    duration_arr = np.asarray(duration, dtype=float)
    event_arr = np.asarray(event, dtype=int)
    risk_arr = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(duration_arr) & np.isfinite(event_arr) & np.isfinite(risk_arr)
    if mask.sum() < 2 or event_arr[mask].sum() == 0:
        return np.nan
    try:
        return float(concordance_index(duration_arr[mask], -risk_arr[mask], event_arr[mask]))
    except Exception:
        return np.nan


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return np.nan
    if np.unique(x_arr[mask]).size < 2 or np.unique(y_arr[mask]).size < 2:
        return np.nan
    try:
        corr, _ = spearmanr(x_arr[mask], y_arr[mask])
        return float(corr)
    except Exception:
        return np.nan


def safe_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def calibration_slope(observed: np.ndarray, pred_risk: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(pred_risk)
    if mask.sum() < 10:
        return np.nan
    y = observed[mask].astype(int)
    if np.unique(y).size < 2:
        return np.nan
    x_logit = safe_logit(pred_risk[mask]).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        model.fit(x_logit, y)
        return float(model.coef_[0][0])
    except Exception:
        return np.nan


def base_metric_row(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    runtime_sec: float,
    failed: bool,
    failure_reason: str = "",
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "model": model_name,
        "cindex": np.nan,
        "spearman_true_lp": np.nan,
        "spearman_true_risk5": np.nan,
        "risk5_mae_vs_true_risk": np.nan,
        "risk5_rmse_vs_true_risk": np.nan,
        "risk5_naive_brier": np.nan,
        "risk5_calibration_slope": np.nan,
        "risk5_available": False,
        "risk5_reason": "",
        "runtime_sec": runtime_sec,
        "failed": failed,
        "failure_reason": failure_reason,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_events_train": int((train_df[STATUS_COL].astype(int) == EVENT_STATUS).sum()),
        "n_events_test": int((test_df[STATUS_COL].astype(int) == EVENT_STATUS).sum()),
        "n_death_competing_test": int((test_df[STATUS_COL].astype(int) == COMPETING_STATUS).sum()),
        "n_censored_test": int((test_df[STATUS_COL].astype(int) == 0).sum()),
    }


def evaluate_predictions(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    risk_score: Sequence[float],
    risk_5y: Optional[Sequence[float]],
    runtime_sec: float,
    risk5_reason: str = "",
) -> Dict[str, Any]:
    row = base_metric_row(
        scenario_id=scenario_id,
        replicate_id=replicate_id,
        model_name=model_name,
        train_df=train_df,
        test_df=test_df,
        runtime_sec=runtime_sec,
        failed=False,
    )
    risk_arr = np.asarray(risk_score, dtype=float)
    event_arr = status_event(test_df[STATUS_COL])
    duration_arr = test_df[DURATION_COL].astype(float).to_numpy()
    row["cindex"] = harrell_c_index(duration_arr, event_arr, risk_arr)

    if "true_lp_carehome" in test_df.columns:
        row["spearman_true_lp"] = safe_spearman(risk_arr, test_df["true_lp_carehome"].to_numpy())
    if TRUTH_RISK_COL in test_df.columns:
        row["spearman_true_risk5"] = safe_spearman(risk_arr, test_df[TRUTH_RISK_COL].to_numpy())

    if risk_5y is None:
        row["risk5_reason"] = risk5_reason or "no_valid_5y_risk_prediction"
        return row

    pred_risk = np.asarray(risk_5y, dtype=float)
    if TRUTH_RISK_COL not in test_df.columns:
        row["risk5_reason"] = "true_risk_carehome_5y_observable_approx_missing"
        return row

    true_risk = test_df[TRUTH_RISK_COL].astype(float).to_numpy()
    valid = np.isfinite(pred_risk) & np.isfinite(true_risk)
    if valid.sum() == 0:
        row["risk5_reason"] = risk5_reason or "no_finite_predicted_or_true_5y_risk"
        return row

    pred_risk = np.clip(pred_risk, 0.0, 1.0)
    diff = pred_risk[valid] - true_risk[valid]
    observed_5y = (
        (test_df[STATUS_COL].astype(int).to_numpy() == EVENT_STATUS)
        & (duration_arr <= EVAL_HORIZON_YEARS)
    ).astype(int)
    row["risk5_available"] = True
    row["risk5_mae_vs_true_risk"] = float(np.mean(np.abs(diff)))
    row["risk5_rmse_vs_true_risk"] = float(np.sqrt(np.mean(diff**2)))
    row["risk5_naive_brier"] = float(np.mean((observed_5y[valid] - pred_risk[valid]) ** 2))
    row["risk5_calibration_slope"] = calibration_slope(observed_5y[valid], pred_risk[valid])
    row["risk5_reason"] = risk5_reason
    return row


def one_hot_encoder() -> Any:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def get_feature_names(transformer: Any, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    names: List[str] = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        try:
            encoder = transformer.named_transformers_["categorical"].named_steps["onehot"]
            names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
        except Exception:
            names.extend([f"categorical_{i}" for i in range(len(categorical_cols))])
    return names


def build_model_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    scale_numeric: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    raw_predictors = [c for c in predictors if c in train_df.columns and c in test_df.columns]
    if not raw_predictors:
        raise ValueError("No requested predictors are present in train/test data.")

    X_train_raw = train_df[raw_predictors].copy()
    X_test_raw = test_df[raw_predictors].copy()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    dropped_all_missing: List[str] = []

    for col in raw_predictors:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]) or pd.api.types.is_bool_dtype(X_train_raw[col]):
            if X_train_raw[col].notna().sum() == 0:
                dropped_all_missing.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    if dropped_all_missing:
        X_train_raw = X_train_raw.drop(columns=dropped_all_missing)
        X_test_raw = X_test_raw.drop(columns=dropped_all_missing)

    transformers = []
    if numeric_cols:
        numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", one_hot_encoder()),
        ]
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_cols))

    if not transformers:
        raise ValueError("No usable predictors remain after dropping all-missing columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    if X_train.shape[1] != len(feature_names):
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if X_train.shape[1] == 0:
        raise ValueError("Preprocessed model matrix has zero columns.")

    std = np.nanstd(X_train, axis=0)
    keep = std > 1e-12
    if not keep.any():
        raise ValueError("All preprocessed predictors are constant in training data.")

    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    feature_names = [name for name, k in zip(feature_names, keep) if k]

    return (
        pd.DataFrame(X_train, columns=feature_names, index=train_df.index),
        pd.DataFrame(X_test, columns=feature_names, index=test_df.index),
        feature_names,
    )


def fit_oracle(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if "true_lp_carehome" not in test_df.columns:
        raise ValueError("true_lp_carehome is required for oracle benchmark.")
    risk_score = test_df["true_lp_carehome"].astype(float).to_numpy()
    if TRUTH_RISK_COL in test_df.columns:
        risk_5y = test_df[TRUTH_RISK_COL].astype(float).to_numpy()
        risk5_reason = "oracle_uses_exported_true_observable_approx_5y_risk"
    else:
        risk_5y = None
        risk5_reason = "true_risk_carehome_5y_observable_approx_missing"
    return risk_score, risk_5y, risk5_reason


def baseline_hazard_at_5y(cph: Any) -> Tuple[float, str]:
    baseline = cph.baseline_cumulative_hazard_
    if baseline is None or baseline.empty:
        return np.nan, "cox_baseline_cumulative_hazard_empty"
    series = baseline.iloc[:, 0].copy()
    index_values = pd.to_numeric(pd.Series(series.index), errors="coerce").to_numpy(dtype=float)
    values = series.to_numpy(dtype=float)
    valid = np.isfinite(index_values) & np.isfinite(values)
    if valid.sum() == 0:
        return np.nan, "cox_baseline_cumulative_hazard_not_finite"
    eligible = valid & (index_values <= EVAL_HORIZON_YEARS)
    if eligible.sum() == 0:
        return np.nan, "cox_no_baseline_cumulative_hazard_at_or_before_5y"
    h0_5 = float(values[np.where(eligible)[0][-1]])
    if not np.isfinite(h0_5) or h0_5 < 0:
        return np.nan, "cox_invalid_baseline_cumulative_hazard_at_5y"
    return h0_5, ""


def fit_cox_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    penalizer_grid: Sequence[float],
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    X_train, X_test, _ = build_model_matrix(train_df, test_df, predictors, scale_numeric=True)
    train_event = status_event(train_df[STATUS_COL])
    cox_train = X_train.copy()
    cox_train[DURATION_COL] = train_df[DURATION_COL].astype(float).to_numpy()
    cox_train["event_carehome_cs"] = train_event

    last_error: Optional[Exception] = None
    fitted_model = None
    for penalizer in penalizer_grid:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(
                    cox_train,
                    duration_col=DURATION_COL,
                    event_col="event_carehome_cs",
                    show_progress=False,
                )
            fitted_model = cph
            break
        except Exception as exc:
            last_error = exc
            fitted_model = None

    if fitted_model is None:
        raise RuntimeError(f"Cox model failed for all penalizers: {last_error}")

    lp = fitted_model.predict_log_partial_hazard(X_test).astype(float).to_numpy()
    h0_5, risk5_reason = baseline_hazard_at_5y(fitted_model)
    if np.isfinite(h0_5):
        risk_5y = 1.0 - np.exp(-h0_5 * np.exp(lp))
        risk_5y = np.clip(risk_5y, 0.0, 1.0)
    else:
        risk_5y = None
    return lp, risk_5y, risk5_reason


def fit_cox_dgm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    return fit_cox_model(train_df, test_df, predictors, penalizer_grid=[0.01, 0.05, 0.1, 0.5, 1.0])


def fit_penalised_cox(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    return fit_cox_model(train_df, test_df, predictors, penalizer_grid=[0.1, 0.5, 1.0, 2.0, 5.0])


def fit_xgb_survival_strict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    X_train, X_test, _ = build_model_matrix(train_df, test_df, predictors, scale_numeric=False)
    duration_train = train_df[DURATION_COL].astype(float).to_numpy()
    event_train = status_event(train_df[STATUS_COL]).astype(bool)
    labels = np.where(event_train, duration_train, -duration_train)

    model = xgb.XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=10,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
        verbosity=0,
    )
    model.fit(X_train, labels)
    try:
        risk_score = model.predict(X_test, output_margin=True)
    except TypeError:
        risk_score = model.predict(X_test)
    return np.asarray(risk_score, dtype=float), None, "xgboost_survival_cox_outputs_risk_score_only_no_5y_calibration"


def prediction_sample_rows(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    test_df: pd.DataFrame,
    risk_score: Sequence[float],
    risk_5y: Optional[Sequence[float]],
    seed: int,
) -> pd.DataFrame:
    n = min(DEBUG_PRED_SAMPLE_N, len(test_df))
    if n == 0:
        return pd.DataFrame()
    sample = test_df.sample(n=n, random_state=seed).copy()
    positions = test_df.index.get_indexer(sample.index)
    risk_arr = np.asarray(risk_score, dtype=float)
    out = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "model": model_name,
            "synthetic_id": sample["synthetic_id"].to_numpy() if "synthetic_id" in sample.columns else "",
            DURATION_COL: sample[DURATION_COL].to_numpy(),
            STATUS_COL: sample[STATUS_COL].to_numpy(),
            "risk_score": risk_arr[positions],
            "true_lp_carehome": sample["true_lp_carehome"].to_numpy()
            if "true_lp_carehome" in sample.columns
            else np.nan,
            TRUTH_RISK_COL: sample[TRUTH_RISK_COL].to_numpy() if TRUTH_RISK_COL in sample.columns else np.nan,
        }
    )
    if risk_5y is not None:
        risk5_arr = np.asarray(risk_5y, dtype=float)
        out["predicted_risk5"] = risk5_arr[positions]
    else:
        out["predicted_risk5"] = np.nan
    return out


def run_model_with_capture(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    fit_func: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_collector: List[pd.DataFrame],
    config: Config,
) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        risk_score, risk_5y, risk5_reason = fit_func()
        runtime_sec = time.perf_counter() - start
        row = evaluate_predictions(
            scenario_id=scenario_id,
            replicate_id=replicate_id,
            model_name=model_name,
            train_df=train_df,
            test_df=test_df,
            risk_score=risk_score,
            risk_5y=risk_5y,
            runtime_sec=runtime_sec,
            risk5_reason=risk5_reason,
        )
        if config.debug_mode:
            sample_collector.append(
                prediction_sample_rows(
                    scenario_id,
                    replicate_id,
                    model_name,
                    test_df,
                    risk_score,
                    risk_5y,
                    seed=replicate_random_state(config.random_seed, replicate_id),
                )
            )
        return row
    except Exception as exc:
        runtime_sec = time.perf_counter() - start
        reason = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {scenario_id} rep={replicate_id} model={model_name}: {reason}")
        traceback.print_exc(limit=2)
        return base_metric_row(
            scenario_id=scenario_id,
            replicate_id=replicate_id,
            model_name=model_name,
            train_df=train_df,
            test_df=test_df,
            runtime_sec=runtime_sec,
            failed=True,
            failure_reason=reason,
        )


def result_key(scenario_id: Any, replicate_id: Any, model_name: Any) -> Tuple[str, str, str]:
    return (str(scenario_id), str(replicate_id), str(model_name))


def dedupe_performance(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return perf.copy()
    key_cols = ["scenario_id", "replicate_id", "model"]
    if not set(key_cols).issubset(perf.columns):
        return perf.copy()
    out = perf.copy()
    out["_replicate_id_key"] = out["replicate_id"].astype(str)
    out = out.drop_duplicates(
        subset=["scenario_id", "_replicate_id_key", "model"],
        keep="last",
    ).drop(columns=["_replicate_id_key"])
    return out.reset_index(drop=True)


def load_existing_performance(paths: Dict[str, Path]) -> pd.DataFrame:
    perf_path = paths["tables"] / "replicate_model_performance.csv"
    if not perf_path.exists():
        return pd.DataFrame()
    try:
        perf = pd.read_csv(perf_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    required = {"scenario_id", "replicate_id", "model", "failed"}
    if not required.issubset(perf.columns):
        print(f"Ignoring existing performance file with missing columns: {perf_path}")
        return pd.DataFrame()
    perf = dedupe_performance(perf)
    print(f"Loaded existing checkpoint rows: {len(perf)}")
    return perf


def success_keys_from_performance(perf: pd.DataFrame) -> set[Tuple[str, str, str]]:
    if perf.empty:
        return set()
    success = perf.loc[~perf["failed"].astype(bool)].copy()
    return {
        result_key(row.scenario_id, row.replicate_id, row.model)
        for row in success.itertuples(index=False)
    }


def replace_result_row(result_rows: List[Dict[str, Any]], new_row: Dict[str, Any]) -> None:
    new_key = result_key(new_row["scenario_id"], new_row["replicate_id"], new_row["model"])
    kept = [
        row
        for row in result_rows
        if result_key(row.get("scenario_id"), row.get("replicate_id"), row.get("model")) != new_key
    ]
    kept.append(new_row)
    result_rows[:] = kept


def save_performance_checkpoint(result_rows: List[Dict[str, Any]], paths: Dict[str, Path]) -> pd.DataFrame:
    perf = dedupe_performance(pd.DataFrame(result_rows))
    perf.to_csv(paths["tables"] / "replicate_model_performance.csv", index=False)
    save_failure_log(perf, paths)
    return perf


def process_scenario_file(
    file_path: Path,
    predictors: Dict[str, List[str]],
    config: Config,
    sample_collector: List[pd.DataFrame],
    result_rows: List[Dict[str, Any]],
    existing_success_keys: set[Tuple[str, str, str]],
    paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    print(f"Loading scenario file: {file_path.name}")
    scenario_df = pd.read_csv(file_path)
    if scenario_df.empty:
        raise ValueError(f"Scenario file is empty: {file_path}")

    scenario_ids = scenario_df["scenario_id"].dropna().astype(str).unique().tolist()
    scenario_id = scenario_ids[0] if scenario_ids else file_path.stem.replace(".csv", "")
    replicate_ids = sorted(scenario_df["replicate_id"].dropna().unique().tolist())
    if config.max_reps_per_scenario is not None:
        replicate_ids = replicate_ids[: config.max_reps_per_scenario]
    print(f"Processing {scenario_id}: {len(replicate_ids)} replicate(s)")

    new_results: List[Dict[str, Any]] = []
    for replicate_id in replicate_ids:
        if all(result_key(scenario_id, replicate_id, model_name) in existing_success_keys for model_name in MODELS):
            print(f"  replicate {replicate_id}: skip all models already successful")
            continue
        rep_df = scenario_df.loc[scenario_df["replicate_id"] == replicate_id].copy()
        train_df, test_df = split_train_test(rep_df, config)
        print(
            f"  replicate {replicate_id}: train={len(train_df)} test={len(test_df)} "
            f"events_test={(test_df[STATUS_COL].astype(int) == EVENT_STATUS).sum()}"
        )
        seed = replicate_random_state(config.random_seed, replicate_id)
        model_calls = [
            (
                MODEL_ORACLE,
                lambda train_df=train_df, test_df=test_df: fit_oracle(train_df, test_df),
            ),
            (
                MODEL_COX_DGM,
                lambda train_df=train_df, test_df=test_df: fit_cox_dgm(
                    train_df, test_df, predictors["dgm_predictors"]
                ),
            ),
            (
                MODEL_COX_SAFE,
                lambda train_df=train_df, test_df=test_df: fit_penalised_cox(
                    train_df, test_df, predictors["all_safe_predictors"]
                ),
            ),
            (
                MODEL_XGB,
                lambda train_df=train_df, test_df=test_df, seed=seed: fit_xgb_survival_strict(
                    train_df, test_df, predictors["all_safe_predictors"], seed=seed
                ),
            ),
        ]
        for model_name, fit_func in model_calls:
            key = result_key(scenario_id, replicate_id, model_name)
            if key in existing_success_keys:
                print(f"    skip existing successful result: rep={replicate_id} model={model_name}")
                continue
            row = run_model_with_capture(
                scenario_id=scenario_id,
                replicate_id=replicate_id,
                model_name=model_name,
                fit_func=fit_func,
                train_df=train_df,
                test_df=test_df,
                sample_collector=sample_collector,
                config=config,
            )
            replace_result_row(result_rows, row)
            new_results.append(row)
            if not bool(row.get("failed", False)):
                existing_success_keys.add(key)
            checkpoint = save_performance_checkpoint(result_rows, paths)
            print(
                f"    saved checkpoint: rows={len(checkpoint)} "
                f"failures={int(checkpoint['failed'].astype(bool).sum())}"
            )

    del scenario_df
    return new_results


def metric_stats(values: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
        }
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n > 0 else np.nan
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - 1.96 * se if np.isfinite(se) else np.nan,
        "ci95_high": mean + 1.96 * se if np.isfinite(se) else np.nan,
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
    }


def aggregate_results(perf: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    metric_columns = [
        "cindex",
        "spearman_true_lp",
        "spearman_true_risk5",
        "risk5_mae_vs_true_risk",
        "risk5_rmse_vs_true_risk",
        "risk5_naive_brier",
        "risk5_calibration_slope",
        "runtime_sec",
    ]
    rows: List[Dict[str, Any]] = []
    for (scenario_id, model_name), group in perf.groupby(["scenario_id", "model"], sort=True):
        success = group.loc[~group["failed"].astype(bool)].copy()
        n_success = int(success.shape[0])
        n_failed = int(group["failed"].astype(bool).sum())
        row: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "model": model_name,
            "n_successful_reps": n_success,
            "n_failed_reps": n_failed,
            "failure_rate": float(n_failed / group.shape[0]) if group.shape[0] else np.nan,
        }
        for metric in metric_columns:
            stats = metric_stats(success[metric] if metric in success.columns else pd.Series(dtype=float))
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(paths["tables"] / "scenario_model_summary_mean_sd_ci.csv", index=False)

    ranking = summary.copy()
    if not ranking.empty:
        ranking["cindex_rank_desc"] = ranking.groupby("scenario_id")["cindex_mean"].rank(
            ascending=False, method="min"
        )
        ranking["spearman_true_lp_rank_desc"] = ranking.groupby("scenario_id")[
            "spearman_true_lp_mean"
        ].rank(ascending=False, method="min")
        ranking["risk5_mae_rank_asc"] = ranking.groupby("scenario_id")[
            "risk5_mae_vs_true_risk_mean"
        ].rank(ascending=True, method="min")
    ranking.to_csv(paths["tables"] / "model_ranking_by_scenario.csv", index=False)

    pivot_specs = [
        ("cindex_mean", "pivot_cindex_mean.csv"),
        ("spearman_true_lp_mean", "pivot_spearman_true_lp_mean.csv"),
        ("risk5_mae_vs_true_risk_mean", "pivot_risk5_mae_mean.csv"),
    ]
    for value_col, file_name in pivot_specs:
        if value_col in summary.columns:
            pivot = summary.pivot(index="scenario_id", columns="model", values=value_col)
            pivot.to_csv(paths["tables"] / file_name)
        else:
            pd.DataFrame().to_csv(paths["tables"] / file_name)

    return summary


def save_failure_log(perf: pd.DataFrame, paths: Dict[str, Path]) -> None:
    failure_cols = [
        "scenario_id",
        "replicate_id",
        "model",
        "failed",
        "failure_reason",
        "runtime_sec",
        "n_train",
        "n_test",
        "n_events_train",
        "n_events_test",
    ]
    failures = perf.loc[perf["failed"].astype(bool), [c for c in failure_cols if c in perf.columns]]
    failures.to_csv(paths["tables"] / "failure_log.csv", index=False)


def value_from_row(row: pd.Series, column: str) -> float:
    if row.empty or column not in row.index:
        return np.nan
    return float(row[column]) if pd.notna(row[column]) else np.nan


def create_model_difference_vs_oracle(summary: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario_id, group in summary.groupby("scenario_id", sort=True):
        oracle_rows = group.loc[group["model"] == MODEL_ORACLE]
        if oracle_rows.empty:
            continue
        oracle = oracle_rows.iloc[0]
        for _, model_row in group.loc[group["model"] != MODEL_ORACLE].iterrows():
            cindex_gap = value_from_row(model_row, "cindex_mean") - value_from_row(oracle, "cindex_mean")
            spearman_gap = value_from_row(model_row, "spearman_true_lp_mean") - value_from_row(
                oracle, "spearman_true_lp_mean"
            )
            model_risk5 = value_from_row(model_row, "risk5_mae_vs_true_risk_mean")
            oracle_risk5 = value_from_row(oracle, "risk5_mae_vs_true_risk_mean")
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model_row["model"],
                    "oracle_cindex_mean": value_from_row(oracle, "cindex_mean"),
                    "model_cindex_mean": value_from_row(model_row, "cindex_mean"),
                    "cindex_gap_vs_oracle": cindex_gap,
                    "oracle_spearman_true_lp_mean": value_from_row(oracle, "spearman_true_lp_mean"),
                    "model_spearman_true_lp_mean": value_from_row(model_row, "spearman_true_lp_mean"),
                    "spearman_gap_vs_oracle": spearman_gap,
                    "model_risk5_mae_mean": model_risk5,
                    "oracle_risk5_mae_mean": oracle_risk5,
                    "risk5_mae_gap_vs_oracle": model_risk5 - oracle_risk5
                    if np.isfinite(model_risk5) and np.isfinite(oracle_risk5)
                    else np.nan,
                    "failure_rate": value_from_row(model_row, "failure_rate"),
                    "n_successful_reps": int(model_row.get("n_successful_reps", 0)),
                    "flag_model_exceeds_oracle_cindex_by_more_than_0_02": bool(cindex_gap > 0.02)
                    if np.isfinite(cindex_gap)
                    else False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "model_difference_vs_oracle_by_scenario.csv", index=False)
    return out


def create_best_non_oracle_model_by_scenario(summary: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    non_oracle = summary.loc[summary["model"] != MODEL_ORACLE].copy()
    for scenario_id, group in non_oracle.groupby("scenario_id", sort=True):
        ranked = group.copy()
        ranked["_cindex_sort"] = pd.to_numeric(ranked["cindex_mean"], errors="coerce").fillna(-np.inf)
        ranked["_spearman_sort"] = pd.to_numeric(ranked["spearman_true_lp_mean"], errors="coerce").fillna(-np.inf)
        ranked["_risk5_sort"] = pd.to_numeric(
            ranked["risk5_mae_vs_true_risk_mean"], errors="coerce"
        ).fillna(np.inf)
        ranked["_failure_sort"] = pd.to_numeric(ranked["failure_rate"], errors="coerce").fillna(np.inf)
        ranked = ranked.sort_values(
            by=["_cindex_sort", "_spearman_sort", "_risk5_sort", "_failure_sort", "model"],
            ascending=[False, False, True, True, True],
        ).reset_index(drop=True)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else pd.Series(dtype=object)
        best_c = value_from_row(best, "cindex_mean")
        second_c = value_from_row(second, "cindex_mean") if not second.empty else np.nan
        gap = best_c - second_c if np.isfinite(best_c) and np.isfinite(second_c) else np.nan
        note = (
            f"{best['model']} had the highest mean C-index among fitted models."
            if np.isfinite(gap)
            else f"{best['model']} was the only fitted model with available C-index."
        )
        if np.isfinite(gap):
            note += f" The margin over the second fitted model was {gap:.4f}."
        rows.append(
            {
                "scenario_id": scenario_id,
                "best_model_by_cindex": best["model"],
                "best_cindex_mean": best_c,
                "second_best_model": second.get("model", "") if not second.empty else "",
                "second_best_cindex_mean": second_c,
                "cindex_difference_best_minus_second": gap,
                "best_model_spearman_true_lp_mean": value_from_row(best, "spearman_true_lp_mean"),
                "best_model_risk5_mae_mean": value_from_row(best, "risk5_mae_vs_true_risk_mean"),
                "interpretation_note": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "best_non_oracle_model_by_scenario.csv", index=False)
    return out


def create_xgb_oracle_sanity_audit_full(summary: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario_id, group in summary.groupby("scenario_id", sort=True):
        oracle_rows = group.loc[group["model"] == MODEL_ORACLE]
        xgb_rows = group.loc[group["model"] == MODEL_XGB]
        if oracle_rows.empty or xgb_rows.empty:
            continue
        oracle = oracle_rows.iloc[0]
        xgb_row = xgb_rows.iloc[0]
        xgb_minus_oracle = value_from_row(xgb_row, "cindex_mean") - value_from_row(oracle, "cindex_mean")
        xgb_spearman = value_from_row(xgb_row, "spearman_true_lp_mean")
        flag_exceeds = bool(xgb_minus_oracle > 0.02) if np.isfinite(xgb_minus_oracle) else False
        flag_low_alignment = bool(xgb_spearman < 0.70) if np.isfinite(xgb_spearman) else True
        rows.append(
            {
                "scenario_id": scenario_id,
                "oracle_cindex_mean": value_from_row(oracle, "cindex_mean"),
                "xgb_cindex_mean": value_from_row(xgb_row, "cindex_mean"),
                "xgb_minus_oracle": xgb_minus_oracle,
                "xgb_spearman_true_lp_mean": xgb_spearman,
                "oracle_spearman_true_lp_mean": value_from_row(oracle, "spearman_true_lp_mean"),
                "flag_xgb_exceeds_oracle_by_more_than_0_02": flag_exceeds,
                "flag_xgb_low_alignment_with_true_lp": flag_low_alignment,
                "needs_leakage_audit": flag_exceeds or flag_low_alignment,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "xgb_oracle_sanity_audit_full.csv", index=False)
    return out


def get_summary_metric(summary: pd.DataFrame, scenario_id: str, model_name: str, metric: str) -> float:
    rows = summary.loc[(summary["scenario_id"] == scenario_id) & (summary["model"] == model_name)]
    if rows.empty:
        return np.nan
    return value_from_row(rows.iloc[0], metric)


def create_scenario_interpretation_table(
    summary: pd.DataFrame,
    best_table: pd.DataFrame,
    xgb_audit: pd.DataFrame,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    best_lookup = best_table.set_index("scenario_id").to_dict("index") if not best_table.empty else {}
    xgb_lookup = xgb_audit.set_index("scenario_id").to_dict("index") if not xgb_audit.empty else {}
    s0_best_c = (
        float(best_lookup["S0_linear_PH_inst30"]["best_cindex_mean"])
        if "S0_linear_PH_inst30" in best_lookup
        else np.nan
    )

    for scenario_id in sorted(summary["scenario_id"].unique().tolist()):
        best_info = best_lookup.get(scenario_id, {})
        xgb_info = xgb_lookup.get(scenario_id, {})
        best_model = best_info.get("best_model_by_cindex", "")
        best_c = float(best_info.get("best_cindex_mean", np.nan))
        oracle_c = get_summary_metric(summary, scenario_id, MODEL_ORACLE, "cindex_mean")
        cox_c = get_summary_metric(summary, scenario_id, MODEL_COX_DGM, "cindex_mean")
        penal_c = get_summary_metric(summary, scenario_id, MODEL_COX_SAFE, "cindex_mean")
        xgb_c = get_summary_metric(summary, scenario_id, MODEL_XGB, "cindex_mean")
        cox_gap = cox_c - oracle_c if np.isfinite(cox_c) and np.isfinite(oracle_c) else np.nan
        xgb_gap = float(xgb_info.get("xgb_minus_oracle", np.nan))
        observed_pattern = (
            f"Best fitted model was {best_model} with mean C-index {best_c:.3f}; "
            f"oracle mean C-index was {oracle_c:.3f}."
            if np.isfinite(best_c) and np.isfinite(oracle_c)
            else f"Best fitted model was {best_model}; one or more C-index values were unavailable."
        )

        if scenario_id in {"S0_linear_PH_inst30", "S1_linear_PH_inst15", "S2_linear_PH_inst45"}:
            if np.isfinite(cox_gap) and cox_gap >= -0.03:
                short = "Cox DGM stayed close to the oracle benchmark in this linear PH setting."
            else:
                short = "Cox DGM was below oracle by more than the close-to-oracle tolerance used here."
        elif scenario_id == "S3_nonlinear_interaction_inst30":
            short = (
                "The nonlinear XGBoost model was the best fitted model by C-index."
                if best_model == MODEL_XGB
                else f"{best_model} led the fitted models, so XGBoost did not dominate this nonlinear scenario."
            )
        elif scenario_id == "S4_nonPH_inst30":
            short = (
                "The non-PH scenario retained measurable discrimination, but proportional models should be interpreted as approximations."
            )
        elif scenario_id == "S5_MAR_missingness_inst30":
            if np.isfinite(s0_best_c) and np.isfinite(best_c):
                delta = best_c - s0_best_c
                short = f"Best fitted-model C-index differed from S0 by {delta:.3f}, summarising the observed missingness impact."
            else:
                short = "Missingness impact could not be compared directly with S0 because C-index values were unavailable."
        elif scenario_id == "S6_highdim_sparseMRI_inst30":
            if best_model in {MODEL_COX_SAFE, MODEL_XGB}:
                short = "The all-safe/high-dimensional model family was most competitive in the sparse MRI-signal scenario."
            else:
                short = "Cox DGM remained the strongest fitted model despite high-dimensional sparse MRI predictors."
        elif scenario_id == "S7_strong_death_competing_inst30":
            short = (
                f"Under stronger competing death risk, {best_model} led fitted models; cause-specific results remain censoring-based."
            )
        else:
            short = "Observed pattern summarised from the full-run model comparison."

        if bool(xgb_info.get("needs_leakage_audit", False)):
            short += " XGBoost sanity flags require follow-up audit."

        rows.append(
            {
                "scenario_id": scenario_id,
                "primary_question": SCENARIO_PRIMARY_QUESTIONS.get(scenario_id, ""),
                "expected_pattern": SCENARIO_EXPECTED_PATTERNS.get(scenario_id, ""),
                "observed_best_non_oracle_model": best_model,
                "observed_pattern": observed_pattern
                + (
                    f" Cox DGM gap vs oracle was {cox_gap:.3f}; XGB gap vs oracle was {xgb_gap:.3f}."
                    if np.isfinite(cox_gap) and np.isfinite(xgb_gap)
                    else ""
                ),
                "short_interpretation": short,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "scenario_interpretation_table.csv", index=False)
    return out


def any_forbidden_predictor_flag(paths: Dict[str, Path]) -> bool:
    audit_path = paths["tables"] / "predictor_audit_all_safe.csv"
    if not audit_path.exists():
        return True
    audit = pd.read_csv(audit_path)
    if not {"included_in_all_safe_predictors", "column"}.issubset(audit.columns):
        return True
    included = audit.loc[audit["included_in_all_safe_predictors"].astype(bool), "column"].astype(str)
    return any(contains_forbidden_name(col) for col in included)


def create_full_run_sanity_checks(
    perf: pd.DataFrame,
    xgb_audit: pd.DataFrame,
    safety: pd.DataFrame,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    perf = dedupe_performance(perf)
    observed_rows = int(len(perf))
    observed_scenarios = int(perf["scenario_id"].nunique()) if not perf.empty else 0
    reps_by_scenario = perf.groupby("scenario_id")["replicate_id"].nunique() if not perf.empty else pd.Series(dtype=int)
    observed_min_reps = int(reps_by_scenario.min()) if not reps_by_scenario.empty else 0
    observed_max_reps = int(reps_by_scenario.max()) if not reps_by_scenario.empty else 0
    total_failed = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    failure_log_path = paths["tables"] / "failure_log.csv"
    if failure_log_path.exists():
        try:
            failure_log_rows = int(pd.read_csv(failure_log_path).shape[0])
        except pd.errors.EmptyDataError:
            failure_log_rows = 0
    else:
        failure_log_rows = 0
    any_xgb_flag = bool(xgb_audit["needs_leakage_audit"].astype(bool).any()) if not xgb_audit.empty else True
    forbidden_flag = any_forbidden_predictor_flag(paths)
    export_safety_passed = bool_series_all_true(safety["safe_to_export_column_names"])
    acceptable_failure_rate = (
        observed_rows > 0
        and failure_log_rows == total_failed
        and (total_failed / observed_rows) <= 0.05
    )
    full_run_passed = (
        observed_rows == EXPECTED_FULL_REPLICATE_MODEL_ROWS
        and observed_scenarios == EXPECTED_FULL_SCENARIOS
        and observed_min_reps == EXPECTED_REPS_PER_SCENARIO
        and observed_max_reps == EXPECTED_REPS_PER_SCENARIO
        and export_safety_passed
        and not forbidden_flag
        and acceptable_failure_rate
        and not any_xgb_flag
    )

    rows = [
        {
            "check_name": "expected_replicate_model_rows",
            "expected": EXPECTED_FULL_REPLICATE_MODEL_ROWS,
            "observed": EXPECTED_FULL_REPLICATE_MODEL_ROWS,
            "passed": True,
            "detail": "Full-run target is 8 scenarios x 50 reps x 4 models.",
        },
        {
            "check_name": "observed_replicate_model_rows",
            "expected": EXPECTED_FULL_REPLICATE_MODEL_ROWS,
            "observed": observed_rows,
            "passed": observed_rows == EXPECTED_FULL_REPLICATE_MODEL_ROWS,
            "detail": "Rows after de-duplicating by scenario_id, replicate_id, and model.",
        },
        {
            "check_name": "expected_scenarios",
            "expected": EXPECTED_FULL_SCENARIOS,
            "observed": EXPECTED_FULL_SCENARIOS,
            "passed": True,
            "detail": "Expected synthetic Step C scenario count.",
        },
        {
            "check_name": "observed_scenarios",
            "expected": EXPECTED_FULL_SCENARIOS,
            "observed": observed_scenarios,
            "passed": observed_scenarios == EXPECTED_FULL_SCENARIOS,
            "detail": "Unique scenarios represented in replicate_model_performance.csv.",
        },
        {
            "check_name": "expected_reps_per_scenario",
            "expected": EXPECTED_REPS_PER_SCENARIO,
            "observed": EXPECTED_REPS_PER_SCENARIO,
            "passed": True,
            "detail": "Expected repetition count per synthetic scenario.",
        },
        {
            "check_name": "observed_min_reps_per_scenario",
            "expected": EXPECTED_REPS_PER_SCENARIO,
            "observed": observed_min_reps,
            "passed": observed_min_reps == EXPECTED_REPS_PER_SCENARIO,
            "detail": "Minimum unique replicate count over scenarios.",
        },
        {
            "check_name": "observed_max_reps_per_scenario",
            "expected": EXPECTED_REPS_PER_SCENARIO,
            "observed": observed_max_reps,
            "passed": observed_max_reps == EXPECTED_REPS_PER_SCENARIO,
            "detail": "Maximum unique replicate count over scenarios.",
        },
        {
            "check_name": "total_failed_model_fits",
            "expected": "logged and <=5%",
            "observed": total_failed,
            "passed": acceptable_failure_rate,
            "detail": f"failure_log rows={failure_log_rows}; observed rows={observed_rows}.",
        },
        {
            "check_name": "any_xgb_oracle_sanity_flag",
            "expected": False,
            "observed": any_xgb_flag,
            "passed": not any_xgb_flag,
            "detail": "True if XGBoost exceeds oracle by >0.02 or has Spearman true-LP alignment <0.70.",
        },
        {
            "check_name": "any_forbidden_predictor_flag",
            "expected": False,
            "observed": forbidden_flag,
            "passed": not forbidden_flag,
            "detail": "Strict all-safe predictors must not contain forbidden outcome/truth/design names.",
        },
        {
            "check_name": "export_safety_passed",
            "expected": True,
            "observed": export_safety_passed,
            "passed": export_safety_passed,
            "detail": "Based on audit/export_safety_audit.csv.",
        },
        {
            "check_name": "full_run_passed",
            "expected": True,
            "observed": full_run_passed,
            "passed": full_run_passed,
            "detail": "Composite full-run gate.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "full_run_sanity_checks.csv", index=False)
    return out


def create_additional_full_run_tables(
    summary: pd.DataFrame,
    perf: pd.DataFrame,
    inputs: Dict[str, Any],
    paths: Dict[str, Path],
) -> Dict[str, pd.DataFrame]:
    diff = create_model_difference_vs_oracle(summary, paths)
    best = create_best_non_oracle_model_by_scenario(summary, paths)
    xgb_audit = create_xgb_oracle_sanity_audit_full(summary, paths)
    interpretation = create_scenario_interpretation_table(summary, best, xgb_audit, paths)
    sanity = create_full_run_sanity_checks(perf, xgb_audit, inputs["safety"], paths)
    return {
        "model_difference_vs_oracle_by_scenario": diff,
        "best_non_oracle_model_by_scenario": best,
        "xgb_oracle_sanity_audit_full": xgb_audit,
        "scenario_interpretation_table": interpretation,
        "full_run_sanity_checks": sanity,
    }


def grouped_bar_plot(
    summary: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if summary.empty or value_col not in summary.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    plot_df = summary.pivot(index="scenario_id", columns="model", values=value_col)
    scenarios = plot_df.index.tolist()
    models = [m for m in MODELS if m in plot_df.columns]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(models), 1)
    colors = ["#2F6B8E", "#D95F59", "#4C8F5B", "#8A6FB0"]

    fig_width = max(11, len(scenarios) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    for i, model in enumerate(models):
        values = plot_df[model].to_numpy(dtype=float)
        ax.bar(x - 0.4 + width / 2 + i * width, values, width, label=model, color=colors[i % len(colors)])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_figures(summary: pd.DataFrame, perf: pd.DataFrame, paths: Dict[str, Path]) -> None:
    grouped_bar_plot(
        summary,
        "cindex_mean",
        "Mean Harrell C-index",
        "Cause-specific C-index by scenario and model",
        paths["figures"] / "cindex_by_scenario_model.png",
    )
    grouped_bar_plot(
        summary,
        "spearman_true_lp_mean",
        "Mean Spearman correlation",
        "Risk-score correlation with true_lp_carehome",
        paths["figures"] / "spearman_true_lp_by_scenario_model.png",
    )
    grouped_bar_plot(
        summary,
        "risk5_mae_vs_true_risk_mean",
        "Mean absolute error",
        "5-year risk MAE versus exported true risk",
        paths["figures"] / "risk5_mae_by_scenario_model.png",
    )

    if perf.empty:
        failure_rate = pd.Series(dtype=float)
    else:
        failure_rate = perf.groupby("model")["failed"].apply(lambda s: s.astype(bool).mean())
        failure_rate = failure_rate.reindex([m for m in MODELS if m in failure_rate.index])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if failure_rate.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(failure_rate.index))
        ax.bar(x, failure_rate.to_numpy(dtype=float), color="#6E7F80")
        ax.set_ylim(0, max(1.0, float(failure_rate.max()) * 1.15))
        ax.set_ylabel("Failure rate")
        ax.set_title("Failure rate by model")
        ax.set_xticks(x)
        ax.set_xticklabels(failure_rate.index.tolist(), rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "failure_rate_by_model.png", dpi=180)
    plt.close(fig)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a compact markdown table without depending on tabulate."""
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))

    headers = [str(c) for c in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def write_readme(
    config: Config,
    paths: Dict[str, Path],
    predictors: Dict[str, List[str]],
    perf: pd.DataFrame,
    summary: pd.DataFrame,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    n_fail = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    n_rows = int(perf.shape[0])
    lines = [
        "# Step C2 Fully Synthetic Model Comparison",
        "",
        f"Run started: {started_at.isoformat(timespec='seconds')}",
        f"Run finished: {finished_at.isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "This pipeline uses only `fully_synthetic_stepC_v1/`. It does not use real SLAM data, raw files, Step B files, death spreadsheets, WMH spreadsheets, or the generator notebook as an input data source.",
        "",
        "Primary endpoint: time to care-home entry / institutionalisation.",
        "",
        "Outcome interpretation: `status=1` is the event of interest; `status=0` and `status=2` are treated as censored in cause-specific survival models. `death_after_carehome` is not treated as a competing event for this endpoint.",
        "",
        "## Configuration",
        "",
        f"- `DEBUG_MODE = {config.debug_mode}`",
        f"- `MAX_REPS_PER_SCENARIO = {config.max_reps_per_scenario}`",
        f"- `TEST_SIZE = {config.test_size}`",
        f"- `RANDOM_SEED = {config.random_seed}`",
        f"- `DATA_DIR = {config.data_dir}`",
        f"- `OUT_DIR = {config.out_dir}`",
        "",
        "To switch to a full run, edit the script constants to `DEBUG_MODE = False` and `MAX_REPS_PER_SCENARIO = None`, or run:",
        "",
        "```bash",
        "python stepC2_fully_synthetic_model_comparison.py --full",
        "```",
        "",
        "## Models",
        "",
        "- `oracle_true_lp_not_a_model`: benchmark using exported simulation truth, not a fitted model.",
        "- `cox_dgm_features`: cause-specific Cox model using the saved DGM fallback predictor list.",
        "- `penalised_cox_all_safe_predictors`: penalised cause-specific Cox model using strict safe baseline predictors.",
        "- `xgb_survival_cox_strict`: leakage-guarded XGBoost `survival:cox` model using strict safe baseline predictors and returning risk scores only.",
        "",
        "No DeepSurv, DeepHit, random survival forest, TabPFN, Fine-Gray, or neural-network models are included in this first Step C2 baseline.",
        "",
        "## Predictor Audits",
        "",
        f"- All-safe predictor count: {len(predictors.get('all_safe_predictors', []))}",
        f"- DGM fallback predictor count: {len(predictors.get('dgm_predictors', []))}",
        "- `tables/predictor_audit_all_safe.csv` records exact inclusion and exclusion decisions.",
        "- `tables/predictor_audit_dgm_features.csv` records exact DGM fallback features and any missing intended features.",
        "",
        "The DGM definition table in this export describes scenario-level DGM settings but does not enumerate exact predictor coefficients. Therefore the Cox DGM model uses a conservative clinical fallback list and records that decision explicitly.",
        "",
        "## Outputs",
        "",
        "- `tables/data_file_audit.csv`",
        "- `tables/predictor_audit_all_safe.csv`",
        "- `tables/predictor_audit_dgm_features.csv`",
        "- `tables/replicate_model_performance.csv`",
        "- `tables/scenario_model_summary_mean_sd_ci.csv`",
        "- `tables/model_ranking_by_scenario.csv`",
        "- `tables/pivot_cindex_mean.csv`",
        "- `tables/pivot_spearman_true_lp_mean.csv`",
        "- `tables/pivot_risk5_mae_mean.csv`",
        "- `tables/failure_log.csv`",
        "- `figures/cindex_by_scenario_model.png`",
        "- `figures/spearman_true_lp_by_scenario_model.png`",
        "- `figures/risk5_mae_by_scenario_model.png`",
        "- `figures/failure_rate_by_model.png`",
        "",
        "Debug mode also saves small sampled predictions under `predictions/debug_prediction_samples.csv`; full per-person predictions are not saved by default.",
        "",
        "## Run Summary",
        "",
        f"- Replicate-model rows: {n_rows}",
        f"- Failed replicate-model rows: {n_fail}",
        "",
    ]

    if not summary.empty:
        display_cols = [
            "scenario_id",
            "model",
            "n_successful_reps",
            "n_failed_reps",
            "failure_rate",
            "cindex_mean",
            "spearman_true_lp_mean",
            "risk5_mae_vs_true_risk_mean",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]
        lines.extend(
            [
                "## Compact Summary",
                "",
                dataframe_to_markdown_table(summary[display_cols]),
                "",
            ]
        )

    lines.extend(
        [
            "## Scientific Interpretation Notes",
            "",
            "- `oracle_true_lp_not_a_model` is an upper benchmark under the simulated DGM, not a real deployable model.",
            "- `cox_dgm_features` should perform close to oracle in S0/S1/S2 if the pipeline and fallback DGM feature list align well with the simulated DGM.",
            "- `penalised_cox_all_safe_predictors` tests whether high-dimensional synthetic predictors add noise or recover extra signal.",
            "- `xgb_survival_cox_strict` should not unrealistically exceed oracle. If it does, leakage or evaluation artefacts must be audited.",
            "- S3 tests nonlinear and interaction effects.",
            "- S4 tests non-proportional hazards.",
            "- S5 tests MAR missingness.",
            "- S6 tests high-dimensional sparse MRI signal.",
            "- S7 tests stronger death competing risk.",
            "",
            "## Dependency Notes",
            "",
            "If dependencies are missing, install them with:",
            "",
            "```bash",
            "pip install lifelines xgboost",
            "```",
            "",
        ]
    )

    (paths["root"] / "README_stepC2_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def write_full_readme(
    config: Config,
    paths: Dict[str, Path],
    predictors: Dict[str, List[str]],
    perf: pd.DataFrame,
    summary: pd.DataFrame,
    additional: Dict[str, pd.DataFrame],
    started_at: datetime,
    finished_at: datetime,
) -> None:
    perf = dedupe_performance(perf)
    n_rows = int(perf.shape[0])
    n_fail = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    best = additional.get("best_non_oracle_model_by_scenario", pd.DataFrame())
    diff = additional.get("model_difference_vs_oracle_by_scenario", pd.DataFrame())
    xgb_audit = additional.get("xgb_oracle_sanity_audit_full", pd.DataFrame())
    sanity = additional.get("full_run_sanity_checks", pd.DataFrame())
    interpretation = additional.get("scenario_interpretation_table", pd.DataFrame())

    def model_gap(scenario: str, model: str, metric: str = "cindex_gap_vs_oracle") -> float:
        rows = diff.loc[(diff["scenario_id"] == scenario) & (diff["model"] == model)] if not diff.empty else pd.DataFrame()
        if rows.empty or metric not in rows.columns:
            return np.nan
        return float(rows.iloc[0][metric])

    def best_model(scenario: str) -> str:
        rows = best.loc[best["scenario_id"] == scenario] if not best.empty else pd.DataFrame()
        return str(rows.iloc[0]["best_model_by_cindex"]) if not rows.empty else "NA"

    linear_gaps = {
        scenario: model_gap(scenario, MODEL_COX_DGM)
        for scenario in ["S0_linear_PH_inst30", "S1_linear_PH_inst15", "S2_linear_PH_inst45"]
    }
    linear_close = all(np.isfinite(gap) and gap >= -0.03 for gap in linear_gaps.values())
    s6_best = best_model("S6_highdim_sparseMRI_inst30")
    s3_best = best_model("S3_nonlinear_interaction_inst30")
    s5_best_c = (
        float(best.loc[best["scenario_id"] == "S5_MAR_missingness_inst30", "best_cindex_mean"].iloc[0])
        if not best.empty and (best["scenario_id"] == "S5_MAR_missingness_inst30").any()
        else np.nan
    )
    s0_best_c = (
        float(best.loc[best["scenario_id"] == "S0_linear_PH_inst30", "best_cindex_mean"].iloc[0])
        if not best.empty and (best["scenario_id"] == "S0_linear_PH_inst30").any()
        else np.nan
    )
    s7_best = best_model("S7_strong_death_competing_inst30")
    any_xgb_flag = (
        bool(xgb_audit["needs_leakage_audit"].astype(bool).any())
        if not xgb_audit.empty and "needs_leakage_audit" in xgb_audit.columns
        else True
    )
    full_run_passed = False
    if not sanity.empty:
        rows = sanity.loc[sanity["check_name"] == "full_run_passed"]
        if not rows.empty:
            full_run_passed = bool(rows.iloc[0]["observed"])

    if np.isfinite(s5_best_c) and np.isfinite(s0_best_c):
        s5_delta_text = f"S5 best fitted-model C-index was {fmt_float(s5_best_c)} versus S0 {fmt_float(s0_best_c)} (delta {fmt_float(s5_best_c - s0_best_c)})."
    else:
        s5_delta_text = "S5 versus S0 fitted-model C-index comparison was unavailable."

    linear_gap_text = ", ".join([f"{k}: {fmt_float(v)}" for k, v in linear_gaps.items()])
    lines = [
        "# Step C2 Fully Synthetic Model Comparison: Full Run",
        "",
        f"Run started: {started_at.isoformat(timespec='seconds')}",
        f"Run finished: {finished_at.isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "This analysis uses only `fully_synthetic_stepC_v1/`.",
        "",
        "No real SLAM data, Step B data, raw CSVs, death spreadsheets, WMH spreadsheets, or files containing real-project identifiers were used. The generator notebook under `code/` is treated as documentation only, not as an input data source.",
        "",
        "## Configuration",
        "",
        f"- `DEBUG_MODE = {config.debug_mode}`",
        f"- `MAX_REPS_PER_SCENARIO = {config.max_reps_per_scenario}`",
        f"- `TEST_SIZE = {config.test_size:.2f}`",
        f"- `RANDOM_SEED = {config.random_seed}`",
        "",
        "## Expected vs Observed Run Size",
        "",
        f"- Expected replicate-model rows: {EXPECTED_FULL_REPLICATE_MODEL_ROWS}",
        f"- Observed replicate-model rows: {n_rows}",
        f"- Failed replicate-model rows: {n_fail}",
        f"- Full-run sanity gate passed: {full_run_passed}",
        "",
        "## Models",
        "",
        "- `oracle_true_lp_not_a_model`",
        "- `cox_dgm_features`",
        "- `penalised_cox_all_safe_predictors`",
        "- `xgb_survival_cox_strict`",
        "",
        "No DeepSurv, DeepHit, random survival forest, TabPFN, Fine-Gray, or neural-network models were added in this Step C2 full run.",
        "",
        "## Main Findings",
        "",
        f"- Cox DGM closeness in S0/S1/S2: {'stayed within the pre-specified 0.03 C-index gap tolerance' if linear_close else 'did not stay within the 0.03 C-index gap tolerance in all three linear PH scenarios'}. Gaps versus oracle were {linear_gap_text}.",
        f"- S6 high-dimensional sparse MRI: the best fitted model by C-index was `{s6_best}`. Penalised Cox and XGBoost competitiveness should be read from `best_non_oracle_model_by_scenario.csv` and `model_difference_vs_oracle_by_scenario.csv`.",
        f"- XGBoost oracle sanity: {'at least one XGBoost sanity flag was raised and needs audit' if any_xgb_flag else 'no XGBoost scenario exceeded oracle by >0.02 or had low true-LP alignment under the configured checks'}.",
        f"- S7 stronger competing death: the best fitted model by C-index was `{s7_best}`; these are still cause-specific results where death-before-care-home is treated as censoring.",
        f"- S5 missingness: {s5_delta_text}",
        f"- S3 nonlinear/interaction: the best fitted model by C-index was `{s3_best}`; this indicates whether the nonlinear XGBoost baseline was favoured in the actual full-run results.",
        "",
        "## Key Tables",
        "",
        "- `tables/scenario_model_summary_mean_sd_ci.csv`",
        "- `tables/model_difference_vs_oracle_by_scenario.csv`",
        "- `tables/best_non_oracle_model_by_scenario.csv`",
        "- `tables/xgb_oracle_sanity_audit_full.csv`",
        "- `tables/scenario_interpretation_table.csv`",
        "- `tables/full_run_sanity_checks.csv`",
        "",
        "## Compact Scenario Interpretation",
        "",
        dataframe_to_markdown_table(
            interpretation[
                [
                    "scenario_id",
                    "observed_best_non_oracle_model",
                    "short_interpretation",
                ]
            ]
        )
        if not interpretation.empty
        else "_No scenario interpretation rows were generated._",
        "",
        "## Limitations",
        "",
        "- Fully synthetic data are for method testing, not clinical inference.",
        "- Cox DGM uses a fallback DGM-relevant predictor list unless exact DGM coefficients are exported.",
        "- This first Step C2 uses cause-specific modelling only; Fine-Gray and CIF evaluation are not included.",
        "- XGBoost outputs risk scores only, not calibrated 5-year absolute risk.",
        "",
        "## Next Recommended Step",
        "",
        "If the full run passes, proceed to Step C3: add competing-risk-specific evaluation / Fine-Gray or CIF reconstruction, or prepare the current summary for Daniel/Ash.",
        "",
        "## Predictor Lists",
        "",
        f"- All-safe predictor count: {len(predictors.get('all_safe_predictors', []))}",
        f"- DGM fallback predictor count: {len(predictors.get('dgm_predictors', []))}",
        "- Exact lists are saved in `tables/predictor_list_all_safe.csv` and `tables/predictor_list_dgm_features.csv`.",
        "",
    ]
    (paths["root"] / "README_stepC2_model_comparison_FULL.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    started_at = datetime.now()
    start_perf = time.perf_counter()
    config = parse_args()
    np.random.seed(config.random_seed)
    paths = make_output_dirs(config)

    print("Step C2 fully synthetic model comparison")
    print(f"DATA_DIR={config.data_dir}")
    print(f"OUT_DIR={config.out_dir}")
    print(f"DEBUG_MODE={config.debug_mode}")
    print(f"MAX_REPS_PER_SCENARIO={config.max_reps_per_scenario}")

    inputs = load_and_audit_inputs(config, paths)
    first_columns = json.loads(inputs["data_file_audit"].iloc[0]["columns_json"])
    predictors = build_predictor_lists(inputs["feature_dictionary"], first_columns, paths)

    existing_perf = load_existing_performance(paths)
    results: List[Dict[str, Any]] = existing_perf.to_dict("records") if not existing_perf.empty else []
    existing_success_keys = success_keys_from_performance(existing_perf)
    if existing_success_keys:
        print(f"Existing successful model fits available for resume: {len(existing_success_keys)}")

    prediction_samples: List[pd.DataFrame] = []
    for file_path in inputs["scenario_files"]:
        scenario_results = process_scenario_file(
            file_path,
            predictors,
            config,
            prediction_samples,
            results,
            existing_success_keys,
            paths,
        )
        print(f"Finished {file_path.name}: new model rows this pass={len(scenario_results)}")

    perf = save_performance_checkpoint(results, paths)
    save_failure_log(perf, paths)
    summary = aggregate_results(perf, paths)
    additional = create_additional_full_run_tables(summary, perf, inputs, paths)
    make_figures(summary, perf, paths)

    if config.debug_mode and prediction_samples:
        samples = pd.concat([s for s in prediction_samples if not s.empty], ignore_index=True)
        samples.to_csv(paths["predictions"] / "debug_prediction_samples.csv", index=False)

    finished_at = datetime.now()
    write_readme(config, paths, predictors, perf, summary, started_at, finished_at)
    if not config.debug_mode:
        write_full_readme(config, paths, predictors, perf, summary, additional, started_at, finished_at)
    total_runtime = time.perf_counter() - start_perf
    failure_count = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    print("Done.")
    print(f"Total runtime seconds: {total_runtime:.1f}")
    print(f"replicate_model_performance row count: {len(perf)}")
    print(f"failure count: {failure_count}")
    print("Replicate performance:", paths["tables"] / "replicate_model_performance.csv")
    print("Scenario summary:", paths["tables"] / "scenario_model_summary_mean_sd_ci.csv")
    print(
        "Model difference vs oracle:",
        paths["tables"] / "model_difference_vs_oracle_by_scenario.csv",
    )
    print(
        "Best non-oracle model:",
        paths["tables"] / "best_non_oracle_model_by_scenario.csv",
    )
    print(
        "XGB oracle sanity audit:",
        paths["tables"] / "xgb_oracle_sanity_audit_full.csv",
    )
    print("README:", paths["root"] / "README_stepC2_model_comparison.md")
    if not config.debug_mode:
        print("Full README:", paths["root"] / "README_stepC2_model_comparison_FULL.md")


if __name__ == "__main__":
    main()
