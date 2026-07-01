#!/usr/bin/env python3
"""Step C3 fully synthetic competing-risk 5-year risk evaluation.

Inputs
------
Expected local input directory: ``fully_synthetic_stepC_v1/``. The script can
also reuse C2 predictor-list outputs from
``fully_synthetic_stepC2_model_comparison/`` when available.

Outputs
-------
Writes aggregate competing-risk performance tables, calibration summaries,
audits, plots, and optional debug prediction samples to
``fully_synthetic_stepC3_competing_risk_evaluation/``.

Data governance
---------------
This script uses exportable fully synthetic Step C files only. It does not use
real SLAM data, internal Step B semi-synthetic data, raw spreadsheets, or real
patient identifiers.

Run
---
Debug mode: ``python src/stepC3_competing_risk/stepC3_competing_risk_evaluation.py --debug``.
Full mode: ``python src/stepC3_competing_risk/stepC3_competing_risk_evaluation.py --full``.

Models
------
Oracle synthetic risk benchmark, Aalen-Johansen null, DGM-feature
cause-specific Cox CIF, and strict all-safe penalised cause-specific Cox CIF.
Fine-Gray is optional and requires R package ``cmprsk``.

Notes
-----
For C3, status=1 is care-home entry and status=2 is death before care home as
the competing event. The default horizon is 5 years.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("fully_synthetic_stepC_v1")
C2_DIR = Path("fully_synthetic_stepC2_model_comparison")
OUT_DIR = Path("fully_synthetic_stepC3_competing_risk_evaluation")

DEBUG_MODE = True
MAX_REPS_PER_SCENARIO: Optional[int] = 2
TEST_SIZE = 0.30
RANDOM_SEED = 42
HORIZON_YEARS = 5.0
DEBUG_PRED_SAMPLE_N = 100

DURATION_COL = "duration_years"
STATUS_COL = "status"
EVENT1_STATUS = 1
EVENT2_STATUS = 2
TRUE_RISK_COL = "true_risk_carehome_5y_observable_approx"
TRUE_LP_COL = "true_lp_carehome"

MODEL_ORACLE = "oracle_true_risk_not_a_model"
MODEL_AJ = "nonparametric_aj_null"
MODEL_CS_DGM = "cs_cox_dgm_cif"
MODEL_CS_SAFE = "cs_penalised_cox_all_safe_cif"
MODEL_FINEGRAY = "finegray_dgm_optional"
MANDATORY_MODELS = [MODEL_ORACLE, MODEL_AJ, MODEL_CS_DGM, MODEL_CS_SAFE]

REQUIRED_COLUMNS = [
    "scenario_id",
    "replicate_id",
    "synthetic_id",
    DURATION_COL,
    STATUS_COL,
    "event_carehome",
    "event_death_before_carehome",
    "event_free_or_censored",
    TRUE_LP_COL,
    TRUE_RISK_COL,
]

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
    "other",
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


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    c2_dir: Path = C2_DIR
    out_dir: Path = OUT_DIR
    debug_mode: bool = DEBUG_MODE
    max_reps_per_scenario: Optional[int] = MAX_REPS_PER_SCENARIO
    test_size: float = TEST_SIZE
    random_seed: int = RANDOM_SEED
    horizon_years: float = HORIZON_YEARS
    rerun_failed: bool = False


def ensure_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
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
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        missing.append("scipy")
        pearsonr = None
        spearmanr = None
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        missing.append("scikit-learn")
        ColumnTransformer = None
        SimpleImputer = None
        LogisticRegression = None
        roc_auc_score = None
        train_test_split = None
        Pipeline = None
        OneHotEncoder = None
        StandardScaler = None
    if missing:
        unique_missing = sorted(set(missing))
        print("Missing required Python package(s): " + ", ".join(unique_missing), file=sys.stderr)
        print("pip install " + " ".join(unique_missing), file=sys.stderr)
        sys.exit(1)
    return (
        plt,
        CoxPHFitter,
        concordance_index,
        pearsonr,
        spearmanr,
        ColumnTransformer,
        SimpleImputer,
        LogisticRegression,
        roc_auc_score,
        train_test_split,
        Pipeline,
        OneHotEncoder,
        StandardScaler,
    )


(
    plt,
    CoxPHFitter,
    concordance_index,
    pearsonr,
    spearmanr,
    ColumnTransformer,
    SimpleImputer,
    LogisticRegression,
    roc_auc_score,
    train_test_split,
    Pipeline,
    OneHotEncoder,
    StandardScaler,
) = ensure_dependencies()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--c2-dir", type=Path, default=C2_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-reps", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--horizon", type=float, default=HORIZON_YEARS)
    parser.add_argument("--rerun-failed", action="store_true")
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
        c2_dir=args.c2_dir,
        out_dir=args.out_dir,
        debug_mode=debug_mode,
        max_reps_per_scenario=max_reps,
        test_size=args.test_size,
        random_seed=args.seed,
        horizon_years=args.horizon,
        rerun_failed=args.rerun_failed,
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


def bool_series_all_true(series: pd.Series) -> bool:
    values = series.map(lambda x: str(x).strip().lower() in {"true", "1", "yes"})
    return bool(values.all())


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def contains_forbidden_name(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in FORBIDDEN_NAME_SUBSTRINGS)


def find_true_risk_column(columns: Sequence[str]) -> str:
    if TRUE_RISK_COL in columns:
        return TRUE_RISK_COL
    candidates = [
        c
        for c in columns
        if "true" in c.lower()
        and "risk" in c.lower()
        and "carehome" in c.lower()
        and ("5y" in c.lower() or "5" in c.lower())
    ]
    if candidates:
        return candidates[0]
    raise RuntimeError("No true 5-year care-home risk column found; C3 cannot compare absolute risk.")


def audit_scenario_file(path: Path) -> Dict[str, Any]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    risk_col = find_true_risk_column(columns)
    required = [c if c != TRUE_RISK_COL else risk_col for c in REQUIRED_COLUMNS]
    missing_required = [c for c in required if c not in columns]
    if missing_required:
        return {
            "file_name": path.name,
            "scenario_id": "",
            "n_rows": np.nan,
            "n_columns": len(columns),
            "n_replicate_ids": np.nan,
            "status_distribution_json": "{}",
            "carehome_event_rate": np.nan,
            "death_before_carehome_rate": np.nan,
            "event_free_or_censored_rate": np.nan,
            "true_risk_column": risk_col,
            "columns_json": json.dumps(columns),
            "missing_required_columns_json": json.dumps(missing_required),
        }

    n_rows = 0
    scenario_ids: set[str] = set()
    replicate_ids: set[Any] = set()
    status_counts: Dict[str, int] = {}
    usecols = ["scenario_id", "replicate_id", STATUS_COL]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        n_rows += len(chunk)
        scenario_ids.update(chunk["scenario_id"].dropna().astype(str).unique().tolist())
        replicate_ids.update(chunk["replicate_id"].dropna().unique().tolist())
        counts = chunk[STATUS_COL].value_counts(dropna=False).to_dict()
        for key, value in counts.items():
            status_counts[str(int(key)) if pd.notna(key) else "nan"] = status_counts.get(
                str(int(key)) if pd.notna(key) else "nan", 0
            ) + int(value)
    carehome = status_counts.get("1", 0)
    death = status_counts.get("2", 0)
    censored = status_counts.get("0", 0)
    denom = max(n_rows, 1)
    return {
        "file_name": path.name,
        "scenario_id": ";".join(sorted(scenario_ids)),
        "n_rows": n_rows,
        "n_columns": len(columns),
        "n_replicate_ids": len(replicate_ids),
        "status_distribution_json": json.dumps(status_counts, sort_keys=True),
        "carehome_event_rate": carehome / denom,
        "death_before_carehome_rate": death / denom,
        "event_free_or_censored_rate": censored / denom,
        "true_risk_column": risk_col,
        "columns_json": json.dumps(columns),
        "missing_required_columns_json": json.dumps(missing_required),
    }


def load_and_audit_inputs(config: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    if not config.data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {config.data_dir}")
    safety = read_csv_required(config.data_dir / "audit" / "export_safety_audit.csv")
    if "safe_to_export_column_names" not in safety.columns:
        raise ValueError("export_safety_audit.csv lacks safe_to_export_column_names")
    if not bool_series_all_true(safety["safe_to_export_column_names"]):
        raise RuntimeError("Export safety audit failed; refusing to run C3.")

    feature_dictionary = read_csv_required(config.data_dir / "tables" / "feature_dictionary.csv")
    dgm_definition = read_csv_required(config.data_dir / "tables" / "dgm_definition_table.csv")
    scenario_summary = read_csv_required(config.data_dir / "tables" / "scenario_summary.csv")
    repetition_summary = read_csv_required(config.data_dir / "tables" / "repetition_summary.csv")

    scenario_files = sorted((config.data_dir / "scenario_datasets").glob("*.csv.gz"))
    if not scenario_files:
        raise FileNotFoundError("No scenario files found.")
    print(f"Found {len(scenario_files)} scenario files.", flush=True)

    audit_rows = []
    for file_path in scenario_files:
        print(f"Auditing {file_path.name} ...", flush=True)
        audit_rows.append(audit_scenario_file(file_path))
    data_file_audit = pd.DataFrame(audit_rows)
    data_file_audit.to_csv(paths["tables"] / "data_file_audit.csv", index=False)
    if data_file_audit["missing_required_columns_json"].ne("[]").any():
        bad = data_file_audit.loc[
            data_file_audit["missing_required_columns_json"].ne("[]"),
            ["file_name", "missing_required_columns_json"],
        ]
        raise RuntimeError("Missing required columns:\n" + bad.to_string(index=False))
    print("Input audit saved:", paths["tables"] / "data_file_audit.csv", flush=True)
    return {
        "safety": safety,
        "feature_dictionary": feature_dictionary,
        "dgm_definition": dgm_definition,
        "scenario_summary": scenario_summary,
        "repetition_summary": repetition_summary,
        "scenario_files": scenario_files,
        "data_file_audit": data_file_audit,
        "true_risk_column": data_file_audit["true_risk_column"].iloc[0],
    }


def read_predictor_list(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return []
    col = "predictor" if "predictor" in df.columns else df.columns[0]
    return df[col].dropna().astype(str).tolist()


def derived_block_for_column(column: str, block: str) -> str:
    if column in KNOWN_COMORBIDITY_COLUMNS:
        return "comorbidity"
    return block


def rebuild_predictors_from_feature_dictionary(
    feature_dictionary: pd.DataFrame,
    all_columns: Sequence[str],
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
    colset = set(all_columns)
    safe_rows: List[Dict[str, Any]] = []
    safe_predictors: List[str] = []
    for _, row in feature_dictionary.iterrows():
        col = str(row["column"])
        block = str(row["block"])
        role = str(row["role"])
        derived_block = derived_block_for_column(col, block)
        present = col in colset
        forbidden = contains_forbidden_name(col)
        reasons: List[str] = []
        included = False
        if not present:
            reasons.append("not_present")
        if block in EXCLUDED_BLOCKS or role in EXCLUDED_BLOCKS:
            reasons.append("excluded_block_or_role")
        if forbidden:
            reasons.append("forbidden_name")
        if col == "sex" and "sex_Female" in colset:
            reasons.append("excluded_redundant_with_sex_Female")
        if not reasons and derived_block in ALLOWED_BLOCKS:
            included = True
            reasons.append("included_allowed_block")
        elif not reasons:
            reasons.append("block_not_allowed")
        if included:
            safe_predictors.append(col)
        safe_rows.append(
            {
                "column": col,
                "source": "rebuilt_from_feature_dictionary",
                "present_in_scenario_files": present,
                "feature_dictionary_block": block,
                "feature_dictionary_role": role,
                "derived_predictor_block": derived_block,
                "included_in_all_safe_predictors": included,
                "contains_forbidden_name_substring": forbidden,
                "reason": ";".join(reasons),
            }
        )

    dgm_rows: List[Dict[str, Any]] = []
    dgm_predictors: List[str] = []
    safe_set = set(safe_predictors)
    for col in CLINICAL_DGM_INTENDED_FEATURES:
        present = col in colset
        included = present and col in safe_set
        if included:
            dgm_predictors.append(col)
        dgm_rows.append(
            {
                "intended_feature": col,
                "source": "clinical_fallback",
                "present_in_scenario_files": present,
                "included_in_dgm_features": included,
                "reason": "included" if included else ("missing" if not present else "not_in_safe_list"),
            }
        )
    return safe_predictors, dgm_predictors, pd.DataFrame(safe_rows), pd.DataFrame(dgm_rows)


def build_predictor_lists(
    config: Config,
    feature_dictionary: pd.DataFrame,
    all_columns: Sequence[str],
    paths: Dict[str, Path],
) -> Dict[str, List[str]]:
    c2_safe = read_predictor_list(config.c2_dir / "tables" / "predictor_list_all_safe.csv")
    c2_dgm = read_predictor_list(config.c2_dir / "tables" / "predictor_list_dgm_features.csv")
    colset = set(all_columns)

    if c2_safe is not None and c2_dgm is not None:
        safe_predictors = [c for c in c2_safe if c in colset]
        dgm_predictors = [c for c in c2_dgm if c in colset]
        safe_rows = []
        for col in c2_safe:
            present = col in colset
            forbidden = contains_forbidden_name(col)
            safe_rows.append(
                {
                    "column": col,
                    "source": "C2_predictor_list_all_safe.csv",
                    "present_in_scenario_files": present,
                    "included_in_all_safe_predictors": present and not forbidden,
                    "contains_forbidden_name_substring": forbidden,
                    "reason": "used_from_C2_list" if present and not forbidden else "missing_or_forbidden",
                }
            )
        dgm_rows = []
        for col in c2_dgm:
            present = col in colset
            forbidden = contains_forbidden_name(col)
            dgm_rows.append(
                {
                    "intended_feature": col,
                    "source": "C2_predictor_list_dgm_features.csv",
                    "present_in_scenario_files": present,
                    "included_in_dgm_features": present and not forbidden,
                    "reason": "used_from_C2_list" if present and not forbidden else "missing_or_forbidden",
                }
            )
        predictor_audit_all_safe = pd.DataFrame(safe_rows)
        predictor_audit_dgm = pd.DataFrame(dgm_rows)
    else:
        safe_predictors, dgm_predictors, predictor_audit_all_safe, predictor_audit_dgm = (
            rebuild_predictors_from_feature_dictionary(feature_dictionary, all_columns)
        )

    suspicious = [c for c in safe_predictors + dgm_predictors if contains_forbidden_name(c)]
    if suspicious:
        raise RuntimeError("Suspicious predictors in C3 list: " + ", ".join(sorted(set(suspicious))))
    if not safe_predictors:
        raise RuntimeError("No all-safe predictors available for C3.")
    if not dgm_predictors:
        raise RuntimeError("No DGM predictors available for C3.")

    predictor_audit_all_safe.to_csv(paths["tables"] / "predictor_audit_all_safe.csv", index=False)
    predictor_audit_dgm.to_csv(paths["tables"] / "predictor_audit_dgm_features.csv", index=False)
    pd.DataFrame({"predictor": safe_predictors}).to_csv(
        paths["tables"] / "predictor_list_all_safe_C3.csv", index=False
    )
    pd.DataFrame({"predictor": dgm_predictors}).to_csv(
        paths["tables"] / "predictor_list_dgm_features_C3.csv", index=False
    )
    print(f"C3 all-safe predictors: {len(safe_predictors)}", flush=True)
    print(f"C3 DGM predictors: {len(dgm_predictors)}", flush=True)
    return {"all_safe_predictors": safe_predictors, "dgm_predictors": dgm_predictors}


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


def event_indicator(status: pd.Series, event_status: int) -> np.ndarray:
    return (status.astype(int).to_numpy() == event_status).astype(int)


def compute_aalen_johansen_cif1_at_horizon(
    time_values: Sequence[float],
    status_values: Sequence[int],
    horizon: float,
) -> Tuple[float, float, float]:
    df = pd.DataFrame(
        {
            "time": np.asarray(time_values, dtype=float),
            "status": np.asarray(status_values, dtype=int),
        }
    )
    df = df.loc[np.isfinite(df["time"]) & (df["time"] >= 0)].copy()
    if df.empty:
        return np.nan, np.nan, np.nan
    event_times = np.sort(df.loc[(df["status"].isin([1, 2])) & (df["time"] <= horizon), "time"].unique())
    survival = 1.0
    cif1 = 0.0
    cif2 = 0.0
    for t in event_times:
        at_risk = int((df["time"] >= t).sum())
        if at_risk <= 0:
            continue
        d1 = int(((df["time"] == t) & (df["status"] == 1)).sum())
        d2 = int(((df["time"] == t) & (df["status"] == 2)).sum())
        h1 = d1 / at_risk
        h2 = d2 / at_risk
        cif1 += survival * h1
        cif2 += survival * h2
        survival *= max(0.0, 1.0 - h1 - h2)
    return float(np.clip(cif1, 0.0, 1.0)), float(np.clip(cif2, 0.0, 1.0)), float(np.clip(survival, 0.0, 1.0))


def one_hot_encoder() -> Any:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def get_feature_names(preprocessor: Any, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    names: List[str] = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        try:
            encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
            names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
        except Exception:
            names.extend([f"categorical_{i}" for i in range(len(categorical_cols))])
    return names


def build_model_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    raw_predictors = [c for c in predictors if c in train_df.columns and c in test_df.columns]
    if not raw_predictors:
        raise ValueError("No requested predictors are present.")
    X_train_raw = train_df[raw_predictors].copy()
    X_test_raw = test_df[raw_predictors].copy()
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in raw_predictors:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]) or pd.api.types.is_bool_dtype(X_train_raw[col]):
            if X_train_raw[col].notna().sum() > 0:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    transformers = []
    if numeric_cols:
        steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No usable predictors after preprocessing.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)
    names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    if X_train.shape[1] != len(names):
        names = [f"x{i}" for i in range(X_train.shape[1])]
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nanstd(X_train, axis=0)
    keep = std > 1e-12
    if not keep.any():
        raise ValueError("All preprocessed predictors are constant.")
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    names = [name for name, k in zip(names, keep) if k]
    return (
        pd.DataFrame(X_train, columns=names, index=train_df.index),
        pd.DataFrame(X_test, columns=names, index=test_df.index),
        names,
    )


def fit_cause_specific_cox(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    event_status: int,
    penalizer_grid: Sequence[float],
) -> Any:
    cox_train = X_train.copy()
    cox_train[DURATION_COL] = train_df[DURATION_COL].astype(float).to_numpy()
    cox_train["event"] = event_indicator(train_df[STATUS_COL], event_status)
    last_error: Optional[Exception] = None
    for penalizer in penalizer_grid:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(cox_train, duration_col=DURATION_COL, event_col="event", show_progress=False)
            return cph
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Cause-specific Cox failed for event {event_status}: {last_error}")


def baseline_cumulative_values_at_times(cph: Any, times: np.ndarray) -> np.ndarray:
    baseline = cph.baseline_cumulative_hazard_.iloc[:, 0]
    base_times = pd.to_numeric(pd.Series(baseline.index), errors="coerce").to_numpy(dtype=float)
    base_values = baseline.to_numpy(dtype=float)
    valid = np.isfinite(base_times) & np.isfinite(base_values)
    base_times = base_times[valid]
    base_values = base_values[valid]
    if len(base_times) == 0:
        return np.zeros_like(times, dtype=float)
    order = np.argsort(base_times)
    base_times = base_times[order]
    base_values = base_values[order]
    positions = np.searchsorted(base_times, times, side="right") - 1
    out = np.zeros_like(times, dtype=float)
    mask = positions >= 0
    out[mask] = base_values[positions[mask]]
    return np.maximum.accumulate(out)


def predict_cif_from_two_cause_specific_cox(
    cox_event1: Any,
    cox_event2: Any,
    X_test: pd.DataFrame,
    horizon: float,
) -> Tuple[np.ndarray, List[str]]:
    warnings_out: List[str] = []
    t1 = pd.to_numeric(
        pd.Series(cox_event1.baseline_cumulative_hazard_.index), errors="coerce"
    ).to_numpy(dtype=float)
    t2 = pd.to_numeric(
        pd.Series(cox_event2.baseline_cumulative_hazard_.index), errors="coerce"
    ).to_numpy(dtype=float)
    times = np.union1d(t1[np.isfinite(t1)], t2[np.isfinite(t2)])
    times = np.sort(times[(times > 0) & (times <= horizon)])
    n = len(X_test)
    if len(times) == 0:
        return np.zeros(n, dtype=float), ["no_baseline_event_times_at_or_before_horizon"]

    H01 = baseline_cumulative_values_at_times(cox_event1, times)
    H02 = baseline_cumulative_values_at_times(cox_event2, times)
    dH01 = np.diff(np.concatenate([[0.0], H01]))
    dH02 = np.diff(np.concatenate([[0.0], H02]))
    dH01 = np.clip(dH01, 0.0, None)
    dH02 = np.clip(dH02, 0.0, None)

    lp1 = cox_event1.predict_log_partial_hazard(X_test).astype(float).to_numpy()
    lp2 = cox_event2.predict_log_partial_hazard(X_test).astype(float).to_numpy()
    hr1 = np.exp(np.clip(lp1, -30, 30))
    hr2 = np.exp(np.clip(lp2, -30, 30))

    pred = np.zeros(n, dtype=float)
    prev_H1 = np.zeros(n, dtype=float)
    prev_H2 = np.zeros(n, dtype=float)
    for inc1, inc2 in zip(dH01, dH02):
        S_prev = np.exp(-prev_H1 - prev_H2)
        pred += S_prev * (inc1 * hr1)
        prev_H1 += inc1 * hr1
        prev_H2 += inc2 * hr2
    if np.isnan(pred).any():
        warnings_out.append("predicted_CIF_contains_nan")
    if (pred < 0).any() or (pred > 1).any():
        warnings_out.append("predicted_CIF_outside_0_1_clipped")
    pred = np.nan_to_num(pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pred = np.clip(pred, 0.0, 1.0)
    return pred, warnings_out


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3 or np.unique(x_arr[mask]).size < 2 or np.unique(y_arr[mask]).size < 2:
        return np.nan
    try:
        corr, _ = spearmanr(x_arr[mask], y_arr[mask])
        return float(corr)
    except Exception:
        return np.nan


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3 or np.unique(x_arr[mask]).size < 2 or np.unique(y_arr[mask]).size < 2:
        return np.nan
    try:
        corr, _ = pearsonr(x_arr[mask], y_arr[mask])
        return float(corr)
    except Exception:
        return np.nan


def safe_auc(y: Sequence[int], score: Sequence[float]) -> float:
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    mask = np.isfinite(score_arr)
    if mask.sum() < 3 or np.unique(y_arr[mask]).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_arr[mask], score_arr[mask]))
    except Exception:
        return np.nan


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


def safe_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def calibration_intercept_slope(observed: np.ndarray, pred: np.ndarray) -> Tuple[float, float, str]:
    mask = np.isfinite(observed) & np.isfinite(pred)
    if mask.sum() < 10:
        return np.nan, np.nan, "too_few_valid_rows"
    y = observed[mask].astype(int)
    if np.unique(y).size < 2:
        return np.nan, np.nan, "observed_event_has_one_class"
    x = safe_logit(pred[mask]).reshape(-1, 1)
    if np.unique(x).size < 2:
        return np.nan, np.nan, "predicted_risk_has_no_variation"
    try:
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        return float(model.intercept_[0]), float(model.coef_[0][0]), ""
    except Exception as exc:
        return np.nan, np.nan, f"{type(exc).__name__}: {exc}"


def base_result_row(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    runtime_sec: float,
    failed: bool,
    failure_reason: str = "",
) -> Dict[str, Any]:
    train_status = train_df[STATUS_COL].astype(int)
    test_status = test_df[STATUS_COL].astype(int)
    return {
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_event1_train": int((train_status == 1).sum()),
        "n_event2_train": int((train_status == 2).sum()),
        "n_censored_train": int((train_status == 0).sum()),
        "n_event1_test": int((test_status == 1).sum()),
        "n_event2_test": int((test_status == 2).sum()),
        "n_censored_test": int((test_status == 0).sum()),
        "runtime_sec": runtime_sec,
        "failed": failed,
        "failure_reason": failure_reason,
        "predicted_risk5_available": False,
        "mean_predicted_risk5": np.nan,
        "mean_true_risk5": np.nan,
        "observed_event1_5y_rate": np.nan,
        "observed_event2_5y_rate": np.nan,
        "risk5_mae_vs_true_risk": np.nan,
        "risk5_rmse_vs_true_risk": np.nan,
        "risk5_spearman_with_true_risk": np.nan,
        "risk5_pearson_with_true_risk": np.nan,
        "brier_5y_naive": np.nan,
        "auc_5y_observed_event1": np.nan,
        "calibration_intercept_5y": np.nan,
        "calibration_slope_5y": np.nan,
        "calibration_failure_reason": "",
        "cause_specific_cindex_event1": np.nan,
        "prediction_warning": "",
        "finegray_available": False,
        "finegray_ran": False,
        "finegray_failure_reason": "",
    }


def calibration_decile_rows(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    test_df: pd.DataFrame,
    pred_risk: np.ndarray,
    true_risk_col: str,
    horizon: float,
) -> List[Dict[str, Any]]:
    valid = np.isfinite(pred_risk)
    if valid.sum() == 0:
        return []
    tmp = pd.DataFrame(
        {
            "pred": pred_risk,
            "true": test_df[true_risk_col].astype(float).to_numpy(),
            "event1": ((test_df[STATUS_COL].astype(int).to_numpy() == 1) & (test_df[DURATION_COL].astype(float).to_numpy() <= horizon)).astype(int),
            "event2": ((test_df[STATUS_COL].astype(int).to_numpy() == 2) & (test_df[DURATION_COL].astype(float).to_numpy() <= horizon)).astype(int),
        }
    ).loc[valid].copy()
    n_bins = min(10, len(tmp))
    ranks = tmp["pred"].rank(method="first")
    tmp["decile"] = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop") + 1
    rows: List[Dict[str, Any]] = []
    for decile, group in tmp.groupby("decile", sort=True):
        rows.append(
            {
                "scenario_id": scenario_id,
                "replicate_id": replicate_id,
                "model": model_name,
                "decile": int(decile),
                "n": int(len(group)),
                "mean_predicted_risk5": float(group["pred"].mean()),
                "mean_true_risk5": float(group["true"].mean()),
                "observed_event1_5y_rate": float(group["event1"].mean()),
                "observed_event2_5y_rate": float(group["event2"].mean()),
            }
        )
    return rows


def evaluate_predictions(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_risk5: Optional[Sequence[float]],
    risk_score: Optional[Sequence[float]],
    runtime_sec: float,
    true_risk_col: str,
    horizon: float,
    prediction_warning: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    row = base_result_row(scenario_id, replicate_id, model_name, train_df, test_df, runtime_sec, False)
    true_risk = test_df[true_risk_col].astype(float).to_numpy()
    duration = test_df[DURATION_COL].astype(float).to_numpy()
    status = test_df[STATUS_COL].astype(int).to_numpy()
    observed_event1_5y = ((status == 1) & (duration <= horizon)).astype(int)
    observed_event2_5y = ((status == 2) & (duration <= horizon)).astype(int)
    row["mean_true_risk5"] = float(np.nanmean(true_risk))
    row["observed_event1_5y_rate"] = float(np.mean(observed_event1_5y))
    row["observed_event2_5y_rate"] = float(np.mean(observed_event2_5y))
    row["prediction_warning"] = prediction_warning

    if pred_risk5 is None:
        return row, []
    pred = np.asarray(pred_risk5, dtype=float)
    pred = np.clip(pred, 0.0, 1.0)
    valid = np.isfinite(pred) & np.isfinite(true_risk)
    if valid.sum() == 0:
        row["prediction_warning"] = (prediction_warning + ";no_finite_predicted_risk").strip(";")
        return row, []

    row["predicted_risk5_available"] = True
    row["mean_predicted_risk5"] = float(np.nanmean(pred))
    diff = pred[valid] - true_risk[valid]
    row["risk5_mae_vs_true_risk"] = float(np.mean(np.abs(diff)))
    row["risk5_rmse_vs_true_risk"] = float(np.sqrt(np.mean(diff**2)))
    row["risk5_spearman_with_true_risk"] = safe_spearman(pred, true_risk)
    row["risk5_pearson_with_true_risk"] = safe_pearson(pred, true_risk)
    row["brier_5y_naive"] = float(np.mean((observed_event1_5y[valid] - pred[valid]) ** 2))
    row["auc_5y_observed_event1"] = safe_auc(observed_event1_5y, pred)
    intercept, slope, cal_reason = calibration_intercept_slope(observed_event1_5y, pred)
    row["calibration_intercept_5y"] = intercept
    row["calibration_slope_5y"] = slope
    row["calibration_failure_reason"] = cal_reason
    score = np.asarray(risk_score, dtype=float) if risk_score is not None else pred
    row["cause_specific_cindex_event1"] = harrell_c_index(duration, (status == 1).astype(int), score)
    deciles = calibration_decile_rows(scenario_id, replicate_id, model_name, test_df, pred, true_risk_col, horizon)
    return row, deciles


def fit_oracle(test_df: pd.DataFrame, true_risk_col: str) -> Tuple[np.ndarray, np.ndarray, str]:
    return (
        test_df[true_risk_col].astype(float).to_numpy(),
        test_df[TRUE_LP_COL].astype(float).to_numpy(),
        "",
    )


def fit_nonparametric_aj_null(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: float) -> Tuple[np.ndarray, np.ndarray, str]:
    cif1, _, _ = compute_aalen_johansen_cif1_at_horizon(
        train_df[DURATION_COL].astype(float).to_numpy(),
        train_df[STATUS_COL].astype(int).to_numpy(),
        horizon,
    )
    pred = np.repeat(cif1, len(test_df))
    return pred, pred.copy(), ""


def fit_cs_cox_cif(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    penalizer_grid: Sequence[float],
    horizon: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    X_train, X_test, _ = build_model_matrix(train_df, test_df, predictors, scale_numeric=True)
    cox1 = fit_cause_specific_cox(train_df, X_train, EVENT1_STATUS, penalizer_grid)
    cox2 = fit_cause_specific_cox(train_df, X_train, EVENT2_STATUS, penalizer_grid)
    pred, warnings_out = predict_cif_from_two_cause_specific_cox(cox1, cox2, X_test, horizon)
    risk_score = cox1.predict_log_partial_hazard(X_test).astype(float).to_numpy()
    return pred, risk_score, ";".join(warnings_out)


def result_key(scenario_id: Any, replicate_id: Any, model_name: Any) -> Tuple[str, str, str]:
    return (str(scenario_id), str(replicate_id), str(model_name))


def dedupe_performance(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or not {"scenario_id", "replicate_id", "model"}.issubset(perf.columns):
        return perf.copy()
    out = perf.copy()
    out["_rep_key"] = out["replicate_id"].astype(str)
    out = out.drop_duplicates(["scenario_id", "_rep_key", "model"], keep="last").drop(columns=["_rep_key"])
    return out.reset_index(drop=True)


def load_existing_performance(paths: Dict[str, Path]) -> pd.DataFrame:
    path = paths["tables"] / "replicate_competing_risk_performance.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        perf = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if not {"scenario_id", "replicate_id", "model", "failed"}.issubset(perf.columns):
        return pd.DataFrame()
    perf = dedupe_performance(perf)
    print(f"Loaded existing C3 checkpoint rows: {len(perf)}", flush=True)
    return perf


def skip_keys_from_performance(perf: pd.DataFrame, rerun_failed: bool) -> set[Tuple[str, str, str]]:
    if perf.empty:
        return set()
    if rerun_failed:
        to_skip = perf.loc[~perf["failed"].astype(bool)].copy()
    else:
        to_skip = perf.copy()
    return {result_key(row.scenario_id, row.replicate_id, row.model) for row in to_skip.itertuples(index=False)}


def replace_row(rows: List[Dict[str, Any]], new_row: Dict[str, Any]) -> None:
    key = result_key(new_row["scenario_id"], new_row["replicate_id"], new_row["model"])
    kept = [r for r in rows if result_key(r.get("scenario_id"), r.get("replicate_id"), r.get("model")) != key]
    kept.append(new_row)
    rows[:] = kept


def save_failure_log(perf: pd.DataFrame, paths: Dict[str, Path]) -> None:
    cols = [
        "scenario_id",
        "replicate_id",
        "model",
        "failed",
        "failure_reason",
        "runtime_sec",
        "n_train",
        "n_test",
        "n_event1_train",
        "n_event2_train",
        "n_event1_test",
        "n_event2_test",
    ]
    failures = perf.loc[perf["failed"].astype(bool), [c for c in cols if c in perf.columns]]
    failures.to_csv(paths["tables"] / "failure_log.csv", index=False)


def save_checkpoint(
    result_rows: List[Dict[str, Any]],
    calibration_rows: List[Dict[str, Any]],
    paths: Dict[str, Path],
) -> pd.DataFrame:
    perf = dedupe_performance(pd.DataFrame(result_rows))
    perf.to_csv(paths["tables"] / "replicate_competing_risk_performance.csv", index=False)
    save_failure_log(perf, paths)
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            paths["tables"] / "calibration_deciles_replicate_level.csv", index=False
        )
    return perf


def base_failure_row(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    runtime_sec: float,
    failure_reason: str,
) -> Dict[str, Any]:
    return base_result_row(
        scenario_id=scenario_id,
        replicate_id=replicate_id,
        model_name=model_name,
        train_df=train_df,
        test_df=test_df,
        runtime_sec=runtime_sec,
        failed=True,
        failure_reason=failure_reason,
    )


def run_model(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    fit_func: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    true_risk_col: str,
    horizon: float,
    debug_sample: bool = False,
    sample_seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[pd.DataFrame]]:
    start = time.perf_counter()
    try:
        pred_risk5, risk_score, warning = fit_func()
        runtime = time.perf_counter() - start
        row, deciles = evaluate_predictions(
            scenario_id,
            replicate_id,
            model_name,
            train_df,
            test_df,
            pred_risk5,
            risk_score,
            runtime,
            true_risk_col,
            horizon,
            warning,
        )
        sample = (
            prediction_sample(
                scenario_id,
                replicate_id,
                model_name,
                test_df,
                np.asarray(pred_risk5, dtype=float),
                true_risk_col,
                sample_seed,
            )
            if debug_sample and pred_risk5 is not None
            else None
        )
        return row, deciles, sample
    except Exception as exc:
        runtime = time.perf_counter() - start
        reason = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {scenario_id} rep={replicate_id} model={model_name}: {reason}", flush=True)
        traceback.print_exc(limit=2)
        return base_failure_row(scenario_id, replicate_id, model_name, train_df, test_df, runtime, reason), [], None


def prediction_sample(
    scenario_id: str,
    replicate_id: Any,
    model_name: str,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    true_risk_col: str,
    seed: int,
) -> pd.DataFrame:
    n = min(DEBUG_PRED_SAMPLE_N, len(test_df))
    sample = test_df.sample(n=n, random_state=seed)
    positions = test_df.index.get_indexer(sample.index)
    return pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "model": model_name,
            "synthetic_id": sample["synthetic_id"].to_numpy(),
            DURATION_COL: sample[DURATION_COL].to_numpy(),
            STATUS_COL: sample[STATUS_COL].to_numpy(),
            "predicted_risk5": pred[positions],
            "true_risk5": sample[true_risk_col].to_numpy(),
        }
    )


def process_scenario_file(
    file_path: Path,
    predictors: Dict[str, List[str]],
    config: Config,
    paths: Dict[str, Path],
    result_rows: List[Dict[str, Any]],
    calibration_rows: List[Dict[str, Any]],
    skip_keys: set[Tuple[str, str, str]],
    true_risk_col: str,
    prediction_samples: List[pd.DataFrame],
) -> None:
    print(f"Loading scenario file: {file_path.name}", flush=True)
    scenario_df = pd.read_csv(file_path)
    scenario_id = str(scenario_df["scenario_id"].dropna().iloc[0])
    replicate_ids = sorted(scenario_df["replicate_id"].dropna().unique().tolist())
    if config.max_reps_per_scenario is not None:
        replicate_ids = replicate_ids[: config.max_reps_per_scenario]
    print(f"Processing {scenario_id}: {len(replicate_ids)} replicate(s)", flush=True)

    for replicate_id in replicate_ids:
        if all(result_key(scenario_id, replicate_id, m) in skip_keys for m in MANDATORY_MODELS):
            print(f"  replicate {replicate_id}: skip all mandatory models already checkpointed", flush=True)
            continue
        rep_df = scenario_df.loc[scenario_df["replicate_id"] == replicate_id].copy()
        train_df, test_df = split_train_test(rep_df, config)
        print(
            f"  replicate {replicate_id}: train={len(train_df)} test={len(test_df)} "
            f"event1_test={(test_df[STATUS_COL].astype(int) == 1).sum()} "
            f"event2_test={(test_df[STATUS_COL].astype(int) == 2).sum()}",
            flush=True,
        )
        model_calls = [
            (
                MODEL_ORACLE,
                lambda test_df=test_df: fit_oracle(test_df, true_risk_col),
            ),
            (
                MODEL_AJ,
                lambda train_df=train_df, test_df=test_df: fit_nonparametric_aj_null(
                    train_df, test_df, config.horizon_years
                ),
            ),
            (
                MODEL_CS_DGM,
                lambda train_df=train_df, test_df=test_df: fit_cs_cox_cif(
                    train_df,
                    test_df,
                    predictors["dgm_predictors"],
                    penalizer_grid=[0.01, 0.05, 0.1, 0.5, 1.0],
                    horizon=config.horizon_years,
                ),
            ),
            (
                MODEL_CS_SAFE,
                lambda train_df=train_df, test_df=test_df: fit_cs_cox_cif(
                    train_df,
                    test_df,
                    predictors["all_safe_predictors"],
                    penalizer_grid=[0.1, 1.0, 5.0],
                    horizon=config.horizon_years,
                ),
            ),
        ]
        for model_name, fit_func in model_calls:
            key = result_key(scenario_id, replicate_id, model_name)
            if key in skip_keys:
                print(f"    skip checkpointed row: rep={replicate_id} model={model_name}", flush=True)
                continue
            row, deciles, _ = run_model(
                scenario_id,
                replicate_id,
                model_name,
                fit_func,
                train_df,
                test_df,
                true_risk_col,
                config.horizon_years,
                debug_sample=config.debug_mode,
                sample_seed=replicate_random_state(config.random_seed, replicate_id),
            )
            replace_row(result_rows, row)
            calibration_rows.extend(deciles)
            if config.debug_mode and _ is not None:
                prediction_samples.append(_)
            skip_keys.add(key)
            perf = save_checkpoint(result_rows, calibration_rows, paths)
            print(
                f"    saved checkpoint: rows={len(perf)} failures={int(perf['failed'].astype(bool).sum())}",
                flush=True,
            )
    del scenario_df


def metric_stats(values: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
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
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
    }


def aggregate_results(perf: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    metric_cols = [
        "risk5_mae_vs_true_risk",
        "risk5_rmse_vs_true_risk",
        "risk5_spearman_with_true_risk",
        "risk5_pearson_with_true_risk",
        "brier_5y_naive",
        "auc_5y_observed_event1",
        "calibration_slope_5y",
        "calibration_intercept_5y",
        "cause_specific_cindex_event1",
        "mean_predicted_risk5",
        "mean_true_risk5",
        "observed_event1_5y_rate",
        "observed_event2_5y_rate",
        "runtime_sec",
    ]
    rows: List[Dict[str, Any]] = []
    for (scenario_id, model), group in perf.groupby(["scenario_id", "model"], sort=True):
        success = group.loc[~group["failed"].astype(bool)].copy()
        row: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "model": model,
            "n_successful_reps": int(len(success)),
            "n_failed_reps": int(group["failed"].astype(bool).sum()),
            "failure_rate": float(group["failed"].astype(bool).mean()) if len(group) else np.nan,
        }
        for metric in metric_cols:
            stats = metric_stats(success[metric] if metric in success.columns else pd.Series(dtype=float))
            for stat, value in stats.items():
                row[f"{metric}_{stat}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(paths["tables"] / "scenario_competing_risk_summary_mean_sd_ci.csv", index=False)
    return summary


def aggregate_calibration(paths: Dict[str, Path]) -> pd.DataFrame:
    path = paths["tables"] / "calibration_deciles_replicate_level.csv"
    if not path.exists():
        out = pd.DataFrame()
        out.to_csv(paths["tables"] / "calibration_deciles_summary.csv", index=False)
        return out
    cal = pd.read_csv(path)
    if cal.empty:
        cal.to_csv(paths["tables"] / "calibration_deciles_summary.csv", index=False)
        return cal
    rows = []
    for (scenario_id, model, decile), group in cal.groupby(["scenario_id", "model", "decile"], sort=True):
        rows.append(
            {
                "scenario_id": scenario_id,
                "model": model,
                "decile": decile,
                "mean_n": float(group["n"].mean()),
                "mean_predicted_risk5": float(group["mean_predicted_risk5"].mean()),
                "mean_true_risk5": float(group["mean_true_risk5"].mean()),
                "observed_event1_5y_rate": float(group["observed_event1_5y_rate"].mean()),
                "observed_event2_5y_rate": float(group["observed_event2_5y_rate"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "calibration_deciles_summary.csv", index=False)
    return out


def get_summary_value(summary: pd.DataFrame, scenario_id: str, model: str, col: str) -> float:
    rows = summary.loc[(summary["scenario_id"] == scenario_id) & (summary["model"] == model)]
    if rows.empty or col not in rows.columns:
        return np.nan
    value = rows.iloc[0][col]
    return float(value) if pd.notna(value) else np.nan


def create_model_difference_vs_oracle(summary: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario_id, group in summary.groupby("scenario_id", sort=True):
        oracle = group.loc[group["model"] == MODEL_ORACLE]
        if oracle.empty:
            continue
        oracle = oracle.iloc[0]
        for _, model_row in group.loc[group["model"] != MODEL_ORACLE].iterrows():
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model_row["model"],
                    "oracle_mae_mean": oracle["risk5_mae_vs_true_risk_mean"],
                    "model_mae_mean": model_row["risk5_mae_vs_true_risk_mean"],
                    "mae_gap_vs_oracle": model_row["risk5_mae_vs_true_risk_mean"] - oracle["risk5_mae_vs_true_risk_mean"],
                    "oracle_rmse_mean": oracle["risk5_rmse_vs_true_risk_mean"],
                    "model_rmse_mean": model_row["risk5_rmse_vs_true_risk_mean"],
                    "rmse_gap_vs_oracle": model_row["risk5_rmse_vs_true_risk_mean"] - oracle["risk5_rmse_vs_true_risk_mean"],
                    "oracle_brier_mean": oracle["brier_5y_naive_mean"],
                    "model_brier_mean": model_row["brier_5y_naive_mean"],
                    "brier_gap_vs_oracle": model_row["brier_5y_naive_mean"] - oracle["brier_5y_naive_mean"],
                    "oracle_auc5_mean": oracle["auc_5y_observed_event1_mean"],
                    "model_auc5_mean": model_row["auc_5y_observed_event1_mean"],
                    "auc5_gap_vs_oracle": model_row["auc_5y_observed_event1_mean"] - oracle["auc_5y_observed_event1_mean"],
                    "oracle_calibration_slope_mean": oracle["calibration_slope_5y_mean"],
                    "model_calibration_slope_mean": model_row["calibration_slope_5y_mean"],
                    "failure_rate": model_row["failure_rate"],
                    "n_successful_reps": model_row["n_successful_reps"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "model_difference_vs_oracle_C3_by_scenario.csv", index=False)
    return out


def create_oracle_sanity_audit(diff: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for _, row in diff.iterrows():
        mae_gap = row["mae_gap_vs_oracle"]
        auc_gap = row["auc5_gap_vs_oracle"]
        flag_mae = bool(np.isfinite(mae_gap) and mae_gap < -0.005)
        flag_auc = bool(np.isfinite(auc_gap) and auc_gap > 0.02)
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "model": row["model"],
                "model_mae_minus_oracle": mae_gap,
                "model_auc_minus_oracle": auc_gap,
                "flag_model_beats_oracle_mae_by_more_than_0_005": flag_mae,
                "flag_model_exceeds_oracle_auc_by_more_than_0_02": flag_auc,
                "needs_audit": flag_mae or flag_auc,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "oracle_sanity_audit_C3.csv", index=False)
    return out


def create_best_model_table(summary: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    non_oracle = summary.loc[summary["model"] != MODEL_ORACLE].copy()
    for scenario_id, group in non_oracle.groupby("scenario_id", sort=True):
        ranked = group.copy()
        ranked["_mae"] = pd.to_numeric(ranked["risk5_mae_vs_true_risk_mean"], errors="coerce").fillna(np.inf)
        ranked["_brier"] = pd.to_numeric(ranked["brier_5y_naive_mean"], errors="coerce").fillna(np.inf)
        ranked["_auc"] = pd.to_numeric(ranked["auc_5y_observed_event1_mean"], errors="coerce").fillna(-np.inf)
        ranked["_failure"] = pd.to_numeric(ranked["failure_rate"], errors="coerce").fillna(np.inf)
        min_mae = ranked["_mae"].min()
        candidates = ranked.loc[ranked["_mae"] <= min_mae + 0.005].copy()
        candidates = candidates.sort_values(["_brier", "_auc", "_failure", "_mae", "model"], ascending=[True, False, True, True, True])
        rest = ranked.loc[~ranked.index.isin(candidates.index)].copy()
        rest = rest.sort_values(["_mae", "_brier", "_auc", "_failure", "model"], ascending=[True, True, False, True, True])
        ordered = pd.concat([candidates, rest], axis=0).reset_index(drop=True)
        if ordered.empty:
            continue
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else pd.Series(dtype=object)
        best_mae = float(best["risk5_mae_vs_true_risk_mean"])
        second_mae = float(second["risk5_mae_vs_true_risk_mean"]) if not second.empty else np.nan
        note = f"{best['model']} selected by lowest MAE with Brier/AUC/failure tie-breakers."
        rows.append(
            {
                "scenario_id": scenario_id,
                "best_model_by_mae": best["model"],
                "best_mae_mean": best_mae,
                "second_best_model": second.get("model", "") if not second.empty else "",
                "second_best_mae_mean": second_mae,
                "mae_difference_best_minus_second": best_mae - second_mae if np.isfinite(second_mae) else np.nan,
                "best_brier_mean": best["brier_5y_naive_mean"],
                "best_auc5_mean": best["auc_5y_observed_event1_mean"],
                "best_calibration_slope_mean": best["calibration_slope_5y_mean"],
                "interpretation_note": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "best_C3_model_by_scenario.csv", index=False)
    return out


def check_finegray_optional(paths: Dict[str, Path]) -> pd.DataFrame:
    r_script = Path("stepC3_finegray_optional.R")
    r_script.write_text(
        "\n".join(
            [
                "# Optional Fine-Gray placeholder for Step C3.",
                "# The Python pipeline checks Rscript and cmprsk availability.",
                "# A production version can call cmprsk::crr and export 5-year CIF predictions.",
                "if (!requireNamespace('cmprsk', quietly = TRUE)) {",
                "  stop('cmprsk is not installed')",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    rscript = shutil.which("Rscript")
    available = False
    reason = ""
    if rscript is None:
        reason = "Rscript_not_found"
    else:
        try:
            proc = subprocess.run(
                [rscript, "-e", "if (!requireNamespace('cmprsk', quietly=TRUE)) quit(status=2)"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                available = True
                reason = "cmprsk_available_but_prediction_not_implemented_in_C3_v1"
            else:
                reason = "cmprsk_not_installed_or_failed"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    out = pd.DataFrame(
        [
            {
                "finegray_available": available,
                "finegray_ran": False,
                "finegray_failure_reason": reason,
                "optional_status": "skipped_optional_not_part_of_mandatory_C3_v1",
                "r_script_path": str(r_script),
            }
        ]
    )
    out.to_csv(paths["tables"] / "finegray_optional_status.csv", index=False)
    return out


def any_forbidden_predictor_flag(paths: Dict[str, Path]) -> bool:
    path = paths["tables"] / "predictor_list_all_safe_C3.csv"
    if not path.exists():
        return True
    predictors = pd.read_csv(path)["predictor"].dropna().astype(str).tolist()
    return any(contains_forbidden_name(c) for c in predictors)


def create_full_run_sanity_checks(
    config: Config,
    perf: pd.DataFrame,
    safety: pd.DataFrame,
    oracle_audit: pd.DataFrame,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    expected_reps = 2 if config.debug_mode else 50
    expected_rows = 8 * expected_reps * len(MANDATORY_MODELS)
    observed_rows = int(len(perf))
    observed_scenarios = int(perf["scenario_id"].nunique()) if not perf.empty else 0
    reps_by_scenario = perf.groupby("scenario_id")["replicate_id"].nunique() if not perf.empty else pd.Series(dtype=int)
    observed_min_reps = int(reps_by_scenario.min()) if not reps_by_scenario.empty else 0
    observed_max_reps = int(reps_by_scenario.max()) if not reps_by_scenario.empty else 0
    failures = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    export_safety = bool_series_all_true(safety["safe_to_export_column_names"])
    forbidden = any_forbidden_predictor_flag(paths)
    oracle_flag = bool(oracle_audit["needs_audit"].astype(bool).any()) if not oracle_audit.empty else True
    failure_rate_ok = observed_rows > 0 and failures / observed_rows <= 0.05
    full_pass = (
        observed_rows == expected_rows
        and observed_scenarios == 8
        and observed_min_reps == expected_reps
        and observed_max_reps == expected_reps
        and export_safety
        and not forbidden
        and not oracle_flag
        and failure_rate_ok
    )
    rows = [
        ("expected_replicate_model_rows", expected_rows, expected_rows, True),
        ("observed_replicate_model_rows", expected_rows, observed_rows, observed_rows == expected_rows),
        ("expected_scenarios", 8, 8, True),
        ("observed_scenarios", 8, observed_scenarios, observed_scenarios == 8),
        ("expected_reps_per_scenario", expected_reps, expected_reps, True),
        ("observed_min_reps_per_scenario", expected_reps, observed_min_reps, observed_min_reps == expected_reps),
        ("observed_max_reps_per_scenario", expected_reps, observed_max_reps, observed_max_reps == expected_reps),
        ("total_failed_model_fits", "logged and <=5%", failures, failure_rate_ok),
        ("export_safety_passed", True, export_safety, export_safety),
        ("any_forbidden_predictor_flag", False, forbidden, not forbidden),
        ("any_oracle_sanity_flag", False, oracle_flag, not oracle_flag),
        ("full_run_passed", True, full_pass, full_pass),
    ]
    out = pd.DataFrame(
        [{"check_name": n, "expected": e, "observed": o, "passed": p} for n, e, o, p in rows]
    )
    out.to_csv(paths["tables"] / "full_run_sanity_checks_C3.csv", index=False)
    return out


def grouped_bar_plot(summary: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    if summary.empty or value_col not in summary.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = summary.pivot(index="scenario_id", columns="model", values=value_col)
        scenarios = plot_df.index.tolist()
        models = [m for m in MANDATORY_MODELS if m in plot_df.columns]
        x = np.arange(len(scenarios))
        width = 0.8 / max(len(models), 1)
        colors = ["#2F6B8E", "#6E7F80", "#D95F59", "#4C8F5B"]
        for i, model in enumerate(models):
            ax.bar(x - 0.4 + width / 2 + i * width, plot_df[model].to_numpy(dtype=float), width, label=model, color=colors[i % len(colors)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_figures(summary: pd.DataFrame, diff: pd.DataFrame, calibration_summary: pd.DataFrame, paths: Dict[str, Path]) -> None:
    specs = [
        ("risk5_mae_vs_true_risk_mean", "Mean MAE vs true risk", "5-year risk MAE by scenario and model", "risk5_mae_by_scenario_model.png"),
        ("risk5_rmse_vs_true_risk_mean", "Mean RMSE vs true risk", "5-year risk RMSE by scenario and model", "risk5_rmse_by_scenario_model.png"),
        ("brier_5y_naive_mean", "Mean naive Brier", "Naive 5-year Brier by scenario and model", "brier_5y_by_scenario_model.png"),
        ("auc_5y_observed_event1_mean", "Mean observed 5-year AUC", "Observed 5-year AUC by scenario and model", "auc5_by_scenario_model.png"),
        ("calibration_slope_5y_mean", "Mean calibration slope", "5-year calibration slope by scenario and model", "calibration_slope_by_scenario_model.png"),
    ]
    for col, ylabel, title, file_name in specs:
        grouped_bar_plot(summary, col, ylabel, title, paths["figures"] / file_name)

    if diff.empty:
        grouped_bar_plot(pd.DataFrame(), "", "MAE gap", "MAE gap vs oracle", paths["figures"] / "oracle_gap_mae_by_scenario_model.png")
    else:
        tmp = diff.rename(columns={"mae_gap_vs_oracle": "oracle_mae_gap_mean"})
        grouped_bar_plot(tmp, "oracle_mae_gap_mean", "MAE gap vs oracle", "MAE gap vs oracle by scenario and model", paths["figures"] / "oracle_gap_mae_by_scenario_model.png")

    for scenario in ["S0_linear_PH_inst30", "S6_highdim_sparseMRI_inst30", "S7_strong_death_competing_inst30"]:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        sub = calibration_summary.loc[calibration_summary["scenario_id"] == scenario] if not calibration_summary.empty else pd.DataFrame()
        if sub.empty:
            ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
            ax.axis("off")
        else:
            for model in [m for m in MANDATORY_MODELS if m in sub["model"].unique()]:
                g = sub.loc[sub["model"] == model].sort_values("decile")
                ax.plot(g["mean_predicted_risk5"], g["observed_event1_5y_rate"], marker="o", label=model)
            lim = max(0.55, float(sub[["mean_predicted_risk5", "observed_event1_5y_rate"]].max().max()) * 1.05)
            ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Mean predicted 5-year risk")
            ax.set_ylabel("Observed 5-year care-home rate")
            ax.set_title(f"Calibration deciles: {scenario}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        short = scenario.split("_")[0]
        fig.savefig(paths["figures"] / f"calibration_deciles_{short}.png", dpi=180)
        plt.close(fig)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.astype(str).values.tolist():
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def write_readme(
    config: Config,
    paths: Dict[str, Path],
    perf: pd.DataFrame,
    summary: pd.DataFrame,
    best: pd.DataFrame,
    oracle_audit: pd.DataFrame,
    sanity: pd.DataFrame,
    finegray_status: pd.DataFrame,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    n_rows = int(len(perf))
    n_fail = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    full_pass = False
    rows = sanity.loc[sanity["check_name"] == "full_run_passed"] if not sanity.empty else pd.DataFrame()
    if not rows.empty:
        full_pass = bool(rows.iloc[0]["observed"])
    fine_reason = finegray_status["finegray_failure_reason"].iloc[0] if not finegray_status.empty else "not_checked"
    any_oracle_flag = bool(oracle_audit["needs_audit"].astype(bool).any()) if not oracle_audit.empty else True
    best_lookup = best.set_index("scenario_id").to_dict("index") if not best.empty else {}

    def best_model(s: str) -> str:
        return str(best_lookup.get(s, {}).get("best_model_by_mae", "NA"))

    def summary_metric(scenario: str, model: str, metric: str) -> float:
        rows = summary.loc[(summary["scenario_id"] == scenario) & (summary["model"] == model)]
        if rows.empty or metric not in rows.columns:
            return np.nan
        value = rows.iloc[0][metric]
        return float(value) if pd.notna(value) else np.nan

    c2_consistency = "C2 comparison table was not available."
    c2_best_path = config.c2_dir / "tables" / "best_non_oracle_model_by_scenario.csv"
    if c2_best_path.exists() and not best.empty:
        c2_best = pd.read_csv(c2_best_path)
        model_map = {
            "cox_dgm_features": MODEL_CS_DGM,
            "penalised_cox_all_safe_predictors": MODEL_CS_SAFE,
            "xgb_survival_cox_strict": "xgb_survival_cox_strict_not_in_C3",
        }
        matches = 0
        compared = 0
        mismatch_notes = []
        for _, c2_row in c2_best.iterrows():
            scenario = c2_row["scenario_id"]
            c2_model = c2_row["best_model_by_cindex"]
            mapped = model_map.get(c2_model, c2_model)
            c3_model = best_model(scenario)
            compared += 1
            if mapped == c3_model:
                matches += 1
            else:
                mismatch_notes.append(f"{scenario}: C2={c2_model}, C3={c3_model}")
        c2_consistency = (
            f"Broadly consistent after mapping model families: {matches}/{compared} scenarios selected the same best fitted-model family."
        )
        if mismatch_notes:
            c2_consistency += " Differences: " + "; ".join(mismatch_notes) + "."

    s7_best_mae = float(best_lookup.get("S7_strong_death_competing_inst30", {}).get("best_mae_mean", np.nan))
    s7_aj_mae = summary_metric(
        "S7_strong_death_competing_inst30",
        MODEL_AJ,
        "risk5_mae_vs_true_risk_mean",
    )
    if np.isfinite(s7_best_mae) and np.isfinite(s7_aj_mae):
        s7_competing_text = (
            f"In S7, the best C3 competing-risk model MAE was {s7_best_mae:.4f} versus AJ-null MAE {s7_aj_mae:.4f}, "
            f"an improvement of {s7_aj_mae - s7_best_mae:.4f} over the non-individualised competing-risk baseline."
        )
    else:
        s7_competing_text = "S7 MAE comparison with the AJ null was unavailable."

    lines = [
        "# Step C3 Competing-Risk Evaluation",
        "",
        f"Run started: {started_at.isoformat(timespec='seconds')}",
        f"Run finished: {finished_at.isoformat(timespec='seconds')}",
        "",
        "## Aim",
        "",
        "C3 extends C2 by moving from cause-specific ranking/model comparison to 5-year absolute-risk evaluation under competing risk. C3 treats `status=2` death before care home as a competing event and reconstructs the care-home cumulative incidence function using cause-specific Cox models.",
        "",
        "## Data Source",
        "",
        "- Uses only `fully_synthetic_stepC_v1/`.",
        "- Optionally reads predictor lists saved by C2.",
        "- Does not use real SLAM data.",
        "- Does not use Step B data.",
        "- Does not use raw CSV files, death spreadsheets, or WMH spreadsheets.",
        "",
        "## Endpoint Definition",
        "",
        "- `status=1` is care-home entry / institutionalisation.",
        "- `status=2` is death before care home as the competing event.",
        "- `death_after_carehome` is not the competing event for the primary endpoint.",
        f"- Horizon: `{config.horizon_years}` years.",
        "",
        "## Models",
        "",
        "- `oracle_true_risk_not_a_model`",
        "- `nonparametric_aj_null`",
        "- `cs_cox_dgm_cif`",
        "- `cs_penalised_cox_all_safe_cif`",
        f"- `finegray_dgm_optional`: skipped/optional; reason = `{fine_reason}`",
        "",
        "## Metrics",
        "",
        "- MAE/RMSE versus true synthetic 5-year risk.",
        "- Naive Brier score at 5 years.",
        "- Observed 5-year AUC.",
        "- Calibration slope/intercept.",
        "- Calibration deciles.",
        "- Cause-specific C-index as a continuity metric with C2.",
        "",
        "## Main Results",
        "",
        f"- replicate-level rows: {n_rows}; failures: {n_fail}; full_run_passed: {full_pass}.",
        f"- In S0/S1/S2, the best non-oracle models were `{best_model('S0_linear_PH_inst30')}`, `{best_model('S1_linear_PH_inst15')}`, and `{best_model('S2_linear_PH_inst45')}`.",
        f"- In S6 high-dimensional sparse MRI, the best non-oracle model was `{best_model('S6_highdim_sparseMRI_inst30')}`.",
        f"- In S7 strong death competing risk, the best non-oracle model was `{best_model('S7_strong_death_competing_inst30')}`. {s7_competing_text}",
        f"- Any model exceeded the oracle unexpectedly: {any_oracle_flag}.",
        f"- Fine-Gray ran successfully: {bool(finegray_status['finegray_ran'].iloc[0]) if not finegray_status.empty else False}; status is recorded in `tables/finegray_optional_status.csv`.",
        f"- C3/C2 conclusion consistency: {c2_consistency}",
        "",
        "## Best Model Table",
        "",
        dataframe_to_markdown_table(
            best[
                [
                    "scenario_id",
                    "best_model_by_mae",
                    "best_mae_mean",
                    "best_brier_mean",
                    "best_auc5_mean",
                    "best_calibration_slope_mean",
                ]
            ]
        )
        if not best.empty
        else "_No best-model rows._",
        "",
        "## Limitations",
        "",
        "- The data are fully synthetic and intended for methodological testing, not clinical inference.",
        "- Fine-Gray status is explicitly logged when the optional model does not run.",
        "- The naive Brier score is not an IPCW Brier score.",
        "- This first C3 version focuses on 5-year risk rather than fully dynamic risk over time.",
        "- Cox DGM still uses a fallback predictor list unless exact DGM coefficients have been exported.",
        "",
        "## Suggested Next Steps",
        "",
        "- Prepare a supervisor-facing summary if C3 passes.",
        "- Optional C4: implement IPCW Brier score, time-dependent AUC, and a formal Fine-Gray pipeline.",
        "- Optional Step C generator improvement: export exact DGM predictor coefficients.",
        "",
    ]
    (paths["root"] / "README_stepC3_competing_risk_evaluation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    started_at = datetime.now()
    start_perf = time.perf_counter()
    config = parse_args()
    np.random.seed(config.random_seed)
    paths = make_output_dirs(config)
    print("Step C3 competing-risk evaluation", flush=True)
    print(f"mode={'debug' if config.debug_mode else 'full'}", flush=True)
    print(f"DATA_DIR={config.data_dir}", flush=True)
    print(f"OUT_DIR={config.out_dir}", flush=True)
    print(f"MAX_REPS_PER_SCENARIO={config.max_reps_per_scenario}", flush=True)

    inputs = load_and_audit_inputs(config, paths)
    first_columns = json.loads(inputs["data_file_audit"].iloc[0]["columns_json"])
    true_risk_col = inputs["true_risk_column"]
    predictors = build_predictor_lists(config, inputs["feature_dictionary"], first_columns, paths)
    finegray_status = check_finegray_optional(paths)

    existing_perf = load_existing_performance(paths)
    result_rows = existing_perf.to_dict("records") if not existing_perf.empty else []
    skip_keys = skip_keys_from_performance(existing_perf, config.rerun_failed)
    calibration_rows: List[Dict[str, Any]] = []
    cal_path = paths["tables"] / "calibration_deciles_replicate_level.csv"
    if cal_path.exists():
        try:
            calibration_rows = pd.read_csv(cal_path).to_dict("records")
        except pd.errors.EmptyDataError:
            calibration_rows = []
    prediction_samples: List[pd.DataFrame] = []

    for file_path in inputs["scenario_files"]:
        process_scenario_file(
            file_path,
            predictors,
            config,
            paths,
            result_rows,
            calibration_rows,
            skip_keys,
            true_risk_col,
            prediction_samples,
        )

    perf = save_checkpoint(result_rows, calibration_rows, paths)
    summary = aggregate_results(perf, paths)
    calibration_summary = aggregate_calibration(paths)
    diff = create_model_difference_vs_oracle(summary, paths)
    oracle_audit = create_oracle_sanity_audit(diff, paths)
    best = create_best_model_table(summary, paths)
    sanity = create_full_run_sanity_checks(config, perf, inputs["safety"], oracle_audit, paths)
    make_figures(summary, diff, calibration_summary, paths)

    if config.debug_mode and prediction_samples:
        pd.concat(prediction_samples, ignore_index=True).to_csv(
            paths["predictions"] / "debug_prediction_samples.csv", index=False
        )

    finished_at = datetime.now()
    write_readme(config, paths, perf, summary, best, oracle_audit, sanity, finegray_status, started_at, finished_at)

    total_runtime = time.perf_counter() - start_perf
    failure_count = int(perf["failed"].astype(bool).sum()) if not perf.empty else 0
    full_pass_row = sanity.loc[sanity["check_name"] == "full_run_passed"]
    full_pass = bool(full_pass_row.iloc[0]["observed"]) if not full_pass_row.empty else False
    print("Done.", flush=True)
    print(f"mode: {'debug' if config.debug_mode else 'full'}", flush=True)
    print(f"total runtime seconds: {total_runtime:.1f}", flush=True)
    print(f"replicate-level result rows: {len(perf)}", flush=True)
    print(f"failure count: {failure_count}", flush=True)
    print(f"full_run_passed: {full_pass}", flush=True)
    print("Core summary:", paths["tables"] / "scenario_competing_risk_summary_mean_sd_ci.csv", flush=True)
    print("Oracle sanity audit:", paths["tables"] / "oracle_sanity_audit_C3.csv", flush=True)
    print("Best model:", paths["tables"] / "best_C3_model_by_scenario.csv", flush=True)
    print("README:", paths["root"] / "README_stepC3_competing_risk_evaluation.md", flush=True)


if __name__ == "__main__":
    main()
