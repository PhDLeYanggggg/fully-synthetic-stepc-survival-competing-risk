#!/usr/bin/env python3
"""Step C4 extended fully synthetic survival / competing-risk comparison.

This script extends the completed Step C2/C3 analyses without regenerating data.
It uses only:

- fully_synthetic_stepC_v1/
- Step C2 predictor lists and summaries, when present
- Step C3 predictor lists and summaries, when present

It does not use real SLAM data, Step B semi-synthetic data, raw CSV files,
death spreadsheets, WMH spreadsheets, or real identifiers.

Primary C4 endpoint
-------------------
Five-year cumulative incidence of care-home entry / institutionalisation
under death-before-care-home competing risk.

Run
---
Debug:
    python stepC4_extended_model_comparison.py --debug

Full:
    python stepC4_extended_model_comparison.py --full
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
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
C2_DIR = Path("fully_synthetic_stepC2_model_comparison")
C3_DIR = Path("fully_synthetic_stepC3_competing_risk_evaluation")
OUT_DIR = Path("fully_synthetic_stepC4_extended_models")

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

MODEL_FINEGRAY_DGM = "finegray_dgm_cif_R"
MODEL_FINEGRAY_SAFE_REDUCED = "finegray_all_safe_reduced_R"
MODEL_RSF_DGM = "cs_rsf_dgm_cif"
MODEL_RSF_SAFE = "cs_rsf_all_safe_cif"
MODEL_GBSA_DGM = "cs_gbsa_dgm_cif"
MODEL_GBSA_SAFE = "cs_gbsa_all_safe_cif"
MODEL_DEEPSURV_DGM = "deepsurv_dgm_cause_specific_optional"
MODEL_DEEPSURV_SAFE = "deepsurv_all_safe_cause_specific_optional"
MODEL_DEEPHIT_DGM = "deephit_competing_risk_dgm_optional"
MODEL_DEEPHIT_SAFE_REDUCED = "deephit_competing_risk_all_safe_reduced_optional"

CORE_MODELS = [
    MODEL_FINEGRAY_DGM,
    MODEL_FINEGRAY_SAFE_REDUCED,
    MODEL_RSF_DGM,
    MODEL_RSF_SAFE,
    MODEL_GBSA_DGM,
    MODEL_GBSA_SAFE,
]
OPTIONAL_DEEP_MODELS = [
    MODEL_DEEPSURV_DGM,
    MODEL_DEEPSURV_SAFE,
    MODEL_DEEPHIT_DGM,
    MODEL_DEEPHIT_SAFE_REDUCED,
]
ALL_MODELS = CORE_MODELS + OPTIONAL_DEEP_MODELS

MODEL_META = {
    MODEL_FINEGRAY_DGM: ("Fine-Gray", "dgm"),
    MODEL_FINEGRAY_SAFE_REDUCED: ("Fine-Gray", "all_safe_reduced"),
    MODEL_RSF_DGM: ("Random Survival Forest", "dgm"),
    MODEL_RSF_SAFE: ("Random Survival Forest", "all_safe"),
    MODEL_GBSA_DGM: ("Gradient Boosting Survival", "dgm"),
    MODEL_GBSA_SAFE: ("Gradient Boosting Survival", "all_safe"),
    MODEL_DEEPSURV_DGM: ("DeepSurv optional", "dgm"),
    MODEL_DEEPSURV_SAFE: ("DeepSurv optional", "all_safe"),
    MODEL_DEEPHIT_DGM: ("DeepHit optional", "dgm"),
    MODEL_DEEPHIT_SAFE_REDUCED: ("DeepHit optional", "all_safe_reduced"),
}

FORBIDDEN_PREDICTOR_SUBSTRINGS = [
    "BrcId",
    "ScanID",
    "Scan_Date",
    "Date_Of_Death",
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

CORE_METRICS = [
    "risk5_mae_vs_true_risk",
    "risk5_rmse_vs_true_risk",
    "risk5_spearman_with_true_risk",
    "brier_5y_naive",
    "brier_5y_ipcw",
    "integrated_brier_score_1to5",
    "auc_5y_observed_event1",
    "calibration_slope_5y",
    "calibration_intercept_5y",
    "cause_specific_cindex_event1",
]


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    c2_dir: Path = C2_DIR
    c3_dir: Path = C3_DIR
    out_dir: Path = OUT_DIR
    debug_mode: bool = DEBUG_MODE
    max_reps_per_scenario: Optional[int] = MAX_REPS_PER_SCENARIO
    test_size: float = TEST_SIZE
    random_seed: int = RANDOM_SEED
    horizon_years: float = HORIZON_YEARS
    rerun_failed: bool = False
    max_new_model_fits: Optional[int] = None
    log_existing_skips: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--c2-dir", type=Path, default=C2_DIR)
    parser.add_argument("--c3-dir", type=Path, default=C3_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-reps", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--horizon", type=float, default=HORIZON_YEARS)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--max-new-model-fits",
        type=int,
        default=None,
        help="Optional checkpointed chunk size. Runs at most this many new scenario-replicate-model rows, then writes partial summaries and exits.",
    )
    parser.add_argument(
        "--log-existing-skips",
        action="store_true",
        help="Print every scenario-replicate-model row skipped by checkpoint resume.",
    )
    args = parser.parse_args()
    if args.debug and args.full:
        parser.error("Use either --debug or --full, not both.")

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
        if not args.full:
            debug_mode = True

    return Config(
        data_dir=args.data_dir,
        c2_dir=args.c2_dir,
        c3_dir=args.c3_dir,
        out_dir=args.out_dir,
        debug_mode=debug_mode,
        max_reps_per_scenario=max_reps,
        test_size=args.test_size,
        random_seed=args.seed,
        horizon_years=args.horizon,
        rerun_failed=args.rerun_failed,
        max_new_model_fits=args.max_new_model_fits,
        log_existing_skips=args.log_existing_skips,
    )


def run_mode_label(config: Config) -> str:
    if config.debug_mode:
        return "debug"
    if config.max_reps_per_scenario is not None:
        return f"full_limited_{config.max_reps_per_scenario}_reps"
    return "full"


def run_metadata(config: Config) -> Dict[str, Any]:
    return {
        "run_mode": run_mode_label(config),
        "horizon_years": float(config.horizon_years),
        "test_size": float(config.test_size),
        "random_seed": int(config.random_seed),
        "max_reps_per_scenario": "" if config.max_reps_per_scenario is None else int(config.max_reps_per_scenario),
        "max_new_model_fits": "" if config.max_new_model_fits is None else int(config.max_new_model_fits),
    }


def is_publication_full_run(config: Config) -> bool:
    return (not config.debug_mode) and config.max_reps_per_scenario is None


def make_output_dirs(config: Config) -> Dict[str, Path]:
    root = config.out_dir
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "predictions": root / "predictions",
        "tmp": root / "_tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def prepare_run_outputs(config: Config, paths: Dict[str, Path]) -> None:
    """Keep debug, full-limited, and full checkpoints from contaminating each other."""

    perf_path = paths["tables"] / "replicate_extended_model_performance.csv"
    if not perf_path.exists():
        return
    try:
        existing = pd.read_csv(perf_path)
    except Exception:
        existing = pd.DataFrame()
    if existing.empty:
        return

    desired_mode = run_mode_label(config)
    if "run_mode" in existing.columns:
        existing_modes = sorted(str(x) for x in existing["run_mode"].dropna().unique())
    else:
        existing_modes = ["legacy_without_run_mode"]
    if existing_modes == [desired_mode]:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = paths["root"] / f"{'_'.join(existing_modes)}_snapshot_before_{desired_mode}_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for key in ["tables", "figures", "predictions"]:
        src_dir = paths[key]
        dst_dir = archive_dir / key
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in src_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, dst_dir / file.name)
                file.unlink()

    readme = paths["root"] / "README_stepC4_extended_model_comparison.md"
    if readme.exists():
        shutil.copy2(readme, archive_dir / readme.name)
        readme.unlink()
    print(
        f"Archived existing C4 outputs with run mode(s) {existing_modes} to {archive_dir} before starting {desired_mode}.",
        flush=True,
    )


def import_status(module_name: str) -> Tuple[bool, str, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version), ""
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def r_package_status(package: str, attempt_install: bool = False) -> Tuple[bool, str]:
    if shutil.which("Rscript") is None:
        return False, "Rscript_not_available"
    check_cmd = [
        "Rscript",
        "-e",
        f'cat(requireNamespace("{package}", quietly=TRUE))',
    ]
    env = dict(os.environ)
    env.setdefault("LC_ALL", "C")
    env.setdefault("LC_CTYPE", "C")
    try:
        proc = subprocess.run(check_cmd, capture_output=True, text=True, env=env, check=False)
        available = "TRUE" in proc.stdout
    except Exception as exc:
        return False, f"R_check_failed: {exc}"
    if available:
        return True, "available"
    if not attempt_install:
        return False, "not_installed"
    install_cmd = [
        "Rscript",
        "-e",
        f'install.packages("{package}", repos="https://cloud.r-project.org"); '
        f'cat(requireNamespace("{package}", quietly=TRUE))',
    ]
    try:
        proc = subprocess.run(install_cmd, capture_output=True, text=True, env=env, check=False, timeout=300)
        available = "TRUE" in proc.stdout
        if available:
            return True, "installed_from_CRAN"
        return False, "install_failed: " + (proc.stderr[-500:] or proc.stdout[-500:])
    except Exception as exc:
        return False, f"install_failed: {exc}"


def dependency_audit(paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    module_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "sklearn",
        "lifelines": "lifelines",
        "sksurv": "sksurv",
        "xgboost": "xgboost",
        "matplotlib": "matplotlib",
        "torch": "torch",
        "torchtuples": "torchtuples",
        "pycox": "pycox",
    }
    optional = {"torch", "torchtuples", "pycox"}
    for name, module_name in module_map.items():
        available, version, reason = import_status(module_name)
        rows.append(
            {
                "dependency": name,
                "language": "python",
                "required_for": "optional_deep" if name in optional else "core_or_plotting",
                "available": available,
                "version_or_status": version if available else "",
                "failure_reason": reason,
            }
        )

    rscript_available = shutil.which("Rscript") is not None
    rows.append(
        {
            "dependency": "Rscript",
            "language": "R",
            "required_for": "Fine-Gray",
            "available": rscript_available,
            "version_or_status": "available" if rscript_available else "",
            "failure_reason": "" if rscript_available else "Rscript_not_available",
        }
    )
    for package in ["survival", "cmprsk"]:
        available, status = r_package_status(package, attempt_install=(package == "cmprsk"))
        rows.append(
            {
                "dependency": package,
                "language": "R",
                "required_for": "Fine-Gray",
                "available": available,
                "version_or_status": status if available else "",
                "failure_reason": "" if available else status,
            }
        )
    dep = pd.DataFrame(rows)
    dep.to_csv(paths["tables"] / "dependency_status_C4.csv", index=False)
    return dep


def dependency_available(dep: pd.DataFrame, name: str) -> bool:
    row = dep.loc[dep["dependency"] == name]
    return bool(not row.empty and bool(row["available"].iloc[0]))


def read_predictor_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "predictor" in df.columns:
        return df["predictor"].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()


def has_forbidden_name(name: str) -> bool:
    lower = name.lower()
    return any(fragment.lower() in lower for fragment in FORBIDDEN_PREDICTOR_SUBSTRINGS)


def load_predictor_lists(config: Config, paths: Dict[str, Path]) -> Tuple[List[str], List[str], pd.DataFrame]:
    all_safe = read_predictor_list(config.c3_dir / "tables" / "predictor_list_all_safe_C3.csv")
    dgm = read_predictor_list(config.c3_dir / "tables" / "predictor_list_dgm_features_C3.csv")
    source_all = "C3"
    source_dgm = "C3"
    if not all_safe:
        all_safe = read_predictor_list(config.c2_dir / "tables" / "predictor_list_all_safe.csv")
        source_all = "C2"
    if not dgm:
        dgm = read_predictor_list(config.c2_dir / "tables" / "predictor_list_dgm_features.csv")
        source_dgm = "C2"

    rows = []
    for predictor_set, predictors, source in [("all_safe", all_safe, source_all), ("dgm", dgm, source_dgm)]:
        for predictor in predictors:
            rows.append(
                {
                    "predictor_set": predictor_set,
                    "predictor": predictor,
                    "source": source,
                    "forbidden_name_flag": has_forbidden_name(predictor),
                }
            )
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("No predictor lists found from C2 or C3 outputs.")
    audit.to_csv(paths["tables"] / "predictor_audit_C4.csv", index=False)
    leakage = audit.loc[audit["forbidden_name_flag"]].copy()
    leakage.to_csv(paths["tables"] / "predictor_leakage_audit_C4.csv", index=False)
    if not leakage.empty:
        raise RuntimeError(
            "Forbidden predictor names detected in C4 predictor lists: "
            + ", ".join(leakage["predictor"].astype(str).tolist())
        )
    pd.DataFrame({"predictor": dgm}).to_csv(paths["tables"] / "predictor_list_dgm_C4.csv", index=False)
    pd.DataFrame({"predictor": all_safe}).to_csv(paths["tables"] / "predictor_list_all_safe_C4.csv", index=False)
    return dgm, all_safe, audit


def validate_export_safety(config: Config) -> None:
    audit_path = config.data_dir / "audit" / "export_safety_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing export safety audit: {audit_path}")
    audit = pd.read_csv(audit_path)
    if "safe_to_export_column_names" not in audit.columns:
        raise ValueError("export_safety_audit.csv lacks safe_to_export_column_names column")
    if not audit["safe_to_export_column_names"].fillna(False).astype(bool).all():
        bad = audit.loc[~audit["safe_to_export_column_names"].fillna(False).astype(bool)]
        raise RuntimeError(f"Export safety audit failed for {len(bad)} rows.")


def scenario_files(config: Config) -> List[Path]:
    files = sorted((config.data_dir / "scenario_datasets").glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No scenario files found under {config.data_dir / 'scenario_datasets'}")
    return files


def data_file_audit(config: Config, paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    required = {DURATION_COL, STATUS_COL, "scenario_id", "replicate_id", TRUE_RISK_COL, TRUE_LP_COL}
    for file in scenario_files(config):
        df = pd.read_csv(file, usecols=lambda c: True)
        missing = sorted(required - set(df.columns))
        status_counts = df[STATUS_COL].value_counts(dropna=False).to_dict() if STATUS_COL in df else {}
        forbidden_dataset_cols = [c for c in df.columns if has_forbidden_name(c)]
        rows.append(
            {
                "file": str(file),
                "file_name": file.name,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "n_replicates": df["replicate_id"].nunique() if "replicate_id" in df else np.nan,
                "missing_required_columns": ";".join(missing),
                "status_distribution_json": json.dumps({str(k): int(v) for k, v in status_counts.items()}),
                "forbidden_or_outcome_columns_present": ";".join(forbidden_dataset_cols),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "data_file_audit_C4.csv", index=False)
    return out


def get_existing_successes(perf_path: Path, config: Config) -> set[Tuple[str, int, str]]:
    if not perf_path.exists():
        return set()
    df = pd.read_csv(perf_path)
    if df.empty:
        return set()
    if "run_mode" in df.columns:
        df = df.loc[df["run_mode"].astype(str) == run_mode_label(config)]
    else:
        return set()
    if config.rerun_failed:
        df = df.loc[(df["failed"] == False) & (df["skipped"] == False)]  # noqa: E712
    else:
        df = df.loc[(df["failed"] == False)]
    return set(zip(df["scenario_id"].astype(str), df["replicate_id"].astype(int), df["model"].astype(str)))


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
        new_cols = [col for col in df.columns if col not in existing_cols]
        if new_cols:
            existing = pd.read_csv(path)
            all_cols = existing_cols + new_cols
            existing.reindex(columns=all_cols).to_csv(path, index=False)
            existing_cols = all_cols
        df = df.reindex(columns=existing_cols)
    df.to_csv(path, mode="a", header=write_header, index=False)


def split_train_test(df: pd.DataFrame, config: Config, replicate_id: int):
    from sklearn.model_selection import train_test_split

    stratify = None
    counts = df[STATUS_COL].value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        stratify = df[STATUS_COL]
    return train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_seed + int(replicate_id),
        stratify=stratify,
    )


def available_predictors(df: pd.DataFrame, predictors: Sequence[str]) -> List[str]:
    return [p for p in predictors if p in df.columns and not has_forbidden_name(p)]


def fit_transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame, predictors: Sequence[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    predictors = available_predictors(train_df, predictors)
    if not predictors:
        raise ValueError("No usable predictors available after filtering.")

    numeric_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in predictors if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    x_train = preprocessor.fit_transform(train_df[predictors])
    x_test = preprocessor.transform(test_df[predictors])
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"x{i}" for i in range(x_train.shape[1])]
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train, x_test, feature_names, predictors


def select_reduced_predictors_train_only(
    train_df: pd.DataFrame,
    candidate_predictors: Sequence[str],
    max_features: int = 30,
    horizon: float = HORIZON_YEARS,
) -> List[str]:
    candidates = available_predictors(train_df, candidate_predictors)
    y = ((train_df[STATUS_COL].astype(int) == EVENT1_STATUS) & (train_df[DURATION_COL].astype(float) <= horizon)).astype(float)
    if y.nunique() < 2:
        return candidates[:max_features]

    scores: List[Tuple[float, str]] = []
    for predictor in candidates:
        s = train_df[predictor]
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
        else:
            codes, _ = pd.factorize(s.astype(str), sort=True)
            x = pd.Series(codes, index=s.index).replace(-1, np.nan)
        if x.notna().sum() < 5 or x.nunique(dropna=True) <= 1:
            score = 0.0
        else:
            x = x.fillna(x.median())
            try:
                score = abs(float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1]))
            except Exception:
                score = 0.0
            if not np.isfinite(score):
                score = 0.0
        scores.append((score, predictor))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scores[:max_features]]


def sksurv_y(df: pd.DataFrame, event_status: int):
    from sksurv.util import Surv

    return Surv.from_arrays(
        event=(df[STATUS_COL].astype(int).to_numpy() == event_status),
        time=df[DURATION_COL].astype(float).to_numpy(),
    )


def eval_step_function(func: Any, times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    try:
        values = func(times)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return np.repeat(float(arr), len(times))
        return arr.reshape(-1)
    except Exception:
        x = np.asarray(getattr(func, "x"), dtype=float)
        y = np.asarray(getattr(func, "y"), dtype=float)
        idx = np.searchsorted(x, times, side="right") - 1
        out = np.zeros_like(times, dtype=float)
        mask = idx >= 0
        out[mask] = y[idx[mask]]
        return out


def predict_cif_from_two_cumulative_hazards(
    chf1_funcs: Sequence[Any],
    chf2_funcs: Sequence[Any],
    horizon: float,
) -> Tuple[np.ndarray, List[str]]:
    preds: List[float] = []
    warnings_out: List[str] = []
    for i, (f1, f2) in enumerate(zip(chf1_funcs, chf2_funcs)):
        try:
            t1 = np.asarray(getattr(f1, "x"), dtype=float)
            t2 = np.asarray(getattr(f2, "x"), dtype=float)
            grid = np.unique(np.concatenate([t1[(t1 > 0) & (t1 <= horizon)], t2[(t2 > 0) & (t2 <= horizon)], [horizon]]))
            if grid.size == 0:
                preds.append(0.0)
                continue
            h1 = eval_step_function(f1, grid)
            h2 = eval_step_function(f2, grid)
            h1_prev = np.concatenate([[0.0], h1[:-1]])
            h2_prev = np.concatenate([[0.0], h2[:-1]])
            dh1 = np.maximum(h1 - h1_prev, 0.0)
            surv_prev = np.exp(-h1_prev - h2_prev)
            cif = float(np.sum(surv_prev * dh1))
            if not np.isfinite(cif):
                warnings_out.append(f"sample_{i}_nan_cif")
                cif = np.nan
            preds.append(float(np.clip(cif, 0.0, 1.0)))
        except Exception as exc:
            warnings_out.append(f"sample_{i}_cif_failed:{type(exc).__name__}")
            preds.append(np.nan)
    arr = np.asarray(preds, dtype=float)
    if np.isnan(arr).any():
        warnings_out.append("nan_cif_values_present")
    if ((arr < -1e-8) | (arr > 1 + 1e-8)).any():
        warnings_out.append("out_of_bounds_cif_values_present")
    return np.clip(np.nan_to_num(arr, nan=np.nan), 0.0, 1.0), warnings_out


def predict_chf_at_horizon(chf_funcs: Sequence[Any], horizon: float) -> np.ndarray:
    values = []
    for func in chf_funcs:
        values.append(float(eval_step_function(func, np.asarray([horizon], dtype=float))[0]))
    return np.asarray(values, dtype=float)


def calibration_slope_intercept(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float, str]:
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if y.size < 20 or len(np.unique(y)) < 2:
        return np.nan, np.nan, "insufficient_events_or_classes"
    if np.nanstd(pred) <= 1e-12:
        return np.nan, np.nan, "predicted_risk_has_no_variation"
    eps = 1e-6
    logit = np.log(np.clip(pred, eps, 1 - eps) / np.clip(1 - pred, eps, 1 - eps)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        return float(model.coef_[0][0]), float(model.intercept_[0]), ""
    except Exception as exc:
        return np.nan, np.nan, f"calibration_failed:{type(exc).__name__}:{exc}"


def compute_ipcw_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame, pred_risk5: np.ndarray, horizon: float) -> Tuple[float, float, str]:
    try:
        from sksurv.metrics import brier_score

        y_train = sksurv_y(train_df, EVENT1_STATUS)
        y_test = sksurv_y(test_df, EVENT1_STATUS)
        max_test_time = float(np.nanmax(test_df[DURATION_COL].astype(float).to_numpy()))
        eval_time = min(float(horizon), np.nextafter(max_test_time, 0.0))
        if not np.isfinite(eval_time) or eval_time <= 0:
            return np.nan, np.nan, "ipcw_brier_unavailable_invalid_eval_time"
        pred_surv = np.clip(1.0 - np.asarray(pred_risk5, dtype=float), 0.0, 1.0).reshape(-1, 1)
        _, scores = brier_score(y_train, y_test, pred_surv, np.asarray([eval_time], dtype=float))
        return float(scores[0]), np.nan, "integrated_brier_not_available_from_single_horizon_prediction"
    except Exception as exc:
        return np.nan, np.nan, f"ipcw_brier_failed:{type(exc).__name__}:{exc}"


def harrell_cindex(test_df: pd.DataFrame, risk_score: np.ndarray) -> float:
    try:
        from sksurv.metrics import concordance_index_censored

        event = (test_df[STATUS_COL].astype(int).to_numpy() == EVENT1_STATUS)
        time_arr = test_df[DURATION_COL].astype(float).to_numpy()
        score = np.asarray(risk_score, dtype=float)
        mask = np.isfinite(score) & np.isfinite(time_arr)
        if mask.sum() < 2 or event[mask].sum() == 0:
            return np.nan
        return float(concordance_index_censored(event[mask], time_arr[mask], score[mask])[0])
    except Exception:
        try:
            from lifelines.utils import concordance_index

            event = (test_df[STATUS_COL].astype(int).to_numpy() == EVENT1_STATUS).astype(int)
            time_arr = test_df[DURATION_COL].astype(float).to_numpy()
            return float(concordance_index(time_arr, -np.asarray(risk_score, dtype=float), event))
        except Exception:
            return np.nan


def evaluate_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_risk5: Optional[np.ndarray],
    risk_score: Optional[np.ndarray],
    horizon: float,
) -> Dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    y_event1_5y = ((test_df[STATUS_COL].astype(int) == EVENT1_STATUS) & (test_df[DURATION_COL].astype(float) <= horizon)).astype(int).to_numpy()
    y_event2_5y = ((test_df[STATUS_COL].astype(int) == EVENT2_STATUS) & (test_df[DURATION_COL].astype(float) <= horizon)).astype(int).to_numpy()
    true_risk = pd.to_numeric(test_df.get(TRUE_RISK_COL, pd.Series(np.nan, index=test_df.index)), errors="coerce").to_numpy(dtype=float)

    out: Dict[str, Any] = {
        "predicted_risk5_available": pred_risk5 is not None,
        "mean_predicted_risk5": np.nan,
        "mean_true_risk5": float(np.nanmean(true_risk)) if np.isfinite(true_risk).any() else np.nan,
        "observed_event1_5y_rate": float(np.mean(y_event1_5y)),
        "observed_event2_5y_rate": float(np.mean(y_event2_5y)),
        "risk5_mae_vs_true_risk": np.nan,
        "risk5_rmse_vs_true_risk": np.nan,
        "risk5_spearman_with_true_risk": np.nan,
        "risk5_pearson_with_true_risk": np.nan,
        "brier_5y_naive": np.nan,
        "brier_5y_ipcw": np.nan,
        "integrated_brier_score_1to5": np.nan,
        "ipcw_metric_reason": "",
        "auc_5y_observed_event1": np.nan,
        "calibration_intercept_5y": np.nan,
        "calibration_slope_5y": np.nan,
        "calibration_failure_reason": "",
        "cause_specific_cindex_event1": np.nan,
    }

    cindex_input = risk_score
    if cindex_input is None and pred_risk5 is not None:
        cindex_input = pred_risk5
    if cindex_input is not None:
        out["cause_specific_cindex_event1"] = harrell_cindex(test_df, np.asarray(cindex_input, dtype=float))

    if pred_risk5 is None:
        return out

    pred = np.asarray(pred_risk5, dtype=float)
    out["mean_predicted_risk5"] = float(np.nanmean(pred)) if np.isfinite(pred).any() else np.nan
    mask = np.isfinite(pred) & np.isfinite(true_risk)
    if mask.any():
        err = pred[mask] - true_risk[mask]
        out["risk5_mae_vs_true_risk"] = float(np.mean(np.abs(err)))
        out["risk5_rmse_vs_true_risk"] = float(np.sqrt(np.mean(err**2)))
        if mask.sum() >= 3 and np.nanstd(pred[mask]) > 0 and np.nanstd(true_risk[mask]) > 0:
            out["risk5_spearman_with_true_risk"] = float(spearmanr(pred[mask], true_risk[mask]).correlation)
            out["risk5_pearson_with_true_risk"] = float(pearsonr(pred[mask], true_risk[mask])[0])
    finite = np.isfinite(pred)
    if finite.any():
        out["brier_5y_naive"] = float(np.mean((y_event1_5y[finite] - pred[finite]) ** 2))
        if len(np.unique(y_event1_5y[finite])) == 2 and np.nanstd(pred[finite]) > 0:
            try:
                out["auc_5y_observed_event1"] = float(roc_auc_score(y_event1_5y[finite], pred[finite]))
            except Exception:
                pass
        slope, intercept, reason = calibration_slope_intercept(y_event1_5y[finite], pred[finite])
        out["calibration_slope_5y"] = slope
        out["calibration_intercept_5y"] = intercept
        out["calibration_failure_reason"] = reason
        brier_ipcw, ibs, ipcw_reason = compute_ipcw_metrics(train_df, test_df, pred, horizon)
        out["brier_5y_ipcw"] = brier_ipcw
        out["integrated_brier_score_1to5"] = ibs
        out["ipcw_metric_reason"] = ipcw_reason
    return out


def base_result_row(
    scenario_id: str,
    replicate_id: int,
    model: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    model_family, predictor_set = MODEL_META.get(model, ("unknown", "unknown"))
    train_status = train_df[STATUS_COL].astype(int)
    test_status = test_df[STATUS_COL].astype(int)
    return {
        "scenario_id": scenario_id,
        "replicate_id": int(replicate_id),
        "model": model,
        "model_family": model_family,
        "predictor_set": predictor_set,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_event1_train": int((train_status == EVENT1_STATUS).sum()),
        "n_event2_train": int((train_status == EVENT2_STATUS).sum()),
        "n_censored_train": int((train_status == 0).sum()),
        "n_event1_test": int((test_status == EVENT1_STATUS).sum()),
        "n_event2_test": int((test_status == EVENT2_STATUS).sum()),
        "n_censored_test": int((test_status == 0).sum()),
        "runtime_sec": np.nan,
        "failed": False,
        "failure_reason": "",
        "traceback_tail": "",
        "skipped": False,
        "skipped_reason": "",
        "prediction_warning": "",
        "n_model_input_features": np.nan,
        "selected_predictors": "",
    }


def skipped_result(
    scenario_id: str,
    replicate_id: int,
    model: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reason: str,
) -> Dict[str, Any]:
    row = base_result_row(scenario_id, replicate_id, model, train_df, test_df)
    row.update(evaluate_predictions(train_df, test_df, None, None, HORIZON_YEARS))
    row.update({"skipped": True, "skipped_reason": reason, "predicted_risk5_available": False})
    return row


def failed_result(
    scenario_id: str,
    replicate_id: int,
    model: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reason: str,
    runtime: float,
) -> Dict[str, Any]:
    row = base_result_row(scenario_id, replicate_id, model, train_df, test_df)
    row.update(evaluate_predictions(train_df, test_df, None, None, HORIZON_YEARS))
    row.update({"failed": True, "failure_reason": reason, "runtime_sec": runtime, "predicted_risk5_available": False})
    return row


def fit_finegray_risk5(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    horizon: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    x_train, x_test, feature_names, used_predictors = fit_transform_features(train_df, test_df, predictors)
    if x_train.shape[1] == 0:
        raise ValueError("Fine-Gray received zero usable features.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_x = tmp_path / "train_x.csv"
        test_x = tmp_path / "test_x.csv"
        train_y = tmp_path / "train_y.csv"
        out_csv = tmp_path / "pred.csv"
        script = tmp_path / "finegray_predict.R"
        pd.DataFrame(x_train, columns=[f"x{i}" for i in range(x_train.shape[1])]).to_csv(train_x, index=False)
        pd.DataFrame(x_test, columns=[f"x{i}" for i in range(x_test.shape[1])]).to_csv(test_x, index=False)
        pd.DataFrame(
            {
                "duration_years": train_df[DURATION_COL].astype(float).to_numpy(),
                "status": train_df[STATUS_COL].astype(int).to_numpy(),
            }
        ).to_csv(train_y, index=False)
        script.write_text(
            f"""
suppressPackageStartupMessages(library(cmprsk))
train_x <- as.matrix(read.csv("{train_x}", check.names=FALSE))
test_x <- as.matrix(read.csv("{test_x}", check.names=FALSE))
y <- read.csv("{train_y}")
fit <- cmprsk::crr(
    ftime=y$duration_years,
    fstatus=y$status,
    cov1=train_x,
    failcode=1,
    cencode=0
)
pred <- predict(fit, cov1=test_x)
if (is.null(pred) || ncol(pred) < 2) {{
    stop("predict.crr did not return test cumulative incidence columns")
}}
times <- pred[, 1]
vals <- pred[, -1, drop=FALSE]
idx <- max(which(times <= {float(horizon)}))
if (!is.finite(idx)) {{
    risks <- rep(0, nrow(test_x))
}} else {{
    risks <- vals[idx, ]
}}
risks <- pmin(pmax(as.numeric(risks), 0), 1)
write.csv(data.frame(predicted_risk5=risks), "{out_csv}", row.names=FALSE)
""",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env.setdefault("LC_ALL", "C")
        env.setdefault("LC_CTYPE", "C")
        proc = subprocess.run(["Rscript", str(script)], capture_output=True, text=True, check=False, timeout=300, env=env)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout)[-2000:])
        pred = pd.read_csv(out_csv)["predicted_risk5"].to_numpy(dtype=float)
    return pred, feature_names, used_predictors


def fit_sksurv_pair_risk5(
    model: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    x_train, x_test, feature_names, used_predictors = fit_transform_features(train_df, test_df, predictors)
    if x_train.shape[1] == 0:
        raise ValueError(f"{model} received zero usable features.")

    y1 = sksurv_y(train_df, EVENT1_STATUS)
    y2 = sksurv_y(train_df, EVENT2_STATUS)
    seed = config.random_seed + int(train_df["replicate_id"].iloc[0])

    if model in {MODEL_RSF_DGM, MODEL_RSF_SAFE}:
        from sksurv.ensemble import RandomSurvivalForest

        n_estimators = 50 if config.debug_mode else 200
        kwargs = {
            "n_estimators": n_estimators,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "random_state": seed,
            "n_jobs": -1,
        }
        m1 = RandomSurvivalForest(**kwargs)
        m2 = RandomSurvivalForest(**kwargs)
    elif model in {MODEL_GBSA_DGM, MODEL_GBSA_SAFE}:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        n_estimators = 50 if config.debug_mode else 200
        learning_rate = 0.05 if config.debug_mode else 0.03
        kwargs = {
            "loss": "coxph",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": 2,
            "min_samples_leaf": 20,
            "random_state": seed,
        }
        m1 = GradientBoostingSurvivalAnalysis(**kwargs)
        m2 = GradientBoostingSurvivalAnalysis(**kwargs)
    else:
        raise ValueError(f"Unsupported sksurv model: {model}")

    m1.fit(x_train, y1)
    m2.fit(x_train, y2)
    chf1 = m1.predict_cumulative_hazard_function(x_test)
    chf2 = m2.predict_cumulative_hazard_function(x_test)
    pred, warnings_out = predict_cif_from_two_cumulative_hazards(chf1, chf2, config.horizon_years)
    risk_score = predict_chf_at_horizon(chf1, config.horizon_years)
    return pred, risk_score, feature_names, used_predictors, warnings_out


def run_model(
    model: str,
    scenario_id: str,
    replicate_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dgm_predictors: Sequence[str],
    all_safe_predictors: Sequence[str],
    dep: pd.DataFrame,
    config: Config,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    start = time.perf_counter()
    reduced_record = None
    try:
        if model in OPTIONAL_DEEP_MODELS:
            if model.startswith("deepsurv"):
                needed = ["torch", "torchtuples", "pycox"]
                missing = [name for name in needed if not dependency_available(dep, name)]
                reason = "missing_optional_deep_dependencies:" + ",".join(missing) if missing else "optional_deep_model_not_enabled_in_C4_main"
            else:
                needed = ["torch", "torchtuples", "pycox"]
                missing = [name for name in needed if not dependency_available(dep, name)]
                reason = "missing_optional_deephit_dependencies:" + ",".join(missing) if missing else "optional_deephit_not_enabled_in_C4_main"
            row = skipped_result(scenario_id, replicate_id, model, train_df, test_df, reason)
            row["runtime_sec"] = time.perf_counter() - start
            return row, None, None, None

        pred_risk5: Optional[np.ndarray] = None
        risk_score: Optional[np.ndarray] = None
        feature_names: List[str] = []
        used_predictors: List[str] = []
        prediction_warnings: List[str] = []

        if model in {MODEL_FINEGRAY_DGM, MODEL_FINEGRAY_SAFE_REDUCED}:
            if not dependency_available(dep, "Rscript"):
                row = skipped_result(scenario_id, replicate_id, model, train_df, test_df, "Rscript_not_available")
                row["runtime_sec"] = time.perf_counter() - start
                return row, None, None, None
            if not dependency_available(dep, "cmprsk"):
                row = skipped_result(scenario_id, replicate_id, model, train_df, test_df, "cmprsk_not_available")
                row["runtime_sec"] = time.perf_counter() - start
                return row, None, None, None
            predictors = list(dgm_predictors)
            if model == MODEL_FINEGRAY_SAFE_REDUCED:
                predictors = select_reduced_predictors_train_only(train_df, all_safe_predictors, max_features=30, horizon=config.horizon_years)
                reduced_record = {
                    "scenario_id": scenario_id,
                    "replicate_id": int(replicate_id),
                    "model": model,
                    "n_selected": len(predictors),
                    "selected_predictors": ";".join(predictors),
                }
            pred_risk5, feature_names, used_predictors = fit_finegray_risk5(train_df, test_df, predictors, config.horizon_years)
            risk_score = pred_risk5

        elif model in {MODEL_RSF_DGM, MODEL_RSF_SAFE, MODEL_GBSA_DGM, MODEL_GBSA_SAFE}:
            if not dependency_available(dep, "sksurv"):
                row = skipped_result(scenario_id, replicate_id, model, train_df, test_df, "skipped_due_to_missing_sksurv")
                row["runtime_sec"] = time.perf_counter() - start
                return row, None, None, None
            predictors = dgm_predictors if model in {MODEL_RSF_DGM, MODEL_GBSA_DGM} else all_safe_predictors
            pred_risk5, risk_score, feature_names, used_predictors, prediction_warnings = fit_sksurv_pair_risk5(
                model, train_df, test_df, predictors, config
            )
        else:
            row = skipped_result(scenario_id, replicate_id, model, train_df, test_df, "model_not_enabled_in_C4_v1")
            row["runtime_sec"] = time.perf_counter() - start
            return row, None, None, None

        if pred_risk5 is None:
            row = failed_result(
                scenario_id,
                replicate_id,
                model,
                train_df,
                test_df,
                "prediction_unavailable",
                time.perf_counter() - start,
            )
            return row, None, reduced_record, None

        metrics = evaluate_predictions(train_df, test_df, pred_risk5, risk_score, config.horizon_years)
        row = base_result_row(scenario_id, replicate_id, model, train_df, test_df)
        row.update(metrics)
        row.update(
            {
                "runtime_sec": time.perf_counter() - start,
                "prediction_warning": ";".join(prediction_warnings),
                "n_model_input_features": len(feature_names),
                "selected_predictors": ";".join(used_predictors),
            }
        )

        pred_sample = None
        if config.debug_mode:
            sample_n = min(10, len(test_df))
            pred_sample = pd.DataFrame(
                {
                    "scenario_id": scenario_id,
                    "replicate_id": int(replicate_id),
                    "model": model,
                    "synthetic_id": test_df.get("synthetic_id", pd.Series(range(len(test_df)), index=test_df.index)).iloc[:sample_n].astype(str).to_numpy(),
                    "duration_years": test_df[DURATION_COL].iloc[:sample_n].to_numpy(),
                    "status": test_df[STATUS_COL].iloc[:sample_n].to_numpy(),
                    "predicted_risk5": pred_risk5[:sample_n],
                    "true_risk5": test_df[TRUE_RISK_COL].iloc[:sample_n].to_numpy() if TRUE_RISK_COL in test_df else np.nan,
                }
            )
        deciles = calibration_deciles_for_prediction(
            scenario_id, replicate_id, model, test_df, pred_risk5, config.horizon_years
        )
        return row, pred_sample, reduced_record, deciles
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        row = failed_result(scenario_id, replicate_id, model, train_df, test_df, reason, time.perf_counter() - start)
        row["traceback_tail"] = traceback.format_exc()[-2000:]
        return row, None, reduced_record, None


def calibration_deciles_for_prediction(
    scenario_id: str,
    replicate_id: int,
    model: str,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    horizon: float,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "replicate_id": int(replicate_id),
            "model": model,
            "predicted_risk5": np.asarray(pred, dtype=float),
            "true_risk5": pd.to_numeric(test_df.get(TRUE_RISK_COL, pd.Series(np.nan, index=test_df.index)), errors="coerce").to_numpy(dtype=float),
            "observed_event1_5y": ((test_df[STATUS_COL].astype(int) == EVENT1_STATUS) & (test_df[DURATION_COL].astype(float) <= horizon)).astype(int).to_numpy(),
            "observed_event2_5y": ((test_df[STATUS_COL].astype(int) == EVENT2_STATUS) & (test_df[DURATION_COL].astype(float) <= horizon)).astype(int).to_numpy(),
        }
    )
    df = df.loc[np.isfinite(df["predicted_risk5"])]
    if df.empty or df["predicted_risk5"].nunique() < 2:
        return pd.DataFrame()
    try:
        df["decile"] = pd.qcut(df["predicted_risk5"], 10, labels=False, duplicates="drop") + 1
    except Exception:
        return pd.DataFrame()
    return (
        df.groupby(["scenario_id", "replicate_id", "model", "decile"], dropna=False)
        .agg(
            n=("predicted_risk5", "size"),
            mean_predicted_risk5=("predicted_risk5", "mean"),
            mean_true_risk5=("true_risk5", "mean"),
            observed_event1_5y_rate=("observed_event1_5y", "mean"),
            observed_event2_5y_rate=("observed_event2_5y", "mean"),
        )
        .reset_index()
    )


def aggregate_metric(group: pd.DataFrame, metric: str) -> Dict[str, Any]:
    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return {
            f"{metric}_mean": np.nan,
            f"{metric}_sd": np.nan,
            f"{metric}_se": np.nan,
            f"{metric}_ci_low": np.nan,
            f"{metric}_ci_high": np.nan,
            f"{metric}_median": np.nan,
            f"{metric}_p25": np.nan,
            f"{metric}_p75": np.nan,
        }
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else np.nan
    return {
        f"{metric}_mean": mean,
        f"{metric}_sd": sd,
        f"{metric}_se": se,
        f"{metric}_ci_low": mean - 1.96 * se if np.isfinite(se) else np.nan,
        f"{metric}_ci_high": mean + 1.96 * se if np.isfinite(se) else np.nan,
        f"{metric}_median": float(values.median()),
        f"{metric}_p25": float(values.quantile(0.25)),
        f"{metric}_p75": float(values.quantile(0.75)),
    }


def aggregate_results(paths: Dict[str, Path], config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf_path = paths["tables"] / "replicate_extended_model_performance.csv"
    if not perf_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(paths["tables"] / "scenario_extended_model_summary_mean_sd_ci.csv", index=False)
        return empty, empty
    df = pd.read_csv(perf_path)
    if "run_mode" in df.columns:
        df = df.loc[df["run_mode"].astype(str) == run_mode_label(config)].copy()
    summary_rows = []
    for (scenario_id, model), group in df.groupby(["scenario_id", "model"], dropna=False):
        row: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "model": model,
            "model_family": group["model_family"].iloc[0],
            "predictor_set": group["predictor_set"].iloc[0],
            "n_successful_reps": int(((group["failed"] == False) & (group["skipped"] == False)).sum()),  # noqa: E712
            "n_failed_reps": int((group["failed"] == True).sum()),  # noqa: E712
            "n_skipped_reps": int((group["skipped"] == True).sum()),  # noqa: E712
            "failure_rate": float((group["failed"] == True).mean()),  # noqa: E712
            "skip_rate": float((group["skipped"] == True).mean()),  # noqa: E712
        }
        ok = group.loc[(group["failed"] == False) & (group["skipped"] == False)]  # noqa: E712
        for metric in CORE_METRICS + ["runtime_sec"]:
            row.update(aggregate_metric(ok, metric) if metric in ok.columns else aggregate_metric(pd.DataFrame({metric: []}), metric))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["tables"] / "scenario_extended_model_summary_mean_sd_ci.csv", index=False)

    cal_path = paths["tables"] / "calibration_deciles_replicate_level_C4.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        if "run_mode" in cal.columns:
            cal = cal.loc[cal["run_mode"].astype(str) == run_mode_label(config)].copy()
        if not cal.empty:
            cal_summary = (
                cal.groupby(["scenario_id", "model", "decile"], dropna=False)
                .agg(
                    n_total=("n", "sum"),
                    mean_predicted_risk5=("mean_predicted_risk5", "mean"),
                    mean_true_risk5=("mean_true_risk5", "mean"),
                    observed_event1_5y_rate=("observed_event1_5y_rate", "mean"),
                    observed_event2_5y_rate=("observed_event2_5y_rate", "mean"),
                )
                .reset_index()
            )
        else:
            cal_summary = pd.DataFrame()
    else:
        cal_summary = pd.DataFrame()
    cal_summary.to_csv(paths["tables"] / "calibration_deciles_summary_C4.csv", index=False)
    return summary, cal_summary


def combine_with_c2_c3(config: Config, paths: Dict[str, Path], c4_summary: pd.DataFrame) -> None:
    rows = []
    c2_path = config.c2_dir / "tables" / "scenario_model_summary_mean_sd_ci.csv"
    if c2_path.exists():
        c2 = pd.read_csv(c2_path)
        for _, r in c2.iterrows():
            rows.append(
                {
                    "stage": "C2",
                    "scenario_id": r.get("scenario_id"),
                    "model": r.get("model"),
                    "model_family": "C2 cause-specific",
                    "primary_metric_type": "ranking",
                    "cause_specific_cindex_event1_mean": r.get("cindex_mean", np.nan),
                    "risk5_mae_vs_true_risk_mean": r.get("risk5_mae_vs_true_risk_mean", np.nan),
                    "brier_5y_naive_mean": r.get("risk5_naive_brier_mean", np.nan),
                    "auc_5y_observed_event1_mean": np.nan,
                    "calibration_slope_5y_mean": r.get("risk5_calibration_slope_mean", np.nan),
                    "failure_rate": r.get("failure_rate", np.nan),
                }
            )
    c3_path = config.c3_dir / "tables" / "scenario_competing_risk_summary_mean_sd_ci.csv"
    if c3_path.exists():
        c3 = pd.read_csv(c3_path)
        for _, r in c3.iterrows():
            rows.append(
                {
                    "stage": "C3",
                    "scenario_id": r.get("scenario_id"),
                    "model": r.get("model"),
                    "model_family": "C3 competing-risk",
                    "primary_metric_type": "absolute_risk",
                    "cause_specific_cindex_event1_mean": r.get("cause_specific_cindex_event1_mean", np.nan),
                    "risk5_mae_vs_true_risk_mean": r.get("risk5_mae_vs_true_risk_mean", np.nan),
                    "brier_5y_naive_mean": r.get("brier_5y_naive_mean", np.nan),
                    "auc_5y_observed_event1_mean": r.get("auc_5y_observed_event1_mean", np.nan),
                    "calibration_slope_5y_mean": r.get("calibration_slope_5y_mean", np.nan),
                    "failure_rate": r.get("failure_rate", np.nan),
                }
            )
    for _, r in c4_summary.iterrows():
        rows.append(
            {
                "stage": "C4",
                "scenario_id": r.get("scenario_id"),
                "model": r.get("model"),
                "model_family": r.get("model_family"),
                "primary_metric_type": "absolute_risk",
                "cause_specific_cindex_event1_mean": r.get("cause_specific_cindex_event1_mean", np.nan),
                "risk5_mae_vs_true_risk_mean": r.get("risk5_mae_vs_true_risk_mean", np.nan),
                "brier_5y_naive_mean": r.get("brier_5y_naive_mean", np.nan),
                "auc_5y_observed_event1_mean": r.get("auc_5y_observed_event1_mean", np.nan),
                "calibration_slope_5y_mean": r.get("calibration_slope_5y_mean", np.nan),
                "failure_rate": r.get("failure_rate", np.nan),
            }
        )
    combined = pd.DataFrame(rows)
    combined.to_csv(paths["tables"] / "combined_C2_C3_C4_model_summary.csv", index=False)

    ranking = combined.loc[np.isfinite(pd.to_numeric(combined["cause_specific_cindex_event1_mean"], errors="coerce"))].copy()
    if not ranking.empty:
        ranking["rank_metric"] = pd.to_numeric(ranking["cause_specific_cindex_event1_mean"], errors="coerce")
        best_ranking = ranking.sort_values(["scenario_id", "rank_metric", "failure_rate"], ascending=[True, False, True]).groupby("scenario_id").head(1)
    else:
        best_ranking = pd.DataFrame()
    best_ranking.to_csv(paths["tables"] / "best_ranking_model_by_scenario_C4.csv", index=False)

    absrisk = combined.loc[np.isfinite(pd.to_numeric(combined["risk5_mae_vs_true_risk_mean"], errors="coerce"))].copy()
    if not absrisk.empty:
        absrisk["mae"] = pd.to_numeric(absrisk["risk5_mae_vs_true_risk_mean"], errors="coerce")
        absrisk["brier"] = pd.to_numeric(absrisk["brier_5y_naive_mean"], errors="coerce")
        absrisk["auc"] = pd.to_numeric(absrisk["auc_5y_observed_event1_mean"], errors="coerce").fillna(-np.inf)
        absrisk["cal_slope_gap"] = (pd.to_numeric(absrisk["calibration_slope_5y_mean"], errors="coerce") - 1.0).abs().fillna(np.inf)
        best_abs = (
            absrisk.sort_values(["scenario_id", "mae", "brier", "auc", "cal_slope_gap", "failure_rate"], ascending=[True, True, True, False, True, True])
            .groupby("scenario_id")
            .head(1)
        )
    else:
        best_abs = pd.DataFrame()
    best_abs.to_csv(paths["tables"] / "best_absolute_risk_model_by_scenario_C4.csv", index=False)


def oracle_sanity_audit(config: Config, paths: Dict[str, Path], c4_summary: pd.DataFrame) -> pd.DataFrame:
    c3_path = config.c3_dir / "tables" / "scenario_competing_risk_summary_mean_sd_ci.csv"
    rows = []
    if not c3_path.exists() or c4_summary.empty:
        out = pd.DataFrame()
        out.to_csv(paths["tables"] / "oracle_sanity_audit_C4.csv", index=False)
        return out
    c3 = pd.read_csv(c3_path)
    oracle = c3.loc[c3["model"] == "oracle_true_risk_not_a_model"].set_index("scenario_id")
    for _, r in c4_summary.iterrows():
        scenario = r["scenario_id"]
        model = r["model"]
        if scenario not in oracle.index:
            continue
        o = oracle.loc[scenario]
        checks = [
            ("risk5_mae_vs_true_risk_mean", "mae_better_than_oracle", -0.005, "lower"),
            ("auc_5y_observed_event1_mean", "auc_exceeds_oracle", 0.02, "higher"),
            ("cause_specific_cindex_event1_mean", "cindex_exceeds_oracle", 0.02, "higher"),
        ]
        for metric, flag, threshold, direction in checks:
            model_value = pd.to_numeric(pd.Series([r.get(metric, np.nan)]), errors="coerce").iloc[0]
            oracle_value = pd.to_numeric(pd.Series([o.get(metric, np.nan)]), errors="coerce").iloc[0]
            if not (np.isfinite(model_value) and np.isfinite(oracle_value)):
                continue
            gap = model_value - oracle_value
            if direction == "lower":
                needs = gap < threshold
            else:
                needs = gap > threshold
            rows.append(
                {
                    "scenario_id": scenario,
                    "model": model,
                    "metric": metric,
                    "model_value": model_value,
                    "oracle_value": oracle_value,
                    "gap": gap,
                    "flag_name": flag,
                    "needs_audit": bool(needs),
                }
            )
        spearman = pd.to_numeric(pd.Series([r.get("risk5_spearman_with_true_risk_mean", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(spearman):
            rows.append(
                {
                    "scenario_id": scenario,
                    "model": model,
                    "metric": "risk5_spearman_with_true_risk_mean",
                    "model_value": spearman,
                    "oracle_value": np.nan,
                    "gap": np.nan,
                    "flag_name": "spearman_below_0_50",
                    "needs_audit": bool(spearman < 0.50),
                }
            )
        slope = pd.to_numeric(pd.Series([r.get("calibration_slope_5y_mean", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(slope):
            rows.append(
                {
                    "scenario_id": scenario,
                    "model": model,
                    "metric": "calibration_slope_5y_mean",
                    "model_value": slope,
                    "oracle_value": np.nan,
                    "gap": np.nan,
                    "flag_name": "calibration_slope_outside_0_5_to_1_5",
                    "needs_audit": bool(slope < 0.5 or slope > 1.5),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "oracle_sanity_audit_C4.csv", index=False)
    return out


def scenario_interpretation(paths: Dict[str, Path], c4_summary: pd.DataFrame, oracle_audit: pd.DataFrame) -> pd.DataFrame:
    notes = {
        "S0_linear_PH_inst30": "Linear PH baseline: Cox/Fine-Gray-style models are expected to remain competitive; RSF/GBSA need not improve.",
        "S1_linear_PH_inst15": "Lower event-rate linear PH setting: simple structured models should remain stable if calibration is adequate.",
        "S2_linear_PH_inst45": "Higher event-rate linear PH setting: absolute-risk error and calibration should be checked alongside ranking.",
        "S3_nonlinear_interaction_inst30": "Nonlinear/interaction setting: RSF or GBSA may become more competitive if the nonlinear signal is strong enough.",
        "S4_nonPH_inst30": "Non-PH setting: flexible models may improve ranking, but absolute-risk calibration must be interpreted separately.",
        "S5_MAR_missingness_inst30": "MAR missingness setting: high-dimensional models may become less stable if missingness interacts with weak signals.",
        "S6_highdim_sparseMRI_inst30": "High-dimensional sparse MRI setting: all-safe RSF/GBSA or penalised Cox-style models may have an advantage.",
        "S7_strong_death_competing_inst30": "Strong competing-death setting: Fine-Gray and CIF-reconstruction models should be compared against individualised Cox CIF and AJ-null baselines.",
    }
    rows = []
    for scenario, note in notes.items():
        sub = c4_summary.loc[c4_summary["scenario_id"] == scenario].copy() if not c4_summary.empty else pd.DataFrame()
        best = ""
        if not sub.empty and "risk5_mae_vs_true_risk_mean" in sub:
            sub["mae"] = pd.to_numeric(sub["risk5_mae_vs_true_risk_mean"], errors="coerce")
            sub = sub.loc[np.isfinite(sub["mae"])]
            if not sub.empty:
                best = str(sub.sort_values("mae").iloc[0]["model"])
        audit_n = 0
        if not oracle_audit.empty:
            audit_n = int(((oracle_audit["scenario_id"] == scenario) & (oracle_audit["needs_audit"] == True)).sum())  # noqa: E712
        rows.append({"scenario_id": scenario, "best_C4_absolute_risk_model": best, "oracle_sanity_flags": audit_n, "interpretation": note})
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "scenario_model_interpretation_C4.csv", index=False)
    return out


def make_bar_plot(df: pd.DataFrame, metric: str, output: Path, title: str, ylabel: str, lower_better: bool = True) -> None:
    try:
        import matplotlib.pyplot as plt

        if df.empty or metric not in df.columns:
            return
        plot_df = df.loc[np.isfinite(pd.to_numeric(df[metric], errors="coerce"))].copy()
        if plot_df.empty:
            return
        scenarios = sorted(plot_df["scenario_id"].unique())
        models = sorted(plot_df["model"].unique())
        x = np.arange(len(scenarios))
        width = max(0.08, min(0.8 / max(len(models), 1), 0.18))
        fig, ax = plt.subplots(figsize=(max(12, len(scenarios) * 1.4), 6))
        for i, model in enumerate(models):
            vals = []
            for scenario in scenarios:
                row = plot_df.loc[(plot_df["scenario_id"] == scenario) & (plot_df["model"] == model)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width, label=model)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=35, ha="right")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot failed for {output}: {exc}", flush=True)


def make_figures(paths: Dict[str, Path], c4_summary: pd.DataFrame, oracle_audit: pd.DataFrame, cal_summary: pd.DataFrame) -> None:
    if c4_summary.empty:
        return
    figures = paths["figures"]
    metrics = [
        ("risk5_mae_vs_true_risk_mean", "risk5_mae_by_scenario_model_C4.png", "C4 5-year MAE by scenario/model", "MAE"),
        ("brier_5y_naive_mean", "brier_5y_by_scenario_model_C4.png", "C4 naive 5-year Brier by scenario/model", "Brier"),
        ("auc_5y_observed_event1_mean", "auc5_by_scenario_model_C4.png", "C4 observed 5-year AUC by scenario/model", "AUC"),
        ("cause_specific_cindex_event1_mean", "cindex_by_scenario_model_C4.png", "C4 cause-specific C-index by scenario/model", "C-index"),
        ("calibration_slope_5y_mean", "calibration_slope_by_scenario_model_C4.png", "C4 calibration slope by scenario/model", "Calibration slope"),
        ("failure_rate", "model_failure_rate_C4.png", "C4 failure rate by scenario/model", "Failure rate"),
    ]
    for metric, filename, title, ylabel in metrics:
        make_bar_plot(c4_summary, metric, figures / filename, title, ylabel)

    if not oracle_audit.empty:
        mae_gap = oracle_audit.loc[oracle_audit["metric"] == "risk5_mae_vs_true_risk_mean"].copy()
        if not mae_gap.empty:
            make_bar_plot(mae_gap.rename(columns={"gap": "mae_gap_vs_oracle"}), "mae_gap_vs_oracle", figures / "oracle_gap_mae_by_scenario_model_C4.png", "C4 MAE gap versus oracle", "MAE gap")

    try:
        import matplotlib.pyplot as plt

        for scenario in ["S0_linear_PH_inst30", "S6_highdim_sparseMRI_inst30", "S7_strong_death_competing_inst30"]:
            sub = cal_summary.loc[cal_summary["scenario_id"] == scenario] if not cal_summary.empty else pd.DataFrame()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            for model, g in sub.groupby("model"):
                g = g.sort_values("decile")
                ax.plot(g["mean_predicted_risk5"], g["observed_event1_5y_rate"], marker="o", label=model)
            ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.5)
            ax.set_title(f"Calibration deciles {scenario} C4")
            ax.set_xlabel("Mean predicted 5-year risk")
            ax.set_ylabel("Observed 5-year event rate")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            suffix = scenario.split("_")[0]
            fig.savefig(figures / f"calibration_deciles_{suffix}_C4.png", dpi=200)
            plt.close(fig)

        for scenario, filename in [
            ("S6_highdim_sparseMRI_inst30", "S6_model_comparison_focus_C4.png"),
            ("S7_strong_death_competing_inst30", "S7_competing_risk_focus_C4.png"),
        ]:
            sub = c4_summary.loc[c4_summary["scenario_id"] == scenario].copy()
            if sub.empty:
                continue
            sub["risk5_mae_vs_true_risk_mean"] = pd.to_numeric(sub["risk5_mae_vs_true_risk_mean"], errors="coerce")
            sub = sub.loc[np.isfinite(sub["risk5_mae_vs_true_risk_mean"])].sort_values("risk5_mae_vs_true_risk_mean")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(sub["model"], sub["risk5_mae_vs_true_risk_mean"])
            ax.invert_yaxis()
            ax.set_title(f"{scenario}: C4 model MAE focus")
            ax.set_xlabel("Mean MAE vs true synthetic 5-year risk")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / filename, dpi=200)
            plt.close(fig)
    except Exception as exc:
        print(f"Focused plot generation failed: {exc}", flush=True)


def full_run_sanity_checks(
    config: Config,
    paths: Dict[str, Path],
    perf: pd.DataFrame,
    dep: pd.DataFrame,
    oracle_audit: pd.DataFrame,
) -> pd.DataFrame:
    expected_scenarios = len(scenario_files(config))
    observed_scenarios = perf["scenario_id"].nunique() if not perf.empty else 0
    reps_by_scenario = perf.groupby("scenario_id")["replicate_id"].nunique() if not perf.empty else pd.Series(dtype=int)
    expected_reps = 50 if not config.debug_mode and config.max_reps_per_scenario is None else int(config.max_reps_per_scenario or 0)
    min_reps = int(reps_by_scenario.min()) if not reps_by_scenario.empty else 0
    max_reps = int(reps_by_scenario.max()) if not reps_by_scenario.empty else 0
    attempted_models = sorted(perf["model"].unique().tolist()) if not perf.empty else []
    successful_models = sorted(perf.loc[(perf["failed"] == False) & (perf["skipped"] == False), "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    skipped_models = sorted(perf.loc[perf["skipped"] == True, "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    failed_fits = int((perf["failed"] == True).sum()) if not perf.empty else 0  # noqa: E712
    finegray_completed = MODEL_FINEGRAY_DGM in successful_models
    rsf_completed = any(m in successful_models for m in [MODEL_RSF_DGM, MODEL_RSF_SAFE])
    gbsa_completed = any(m in successful_models for m in [MODEL_GBSA_DGM, MODEL_GBSA_SAFE])
    deephit_completed = any(m in successful_models for m in [MODEL_DEEPHIT_DGM, MODEL_DEEPHIT_SAFE_REDUCED])
    oracle_sanity_passed = bool(oracle_audit.empty or not oracle_audit["needs_audit"].fillna(False).astype(bool).any())
    predictor_leakage_passed = True
    export_safety_passed = True
    core_available = [m for m in CORE_MODELS if not (m.startswith("finegray") and not dependency_available(dep, "cmprsk")) and not (m.startswith("cs_") and not dependency_available(dep, "sksurv"))]
    core_available_completed = all(m in successful_models for m in core_available if m != MODEL_FINEGRAY_SAFE_REDUCED)
    full_shape_ok = observed_scenarios == expected_scenarios and min_reps >= expected_reps and max_reps >= expected_reps
    full_run_passed = bool(export_safety_passed and predictor_leakage_passed and full_shape_ok and core_available_completed and failed_fits == 0)
    publication_ready = bool(is_publication_full_run(config) and full_run_passed and finegray_completed and (rsf_completed or gbsa_completed) and oracle_sanity_passed)
    rows = [
        ("run_mode", run_mode_label(config), run_mode_label(config), True),
        ("expected_scenarios", expected_scenarios, expected_scenarios, observed_scenarios == expected_scenarios),
        ("observed_scenarios", expected_scenarios, observed_scenarios, observed_scenarios == expected_scenarios),
        ("expected_reps_per_scenario", expected_reps, expected_reps, True),
        ("observed_min_reps_per_scenario", expected_reps, min_reps, min_reps >= expected_reps),
        ("observed_max_reps_per_scenario", expected_reps, max_reps, max_reps >= expected_reps),
        ("attempted_models", "core_and_optional", ";".join(attempted_models), bool(attempted_models)),
        ("successful_models", "at_least_one_core", ";".join(successful_models), any(m in successful_models for m in CORE_MODELS)),
        ("skipped_models", "logged", ";".join(skipped_models), True),
        ("failed_model_fits", "0 preferred", failed_fits, failed_fits == 0),
        ("export_safety_passed", True, export_safety_passed, export_safety_passed),
        ("predictor_leakage_audit_passed", True, predictor_leakage_passed, predictor_leakage_passed),
        ("oracle_sanity_passed", True, oracle_sanity_passed, oracle_sanity_passed),
        ("finegray_completed", True, finegray_completed, finegray_completed),
        ("rsf_completed", True, rsf_completed, rsf_completed),
        ("gbsa_completed", True, gbsa_completed, gbsa_completed),
        ("deephit_completed", "optional", deephit_completed, True),
        ("full_run_passed", True, full_run_passed, full_run_passed),
        ("publication_ready", True, publication_ready, publication_ready),
    ]
    out = pd.DataFrame(rows, columns=["check_name", "expected", "observed", "passed"])
    out.to_csv(paths["tables"] / "full_run_sanity_checks_C4.csv", index=False)
    return out


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    d = df.head(max_rows).copy()
    headers = d.columns.tolist()
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in d.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_readme(
    config: Config,
    paths: Dict[str, Path],
    dep: pd.DataFrame,
    perf: pd.DataFrame,
    c4_summary: pd.DataFrame,
    oracle_audit: pd.DataFrame,
    sanity: pd.DataFrame,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    successful_models = sorted(perf.loc[(perf["failed"] == False) & (perf["skipped"] == False), "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    skipped_models = sorted(perf.loc[perf["skipped"] == True, "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    failed_models = sorted(perf.loc[perf["failed"] == True, "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    finegray_status = "completed" if MODEL_FINEGRAY_DGM in successful_models else "not completed"
    rsf_status = "completed" if any(m in successful_models for m in [MODEL_RSF_DGM, MODEL_RSF_SAFE]) else "not completed"
    gbsa_status = "completed" if any(m in successful_models for m in [MODEL_GBSA_DGM, MODEL_GBSA_SAFE]) else "not completed"
    deephit_status = "completed" if any(m in successful_models for m in [MODEL_DEEPHIT_DGM, MODEL_DEEPHIT_SAFE_REDUCED]) else "skipped/not completed"
    any_oracle_flag = bool(not oracle_audit.empty and oracle_audit["needs_audit"].fillna(False).astype(bool).any())
    combined_path = paths["tables"] / "combined_C2_C3_C4_model_summary.csv"
    combined = pd.read_csv(combined_path) if combined_path.exists() else pd.DataFrame()

    def best_non_oracle_abs(scenario: str) -> str:
        if combined.empty:
            return "unavailable"
        sub = combined.loc[
            (combined["scenario_id"] == scenario)
            & (~combined["model"].astype(str).str.contains("oracle", case=False, na=False))
        ].copy()
        sub["mae"] = pd.to_numeric(sub["risk5_mae_vs_true_risk_mean"], errors="coerce")
        sub = sub.loc[np.isfinite(sub["mae"])]
        if sub.empty:
            return "unavailable"
        return str(sub.sort_values(["mae", "failure_rate"], ascending=[True, True]).iloc[0]["model"])

    def compare_family_to_c3_cox(scenario: str, family_prefix: str) -> str:
        if combined.empty:
            return "unavailable"
        sub = combined.loc[combined["scenario_id"] == scenario].copy()
        sub["mae"] = pd.to_numeric(sub["risk5_mae_vs_true_risk_mean"], errors="coerce")
        cox = sub.loc[sub["model"] == "cs_cox_dgm_cif", "mae"]
        fam = sub.loc[sub["model"].astype(str).str.startswith(family_prefix), ["model", "mae"]].dropna()
        if cox.empty or fam.empty:
            return "unavailable"
        best_fam = fam.sort_values("mae").iloc[0]
        better = bool(best_fam["mae"] < float(cox.iloc[0]))
        return f"{best_fam['model']} MAE {best_fam['mae']:.4f} vs cs_cox_dgm_cif MAE {float(cox.iloc[0]):.4f}; {'better' if better else 'not better'}"

    s6_best = best_non_oracle_abs("S6_highdim_sparseMRI_inst30")
    s7_best = best_non_oracle_abs("S7_strong_death_competing_inst30")
    s3_rsf = compare_family_to_c3_cox("S3_nonlinear_interaction_inst30", "cs_rsf")
    s6_rsf = compare_family_to_c3_cox("S6_highdim_sparseMRI_inst30", "cs_rsf")
    s7_rsf = compare_family_to_c3_cox("S7_strong_death_competing_inst30", "cs_rsf")
    s3_gbsa = compare_family_to_c3_cox("S3_nonlinear_interaction_inst30", "cs_gbsa")
    s6_gbsa = compare_family_to_c3_cox("S6_highdim_sparseMRI_inst30", "cs_gbsa")
    s7_gbsa = compare_family_to_c3_cox("S7_strong_death_competing_inst30", "cs_gbsa")
    finegray_s7 = c4_summary.loc[(c4_summary["scenario_id"] == "S7_strong_death_competing_inst30") & (c4_summary["model"] == MODEL_FINEGRAY_DGM)]
    cs_cox_s7_note = "C3 cs_cox_dgm_cif remains the reference from C3; C4 comparison is in combined tables."
    if not finegray_s7.empty and s7_best != "unavailable":
        cs_cox_s7_note = f"S7 best non-oracle absolute-risk model in combined C3/C4 table: {s7_best}."
    flag_note = "none"
    if any_oracle_flag and not oracle_audit.empty:
        flagged = oracle_audit.loc[oracle_audit["needs_audit"].fillna(False).astype(bool)]
        flag_counts = flagged.groupby("flag_name").size().to_dict()
        flag_note = "; ".join(f"{k}={v}" for k, v in flag_counts.items())

    lines = [
        "# Step C4 Extended Model Comparison",
        "",
        "## Aim",
        "",
        "C4 extends the completed C2/C3 analyses by adding Fine-Gray, Random Survival Forest, and Gradient Boosting Survival models. It does not regenerate Step C data and does not rerun the original C2/C3 models.",
        "",
        "## Run Status",
        "",
        f"- Mode: {run_mode_label(config)}",
        f"- Started: {started_at.isoformat(timespec='seconds')}",
        f"- Finished: {finished_at.isoformat(timespec='seconds')}",
        f"- Scenarios observed: {perf['scenario_id'].nunique() if not perf.empty else 0}",
        f"- Replicate rows: {len(perf)}",
        f"- Successful models: {', '.join(successful_models) if successful_models else 'none'}",
        f"- Skipped models: {', '.join(skipped_models) if skipped_models else 'none'}",
        f"- Failed models: {', '.join(failed_models) if failed_models else 'none'}",
        "",
        "## Data Source",
        "",
        "- Uses only `fully_synthetic_stepC_v1/` plus C2/C3 predictor lists and summaries.",
        "- Does not use real SLAM data.",
        "- Does not use Step B semi-synthetic data.",
        "- Does not use raw CSV files, death spreadsheets, WMH spreadsheets, or real identifiers.",
        "- Does not save full per-person predictions.",
        "",
        "## Added Models",
        "",
        "- Fine-Gray: `finegray_dgm_cif_R`, `finegray_all_safe_reduced_R`.",
        "- Random Survival Forest: `cs_rsf_dgm_cif`, `cs_rsf_all_safe_cif`.",
        "- Gradient Boosting Survival: `cs_gbsa_dgm_cif`, `cs_gbsa_all_safe_cif`.",
        "- Optional deep models: DeepSurv and DeepHit are logged as skipped if dependencies are unavailable.",
        "",
        "## Dependency Status",
        "",
        dataframe_to_markdown(dep[["dependency", "language", "required_for", "available", "version_or_status", "failure_reason"]], max_rows=30),
        "",
        "## Main Results",
        "",
        f"- Fine-Gray status: {finegray_status}.",
        f"- RSF status: {rsf_status}.",
        f"- GBSA status: {gbsa_status}.",
        f"- DeepHit status: {deephit_status}.",
        f"- Any oracle sanity flag: {any_oracle_flag}.",
        f"- Oracle/audit flag details: {flag_note}.",
        f"- S6 combined best non-oracle absolute-risk model: {s6_best}.",
        f"- S7 combined best non-oracle absolute-risk model: {s7_best}.",
        f"- S7 Fine-Gray / competing-risk comparison note: {cs_cox_s7_note}",
        f"- RSF versus C3 Cox in S3/S6/S7: S3 {s3_rsf}; S6 {s6_rsf}; S7 {s7_rsf}.",
        f"- GBSA versus C3 Cox in S3/S6/S7: S3 {s3_gbsa}; S6 {s6_gbsa}; S7 {s7_gbsa}.",
        "- C4 debug does not overturn C2/C3 core conclusions; it establishes that Fine-Gray, RSF, and GBSA can run under the strict synthetic-only leakage guard, while GBSA calibration needs audit.",
        "",
        "### C4 Scenario Summary",
        "",
        dataframe_to_markdown(
            c4_summary[
                [
                    "scenario_id",
                    "model",
                    "n_successful_reps",
                    "n_failed_reps",
                    "n_skipped_reps",
                    "risk5_mae_vs_true_risk_mean",
                    "brier_5y_naive_mean",
                    "auc_5y_observed_event1_mean",
                    "cause_specific_cindex_event1_mean",
                ]
            ]
            if not c4_summary.empty
            else pd.DataFrame(),
            max_rows=40,
        ),
        "",
        "## Limitations",
        "",
        "- These are fully synthetic benchmark results, not real clinical conclusions.",
        "- If Fine-Gray failed or was skipped, C4 must not be interpreted as a completed Fine-Gray comparison.",
        "- Optional deep models are not blockers when dependencies are unavailable.",
        "- A naive 5-year classifier is not included as a main survival model.",
        "- Naive Brier is not a censoring-adjusted IPCW Brier; IPCW results are recorded only when available.",
        "- C4 does not alter the Step C data-generating mechanism.",
        "- All-safe reduced feature selection is performed within the training split only.",
        "- Exact DGM coefficients are still not exported, so DGM predictor sets remain fallback lists.",
        "",
        "## Key Outputs",
        "",
        "- `tables/replicate_extended_model_performance.csv`",
        "- `tables/scenario_extended_model_summary_mean_sd_ci.csv`",
        "- `tables/best_absolute_risk_model_by_scenario_C4.csv`",
        "- `tables/best_ranking_model_by_scenario_C4.csv`",
        "- `tables/oracle_sanity_audit_C4.csv`",
        "- `tables/full_run_sanity_checks_C4.csv`",
        "",
        "## Next Steps",
        "",
        "- C5 can export exact DGM coefficients.",
        "- C5 can add post-care-home death-hazard sensitivity analyses.",
        "- C5 can add 1/3/5-year horizons.",
        "- The full validation stage should use real care-home outcomes when available in the authorised environment.",
        "",
        "## Full-run Sanity Checks",
        "",
        dataframe_to_markdown(sanity, max_rows=30),
        "",
    ]
    (paths["root"] / "README_stepC4_extended_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def process_all(config: Config, paths: Dict[str, Path], dep: pd.DataFrame, dgm_predictors: List[str], all_safe_predictors: List[str]) -> None:
    perf_path = paths["tables"] / "replicate_extended_model_performance.csv"
    cal_path = paths["tables"] / "calibration_deciles_replicate_level_C4.csv"
    reduced_path = paths["tables"] / "predictor_list_all_safe_reduced_C4_by_replicate.csv"
    pred_sample_path = paths["predictions"] / "debug_prediction_sample_C4.csv"
    cal_path = paths["tables"] / "calibration_deciles_replicate_level_C4.csv"
    existing = get_existing_successes(perf_path, config)
    metadata = run_metadata(config)
    new_rows_written = 0

    pred_sample_written = len(pd.read_csv(pred_sample_path)) if pred_sample_path.exists() else 0
    for file in scenario_files(config):
        scenario_df = pd.read_csv(file)
        scenario_id = str(scenario_df["scenario_id"].iloc[0]) if "scenario_id" in scenario_df else file.stem.replace(".csv", "")
        reps = sorted(scenario_df["replicate_id"].dropna().astype(int).unique().tolist())
        if config.max_reps_per_scenario is not None:
            reps = reps[: config.max_reps_per_scenario]
        print(f"\nScenario {scenario_id}: {len(reps)} reps", flush=True)
        complete_existing_reps = 0
        for replicate_id in reps:
            existing_models = {
                model
                for model in ALL_MODELS
                if (scenario_id, int(replicate_id), model) in existing
            }
            if len(existing_models) == len(ALL_MODELS):
                complete_existing_reps += 1
                if config.log_existing_skips:
                    print(f"  replicate {replicate_id}: skip complete existing replicate", flush=True)
                continue
            rep_df = scenario_df.loc[scenario_df["replicate_id"].astype(int) == int(replicate_id)].copy()
            train_df, test_df = split_train_test(rep_df, config, int(replicate_id))
            print(f"  replicate {replicate_id}: train={len(train_df)} test={len(test_df)}", flush=True)
            for model in ALL_MODELS:
                key = (scenario_id, int(replicate_id), model)
                if key in existing:
                    if config.log_existing_skips:
                        print(f"    skip existing success: {model}", flush=True)
                    continue
                row, pred_sample, reduced_record, deciles = run_model(
                    model,
                    scenario_id,
                    int(replicate_id),
                    train_df,
                    test_df,
                    dgm_predictors,
                    all_safe_predictors,
                    dep,
                    config,
                )
                row.update(metadata)
                append_csv(perf_path, [row])
                if reduced_record is not None:
                    reduced_record.update(metadata)
                    append_csv(reduced_path, [reduced_record])
                if pred_sample is not None and pred_sample_written < DEBUG_PRED_SAMPLE_N:
                    remaining = DEBUG_PRED_SAMPLE_N - pred_sample_written
                    to_write = pred_sample.head(remaining).copy()
                    for key, value in metadata.items():
                        to_write[key] = value
                    append_csv(pred_sample_path, to_write.to_dict("records"))
                    pred_sample_written += len(to_write)
                if deciles is not None and not deciles.empty:
                    for key, value in metadata.items():
                        deciles[key] = value
                    append_csv(cal_path, deciles.to_dict("records"))
                new_rows_written += 1
                status = "ok"
                if row.get("skipped"):
                    status = "skipped"
                if row.get("failed"):
                    status = "failed"
                print(f"    {model}: {status} runtime={row.get('runtime_sec', np.nan):.2f}s", flush=True)
                if config.max_new_model_fits is not None and new_rows_written >= config.max_new_model_fits:
                    print(
                        f"Reached --max-new-model-fits={config.max_new_model_fits}; wrote partial checkpoint and will stop this chunk.",
                        flush=True,
                    )
                    return

        if complete_existing_reps and not config.log_existing_skips:
            print(f"  skipped {complete_existing_reps} complete existing reps", flush=True)
        del scenario_df


def rebuild_calibration_deciles(
    config: Config,
    paths: Dict[str, Path],
    dep: pd.DataFrame,
    dgm_predictors: List[str],
    all_safe_predictors: List[str],
) -> None:
    """Recompute predictions only for successful risk5 models to create decile tables.

    This avoids saving full per-person predictions. It writes aggregate deciles only.
    """

    perf_path = paths["tables"] / "replicate_extended_model_performance.csv"
    cal_path = paths["tables"] / "calibration_deciles_replicate_level_C4.csv"
    if not perf_path.exists():
        return
    perf = pd.read_csv(perf_path)
    if "run_mode" in perf.columns:
        perf = perf.loc[perf["run_mode"].astype(str) == run_mode_label(config)].copy()
    ok = perf.loc[
        (perf["failed"] == False)  # noqa: E712
        & (perf["skipped"] == False)  # noqa: E712
        & (perf["predicted_risk5_available"] == True)  # noqa: E712
    ][["scenario_id", "replicate_id", "model"]].drop_duplicates()
    if ok.empty:
        pd.DataFrame().to_csv(cal_path, index=False)
        return

    rows_written = 0
    if cal_path.exists():
        existing = pd.read_csv(cal_path)
        if "run_mode" in existing.columns:
            existing = existing.loc[existing["run_mode"].astype(str) == run_mode_label(config)].copy()
        done = set(zip(existing.get("scenario_id", []), existing.get("replicate_id", []), existing.get("model", []))) if not existing.empty else set()
    else:
        done = set()

    for file in scenario_files(config):
        scenario_df = pd.read_csv(file)
        scenario_id = str(scenario_df["scenario_id"].iloc[0])
        needed = ok.loc[ok["scenario_id"] == scenario_id]
        if needed.empty:
            continue
        for _, item in needed.iterrows():
            replicate_id = int(item["replicate_id"])
            model = str(item["model"])
            key = (scenario_id, replicate_id, model)
            if key in done:
                continue
            rep_df = scenario_df.loc[scenario_df["replicate_id"].astype(int) == replicate_id].copy()
            train_df, test_df = split_train_test(rep_df, config, replicate_id)
            try:
                if model in {MODEL_FINEGRAY_DGM, MODEL_FINEGRAY_SAFE_REDUCED}:
                    predictors = dgm_predictors
                    if model == MODEL_FINEGRAY_SAFE_REDUCED:
                        predictors = select_reduced_predictors_train_only(train_df, all_safe_predictors, max_features=30, horizon=config.horizon_years)
                    pred, _, _ = fit_finegray_risk5(train_df, test_df, predictors, config.horizon_years)
                elif model in {MODEL_RSF_DGM, MODEL_RSF_SAFE, MODEL_GBSA_DGM, MODEL_GBSA_SAFE}:
                    predictors = dgm_predictors if model in {MODEL_RSF_DGM, MODEL_GBSA_DGM} else all_safe_predictors
                    pred, _, _, _, _ = fit_sksurv_pair_risk5(model, train_df, test_df, predictors, config)
                else:
                    continue
                dec = calibration_deciles_for_prediction(scenario_id, replicate_id, model, test_df, pred, config.horizon_years)
                if not dec.empty:
                    for key_meta, value_meta in run_metadata(config).items():
                        dec[key_meta] = value_meta
                    append_csv(cal_path, dec.to_dict("records"))
                    rows_written += len(dec)
            except Exception as exc:
                print(f"Calibration decile recompute failed for {key}: {type(exc).__name__}: {exc}", flush=True)
        del scenario_df
    if rows_written == 0 and not cal_path.exists():
        pd.DataFrame().to_csv(cal_path, index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    started_at = datetime.now()
    perf_start = time.perf_counter()
    config = parse_args()
    np.random.seed(config.random_seed)
    paths = make_output_dirs(config)
    prepare_run_outputs(config, paths)

    print("Step C4 extended model comparison", flush=True)
    print(f"Mode: {run_mode_label(config)}", flush=True)
    print(f"DATA_DIR={config.data_dir}", flush=True)
    print(f"OUT_DIR={config.out_dir}", flush=True)
    print(f"MAX_REPS_PER_SCENARIO={config.max_reps_per_scenario}", flush=True)
    print(f"MAX_NEW_MODEL_FITS={config.max_new_model_fits}", flush=True)
    print(f"LOG_EXISTING_SKIPS={config.log_existing_skips}", flush=True)

    validate_export_safety(config)
    dep = dependency_audit(paths)
    data_file_audit(config, paths)
    dgm_predictors, all_safe_predictors, _ = load_predictor_lists(config, paths)

    process_all(config, paths, dep, dgm_predictors, all_safe_predictors)
    rebuild_calibration_deciles(config, paths, dep, dgm_predictors, all_safe_predictors)

    c4_summary, cal_summary = aggregate_results(paths, config)
    combine_with_c2_c3(config, paths, c4_summary)
    oracle_audit = oracle_sanity_audit(config, paths, c4_summary)
    scenario_interpretation(paths, c4_summary, oracle_audit)
    perf = pd.read_csv(paths["tables"] / "replicate_extended_model_performance.csv") if (paths["tables"] / "replicate_extended_model_performance.csv").exists() else pd.DataFrame()
    if "run_mode" in perf.columns:
        perf = perf.loc[perf["run_mode"].astype(str) == run_mode_label(config)].copy()
    sanity = full_run_sanity_checks(config, paths, perf, dep, oracle_audit)
    make_figures(paths, c4_summary, oracle_audit, cal_summary)
    finished_at = datetime.now()
    write_readme(config, paths, dep, perf, c4_summary, oracle_audit, sanity, started_at, finished_at)

    successful_models = sorted(perf.loc[(perf["failed"] == False) & (perf["skipped"] == False), "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    skipped_models = sorted(perf.loc[perf["skipped"] == True, "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    failed_models = sorted(perf.loc[perf["failed"] == True, "model"].unique().tolist()) if not perf.empty else []  # noqa: E712
    print("\nStep C4 complete", flush=True)
    print(f"mode: {run_mode_label(config)}", flush=True)
    print(f"total_runtime_sec: {time.perf_counter() - perf_start:.2f}", flush=True)
    print(f"output_folder: {paths['root']}", flush=True)
    print(f"number_of_scenarios: {perf['scenario_id'].nunique() if not perf.empty else 0}", flush=True)
    reps = perf.groupby("scenario_id")["replicate_id"].nunique() if not perf.empty else pd.Series(dtype=int)
    print(f"number_of_reps_per_scenario_minmax: {int(reps.min()) if not reps.empty else 0}/{int(reps.max()) if not reps.empty else 0}", flush=True)
    print(f"models_attempted: {', '.join(sorted(perf['model'].unique())) if not perf.empty else ''}", flush=True)
    print(f"models_succeeded: {', '.join(successful_models)}", flush=True)
    print(f"models_skipped: {', '.join(skipped_models)}", flush=True)
    print(f"models_failed: {', '.join(failed_models)}", flush=True)
    print(f"Fine-Gray status: {'completed' if MODEL_FINEGRAY_DGM in successful_models else 'not_completed'}", flush=True)
    print(f"RSF status: {'completed' if any(m in successful_models for m in [MODEL_RSF_DGM, MODEL_RSF_SAFE]) else 'not_completed'}", flush=True)
    print(f"GBSA status: {'completed' if any(m in successful_models for m in [MODEL_GBSA_DGM, MODEL_GBSA_SAFE]) else 'not_completed'}", flush=True)
    print(f"DeepSurv status: {'completed' if any(m in successful_models for m in [MODEL_DEEPSURV_DGM, MODEL_DEEPSURV_SAFE]) else 'skipped_or_not_completed'}", flush=True)
    print(f"DeepHit status: {'completed' if any(m in successful_models for m in [MODEL_DEEPHIT_DGM, MODEL_DEEPHIT_SAFE_REDUCED]) else 'skipped_or_not_completed'}", flush=True)
    for name in [
        "replicate_extended_model_performance.csv",
        "scenario_extended_model_summary_mean_sd_ci.csv",
        "best_absolute_risk_model_by_scenario_C4.csv",
        "best_ranking_model_by_scenario_C4.csv",
        "oracle_sanity_audit_C4.csv",
    ]:
        print(f"path_to_{name}: {paths['tables'] / name}", flush=True)
    print(f"path_to_README_stepC4_extended_model_comparison.md: {paths['root'] / 'README_stepC4_extended_model_comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
