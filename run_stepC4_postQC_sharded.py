#!/usr/bin/env python3
"""Run and merge the strict post-QC Step C4 comparison by scenario.

The runner preserves only previously completed C4 rows whose actual fitted
predictors pass the current forbidden-name audit. It reruns the affected
all-safe tree/boosting models and all canonical pycox neural models.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import pandas as pd


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

PRESERVED_MODELS = [
    "finegray_dgm_cif_R",
    "finegray_all_safe_reduced_R",
    "cs_rsf_dgm_cif",
    "cs_gbsa_dgm_cif",
]

RERUN_MODELS = [
    "cs_rsf_all_safe_cif",
    "cs_gbsa_all_safe_cif",
    "deepsurv_dgm_cause_specific_pycox",
    "deepsurv_all_safe_cause_specific_pycox",
    "deephit_competing_risk_dgm_pycox",
    "deephit_competing_risk_all_safe_pycox",
]

ALL_FINAL_MODELS = PRESERVED_MODELS + RERUN_MODELS

FORBIDDEN_FRAGMENTS = [
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
    "not_predictor",
    "diagnosis_reference",
    "target_diag",
    "final_diagnosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--legacy-c4-dir",
        type=Path,
        default=Path("fully_synthetic_stepC4_extended_models"),
    )
    parser.add_argument(
        "--c2-dir",
        type=Path,
        default=Path("fully_synthetic_stepC2_model_comparison_postQC"),
    )
    parser.add_argument(
        "--c3-dir",
        type=Path,
        default=Path(
            "fully_synthetic_stepC3_competing_risk_evaluation_postQC_calibrated"
        ),
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=Path("fully_synthetic_stepC4_postQC_shards"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fully_synthetic_stepC4_extended_models_postQC"),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip model fitting and merge existing completed shards.",
    )
    return parser.parse_args()


def absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def suspicious_predictors(values: Iterable[object]) -> List[str]:
    suspicious = set()
    for value in values:
        if pd.isna(value):
            continue
        for predictor in str(value).split(";"):
            lower = predictor.lower()
            if any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
                suspicious.add(predictor)
    return sorted(suspicious)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def run_shard(
    root: Path,
    script: Path,
    c2_dir: Path,
    c3_dir: Path,
    shard_root: Path,
    scenario: str,
) -> str:
    out_dir = shard_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--full",
        "--scenarios",
        scenario,
        "--sksurv-n-jobs",
        "1",
        "--models",
        ",".join(RERUN_MODELS),
        "--c2-dir",
        str(c2_dir),
        "--c3-dir",
        str(c3_dir),
        "--out-dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{scenario} exited with code {proc.returncode}; see {log_path}"
        )
    return scenario


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_performance(
    legacy_dir: Path,
    shard_root: Path,
    out_dir: Path,
) -> pd.DataFrame:
    legacy_path = (
        legacy_dir / "tables" / "replicate_extended_model_performance.csv"
    )
    legacy = read_csv_required(legacy_path)
    preserved = legacy.loc[legacy["model"].isin(PRESERVED_MODELS)].copy()
    bad = suspicious_predictors(preserved["selected_predictors"])
    if bad:
        raise RuntimeError(
            "A preserved C4 checkpoint contains forbidden predictors: "
            + ", ".join(bad)
        )
    preserved["implementation_version"] = (
        "1.0-classical-checkpoint-screened-postQC"
    )
    preserved["source_checkpoint"] = (
        f"{legacy_dir.name}/tables/"
        "replicate_extended_model_performance.csv"
    )

    pieces = [preserved]
    for scenario in SCENARIOS:
        path = (
            shard_root
            / scenario
            / "tables"
            / "replicate_extended_model_performance.csv"
        )
        shard = read_csv_required(path)
        shard = shard.loc[
            (shard["scenario_id"] == scenario)
            & shard["model"].isin(RERUN_MODELS)
        ].copy()
        shard["source_checkpoint"] = (
            f"{shard_root.name}/{scenario}/tables/"
            "replicate_extended_model_performance.csv"
        )
        pieces.append(shard)

    merged = pd.concat(pieces, ignore_index=True, sort=False)
    key = ["scenario_id", "replicate_id", "model"]
    duplicates = merged.duplicated(key, keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate merged C4 keys:\n"
            + merged.loc[duplicates, key].head(20).to_string(index=False)
        )
    expected_rows = len(SCENARIOS) * 50 * len(ALL_FINAL_MODELS)
    if len(merged) != expected_rows:
        raise RuntimeError(
            f"Merged performance has {len(merged)} rows; expected "
            f"{expected_rows}."
        )
    expected_keys = {
        (scenario, replicate_id, model)
        for scenario in SCENARIOS
        for replicate_id in range(1, 51)
        for model in ALL_FINAL_MODELS
    }
    observed_keys = set(
        merged[key].itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)[:20]
        unexpected = sorted(observed_keys - expected_keys)[:20]
        raise RuntimeError(
            "Merged C4 keys do not match the exact scenario x repetition "
            f"x model grid. Missing examples: {missing}; unexpected "
            f"examples: {unexpected}."
        )
    counts = merged.groupby("model").size()
    bad_counts = counts.loc[counts != len(SCENARIOS) * 50]
    if not bad_counts.empty:
        raise RuntimeError(
            "Unexpected model counts:\n" + bad_counts.to_string()
        )
    if merged["failed"].map(parse_bool).any():
        raise RuntimeError("Merged C4 checkpoint contains failed fits.")
    if merged["skipped"].map(parse_bool).any():
        raise RuntimeError("Merged C4 checkpoint contains skipped fits.")
    bad = suspicious_predictors(merged["selected_predictors"])
    if bad:
        raise RuntimeError(
            "Merged C4 checkpoint contains forbidden predictors: "
            + ", ".join(bad)
        )
    merged["brier_5y_ipcw"] = float("nan")
    merged["integrated_brier_score_1to5"] = float("nan")
    merged["ipcw_metric_reason"] = (
        "not_computed_requires_competing_risk_specific_ipcw; "
        "the primary observed-status Brier is valid because the audit confirms "
        "no loss to follow-up before five years"
    )

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        tables / "replicate_extended_model_performance.csv",
        index=False,
    )
    return merged


def merge_calibration(
    legacy_dir: Path,
    shard_root: Path,
    out_dir: Path,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    legacy_path = (
        legacy_dir / "tables" / "calibration_deciles_replicate_level_C4.csv"
    )
    legacy = read_csv_required(legacy_path)
    pieces = [legacy.loc[legacy["model"].isin(PRESERVED_MODELS)].copy()]
    for scenario in SCENARIOS:
        path = (
            shard_root
            / scenario
            / "tables"
            / "calibration_deciles_replicate_level_C4.csv"
        )
        shard = read_csv_required(path)
        pieces.append(
            shard.loc[
                (shard["scenario_id"] == scenario)
                & shard["model"].isin(RERUN_MODELS)
            ].copy()
        )
    merged = pd.concat(pieces, ignore_index=True, sort=False)
    key = ["scenario_id", "replicate_id", "model", "decile"]
    duplicates = merged.duplicated(key, keep=False)
    if duplicates.any():
        raise RuntimeError("Duplicate merged calibration-decile keys.")
    covered = merged[
        ["scenario_id", "replicate_id", "model"]
    ].drop_duplicates()
    expected = performance[
        ["scenario_id", "replicate_id", "model"]
    ].drop_duplicates()
    missing = expected.merge(
        covered,
        how="left",
        indicator=True,
    ).loc[lambda frame: frame["_merge"] == "left_only"]
    if not missing.empty:
        raise RuntimeError(
            "Calibration deciles are missing for completed fits:\n"
            + missing.head(20).to_string(index=False)
        )
    merged.to_csv(
        out_dir / "tables" / "calibration_deciles_replicate_level_C4.csv",
        index=False,
    )
    return merged


def merge_reduced_predictors(
    legacy_dir: Path,
    out_dir: Path,
) -> None:
    path = (
        legacy_dir
        / "tables"
        / "predictor_list_all_safe_reduced_C4_by_replicate.csv"
    )
    reduced = read_csv_required(path)
    reduced = reduced.loc[
        reduced["model"] == "finegray_all_safe_reduced_R"
    ].copy()
    bad = suspicious_predictors(reduced["selected_predictors"])
    if bad:
        raise RuntimeError(
            "Preserved reduced Fine-Gray list contains forbidden predictors: "
            + ", ".join(bad)
        )
    reduced.to_csv(
        out_dir
        / "tables"
        / "predictor_list_all_safe_reduced_C4_by_replicate.csv",
        index=False,
    )


def run_global_aggregation(
    root: Path,
    script: Path,
    c2_dir: Path,
    c3_dir: Path,
    out_dir: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(script),
        "--full",
        "--models",
        ",".join(RERUN_MODELS),
        "--c2-dir",
        str(c2_dir),
        "--c3-dir",
        str(c3_dir),
        "--out-dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    log_path = out_dir / "global_aggregation.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Global aggregation failed with code {proc.returncode}; "
            f"see {log_path}"
        )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    script = root / "stepC4_extended_model_comparison.py"
    legacy_dir = absolute(root, args.legacy_c4_dir)
    c2_dir = absolute(root, args.c2_dir)
    c3_dir = absolute(root, args.c3_dir)
    shard_root = absolute(root, args.shard_root)
    out_dir = absolute(root, args.out_dir)
    shard_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        workers = max(1, min(int(args.max_workers), len(SCENARIOS)))
        print(
            f"Running {len(SCENARIOS)} Step C4 scenario shards with "
            f"{workers} workers.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_shard,
                    root,
                    script,
                    c2_dir,
                    c3_dir,
                    shard_root,
                    scenario,
                ): scenario
                for scenario in SCENARIOS
            }
            for future in as_completed(futures):
                scenario = futures[future]
                future.result()
                print(f"Completed shard: {scenario}", flush=True)

    performance = merge_performance(legacy_dir, shard_root, out_dir)
    calibration = merge_calibration(
        legacy_dir,
        shard_root,
        out_dir,
        performance,
    )
    merge_reduced_predictors(legacy_dir, out_dir)
    run_global_aggregation(root, script, c2_dir, c3_dir, out_dir)

    sanity_path = (
        out_dir / "tables" / "full_run_sanity_checks_C4.csv"
    )
    sanity = read_csv_required(sanity_path)
    failed_checks = sanity.loc[~sanity["passed"].map(parse_bool)]
    print(
        f"Merged {len(performance)} performance rows and "
        f"{len(calibration)} calibration-decile rows.",
        flush=True,
    )
    if failed_checks.empty:
        print("All C4 full-run sanity checks passed.", flush=True)
    else:
        print(
            "C4 completed with sanity checks requiring review:\n"
            + failed_checks.to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
