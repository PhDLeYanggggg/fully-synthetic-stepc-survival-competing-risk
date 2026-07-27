import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_stepc4_cif_test_module",
    ROOT / "stepC4_extended_model_comparison.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CumulativeHazard:
    def __init__(self, rate: float, grid: np.ndarray):
        self.rate = rate
        self.x = grid
        self.y = rate * grid

    def __call__(self, times):
        return self.rate * np.asarray(times, dtype=float)


def test_two_cause_cif_reconstruction_matches_exponential_closed_form():
    horizon = 5.0
    lambda1 = 0.12
    lambda2 = 0.08
    grid = np.linspace(0.005, horizon, 1000)
    expected = (
        lambda1
        / (lambda1 + lambda2)
        * (1.0 - np.exp(-(lambda1 + lambda2) * horizon))
    )

    prediction, warnings = MODULE.predict_cif_from_two_cumulative_hazards(
        [CumulativeHazard(lambda1, grid)],
        [CumulativeHazard(lambda2, grid)],
        horizon,
    )
    assert not warnings
    assert np.isclose(prediction[0], expected, atol=5e-4)

    ch1 = pd.DataFrame({0: lambda1 * grid}, index=grid)
    ch2 = pd.DataFrame({0: lambda2 * grid}, index=grid)
    prediction_df, _, warnings_df = MODULE.cif_from_two_cumulative_hazard_dfs(
        ch1, ch2, horizon
    )
    assert not warnings_df
    assert np.isclose(prediction_df[0], expected, atol=5e-4)


def test_generic_survival_ipcw_brier_is_not_misapplied_to_cif():
    brier, integrated_brier, reason = MODULE.compute_ipcw_metrics(
        pd.DataFrame(),
        pd.DataFrame(),
        np.asarray([0.2, 0.4]),
        5.0,
    )

    assert np.isnan(brier)
    assert np.isnan(integrated_brier)
    assert "competing_risk_specific_ipcw" in reason


def test_full_run_sanity_requires_every_model_scenario_repetition_cell(
    tmp_path, monkeypatch
):
    scenario_ids = ["S0_linear_PH_inst30", "S1_linear_PH_inst15"]
    monkeypatch.setattr(
        MODULE,
        "scenario_files",
        lambda config: [Path(f"{scenario}.csv.gz") for scenario in scenario_ids],
    )
    config = MODULE.Config(
        data_dir=tmp_path,
        out_dir=tmp_path,
        debug_mode=True,
        max_reps_per_scenario=2,
    )
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(columns=["column", "reason"]).to_csv(
        tables / "predictor_leakage_audit_C4.csv", index=False
    )
    dependencies = pd.DataFrame(
        {
            "dependency": [
                "cmprsk",
                "sksurv",
                "torch",
                "torchtuples",
                "pycox",
            ],
            "available": [True, True, True, True, True],
        }
    )
    rows = [
        {
            "scenario_id": scenario,
            "replicate_id": replicate_id,
            "model": model,
            "failed": False,
            "skipped": False,
        }
        for scenario in scenario_ids
        for replicate_id in [1, 2]
        for model in MODULE.ALL_MODELS
    ]
    complete = pd.DataFrame(rows)
    complete_checks = MODULE.full_run_sanity_checks(
        config, {"tables": tables}, complete, dependencies, pd.DataFrame()
    ).set_index("check_name")
    assert MODULE.parse_safe_bool(
        complete_checks.loc["all_available_models_completed", "observed"]
    )
    assert MODULE.parse_safe_bool(
        complete_checks.loc["full_run_passed", "observed"]
    )

    malformed = complete.drop(
        complete.index[
            (complete["scenario_id"] == scenario_ids[0])
            & (complete["replicate_id"] == 1)
            & (complete["model"] == MODULE.MODEL_FINEGRAY_SAFE_REDUCED)
        ][0]
    )
    malformed = pd.concat([malformed, malformed.iloc[[0]]], ignore_index=True)
    malformed_checks = MODULE.full_run_sanity_checks(
        config, {"tables": tables}, malformed, dependencies, pd.DataFrame()
    ).set_index("check_name")
    assert not MODULE.parse_safe_bool(
        malformed_checks.loc["all_available_models_completed", "observed"]
    )
    assert not MODULE.parse_safe_bool(
        malformed_checks.loc["full_run_passed", "observed"]
    )

    wrong_ids = complete.copy()
    wrong_ids.loc[
        (wrong_ids["scenario_id"] == scenario_ids[0])
        & (wrong_ids["replicate_id"] == 2),
        "replicate_id",
    ] = 3
    wrong_id_checks = MODULE.full_run_sanity_checks(
        config, {"tables": tables}, wrong_ids, dependencies, pd.DataFrame()
    ).set_index("check_name")
    assert not MODULE.parse_safe_bool(
        wrong_id_checks.loc[
            "exact_model_scenario_repetition_grid_completed",
            "observed",
        ]
    )
    assert not MODULE.parse_safe_bool(
        wrong_id_checks.loc["full_run_passed", "observed"]
    )
