import importlib.util
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_stepc6_statistics_test_module",
    ROOT / "stepC6_publication_analysis.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_paired_improvement_orientation_for_lower_and_higher_metrics():
    model = pd.DataFrame(
        {
            "replicate_id": [1, 2, 3],
            "mae": [0.10, 0.20, 0.30],
            "cindex": [0.70, 0.72, 0.74],
        }
    )
    reference = pd.DataFrame(
        {
            "replicate_id": [1, 2, 3],
            "mae": [0.20, 0.30, 0.40],
            "cindex": [0.60, 0.62, 0.64],
        }
    )

    lower = MODULE.paired_delta_summary(
        model, reference, "mae", "lower", 200, 42, "lower"
    )
    higher = MODULE.paired_delta_summary(
        model, reference, "cindex", "higher", 200, 42, "higher"
    )

    assert np.isclose(lower["mean_model_minus_reference"], -0.10)
    assert np.isclose(lower["mean_improvement_oriented"], 0.10)
    assert lower["proportion_model_better"] == 1.0
    assert np.isclose(higher["mean_model_minus_reference"], 0.10)
    assert np.isclose(higher["mean_improvement_oriented"], 0.10)
    assert higher["proportion_model_better"] == 1.0


def test_holm_adjustment_matches_known_ordered_values():
    adjusted = MODULE.holm_adjust(pd.Series([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted.to_numpy(), [0.03, 0.06, 0.06])


def test_verify_stage_requires_exact_scenarios_and_repetition_ids(monkeypatch):
    monkeypatch.setattr(MODULE, "SCENARIOS", ["S0", "S1"])
    rows = [
        {
            "scenario_id": scenario,
            "replicate_id": replicate,
            "model": "model",
            "failed": False,
        }
        for scenario in MODULE.SCENARIOS
        for replicate in range(1, 51)
    ]
    valid = pd.DataFrame(rows)
    MODULE.verify_stage(valid, "test", ["model"])

    wrong_scenario = valid.copy()
    wrong_scenario.loc[
        wrong_scenario["scenario_id"] == "S1", "scenario_id"
    ] = "unexpected"
    with pytest.raises(RuntimeError, match="scenario set mismatch"):
        MODULE.verify_stage(wrong_scenario, "test", ["model"])

    wrong_repetition = valid.copy()
    wrong_repetition.loc[
        (wrong_repetition["scenario_id"] == "S1")
        & (wrong_repetition["replicate_id"] == 50),
        "replicate_id",
    ] = 0
    with pytest.raises(RuntimeError, match="repetition identifiers 1-50"):
        MODULE.verify_stage(wrong_repetition, "test", ["model"])


def test_file_integrity_manifest_uses_sha256_and_relative_paths(tmp_path):
    package = tmp_path / "synthetic_package"
    package.mkdir()
    (package / "a.txt").write_text("abc", encoding="utf-8")
    nested = package / "tables"
    nested.mkdir()
    (nested / "b.csv").write_text("x\n1\n", encoding="utf-8")

    manifest = MODULE.file_integrity_manifest([("data", package)])

    assert manifest["relative_path"].tolist() == ["a.txt", "tables/b.csv"]
    assert manifest["sha256"].tolist()[0] == hashlib.sha256(b"abc").hexdigest()
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not manifest["relative_path"].str.startswith("/").any()


def test_truth_target_audit_marks_only_non_ph_s4_as_approximate():
    audit = MODULE.truth_target_validity_audit()
    approximate = audit.loc[
        ~audit["truth_based_metrics_primary"], "scenario_id"
    ].tolist()
    assert approximate == ["S4_nonPH_inst30"]
    assert audit.loc[
        audit["scenario_id"] == "S4_nonPH_inst30",
        "exported_true_risk_status",
    ].iloc[0] == "approximate_nonPH_proxy"
    assert (
        audit.loc[
            audit["scenario_id"] != "S4_nonPH_inst30",
            "exported_true_risk_status",
        ]
        == "closed_form_calibrated_PH_target"
    ).all()


def test_calibration_decile_grid_requires_complete_ten_groups(monkeypatch):
    monkeypatch.setattr(MODULE, "SCENARIOS", ["S0", "S1"])
    monkeypatch.setitem(MODULE.EXPECTED_MODELS, "C3", {"c3_model"})
    monkeypatch.setitem(MODULE.EXPECTED_MODELS, "C4", {"c4_model"})
    rows = [
        {
            "stage": stage,
            "scenario_id": scenario,
            "replicate_id": replicate,
            "model": model,
            "decile": decile,
            "n": 150,
            "mean_predicted_risk5": 0.05 * decile,
            "mean_true_risk5": 0.05 * decile,
            "observed_event1_5y_rate": 0.05 * decile,
        }
        for stage, model in [("C3", "c3_model"), ("C4", "c4_model")]
        for scenario in MODULE.SCENARIOS
        for replicate in range(1, 51)
        for decile in range(1, 11)
    ]
    valid = pd.DataFrame(rows)
    MODULE.verify_calibration_decile_grid(valid)

    missing_decile = valid.drop(valid.index[-1])
    with pytest.raises(RuntimeError, match="deciles 1-10"):
        MODULE.verify_calibration_decile_grid(missing_decile)

    wrong_test_count = valid.copy()
    wrong_test_count.loc[wrong_test_count.index[0], "n"] = 149
    with pytest.raises(RuntimeError, match="1,500 held-out"):
        MODULE.verify_calibration_decile_grid(wrong_test_count)


def test_primary_results_table_gate_rejects_nonfinite_metrics(monkeypatch):
    monkeypatch.setattr(MODULE, "SCENARIOS", ["S0", "S1"])
    monkeypatch.setattr(MODULE, "PRIMARY_ABSOLUTE_MODELS", ["m1", "m2"])
    rows = [
        {
            "scenario_id": scenario,
            "model": model,
            "n_successful_reps": 50,
            "failure_rate": 0.0,
            "cause_specific_cindex_event1_mean": 0.7,
            "risk5_mae_vs_true_risk_mean": 0.05,
            "brier_5y_naive_mean": 0.18,
            "calibration_slope_5y_mean": 1.0,
            "decile_weighted_absolute_calibration_error_vs_truth_mean": 0.02,
            "decile_e90_vs_truth_mean": 0.04,
            "decile_weighted_absolute_calibration_error_vs_observed_mean": 0.03,
            "decile_e90_vs_observed_mean": 0.05,
        }
        for scenario in MODULE.SCENARIOS
        for model in MODULE.PRIMARY_ABSOLUTE_MODELS
    ]
    valid = pd.DataFrame(rows)
    MODULE.verify_primary_results_table(valid)

    invalid = valid.copy()
    invalid.loc[invalid.index[0], "risk5_mae_vs_true_risk_mean"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite required metrics"):
        MODULE.verify_primary_results_table(invalid)


def test_c4_qc_rejects_predictors_outside_locked_sets(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    required_checks = [
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
    ]
    pd.DataFrame(
        {
            "check_name": required_checks,
            "passed": ["True"] * len(required_checks),
        }
    ).to_csv(tables / "full_run_sanity_checks_C4.csv", index=False)
    pd.DataFrame({"predictor": ["age"]}).to_csv(
        tables / "predictor_list_dgm_C4.csv", index=False
    )
    pd.DataFrame({"predictor": ["age", "MMSE"]}).to_csv(
        tables / "predictor_list_all_safe_C4.csv", index=False
    )
    base = {
        "scenario_id": "S0_linear_PH_inst30",
        "replicate_id": 0,
        "model": "test_model",
        "predictor_set": "all_safe",
        "brier_5y_ipcw": np.nan,
        "integrated_brier_score_1to5": np.nan,
        "ipcw_metric_reason": "not_computed_requires_competing_risk_specific_ipcw",
    }

    MODULE.verify_c4_qc(
        tmp_path,
        pd.DataFrame([{**base, "selected_predictors": "age;MMSE"}]),
    )

    with pytest.raises(RuntimeError, match="outside their locked predictor set"):
        MODULE.verify_c4_qc(
            tmp_path,
            pd.DataFrame(
                [{**base, "selected_predictors": "age;innocent_extra"}]
            ),
        )

    with pytest.raises(RuntimeError, match="must be unavailable"):
        MODULE.verify_c4_qc(
            tmp_path,
            pd.DataFrame(
                [
                    {
                        **base,
                        "selected_predictors": "age;MMSE",
                        "brier_5y_ipcw": 0.12,
                    }
                ]
            ),
        )
