"""Regression tests for unpenalised logistic calibration.

These tests use small deterministic arrays only; no project data are loaded.
"""

from __future__ import annotations

import numpy as np

import stepC2_fully_synthetic_model_comparison as c2
import stepC3_competing_risk_evaluation as c3
import run_stepC4_postQC_sharded as c4_runner


def synthetic_calibration_example() -> tuple[np.ndarray, np.ndarray]:
    predicted = np.linspace(0.03, 0.75, 200)
    observed = (np.arange(200) % 5 < np.ceil(predicted * 5)).astype(int)
    return observed, predicted


def test_c2_calibration_slope_is_finite_with_current_sklearn() -> None:
    observed, predicted = synthetic_calibration_example()
    slope = c2.calibration_slope(observed, predicted)
    assert np.isfinite(slope)


def test_c3_calibration_slope_and_intercept_are_finite() -> None:
    observed, predicted = synthetic_calibration_example()
    intercept, slope, reason = c3.calibration_intercept_slope(
        observed,
        predicted,
    )
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert reason == ""


def test_c4_shard_merge_boolean_parser_rejects_false_string() -> None:
    assert c4_runner.parse_bool("True") is True
    assert c4_runner.parse_bool("False") is False
