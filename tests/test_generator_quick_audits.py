import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stepc_generator_quick_audits_test_module",
    ROOT / "src" / "stepC_generator" / "stepC_quick_audits.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_quick_audits_generate_expected_synthetic_checks(tmp_path):
    data_dir = tmp_path / "fully_synthetic_stepC_v1"
    (data_dir / "tables").mkdir(parents=True)
    (data_dir / "scenario_datasets").mkdir()
    pd.DataFrame(
        [
            {
                "column": "audit_truth",
                "block": "DGM_truth",
                "role": "simulation_truth_or_audit",
            },
            {
                "column": "true_lp_carehome",
                "block": "DGM_truth",
                "role": "simulation_truth_or_audit",
            },
            {"column": "duration_years", "block": "outcome", "role": "outcome"},
            {"column": "status", "block": "outcome", "role": "outcome"},
        ]
    ).to_csv(data_dir / "tables" / "feature_dictionary.csv", index=False)
    pd.DataFrame(
        {
            "true_lp_carehome": np.arange(-4.0, 4.0),
            "duration_years": [5.0, 5.0, 5.0, 2.0, 5.0, 2.0, 1.0, 0.5],
            "status": [0, 2, 0, 1, 2, 1, 1, 1],
            "audit_truth": [np.nan] * 8,
        }
    ).to_csv(
        data_dir
        / "scenario_datasets"
        / "S0_linear_PH_inst30.csv.gz",
        index=False,
        compression="gzip",
    )

    outputs = MODULE.run_audits(data_dir)
    high_missing = pd.read_csv(outputs["high_missing"])
    gradients = pd.read_csv(outputs["gradient"])
    monotonicity = pd.read_csv(outputs["monotonicity"])

    assert high_missing["column"].tolist() == ["audit_truth"]
    assert len(gradients) == 4
    assert MODULE.scenario_id_from_path(
        data_dir
        / "scenario_datasets"
        / "S0_linear_PH_inst30.csv.gz"
    ) == "S0_linear_PH_inst30"
    assert bool(monotonicity["monotonic_non_decreasing"].iloc[0])
