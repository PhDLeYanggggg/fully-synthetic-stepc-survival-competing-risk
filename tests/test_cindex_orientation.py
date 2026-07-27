import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C2 = load_module("stepc2_cindex_orientation", "stepC2_fully_synthetic_model_comparison.py")
C3 = load_module("stepc3_cindex_orientation", "stepC3_competing_risk_evaluation.py")
C4 = load_module("stepc4_cindex_orientation", "stepC4_extended_model_comparison.py")


def test_higher_risk_means_earlier_carehome_for_all_stages():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 0, 0])
    perfect = np.array([4.0, 3.0, 2.0, 1.0])
    reverse = perfect[::-1]
    tied = np.ones(4)
    test_df = pd.DataFrame(
        {"duration_years": durations, "status": [1, 1, 0, 2]}
    )

    for module in [C2, C3]:
        assert np.isclose(
            module.harrell_c_index(durations, event, perfect), 1.0
        )
        assert np.isclose(
            module.harrell_c_index(durations, event, reverse), 0.0
        )
        assert np.isclose(
            module.harrell_c_index(durations, event, tied), 0.5
        )

    assert np.isclose(C4.harrell_cindex(test_df, perfect), 1.0)
    assert np.isclose(C4.harrell_cindex(test_df, reverse), 0.0)
    assert np.isclose(C4.harrell_cindex(test_df, tied), 0.5)
