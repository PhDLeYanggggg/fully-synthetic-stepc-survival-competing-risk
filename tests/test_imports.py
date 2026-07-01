from pathlib import Path


def test_core_dependencies_import():
    import lifelines  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sksurv  # noqa: F401
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401


def test_organised_script_paths_exist():
    expected = [
        Path("src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py"),
        Path("src/stepC2_model_comparison/stepC2_fully_synthetic_model_comparison.py"),
        Path("src/stepC3_competing_risk/stepC3_competing_risk_evaluation.py"),
        Path("src/stepC4_extended_models/stepC4_extended_model_comparison.py"),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    assert not missing, f"Missing expected script paths: {missing}"
