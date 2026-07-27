import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


C2_MODULES = [
    load_module("qc_c2_root", "stepC2_fully_synthetic_model_comparison.py"),
    load_module("qc_c2_src", "src/stepC2_model_comparison/stepC2_fully_synthetic_model_comparison.py"),
]
C3_MODULES = [
    load_module("qc_c3_root", "stepC3_competing_risk_evaluation.py"),
    load_module("qc_c3_src", "src/stepC3_competing_risk/stepC3_competing_risk_evaluation.py"),
]
C4_MODULES = [
    load_module("qc_c4_root", "stepC4_extended_model_comparison.py"),
    load_module("qc_c4_src", "src/stepC4_extended_models/stepC4_extended_model_comparison.py"),
]
C5A_MODULES = [
    load_module("qc_c5a_root", "stepC5A_exact_DGM_coefficients.py"),
    load_module("qc_c5a_src", "src/stepC5A_exact_DGM_coefficients/stepC5A_exact_DGM_coefficients.py"),
]


def test_public_repository_text_is_english_only():
    text_suffixes = {
        ".cff",
        ".md",
        ".py",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
    violations = []
    for path in REPO_ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in text_suffixes
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8")
        if cjk.search(text):
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert not violations, (
        "Public repository contains non-English CJK text: "
        + ", ".join(violations)
    )


def test_safe_to_export_string_false_is_not_truthy():
    values = pd.Series(["True", "False"])
    for module in C2_MODULES + C3_MODULES:
        assert module.bool_series_all_true(values) is False
        assert [module.parse_safe_bool(x) for x in values] == [True, False]
        assert module.parse_bool_series(values).tolist() == [True, False]
    for module in C4_MODULES + C5A_MODULES:
        assert [module.parse_safe_bool(x) for x in values] == [True, False]
    for module in C4_MODULES:
        assert module.parse_bool_series(values).tolist() == [True, False]
        assert module.APPROXIMATE_TRUTH_SCENARIOS == {"S4_nonPH_inst30"}


def test_forbidden_predictor_name_filter_excludes_reference_and_truth_columns():
    forbidden = [
        "diagnosis_reference_NOT_PREDICTOR",
        "target_diag_reference_NOT_PREDICTOR",
        "true_risk_carehome_5y",
    ]
    allowed = ["age", "MMSE", "MTL_total_pct"]

    for module in C2_MODULES + C3_MODULES:
        assert all(module.contains_forbidden_name(name) for name in forbidden)
        assert not any(module.contains_forbidden_name(name) for name in allowed)

    for module in C4_MODULES:
        assert all(module.has_forbidden_name(name) for name in forbidden)
        assert not any(module.has_forbidden_name(name) for name in allowed)


def test_feature_dictionary_role_excludes_not_for_prediction_columns(tmp_path):
    feature_dictionary = pd.DataFrame(
        [
            {
                "column": "diagnosis_reference_NOT_PREDICTOR",
                "block": "diagnosis_reference",
                "role": "reference_only_not_for_prediction",
            },
            {
                "column": "target_diag_reference_NOT_PREDICTOR",
                "block": "diagnosis_reference",
                "role": "reference_only_not_for_prediction",
            },
            {"column": "age", "block": "demographic", "role": "synthetic_predictor"},
            {"column": "MMSE", "block": "cognition", "role": "synthetic_predictor"},
            {"column": "MTL_total_pct", "block": "MRI", "role": "synthetic_predictor"},
        ]
    )
    all_columns = feature_dictionary["column"].tolist()

    for module in C2_MODULES:
        paths = {"tables": tmp_path / module.__name__}
        paths["tables"].mkdir()
        result = module.build_predictor_lists(feature_dictionary, all_columns, paths)
        assert "diagnosis_reference_NOT_PREDICTOR" not in result["all_safe_predictors"]
        assert "target_diag_reference_NOT_PREDICTOR" not in result["all_safe_predictors"]
        assert {"age", "MMSE", "MTL_total_pct"}.issubset(result["all_safe_predictors"])

    for module in C3_MODULES:
        safe_predictors, _, audit, _ = module.rebuild_predictors_from_feature_dictionary(
            feature_dictionary, all_columns
        )
        assert "diagnosis_reference_NOT_PREDICTOR" not in safe_predictors
        assert "target_diag_reference_NOT_PREDICTOR" not in safe_predictors
        assert {"age", "MMSE", "MTL_total_pct"}.issubset(safe_predictors)
        excluded = audit.loc[audit["column"].str.contains("NOT_PREDICTOR"), "reason"].tolist()
        assert all("excluded_not_for_prediction_role" in reason for reason in excluded)

    for module in C4_MODULES:
        module_root = tmp_path / module.__name__
        data_dir = module_root / "data"
        c3_dir = module_root / "c3"
        tables_dir = module_root / "out" / "tables"
        (data_dir / "tables").mkdir(parents=True)
        (c3_dir / "tables").mkdir(parents=True)
        tables_dir.mkdir(parents=True)
        feature_dictionary.to_csv(
            data_dir / "tables" / "feature_dictionary.csv",
            index=False,
        )
        pd.DataFrame({"predictor": all_columns}).to_csv(
            c3_dir / "tables" / "predictor_list_all_safe_C3.csv",
            index=False,
        )
        pd.DataFrame({"predictor": all_columns}).to_csv(
            c3_dir / "tables" / "predictor_list_dgm_features_C3.csv",
            index=False,
        )
        config = module.Config(
            data_dir=data_dir,
            c2_dir=module_root / "c2",
            c3_dir=c3_dir,
            out_dir=module_root / "out",
        )
        dgm_predictors, safe_predictors, audit = module.load_predictor_lists(
            config,
            {"tables": tables_dir},
        )
        for predictors in [dgm_predictors, safe_predictors]:
            assert "diagnosis_reference_NOT_PREDICTOR" not in predictors
            assert "target_diag_reference_NOT_PREDICTOR" not in predictors
            assert {"age", "MMSE", "MTL_total_pct"}.issubset(predictors)
        role_flags = audit.loc[
            audit["predictor"].str.contains("NOT_PREDICTOR"),
            "not_for_prediction_role_flag",
        ]
        assert role_flags.all()
