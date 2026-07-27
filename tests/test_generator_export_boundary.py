import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_stepc_generator_boundary_test_module",
    ROOT
    / "src"
    / "stepC_generator"
    / "stepC_fully_synthetic_exportable_generator_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_export_package_never_contains_internal_summary_directory(tmp_path):
    export_dir = tmp_path / "fully_synthetic_stepC_v1"
    MODULE.ensure_dirs(export_dir)

    assert (export_dir / "scenario_datasets").is_dir()
    assert (export_dir / "tables").is_dir()
    assert (export_dir / "audit").is_dir()
    assert not (export_dir / "internal_summary").exists()

    with pytest.raises(RuntimeError, match="must be outside"):
        MODULE.prepare_internal_audit_dir(
            export_dir,
            export_dir / "internal_summary",
        )

    internal_dir = MODULE.prepare_internal_audit_dir(
        export_dir,
        tmp_path / "internal_only_audit",
    )
    assert internal_dir.is_dir()
    assert export_dir.resolve() not in internal_dir.parents
