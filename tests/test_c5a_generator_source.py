from pathlib import Path

import stepC5A_exact_DGM_coefficients as c5a


def test_load_generator_source_falls_back_to_python(monkeypatch, tmp_path):
    missing_notebook = tmp_path / "missing_generator.ipynb"
    python_source = tmp_path / "generator.py"
    python_source.write_text("SCENARIOS = []\n", encoding="utf-8")

    monkeypatch.setattr(c5a, "GENERATOR_NOTEBOOK", missing_notebook)
    monkeypatch.setattr(c5a, "GENERATOR_PYTHON", python_source)

    source_path, source_text, source_note = c5a.load_generator_source()

    assert source_path == python_source
    assert source_text == "SCENARIOS = []\n"
    assert source_note == "Python generator source with 1 lines."


def test_generator_source_audit_locates_all_dgm_components():
    audit = c5a.build_code_location_audit()

    assert len(audit) == 8
    assert audit["found"].all()
    assert audit["source_file"].map(lambda value: not Path(value).is_absolute()).all()
