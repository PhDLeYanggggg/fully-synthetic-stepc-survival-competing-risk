from __future__ import annotations

import inspect

import numpy as np
import pytest

import stepC4_extended_model_comparison as c4


def test_deephit_discretisation_preserves_competing_event_codes() -> None:
    pytest.importorskip("pycox")
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime

    labtrans = LabTransDiscreteTime(
        np.linspace(0.0, 5.0, 8, dtype="float32")
    )
    durations = np.array([5.0, 0.5, 1.5, 2.5, 4.5], dtype="float32")
    events = np.array([0, 1, 2, 1, 2], dtype="int64")

    duration_index, transformed_events = (
        c4.transform_deephit_competing_labels(
            labtrans,
            durations,
            events,
        )
    )

    assert duration_index.dtype == np.int64
    assert transformed_events.dtype == np.int64
    np.testing.assert_array_equal(transformed_events, events)
    assert set(np.unique(transformed_events)) == {0, 1, 2}


def test_deephit_discretisation_rejects_unknown_event_code() -> None:
    pytest.importorskip("pycox")
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime

    labtrans = LabTransDiscreteTime(
        np.linspace(0.0, 5.0, 8, dtype="float32")
    )
    with pytest.raises(ValueError, match="Unsupported"):
        c4.transform_deephit_competing_labels(
            labtrans,
            np.array([1.0], dtype="float32"),
            np.array([3], dtype="int64"),
        )


def test_c4_uses_canonical_pycox_model_paths() -> None:
    deepsurv_source = inspect.getsource(c4.fit_one_deepsurv_cause)
    deephit_source = inspect.getsource(c4.fit_deephit_risk5)

    assert "from pycox.models import CoxPH" in deepsurv_source
    assert "from pycox.models import DeepHit" in deephit_source
    assert "numpy_mlp" not in deepsurv_source
    assert "numpy_mlp" not in deephit_source
    assert c4.MODEL_DEEPSURV_DGM.endswith("_pycox")
    assert c4.MODEL_DEEPHIT_DGM.endswith("_pycox")
