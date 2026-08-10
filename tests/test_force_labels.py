from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from training.force_labels import apply_force_label_mode


def _config(**overrides):
    cfg = {
        "mode": "raw_teacher_blend",
        "teacher_key": "TeacherForce",
        "raw_weight": 2.0,
        "teacher_weight": 1.0,
    }
    cfg.update(overrides)
    return SimpleNamespace(get_force_label_config=lambda: cfg)


def _dataset():
    raw = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    return {
        "R": np.zeros_like(raw),
        "F": raw,
        "TeacherForce": raw + 3.0,
        "mask": np.asarray([[1, 1], [1, 0]], dtype=np.float32),
    }


def test_blend_matches_weighted_mse_optimum_and_masks_padding() -> None:
    dataset = _dataset()
    result = apply_force_label_mode(_config(), dataset)
    expected = (2.0 * dataset["F"] + dataset["TeacherForce"]) / 3.0
    expected[1, 1] = 0.0
    np.testing.assert_allclose(result["F"], expected)
    np.testing.assert_array_equal(result["RawForce"], dataset["F"])
    np.testing.assert_array_equal(dataset["F"], np.arange(12, dtype=np.float32).reshape(2, 2, 3))


def test_teacher_mode_and_validation() -> None:
    dataset = _dataset()
    result = apply_force_label_mode(_config(mode="teacher"), dataset)
    np.testing.assert_allclose(result["F"][0], dataset["TeacherForce"][0])
    with pytest.raises(ValueError, match="requires dataset field"):
        apply_force_label_mode(_config(teacher_key="missing"), dataset)
