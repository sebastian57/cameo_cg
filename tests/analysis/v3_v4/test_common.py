import numpy as np
import pytest

from analysis.physics.v3_v4.common import (
    bootstrap_mean,
    join_labels_by_state,
    rigid_tangent,
    rotate_about_centroid,
    signed_path_cost,
)


def test_signed_path_cost_uses_force_negative_gradient_convention():
    R = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    F = np.array([[[-2.0, 0.0, 0.0]], [[-2.0, 0.0, 0.0]]])
    assert signed_path_cost(R, F) == pytest.approx(2.0)


def test_state_join_preserves_state_ids_and_rejects_duplicates():
    states = {"state": np.array([0, 1]), "R": np.zeros((2, 1, 3))}
    labels = {"state": np.array([1, 0]), "F": np.ones((2, 1, 3))}
    joined = join_labels_by_state(states, labels)
    assert joined["F"][0, 0, 0] == 1.0
    with pytest.raises(ValueError, match="duplicate"):
        join_labels_by_state(
            states, {"state": np.array([0, 0]), "F": np.ones((2, 1, 3))}
        )


def test_rotation_about_centroid_preserves_distances():
    R = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotated = rotate_about_centroid(R, np.array([0.0, 0.0, 1.0]), np.pi / 3)
    assert np.linalg.norm(rotated[1] - rotated[0]) == pytest.approx(
        np.linalg.norm(R[1] - R[0])
    )


def test_rotation_tangent_has_expected_norm():
    R = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    tangent = rigid_tangent(R, np.array([0.0, 0.0, 1.0]))
    assert tangent.shape == R.shape
    assert np.linalg.norm(tangent) > 0


def test_bootstrap_is_deterministic_for_fixed_seed():
    a = bootstrap_mean(np.arange(10.0), 200, 7)
    b = bootstrap_mean(np.arange(10.0), 200, 7)
    assert a == b
