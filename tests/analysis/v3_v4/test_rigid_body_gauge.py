import importlib

import numpy as np
import pytest


gauge = importlib.import_module("02_rigid_body_gauge")


def test_exact_rotation_curve_has_zero_energy_change_for_invariant_quadratic():
    energy = lambda x: float(np.sum((x - x.mean(axis=0)) ** 2))
    R = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert gauge.rotation_invariance_error(
        energy, R, np.array([0.0, 0.0, 1.0]), 0.7
    ) == pytest.approx(0.0)


def test_rotation_second_derivative_is_perpendicular_to_rotation_tangent_at_centroid():
    R = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    q = gauge.rigid_tangent(R, np.array([0.0, 0.0, 1.0]))
    a2 = gauge.rotation_second_derivative(R, np.array([0.0, 0.0, 1.0]))
    assert np.dot(q.ravel(), a2.ravel()) == pytest.approx(0.0)
