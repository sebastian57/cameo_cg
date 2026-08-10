from __future__ import annotations

import numpy as np

from md.projected_force_analysis import (
    aggregate_vector_field,
    empirical_drift_samples,
    pair_distance_jacobians,
    periodic_difference,
    project_force_response,
    ramachandran_values_and_jacobians,
    remove_center_of_mass_force,
    wrap_degrees,
)


def _numpy_dihedral(points: np.ndarray, indices: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = points[list(indices)]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1_unit, v), w), np.dot(v, w))))


def test_periodic_degree_helpers_use_shortest_difference():
    np.testing.assert_allclose(wrap_degrees([180.0, 181.0, -181.0]), [-180.0, -179.0, 179.0])
    values = np.asarray([[179.0, -179.0], [-179.0, 179.0]])
    np.testing.assert_allclose(periodic_difference(values), [[2.0, -2.0]])


def test_ramachandran_values_and_jacobian_match_finite_difference():
    coordinates = np.asarray(
        [[
            [-0.4, 0.8, 0.3],
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [1.5, 0.9, 0.7],
            [2.2, 1.1, -0.2],
        ]],
        dtype=np.float64,
    )
    values, jacobians = ramachandran_values_and_jacobians(coordinates, batch_size=1)
    expected = wrap_degrees(
        [
            _numpy_dihedral(coordinates[0], (0, 1, 2, 3)) + 180.0,
            _numpy_dihedral(coordinates[0], (1, 2, 3, 4)) + 180.0,
        ]
    )
    np.testing.assert_allclose(values[0], expected, atol=2.0e-5)
    epsilon = 1.0e-3
    plus = coordinates.copy()
    minus = coordinates.copy()
    plus[0, 0, 0] += epsilon
    minus[0, 0, 0] -= epsilon
    value_plus, _ = ramachandran_values_and_jacobians(plus, batch_size=1)
    value_minus, _ = ramachandran_values_and_jacobians(minus, batch_size=1)
    finite_difference = wrap_degrees(value_plus - value_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(jacobians[0, :, 0, 0], finite_difference[0], rtol=4.0e-3, atol=4.0e-3)
    np.testing.assert_allclose(jacobians.sum(axis=2), 0.0, atol=1.0e-4)


def test_pair_distance_jacobian_matches_finite_difference():
    coordinates = np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [0.0, 3.0, 0.0]]])
    pairs = np.asarray([[0, 1], [1, 2]])
    coefficients = np.asarray([[2.0, -1.0], [0.5, 3.0]])
    jacobian = pair_distance_jacobians(coordinates, pairs, coefficients)
    epsilon = 1.0e-6
    plus = coordinates.copy()
    minus = coordinates.copy()
    plus[0, 1, 1] += epsilon
    minus[0, 1, 1] -= epsilon

    def evaluate(position: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(
            position[:, pairs[:, 0], :] - position[:, pairs[:, 1], :], axis=-1
        )
        return distances @ coefficients

    finite_difference = (evaluate(plus) - evaluate(minus)) / (2.0 * epsilon)
    np.testing.assert_allclose(jacobian[0, :, 1, 1], finite_difference[0], atol=1.0e-8)


def test_center_of_mass_removal_and_force_projection():
    masses = np.asarray([1.0, 3.0])
    common_acceleration = np.asarray([[[2.0, -1.0, 0.5], [6.0, -3.0, 1.5]]])
    adjusted = remove_center_of_mass_force(common_acceleration, masses)
    np.testing.assert_allclose(adjusted, 0.0, atol=1.0e-15)

    forces = np.asarray([[[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
    jacobian = np.zeros((1, 1, 2, 3))
    jacobian[0, 0, 0, 0] = 1.0
    jacobian[0, 0, 1, 0] = -1.0
    projected = project_force_response(jacobian, forces, np.ones(2), remove_net_force=True)
    np.testing.assert_allclose(projected, [[4.0]])


def test_aggregate_vector_field_and_segmented_periodic_drift():
    values = np.asarray([[0.2, 0.2], [0.3, 0.4], [1.2, 1.2]])
    vectors = np.asarray([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    edges = np.asarray([0.0, 1.0, 2.0])
    field = aggregate_vector_field(values, vectors, edges, edges)
    assert field.count[0, 0] == 2
    np.testing.assert_allclose(field.mean[0, 0], [2.0, 0.0])
    np.testing.assert_allclose(field.standard_error[0, 0], [1.0 / np.sqrt(2.0), 0.0])
    np.testing.assert_allclose(field.coherence[0, 0], 1.0)

    periodic_values = np.asarray([[179.0, 0.0], [-179.0, 1.0], [50.0, 50.0], [51.0, 52.0]])
    starts, drift = empirical_drift_samples(periodic_values, [2, 2], 0.5, periodic=True)
    np.testing.assert_allclose(starts, [[179.0, 0.0], [50.0, 50.0]])
    np.testing.assert_allclose(drift, [[4.0, 2.0], [2.0, 4.0]])
