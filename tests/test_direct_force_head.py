"""Physics tests for the backend-independent central force scatter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models.direct_force import scatter_central_pair_forces


SENDERS = jnp.asarray([0, 1, 0, 2, 1, 2], dtype=jnp.int32)
RECEIVERS = jnp.asarray([1, 0, 2, 0, 2, 1], dtype=jnp.int32)
POSITIONS = jnp.asarray(
    [[0.0, 0.0, 0.0], [0.43, 0.0, 0.0], [0.11, 0.37, 0.0]],
    dtype=jnp.float32,
)
COEFFICIENT = jnp.asarray([0.2, -0.1, 0.7, 0.5, -0.4, 0.3], dtype=jnp.float32)


def _forces(positions, coefficient=COEFFICIENT, valid=None):
    if valid is None:
        valid = jnp.ones((6,), dtype=jnp.bool_)
    vectors = positions[SENDERS] - positions[RECEIVERS]
    return scatter_central_pair_forces(
        coefficient,
        vectors,
        SENDERS,
        RECEIVERS,
        valid,
        num_nodes=3,
        edge_scale=jnp.linspace(0.4, 0.9, 6),
    )


def test_force_conservation_rotation_and_finite_gradients() -> None:
    forces = _forces(POSITIONS)
    np.testing.assert_allclose(np.asarray(forces.sum(axis=0)), 0.0, atol=2.0e-7)
    torque = jnp.sum(jnp.cross(POSITIONS, forces), axis=0)
    np.testing.assert_allclose(np.asarray(torque), 0.0, atol=2.0e-7)

    angle = 0.71
    rotation = jnp.asarray(
        [[jnp.cos(angle), -jnp.sin(angle), 0.0],
         [jnp.sin(angle), jnp.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    rotated = _forces(POSITIONS @ rotation.T)
    np.testing.assert_allclose(rotated, forces @ rotation.T, rtol=2.0e-6, atol=2.0e-7)

    grad = jax.grad(lambda c: jnp.sum(_forces(POSITIONS, c) ** 2))(COEFFICIENT)
    assert np.isfinite(np.asarray(grad)).all()


def test_invalid_edges_contribute_exactly_zero() -> None:
    forces = _forces(POSITIONS, valid=jnp.zeros((6,), dtype=jnp.bool_))
    np.testing.assert_allclose(forces, 0.0, atol=0.0)


def test_pair_symmetrization_matches_explicit_average() -> None:
    positions = POSITIONS[:2]
    senders = jnp.asarray([0, 1], dtype=jnp.int32)
    receivers = jnp.asarray([1, 0], dtype=jnp.int32)
    vectors = positions[senders] - positions[receivers]
    coefficient = jnp.asarray([2.0, 4.0])
    force = scatter_central_pair_forces(
        coefficient,
        vectors,
        senders,
        receivers,
        jnp.ones((2,), dtype=jnp.bool_),
        num_nodes=2,
    )
    direction = vectors[0] / jnp.linalg.norm(vectors[0])
    np.testing.assert_allclose(force[0], 3.0 * direction, atol=2.0e-7)
    np.testing.assert_allclose(force[1], -3.0 * direction, atol=2.0e-7)

