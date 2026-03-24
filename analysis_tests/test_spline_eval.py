import numpy as np
import jax
import jax.numpy as jnp
import pytest

from models.spline_eval import (
    evaluate_cubic_spline,
    evaluate_cubic_spline_periodic,
    evaluate_cubic_spline_by_type,
)


def _extract_coeffs(cs):
    # scipy CubicSpline coeff layout: (4, N-1), descending powers
    c = np.asarray(cs.c)
    coeffs = np.stack([c[3], c[2], c[1], c[0]], axis=1)
    return np.asarray(cs.x), coeffs


def test_periodic_wrap_simple_coeffs():
    knots = jnp.array([-jnp.pi, 0.0, jnp.pi], dtype=jnp.float32)
    # interval polynomials U = x^2 in local dx coordinates for both intervals
    coeffs = jnp.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=jnp.float32)

    x1 = jnp.array([3.0 * jnp.pi + 0.1], dtype=jnp.float32)
    x2 = jnp.array([0.1], dtype=jnp.float32)
    y1 = evaluate_cubic_spline_periodic(x1, knots, coeffs)
    y2 = evaluate_cubic_spline_periodic(x2, knots, coeffs)
    np.testing.assert_allclose(np.asarray(y1), np.asarray(y2), rtol=1e-6, atol=1e-6)


def test_by_type_fallback_logic():
    knots = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    # type 0: U=x, type 1: U=2x
    c0 = jnp.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    c1 = jnp.array([[0.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]], dtype=jnp.float32)
    all_knots = jnp.stack([knots, knots], axis=0)
    all_coeffs = jnp.stack([c0, c1], axis=0)
    type_mask = jnp.array([1, 0], dtype=jnp.int32)

    x = jnp.array([0.25, 0.5], dtype=jnp.float32)
    species = jnp.array([0, 1], dtype=jnp.int32)

    y = evaluate_cubic_spline_by_type(x, species, all_knots, all_coeffs, type_mask, knots, c0)
    # species=0 uses typed (x), species=1 falls back to global (x)
    np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=1e-6, atol=1e-6)


def test_matches_scipy_and_gradient():
    scipy = pytest.importorskip("scipy")
    from scipy.interpolate import CubicSpline

    grid = np.linspace(0.2, 2.7, 64)
    U = np.cos(2.0 * grid) + 0.5 * np.cos(3.0 * grid)
    cs = CubicSpline(grid, U, bc_type="natural")
    knots, coeffs = _extract_coeffs(cs)

    x = np.linspace(0.21, 2.69, 200)
    y_ref = cs(x)
    y_jax = evaluate_cubic_spline(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(knots), jnp.asarray(coeffs))

    np.testing.assert_allclose(np.asarray(y_jax), y_ref, rtol=1e-6, atol=1e-6)

    # Gradient check: d/dx sum U(x)
    def f(x_arr):
        return jnp.sum(evaluate_cubic_spline(x_arr, jnp.asarray(knots), jnp.asarray(coeffs)))

    grad_jax = np.asarray(jax.grad(f)(jnp.asarray(x, dtype=jnp.float64)))
    grad_ref = cs.derivative()(x)
    np.testing.assert_allclose(grad_jax, grad_ref, rtol=1e-5, atol=1e-5)


def test_jit_equivalence():
    knots = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    coeffs = jnp.array([
        [1.0, -2.0, 0.5, 0.25],
        [0.0, 1.0, -0.5, 0.1],
    ], dtype=jnp.float32)
    x = jnp.linspace(0.0, 2.0, 33)

    y = evaluate_cubic_spline(x, knots, coeffs)
    y_jit = jax.jit(evaluate_cubic_spline)(x, knots, coeffs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_jit), rtol=1e-6, atol=1e-6)
