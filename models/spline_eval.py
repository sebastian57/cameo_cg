"""
JAX-Compatible Cubic Spline Evaluator

Pure-JAX implementation of cubic spline evaluation for use in prior energy
computations. No scipy dependency at runtime.

Supports:
- Standard cubic spline evaluation with boundary clamping
- Periodic cubic spline evaluation (for dihedrals)
- Species-indexed evaluation with global fallback (for residue-specific angles)

All functions are JIT-compilable and autodiff-compatible.
"""

import jax
import jax.numpy as jnp


def evaluate_cubic_spline(
    x: jax.Array,
    knots: jax.Array,
    coeffs: jax.Array,
) -> jax.Array:
    """
    Evaluate a cubic spline at given coordinates.

    For each x, finds interval i such that knots[i] <= x < knots[i+1],
    then computes: U = c0 + c1*dx + c2*dx^2 + c3*dx^3  where dx = x - knots[i].

    Clamps x to [knots[0], knots[-1]] for boundary safety.
    JIT-compilable and autodiff-compatible.

    Args:
        x: Coordinates to evaluate, shape (...)
        knots: Sorted knot positions, shape (N,)
        coeffs: Polynomial coefficients per interval, shape (N-1, 4)
                 Each row is [c0, c1, c2, c3] (ascending power order)

    Returns:
        Energy values, same shape as x
    """
    # Clamp to valid range (small epsilon to stay in last interval)
    eps = jnp.float32(1e-7)
    x_clamped = jnp.clip(x, knots[0], knots[-1] - eps)

    # Find interval index: knots[i] <= x < knots[i+1]
    i = jnp.searchsorted(knots, x_clamped, side='right') - 1
    i = jnp.clip(i, 0, knots.shape[0] - 2)

    # Local coordinate within interval
    dx = x_clamped - knots[i]

    # Gather coefficients for each x
    c = coeffs[i]  # (..., 4)

    # Evaluate polynomial: c0 + c1*dx + c2*dx^2 + c3*dx^3
    # Use Horner's method for numerical stability
    U = c[..., 3]
    U = U * dx + c[..., 2]
    U = U * dx + c[..., 1]
    U = U * dx + c[..., 0]

    return U


def evaluate_cubic_spline_periodic(
    x: jax.Array,
    knots: jax.Array,
    coeffs: jax.Array,
    period: float = 2.0 * jnp.pi,
) -> jax.Array:
    """
    Evaluate a periodic cubic spline.

    Wraps x into [knots[0], knots[0] + period) before evaluation.
    Used for dihedral angles where the domain is periodic (e.g., [-pi, pi]).

    Args:
        x: Coordinates to evaluate, shape (...)
        knots: Sorted knot positions, shape (N,)
        coeffs: Polynomial coefficients per interval, shape (N-1, 4)
        period: Period of the function (default: 2*pi)

    Returns:
        Energy values, same shape as x
    """
    # Wrap x into [knots[0], knots[0] + period)
    x_wrapped = knots[0] + (x - knots[0]) % period
    return evaluate_cubic_spline(x_wrapped, knots, coeffs)


def evaluate_cubic_spline_by_type(
    x: jax.Array,
    species: jax.Array,
    all_knots: jax.Array,
    all_coeffs: jax.Array,
    type_mask: jax.Array,
    global_knots: jax.Array,
    global_coeffs: jax.Array,
) -> jax.Array:
    """
    Evaluate type-specific cubic splines with fallback to global.

    For each entry, selects the spline corresponding to its species type.
    If the type's mask is 0 (insufficient data), falls back to the global spline.

    All per-type splines must share the same knot grid (same number of points,
    same domain) to allow array indexing without dynamic shapes.

    Args:
        x: Coordinate values, shape (M,)
        species: Species ID per entry (int), shape (M,)
        all_knots: Knots per type, shape (n_types, N) — same grid for all types
        all_coeffs: Coefficients per type, shape (n_types, N-1, 4)
        type_mask: 1 = use type-specific, 0 = use global, shape (n_types,)
        global_knots: Global fallback knots, shape (N,)
        global_coeffs: Global fallback coefficients, shape (N-1, 4)

    Returns:
        Energy values, shape (M,)
    """
    # Evaluate type-specific splines (gather by species index).
    # We must evaluate entry-wise because each entry may select a different
    # spline parameter set; broadcasting all_knots[species] into
    # evaluate_cubic_spline() is not valid.
    knots_typed = all_knots[species]
    coeffs_typed = all_coeffs[species]

    def _eval_one(xi, ki, ci):
        return evaluate_cubic_spline(xi, ki, ci)

    U_typed = jax.vmap(_eval_one)(x, knots_typed, coeffs_typed)

    # Evaluate global spline
    U_global = evaluate_cubic_spline(x, global_knots, global_coeffs)

    # Select: use type-specific if mask==1, else global
    use_typed = type_mask[species]  # (M,) — 1 or 0
    return jnp.where(use_typed, U_typed, U_global)
