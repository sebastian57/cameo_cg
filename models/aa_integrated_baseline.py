"""Conservative local baseline used for AA-integrated residual training.

The artifact stores a small linear energy basis fitted to correctly mapped CG
force labels.  It is evaluated from CG coordinates only at runtime: omitted AA
coordinates have been integrated out statistically during the offline fit.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


def load_artifact(path: str) -> dict[str, Any]:
    """Load a portable NPZ artifact into JAX arrays for a fixed prior."""
    with np.load(path, allow_pickle=False) as raw:
        required = {
            "pair_indices", "pair_group", "pair_centers", "pair_width",
            "angle_indices", "angle_centers", "angle_width",
            "torsion_indices", "torsion_harmonics", "density_sigma",
            "density_cutoff", "coeff_vector", "n_pair_groups",
        }
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"AA-integrated baseline artifact missing arrays: {missing}")
        payload = {key: jnp.asarray(raw[key]) for key in required if key != "n_pair_groups"}
        payload["n_pair_groups"] = int(raw["n_pair_groups"].item())
        return payload


def coefficient_size(spec: Mapping[str, Any]) -> int:
    n_pair_groups = int(spec["n_pair_groups"])
    n_pair = n_pair_groups * int(spec["pair_centers"].shape[0])
    n_angle = int(spec["angle_indices"].shape[0]) * int(spec["angle_centers"].shape[0])
    n_torsion = int(spec["torsion_indices"].shape[0]) * 2 * int(spec["torsion_harmonics"].shape[0])
    n_density = int(spec["density_sigma"].shape[0]) * 2
    return n_pair + n_angle + n_torsion + n_density


def _unpack_coefficients(spec: Mapping[str, Any], coefficients: jax.Array) -> tuple[jax.Array, ...]:
    n_pair_groups = int(spec["n_pair_groups"])
    n_pair_basis = int(spec["pair_centers"].shape[0])
    n_angle_basis = int(spec["angle_centers"].shape[0])
    n_harmonics = int(spec["torsion_harmonics"].shape[0])
    n_density = int(spec["density_sigma"].shape[0])

    offset = 0
    pair_size = n_pair_groups * n_pair_basis
    pair = coefficients[offset:offset + pair_size].reshape((n_pair_groups, n_pair_basis))
    offset += pair_size
    angle_size = int(spec["angle_indices"].shape[0]) * n_angle_basis
    angle = coefficients[offset:offset + angle_size].reshape((-1, n_angle_basis))
    offset += angle_size
    torsion_size = int(spec["torsion_indices"].shape[0]) * 2 * n_harmonics
    torsion = coefficients[offset:offset + torsion_size].reshape((-1, 2 * n_harmonics))
    offset += torsion_size
    density = coefficients[offset:].reshape((n_density, 2))
    return pair, angle, torsion, density


def _safe_norm(vector: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=-1) + 1.0e-12)


def _smooth_cutoff(distance: jax.Array, cutoff: jax.Array) -> jax.Array:
    scaled = jnp.clip(distance / cutoff, 0.0, 1.0)
    value = 0.5 * (jnp.cos(jnp.pi * scaled) + 1.0)
    return jnp.where(distance < cutoff, value, 0.0)


def _dihedral(R: jax.Array, indices: jax.Array) -> jax.Array:
    p0, p1, p2, p3 = R[indices[0]], R[indices[1]], R[indices[2]], R[indices[3]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_hat = b1 / _safe_norm(b1)
    v = b0 - jnp.sum(b0 * b1_hat) * b1_hat
    w = b2 - jnp.sum(b2 * b1_hat) * b1_hat
    x = jnp.sum(v * w)
    y = jnp.sum(jnp.cross(b1_hat, v) * w)
    return jnp.arctan2(y, x + 1.0e-12)


def energy_components_from_coefficients(
    R: jax.Array,
    mask: jax.Array,
    spec: Mapping[str, Any],
    coefficients: jax.Array,
) -> dict[str, jax.Array]:
    """Return the four physical terms of the fixed-basis baseline energy.

    All terms are scalar, differentiable functions of CG coordinates.  The
    density component is EAM-like and captures a compact many-body local
    environment contribution without introducing hidden runtime variables.
    """
    pair_coeff, angle_coeff, torsion_coeff, density_coeff = _unpack_coefficients(spec, coefficients)
    dtype = R.dtype

    pair_idx = spec["pair_indices"].astype(jnp.int32)
    pair_valid = (mask[pair_idx[:, 0]] > 0) & (mask[pair_idx[:, 1]] > 0)
    dr = R[pair_idx[:, 0]] - R[pair_idx[:, 1]]
    distances = _safe_norm(dr)
    centers = spec["pair_centers"].astype(dtype)
    width = spec["pair_width"].astype(dtype)
    pair_basis = jnp.exp(-0.5 * ((distances[:, None] - centers[None, :]) / width) ** 2)
    pair_basis = pair_basis * _smooth_cutoff(distances, spec["density_cutoff"].astype(dtype))[:, None]
    selected_pair_coeff = pair_coeff[spec["pair_group"].astype(jnp.int32)]
    E_pair = jnp.sum(jnp.where(pair_valid, jnp.sum(pair_basis * selected_pair_coeff, axis=-1), 0.0))

    angle_idx = spec["angle_indices"].astype(jnp.int32)
    angle_valid = jnp.all(mask[angle_idx] > 0, axis=1)
    va = R[angle_idx[:, 0]] - R[angle_idx[:, 1]]
    vb = R[angle_idx[:, 2]] - R[angle_idx[:, 1]]
    cos_theta = jnp.sum(va * vb, axis=-1) / (_safe_norm(va) * _safe_norm(vb))
    angle_basis = jnp.exp(-0.5 * ((cos_theta[:, None] - spec["angle_centers"].astype(dtype)) / spec["angle_width"].astype(dtype)) ** 2)
    E_angle = jnp.sum(jnp.where(angle_valid, jnp.sum(angle_basis * angle_coeff, axis=-1), 0.0))

    torsion_idx = spec["torsion_indices"].astype(jnp.int32)
    torsion_valid = jnp.all(mask[torsion_idx] > 0, axis=1)
    phi = jax.vmap(lambda idx: _dihedral(R, idx))(torsion_idx)
    harmonics = spec["torsion_harmonics"].astype(dtype)
    torsion_basis = jnp.concatenate(
        [jnp.cos(phi[:, None] * harmonics[None, :]), jnp.sin(phi[:, None] * harmonics[None, :])],
        axis=-1,
    )
    E_torsion = jnp.sum(jnp.where(torsion_valid, jnp.sum(torsion_basis * torsion_coeff, axis=-1), 0.0))

    n_atoms = R.shape[0]
    delta = R[:, None, :] - R[None, :, :]
    dist_matrix = _safe_norm(delta)
    neighbor_valid = (mask[:, None] > 0) & (mask[None, :] > 0) & (~jnp.eye(n_atoms, dtype=bool))
    density_kernel = jnp.exp(-0.5 * (dist_matrix / spec["density_sigma"].astype(dtype)[0]) ** 2)
    density_kernel = density_kernel * _smooth_cutoff(dist_matrix, spec["density_cutoff"].astype(dtype))
    rho = jnp.sum(jnp.where(neighbor_valid, density_kernel, 0.0), axis=1)
    density_features = jnp.stack([rho * rho, rho * rho * rho], axis=-1)
    E_density = jnp.sum(jnp.where(mask > 0, jnp.sum(density_features * density_coeff, axis=-1), 0.0))

    return {
        "pair": E_pair,
        "angle": E_angle,
        "torsion": E_torsion,
        "density": E_density,
    }


def energy_from_coefficients(
    R: jax.Array,
    mask: jax.Array,
    spec: Mapping[str, Any],
    coefficients: jax.Array,
    component_scales: Mapping[str, jax.Array] | None = None,
) -> jax.Array:
    """Evaluate the total fixed-basis local AA-integrated energy."""
    components = energy_components_from_coefficients(R, mask, spec, coefficients)
    if component_scales is None:
        return sum(components.values())
    return sum(
        components[name] * jnp.asarray(component_scales.get(name, 1.0), dtype=R.dtype)
        for name in components
    )


def energy_from_artifact(
    R: jax.Array,
    mask: jax.Array,
    spec: Mapping[str, Any],
    component_scales: Mapping[str, jax.Array] | None = None,
) -> jax.Array:
    return energy_from_coefficients(
        R, mask, spec, spec["coeff_vector"].astype(R.dtype), component_scales
    )
