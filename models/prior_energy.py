"""
Prior Energy Terms for Coarse-Grained Proteins

Implements physics-based energy terms:
- Bonds: Harmonic stretching between sequence-separated beads (i, i+4)
- Angles: Fourier series bending potential
- Dihedrals: Periodic torsion potential
- Repulsive: Soft-sphere non-bonded interactions

Supports two evaluation modes:
- Parametric: harmonic bond, Fourier angle, periodic dihedral
- Spline: cubic spline PMF from KDE + Boltzmann inversion
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from .topology import TopologyBuilder, precompute_repulsive_pairs
from .spline_eval import (
    evaluate_cubic_spline,
    evaluate_cubic_spline_periodic,
    evaluate_cubic_spline_by_type,
)
from .aa_integrated_baseline import energy_from_artifact as aa_integrated_baseline_energy
from .aa_integrated_baseline import load_artifact as load_aa_integrated_baseline_artifact
from utils.logging import model_logger


# =============================================================================
# Safe atan2 with well-defined gradients at (0, 0)
# =============================================================================
# The standard atan2(y, x) has undefined gradients when x = y = 0 because:
#   d(atan2)/dy = x / (x² + y²)  →  0/0 = NaN
#   d(atan2)/dx = -y / (x² + y²) →  0/0 = NaN
#
# This causes NaN gradient propagation for padded atoms at the same location.
# Solution: Use a custom VJP that adds epsilon to the denominator.
# =============================================================================

@jax.custom_vjp
def _safe_atan2(y: jax.Array, x: jax.Array) -> jax.Array:
    """atan2 with well-defined gradients at (0, 0)."""
    return jnp.arctan2(y, x)


def _safe_atan2_fwd(y: jax.Array, x: jax.Array):
    """Forward pass: compute atan2 and save inputs for backward."""
    return _safe_atan2(y, x), (y, x)


def _safe_atan2_bwd(res, g):
    """Backward pass: compute gradients with epsilon to avoid 0/0."""
    y, x = res
    # Add epsilon to denominator to ensure well-defined gradients at (0, 0)
    # Standard gradients: dy = x / (x² + y²), dx = -y / (x² + y²)
    denom = x**2 + y**2 + 1e-12  # Small epsilon prevents division by zero
    grad_y = g * x / denom
    grad_x = g * (-y) / denom
    return grad_y, grad_x


_safe_atan2.defvjp(_safe_atan2_fwd, _safe_atan2_bwd)


# =============================================================================
# Safe norm with well-defined gradients at zero vectors
# =============================================================================
# The standard norm gradient d(||v||)/d(v) = v / ||v|| is undefined when v = 0.
# This causes NaN gradient propagation for padded atoms at the same location.
# Solution: Use a custom VJP that handles the zero case.
# =============================================================================

@jax.custom_vjp
def _safe_norm(v: jax.Array) -> jax.Array:
    """Compute norm with well-defined gradients at v = 0."""
    return jnp.linalg.norm(v, axis=-1)


def _safe_norm_fwd(v: jax.Array):
    """Forward pass: compute norm and save for backward."""
    norm = _safe_norm(v)
    return norm, (v, norm)


def _safe_norm_bwd(res, g):
    """Backward pass: compute gradient with safe division."""
    v, norm = res
    # Standard gradient: d(||v||)/d(v) = v / ||v||
    # When ||v|| = 0, the gradient is undefined. We return 0 in this case.
    # Add epsilon to denominator and use where to handle the zero case cleanly.
    safe_norm = jnp.maximum(norm, 1e-12)
    grad_v = g[..., None] * v / safe_norm[..., None]
    # For zero vectors, set gradient to zero (any direction is valid, but 0 is safest)
    grad_v = jnp.where(norm[..., None] > 1e-12, grad_v, 0.0)
    return (grad_v,)


_safe_norm.defvjp(_safe_norm_fwd, _safe_norm_bwd)


_GROUP_INDEX = {
    "POSITIVE": 0,
    "NEGATIVE": 1,
    "POLAR_UNCHARGED": 2,
    "NONPOLAR": 3,
}
_DEFAULT_GROUP_ORDER = ["POSITIVE", "NEGATIVE", "POLAR_UNCHARGED", "NONPOLAR"]

_AA_POSITIVE = {"LYS", "ARG", "HSP"}
_AA_NEGATIVE = {"ASP", "GLU"}
_AA_POLAR_UNCHARGED = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "HSD", "HSE"}
_AA_NONPOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"}
_AA_KNOWN = _AA_POSITIVE | _AA_NEGATIVE | _AA_POLAR_UNCHARGED | _AA_NONPOLAR


def _normalize_resname(value: Any) -> str:
    """Normalize residue name values from NPZ metadata."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().upper()


def _empty_pairs() -> jax.Array:
    """Return an empty pair array with stable shape/dtype."""
    return jnp.zeros((0, 2), dtype=jnp.int32)


def _build_local_sequence_pairs(n_atoms: int, max_sep: int, min_sep: int = 1) -> jax.Array:
    """Build explicit local sequence pairs (i, i+s) for min_sep <= s <= max_sep."""
    if n_atoms <= 1 or max_sep < min_sep:
        return _empty_pairs()

    pairs = []
    for sep in range(min_sep, max_sep + 1):
        for i in range(0, n_atoms - sep):
            pairs.append((i, i + sep))

    if not pairs:
        return _empty_pairs()
    return jnp.asarray(np.asarray(pairs, dtype=np.int32))


def _pair_sequence_separations(pairs: jax.Array) -> jax.Array:
    """Compute |i-j| for each pair index."""
    if pairs.shape[0] == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    return jnp.abs(pairs[:, 1] - pairs[:, 0]).astype(jnp.int32)


def _filter_pairs_by_min_sep(pairs: jax.Array, min_sep: int) -> jax.Array:
    """Filter pair array by minimum sequence separation."""
    if pairs.shape[0] == 0:
        return pairs
    arr = np.asarray(pairs, dtype=np.int32)
    sep = np.abs(arr[:, 1] - arr[:, 0])
    keep = sep >= int(min_sep)
    if not np.any(keep):
        return _empty_pairs()
    return jnp.asarray(arr[keep], dtype=jnp.int32)


def _concat_pairs(a: jax.Array, b: jax.Array) -> jax.Array:
    """Concatenate two pair arrays while preserving empty-array behavior."""
    if a.shape[0] == 0:
        return b
    if b.shape[0] == 0:
        return a
    return jnp.concatenate([a, b], axis=0)


def _build_charge_and_group_by_species(
    id_to_aa: Dict[int, str], his_charge: float
) -> Tuple[jax.Array, jax.Array]:
    """
    Build species-indexed charge and group lookup arrays from dataset AA mapping.
    """
    if not id_to_aa:
        raise ValueError("id_to_aa mapping is required for typed prior terms.")

    species_ids = sorted(int(k) for k in id_to_aa.keys())
    max_sid = int(max(species_ids))

    charge = np.zeros((max_sid + 1,), dtype=np.float32)
    group = np.full((max_sid + 1,), _GROUP_INDEX["NONPOLAR"], dtype=np.int32)

    for sid in species_ids:
        aa = _normalize_resname(id_to_aa[sid])

        # Group assignment
        if aa in _AA_POSITIVE:
            group[sid] = _GROUP_INDEX["POSITIVE"]
        elif aa in _AA_NEGATIVE:
            group[sid] = _GROUP_INDEX["NEGATIVE"]
        elif aa in _AA_POLAR_UNCHARGED:
            group[sid] = _GROUP_INDEX["POLAR_UNCHARGED"]
        elif aa in _AA_NONPOLAR:
            group[sid] = _GROUP_INDEX["NONPOLAR"]
        else:
            group[sid] = _GROUP_INDEX["NONPOLAR"]
            model_logger.warning(
                f"Unknown residue '{aa}' for species {sid}; defaulting group to NONPOLAR and charge to 0."
            )

        # Charge assignment
        if aa in {"LYS", "ARG", "HSP"}:
            charge[sid] = 1.0
        elif aa in {"ASP", "GLU"}:
            charge[sid] = -1.0
        elif aa in {"HSD", "HSE"}:
            charge[sid] = 0.0
        elif aa == "HIS":
            charge[sid] = float(his_charge)
        else:
            charge[sid] = 0.0

    return jnp.asarray(charge, dtype=jnp.float32), jnp.asarray(group, dtype=jnp.int32)


def _lookup_by_species(species: jax.Array, table: jax.Array) -> jax.Array:
    """Safe species lookup with clipping to valid table indices."""
    idx = jnp.clip(species.astype(jnp.int32), 0, table.shape[0] - 1)
    return table[idx]


def _stickiness_alpha_from_free(
    stick_s_free: jax.Array,
    nonref_group_indices: jax.Array,
    reference_group_idx: int,
    n_groups: int = 4,
) -> jax.Array:
    """
    Build full alpha vector from unconstrained free parameters using softplus.
    """
    free = jnp.ravel(stick_s_free)
    alpha = jnp.ones((n_groups,), dtype=free.dtype)
    alpha = alpha.at[nonref_group_indices].set(jax.nn.softplus(free))
    alpha = alpha.at[reference_group_idx].set(jnp.array(1.0, dtype=free.dtype))
    return alpha


def _same_segment_mask(indices: jax.Array, segment_id: Optional[jax.Array]) -> jax.Array:
    """Return true where all topology indices belong to the same packed segment."""
    if segment_id is None:
        return jnp.ones((indices.shape[0],), dtype=jnp.bool_)
    segment_id = jnp.asarray(segment_id, dtype=jnp.int32)
    seg = segment_id[indices]
    valid = jnp.all(seg >= 0, axis=1)
    same = jnp.all(seg == seg[:, :1], axis=1)
    return valid & same


def compute_dh_energy(
    R: jax.Array,
    mask: jax.Array,
    species: jax.Array,
    pairs: jax.Array,
    seq_sep: jax.Array,
    charge_by_species: jax.Array,
    k_dh: jax.Array,
    lambda_d: jax.Array,
    w_by_sep: jax.Array,
    segment_id: Optional[jax.Array] = None,
) -> jax.Array:
    """Debye-Huckel energy over a pair set."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0
    valid = valid & _same_segment_mask(pairs, segment_id)

    dR = R[pi] - R[pj]
    r = _safe_norm(dR)
    r = jnp.where(valid, r, jax.lax.stop_gradient(r))
    r_safe = jnp.maximum(jnp.where(valid, r, 1e6), jnp.array(1e-3, dtype=R.dtype))

    q_i = _lookup_by_species(species[pi], charge_by_species).astype(R.dtype)
    q_j = _lookup_by_species(species[pj], charge_by_species).astype(R.dtype)

    w = jnp.ravel(jnp.asarray(w_by_sep, dtype=R.dtype))
    if w.size == 0:
        w = jnp.asarray([0.0], dtype=R.dtype)
    sep_idx = jnp.clip(seq_sep.astype(jnp.int32), 0, w.shape[0] - 1)
    w_sep = w[sep_idx]

    k_dh = jnp.asarray(k_dh, dtype=R.dtype)
    lambda_safe = jnp.maximum(jnp.asarray(lambda_d, dtype=R.dtype), jnp.array(1e-6, dtype=R.dtype))
    term = k_dh * q_i * q_j * jnp.exp(-r_safe / lambda_safe) / r_safe
    return jnp.sum(jnp.where(valid, term * w_sep, 0.0))


def compute_stickiness_energy(
    R: jax.Array,
    mask: jax.Array,
    species: jax.Array,
    pairs: jax.Array,
    group_by_species: jax.Array,
    alpha: jax.Array,
    r0: jax.Array,
    sigma: jax.Array,
    segment_id: Optional[jax.Array] = None,
) -> jax.Array:
    """Typed nonbonded stickiness energy over a pair set."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0
    valid = valid & _same_segment_mask(pairs, segment_id)

    dR = R[pi] - R[pj]
    r = _safe_norm(dR)
    r = jnp.where(valid, r, jax.lax.stop_gradient(r))
    r_eval = jnp.where(valid, r, 1e6)

    t_i = _lookup_by_species(species[pi], group_by_species).astype(jnp.int32)
    t_j = _lookup_by_species(species[pj], group_by_species).astype(jnp.int32)

    alpha_i = alpha[jnp.clip(t_i, 0, alpha.shape[0] - 1)].astype(R.dtype)
    alpha_j = alpha[jnp.clip(t_j, 0, alpha.shape[0] - 1)].astype(R.dtype)

    r0 = jnp.asarray(r0, dtype=R.dtype)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=R.dtype), jnp.array(1e-6, dtype=R.dtype))
    phi = -jnp.exp(-0.5 * ((r_eval - r0) / sigma) ** 2)

    term = alpha_i * alpha_j * phi
    return jnp.sum(jnp.where(valid, term, 0.0))


def compute_salt_bridge_energy(
    R: jax.Array,
    mask: jax.Array,
    species: jax.Array,
    pairs: jax.Array,
    charge_by_species: jax.Array,
    delta_sb: jax.Array,
    r0_sb: jax.Array,
    sigma_sb: jax.Array,
    segment_id: Optional[jax.Array] = None,
) -> jax.Array:
    """Short-range salt-bridge correction for opposite-charge pairs."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0
    valid = valid & _same_segment_mask(pairs, segment_id)

    dR = R[pi] - R[pj]
    r = _safe_norm(dR)
    r = jnp.where(valid, r, jax.lax.stop_gradient(r))
    r_eval = jnp.where(valid, r, 1e6)

    q_i = _lookup_by_species(species[pi], charge_by_species).astype(R.dtype)
    q_j = _lookup_by_species(species[pj], charge_by_species).astype(R.dtype)
    opposite = (q_i * q_j) == jnp.array(-1.0, dtype=R.dtype)

    r0_sb = jnp.asarray(r0_sb, dtype=R.dtype)
    sigma_sb = jnp.maximum(jnp.asarray(sigma_sb, dtype=R.dtype), jnp.array(1e-6, dtype=R.dtype))
    # Positive short-range shape; attraction/repulsion controlled by delta_sb sign.
    psi = jnp.exp(-0.5 * ((r_eval - r0_sb) / sigma_sb) ** 2)
    delta_sb = jnp.asarray(delta_sb, dtype=R.dtype)

    term = delta_sb * psi
    keep = valid & opposite
    return jnp.sum(jnp.where(keep, term, 0.0))


def compute_fene_energy(
    R: jax.Array,
    mask: jax.Array,
    bonds: jax.Array,
    r0: jax.Array,
    R0: jax.Array,
    k: jax.Array,
    wall_energy: jax.Array,
    eps: jax.Array,
    segment_id: Optional[jax.Array] = None,
) -> jax.Array:
    """FENE energy for consecutive sequence bonds."""
    if bonds.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    bi, bj = bonds[:, 0], bonds[:, 1]
    valid = (mask[bi] * mask[bj]) > 0
    valid = valid & _same_segment_mask(bonds, segment_id)

    dR = R[bi] - R[bj]
    r = _safe_norm(dR)
    r = jnp.where(valid, r, jax.lax.stop_gradient(r))

    r0 = jnp.asarray(r0, dtype=R.dtype)
    R0 = jnp.maximum(jnp.asarray(R0, dtype=R.dtype), jnp.asarray(eps, dtype=R.dtype))
    k = jnp.asarray(k, dtype=R.dtype)
    wall_energy = jnp.asarray(wall_energy, dtype=R.dtype)
    eps = jnp.asarray(eps, dtype=R.dtype)

    x = (r - r0) / R0
    inside = jnp.abs(x) < (1.0 - eps)
    x_safe = jnp.where(inside, x, 0.0)
    U_inside = -0.5 * k * R0**2 * jnp.log1p(-(x_safe**2))
    overshoot = jnp.maximum(jnp.abs(x) - (1.0 - eps), 0.0) * R0
    U_outside = wall_energy + 0.5 * k * overshoot**2
    U = jnp.where(inside, U_inside, U_outside)
    U = jnp.where(k > 0.0, U, 0.0)
    return jnp.sum(jnp.where(valid, U, 0.0))


def compute_leash_energy(
    R: jax.Array,
    mask: jax.Array,
    pairs: jax.Array,
    d_safe: jax.Array,
    k_safe: jax.Array,
    segment_id: Optional[jax.Array] = None,
) -> jax.Array:
    """Flat-bottom pair-distance leash energy."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0
    valid = valid & _same_segment_mask(pairs, segment_id)

    dR = R[pi] - R[pj]
    r = _safe_norm(dR)
    r = jnp.where(valid, r, jax.lax.stop_gradient(r))
    r_eval = jnp.where(valid, r, 0.0)

    d_safe = jnp.asarray(d_safe, dtype=R.dtype)
    k_safe = jnp.asarray(k_safe, dtype=R.dtype)
    dr = jnp.maximum(r_eval - d_safe, 0.0)
    U = 0.5 * k_safe * dr**2
    return jnp.sum(jnp.where(valid, U, 0.0))


def _flat_bottom_quadratic(
    x: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Squared normalized violation outside [lower, upper], zero inside."""
    lo = jnp.minimum(lower, upper)
    hi = jnp.maximum(lower, upper)
    violation = jnp.maximum(lo - x, 0.0) + jnp.maximum(x - hi, 0.0)
    return (violation / (sigma + 1.0e-8)) ** 2


def _angular_fourier_energy(theta: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Compute Fourier series energy for angles.

    E_angle = sum_n [ a_n * cos(n*theta) + b_n * sin(n*theta) ]

    Args:
        theta: Angles in radians, shape (n_angles,)
        a: Fourier cosine coefficients, shape (n_terms,)
        b: Fourier sine coefficients, shape (n_terms,)

    Returns:
        Energy per angle, shape (n_angles,)
    """
    a = jnp.ravel(jnp.asarray(a, dtype=theta.dtype))
    b = jnp.ravel(jnp.asarray(b, dtype=theta.dtype))

    n = jnp.arange(1, a.shape[0] + 1, dtype=theta.dtype)  # Harmonic indices: 1, 2, 3, ...

    # Vectorized computation: (n_angles, n_terms)
    energy = jnp.sum(
        a[None, :] * jnp.cos(n[None, :] * theta[:, None]) +
        b[None, :] * jnp.sin(n[None, :] * theta[:, None]),
        axis=1
    )
    return energy


def _dihedral_periodic_energy(phi: jax.Array, k: jax.Array, gamma: jax.Array) -> jax.Array:
    """
    Compute periodic cosine energy for dihedrals.

    E_dihedral = sum_n [ k_n * (1 + cos(n*phi - gamma_n)) ]

    Args:
        phi: Dihedral angles in radians, shape (n_dihedrals,)
        k: Force constants, shape (n_terms,)
        gamma: Phase offsets, shape (n_terms,)

    Returns:
        Energy per dihedral, shape (n_dihedrals,)
    """
    k = jnp.ravel(jnp.asarray(k, dtype=phi.dtype))
    gamma = jnp.ravel(jnp.asarray(gamma, dtype=phi.dtype))

    n = jnp.arange(1, k.shape[0] + 1, dtype=phi.dtype)  # Harmonic indices: 1, 2, 3, ...

    # Vectorized computation: (n_dihedrals, n_terms)
    energy = jnp.sum(
        k[None, :] * (1.0 + jnp.cos(n[None, :] * phi[:, None] - gamma[None, :])),
        axis=1
    )
    return energy


def _compute_angles(R: jax.Array, angles: jax.Array, displacement) -> jax.Array:
    """
    Compute angles between three consecutive atoms.

    Args:
        R: Coordinates, shape (n_atoms, 3)
        angles: Angle triplet indices, shape (n_angles, 3)
        displacement: JAX-MD displacement function (unused, kept for API compat)

    Returns:
        Angles in radians, shape (n_angles,)
    """
    ia, ib, ic = angles[:, 0], angles[:, 1], angles[:, 2]
    Ra, Rb, Rc = R[ia], R[ib], R[ic]

    # Vectors from central atom (free-space: displacement is subtraction)
    v_ba = Ra - Rb  # b -> a
    v_bc = Rc - Rb  # b -> c

    # Angle via dot product
    # Use _safe_norm to ensure well-defined gradients at zero vectors
    dot = jnp.einsum("ij,ij->i", v_ba, v_bc)
    norm_ba = _safe_norm(v_ba)
    norm_bc = _safe_norm(v_bc)

    cos_theta = dot / (norm_ba * norm_bc + 1e-12)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    return jnp.arccos(cos_theta)


def _compute_dihedrals(R: jax.Array, dihedrals: jax.Array, displacement) -> jax.Array:
    """
    Compute dihedral angles between four consecutive atoms.

    Args:
        R: Coordinates, shape (n_atoms, 3)
        dihedrals: Dihedral quadruplet indices, shape (n_dihedrals, 4)
        displacement: JAX-MD displacement function (unused, kept for API compat)

    Returns:
        Dihedral angles in radians, shape (n_dihedrals,)
    """
    i, j, k, l = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]
    Ri, Rj, Rk, Rl = R[i], R[j], R[k], R[l]

    # Bond vectors (free-space: displacement is subtraction)
    b1 = Rj - Ri  # i -> j
    b2 = Rk - Rj  # j -> k
    b3 = Rl - Rk  # k -> l

    # Normal vectors to planes
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    # Normalize middle bond
    # Use _safe_norm to ensure well-defined gradients at zero vectors
    b2_norm = _safe_norm(b2)
    b2_hat = b2 / (b2_norm[:, None] + 1e-12)

    # Dihedral angle via atan2
    # Use _safe_atan2 to ensure well-defined gradients at (0, 0)
    # which occurs for padded atoms at the same location
    m1 = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)

    return _safe_atan2(y, x)


class PriorEnergy:
    """
    Physics-based prior energy for coarse-grained proteins.

    Computes energy from bonds, angles, dihedrals, and repulsive interactions.
    Parameters are loaded from config (parametric) or from a spline NPZ file.

    Two modes controlled by presence of `spline_file` in config:
    - Parametric (default): harmonic bond, Fourier angle, periodic dihedral
    - Spline: cubic spline PMF from KDE + Boltzmann inversion

    Example:
        >>> config = ConfigManager("config.yaml")
        >>> topology = TopologyBuilder(N_max=100)
        >>> prior = PriorEnergy(config, topology, displacement)
        >>> energies = prior.compute_energy(R, mask)
        >>> total = prior.compute_total_energy(R, mask)
    """

    def __init__(
        self,
        config,
        topology: TopologyBuilder,
        displacement,
        id_to_aa: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize prior energy model.

        Args:
            config: ConfigManager instance
            topology: TopologyBuilder instance
            displacement: JAX-MD displacement function (from space.free())
        """
        self.config = config
        self.topology = topology
        self.displacement = displacement

        self.weights = config.get_prior_weights()

        # Get topology
        self.bonds, self.angles = topology.get_bonds_and_angles()
        self.dihedrals = topology.get_dihedrals()
        self.rep_pairs = topology.get_repulsive_pairs()
        self.fene_pairs = self.bonds
        wca_cfg = config.get("model", "priors", "wca", default=None)
        if wca_cfg is None:
            wca_cfg = config.get("priors", "wca", default={})
        if not isinstance(wca_cfg, dict):
            wca_cfg = {}
        self.wca_min_sep = int(wca_cfg.get("min_sep", wca_cfg.get("min_repulsive_sep", 1)))
        self.wca_pairs = precompute_repulsive_pairs(self.topology.N_max, min_sep=self.wca_min_sep)
        lj_cfg = config.get("model", "priors", "lj", default={}) or {}
        self.lj_min_sep = int(lj_cfg.get("min_sep", 2))
        self.lj_pairs = precompute_repulsive_pairs(self.topology.N_max, min_sep=self.lj_min_sep)
        leash_cfg = config.get("model", "priors", "leash", default=None)
        if leash_cfg is None:
            leash_cfg = config.get("priors", "leash", default={})
        if not isinstance(leash_cfg, dict):
            leash_cfg = {}
        self.leash_min_sep = int(leash_cfg.get("min_sep", 2))
        self.leash_pairs = precompute_repulsive_pairs(self.topology.N_max, min_sep=self.leash_min_sep)
        ev_min = int(config.get("model", "priors", "excluded_volume_min_sep", default=2))
        ev_max = int(config.get("model", "priors", "excluded_volume_max_sep", default=5))
        self.excluded_vol_pairs = topology.get_excluded_volume_pairs(min_sep=ev_min, max_sep=ev_max)
        self._init_explicit_local_priors(config)
        self._init_five_particle_flat_bottom_prior(config)
        self._init_aa_integrated_baseline_prior(config)
        self._init_ala2_feature_recovery_prior(config)
        self._init_ala2_rama_recovery_prior(config)
        self._init_ala2_geometry_support_recovery_prior(config)
        self._init_typed_interaction_metadata(config, id_to_aa)

        # Check for spline-based priors.
        # New path: explicit boolean gate in config.
        # Backward compatibility: if boolean is not set, enable splines when
        # a spline file is provided.
        spline_path = config.get("model", "priors", "spline_file", default=None)
        use_spline_cfg = config.get("model", "priors", "use_spline_priors", default=None)
        use_spline_priors = bool(spline_path is not None) if use_spline_cfg is None else bool(use_spline_cfg)

        if use_spline_priors:
            if spline_path is None:
                raise ValueError(
                    "model.priors.use_spline_priors is true, but model.priors.spline_file is not set."
                )
            self._init_spline_priors(spline_path, config)
        else:
            self._init_parametric_priors(config)

    def _init_explicit_local_priors(self, config) -> None:
        """Initialize explicit local pair priors from config arrays.

        These prototype terms are intended for fixed-topology, one-protein
        datasets. In tiled residual mode the prior forces are computed on the
        untiled dataset before packing, so the explicit indices are valid for
        the current 4zoh residual-cache tests.
        """
        prior_cfg = config.get("model", "priors", default={}) or {}

        def _pairs(name: str) -> jax.Array:
            cfg = prior_cfg.get(name, {}) or {}
            pairs = np.asarray(cfg.get("pairs", []), dtype=np.int32)
            if pairs.size == 0:
                return _empty_pairs()
            return jnp.asarray(pairs.reshape((-1, 2)), dtype=jnp.int32)

        def _vector(name: str, key: str) -> jax.Array:
            cfg = prior_cfg.get(name, {}) or {}
            vals = np.asarray(cfg.get(key, []), dtype=np.float32)
            return jnp.asarray(vals.reshape((-1,)), dtype=jnp.float32)

        self.local_in_pairs = _pairs("local_in")
        self.local_in_r0 = _vector("local_in", "r0")
        self.local_in_k = _vector("local_in", "k")
        self.local_in_margin = jnp.asarray(
            float((prior_cfg.get("local_in", {}) or {}).get("margin", 0.5)),
            dtype=jnp.float32,
        )

        self.local_bond_in_pairs = _pairs("local_bond_in")
        self.local_bond_in_r0 = _vector("local_bond_in", "r0")
        self.local_bond_in_k = _vector("local_bond_in", "k")

        angle_cfg = prior_cfg.get("local_angle_in", {}) or {}
        angle_indices = np.asarray(angle_cfg.get("angles", []), dtype=np.int32)
        self.local_angle_in_angles = (
            jnp.zeros((0, 3), dtype=jnp.int32)
            if angle_indices.size == 0
            else jnp.asarray(angle_indices.reshape((-1, 3)), dtype=jnp.int32)
        )
        self.local_angle_in_cos0 = jnp.asarray(
            np.asarray(angle_cfg.get("cos0", []), dtype=np.float32).reshape((-1,)), dtype=jnp.float32
        )
        self.local_angle_in_k = jnp.asarray(
            np.asarray(angle_cfg.get("k", []), dtype=np.float32).reshape((-1,)), dtype=jnp.float32
        )
        if self.local_angle_in_angles.shape[0] != self.local_angle_in_cos0.shape[0] or self.local_angle_in_angles.shape[0] != self.local_angle_in_k.shape[0]:
            raise ValueError("local_angle_in angles, cos0, and k must have matching lengths.")

        crowd_cfg = prior_cfg.get("crowding_wall", {}) or {}
        self.crowding_r0 = jnp.asarray(float(crowd_cfg.get("r0", 7.0)), dtype=jnp.float32)
        self.crowding_a = jnp.asarray(float(crowd_cfg.get("a", 0.4)), dtype=jnp.float32)
        self.crowding_min_seq_sep = int(crowd_cfg.get("min_seq_sep", crowd_cfg.get("m", 2)))
        crowding_N_max = float(crowd_cfg.get("N_max", crowd_cfg.get("n_max", 0.0)))
        self.crowding_enabled = crowding_N_max > 0.0
        self.crowding_N_max = jnp.asarray(crowding_N_max, dtype=jnp.float32)
        self.crowding_k = jnp.asarray(float(crowd_cfg.get("k", crowd_cfg.get("k_N", 2.0))), dtype=jnp.float32)
        self.crowding_p = jnp.asarray(float(crowd_cfg.get("p", 2.0)), dtype=jnp.float32)

        # Explicit, fixed-fragment torsion expansion.  This is kept separate
        # from the generic topology dihedral term because it can represent the
        # coupled phi/psi geometry of a small mapped molecule.
        torsion_cfg = prior_cfg.get("local_torsion_fourier", {}) or {}

        def _quadruplet(key: str) -> jax.Array:
            values = np.asarray(torsion_cfg.get(key, []), dtype=np.int32)
            if values.size == 0:
                return jnp.zeros((0, 4), dtype=jnp.int32)
            return jnp.asarray(values.reshape((-1, 4)), dtype=jnp.int32)

        self.local_torsion_phi_indices = _quadruplet("phi_indices")
        self.local_torsion_psi_indices = _quadruplet("psi_indices")
        self.local_torsion_order = int(torsion_cfg.get("order", 0))
        n_marginal = 4 * self.local_torsion_order
        n_coupled = 4 * self.local_torsion_order * self.local_torsion_order
        self.local_torsion_marginal_coeff = jnp.asarray(
            np.asarray(torsion_cfg.get("marginal_coeff", np.zeros(n_marginal)), dtype=np.float32).reshape((-1,)),
            dtype=jnp.float32,
        )
        self.local_torsion_coupled_coeff = jnp.asarray(
            np.asarray(torsion_cfg.get("coupled_coeff", np.zeros(n_coupled)), dtype=np.float32).reshape((-1,)),
            dtype=jnp.float32,
        )
        if self.local_torsion_marginal_coeff.shape[0] != n_marginal:
            raise ValueError(
                "local_torsion_fourier.marginal_coeff must contain 4 * order values."
            )
        if self.local_torsion_coupled_coeff.shape[0] != n_coupled:
            raise ValueError(
                "local_torsion_fourier.coupled_coeff must contain 4 * order**2 values."
            )
        self.local_torsion_marginal_scale = jnp.asarray(
            float(torsion_cfg.get("marginal_scale", 1.0)), dtype=jnp.float32
        )
        self.local_torsion_coupled_scale = jnp.asarray(
            float(torsion_cfg.get("coupled_scale", 1.0)), dtype=jnp.float32
        )

    def _init_five_particle_flat_bottom_prior(self, config) -> None:
        """Load the fitted 5-particle flat-bottom safety prior, when enabled."""
        prior_cfg = config.get("model", "priors", default={}) or {}
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        flat_cfg = prior_cfg.get("five_particle_flat_bottom", {}) or {}
        if not isinstance(flat_cfg, dict):
            flat_cfg = {"enabled": bool(flat_cfg)}

        self.five_particle_flat_enabled = bool(flat_cfg.get("enabled", False))
        self.five_particle_flat_pairs = _empty_pairs()
        self.five_particle_flat_bonds = _empty_pairs()
        self.five_particle_flat_angles = jnp.zeros((0, 3), dtype=jnp.int32)
        self.five_particle_flat_fragments: Dict[int, Dict[str, jax.Array]] = {}
        self.five_particle_flat_params: Dict[str, jax.Array] = {}

        if not self.five_particle_flat_enabled:
            return

        json_path = flat_cfg.get("json_path", None)
        if json_path is None:
            raise ValueError(
                "model.priors.five_particle_flat_bottom.enabled=true requires json_path."
            )

        path = Path(json_path)
        if not path.is_absolute():
            from_config = config.config_path.parent / path
            from_cwd = Path.cwd() / path
            path = from_config if from_config.exists() else from_cwd
        if not path.exists():
            raise FileNotFoundError(f"5-particle flat-bottom prior JSON not found: {path}")

        with path.open("r") as f:
            data = json.load(f)
        stats = data["statistics"]
        hyper = data["hyperparameters"]

        bond_stats = stats["bonds"]["per_bond"]
        self.five_particle_flat_bonds = jnp.asarray(
            [b["bond_index"] for b in bond_stats], dtype=jnp.int32
        )
        self.five_particle_flat_params["bond_sigma"] = jnp.asarray(
            [b["sigma"] for b in bond_stats], dtype=jnp.float32
        )
        self.five_particle_flat_params["bond_p01"] = jnp.asarray(
            [b["p01"] for b in bond_stats], dtype=jnp.float32
        )
        self.five_particle_flat_params["bond_p99"] = jnp.asarray(
            [b["p99"] for b in bond_stats], dtype=jnp.float32
        )
        self.five_particle_flat_params["bond_k"] = jnp.asarray(
            hyper["bond_prior"]["k_bond"], dtype=jnp.float32
        )

        angle_stats = stats["angles"]["per_angle"]
        self.five_particle_flat_angles = jnp.asarray(
            [a["angle_index"] for a in angle_stats], dtype=jnp.int32
        )
        self.five_particle_flat_params["angle_sigma"] = jnp.asarray(
            [a["sigma"] for a in angle_stats], dtype=jnp.float32
        )
        self.five_particle_flat_params["angle_cos_p01"] = jnp.asarray(
            [a["p01"] for a in angle_stats], dtype=jnp.float32
        )
        self.five_particle_flat_params["angle_cos_p99"] = jnp.asarray(
            [a["p99"] for a in angle_stats], dtype=jnp.float32
        )
        self.five_particle_flat_params["angle_k"] = jnp.asarray(
            hyper["angle_prior"]["k_angle"], dtype=jnp.float32
        )

        frag_hyper = hyper["fragment_distance_prior"]
        self.five_particle_flat_params["fragment_k_base"] = jnp.asarray(
            frag_hyper["k_fragment_base"], dtype=jnp.float32
        )
        weights = {int(k): float(v) for k, v in frag_hyper["weights"].items()}
        for sep_key, sep_stats in stats["fragments"].items():
            sep = int(sep_key.split("_")[1])
            per_pair = sep_stats["per_pair"]
            self.five_particle_flat_fragments[sep] = {
                "pairs": jnp.asarray([p["pair_index"] for p in per_pair], dtype=jnp.int32),
                "sigma": jnp.asarray([p["sigma"] for p in per_pair], dtype=jnp.float32),
                "p01": jnp.asarray([p["p01"] for p in per_pair], dtype=jnp.float32),
                "p99": jnp.asarray([p["p99"] for p in per_pair], dtype=jnp.float32),
                "weight": jnp.asarray(weights.get(sep, 1.0), dtype=jnp.float32),
            }

        all_pairs = stats["nonbonded"]["per_pair"]
        self.five_particle_flat_pairs = jnp.asarray(
            [p["pair"] for p in all_pairs], dtype=jnp.int32
        )
        wca = hyper["wca_repulsion"]
        collapse = hyper["collapse_guard"]
        shell = hyper["type_pair_shell"]
        global_nb = stats["nonbonded"]["global"]
        self.five_particle_flat_params.update(
            {
                "wca_epsilon": jnp.asarray(wca["epsilon"], dtype=jnp.float32),
                "wca_sigma": jnp.asarray(wca["sigma"], dtype=jnp.float32),
                "wca_r_cut": jnp.asarray(wca["r_guard"], dtype=jnp.float32),
                "wca_r_floor": jnp.asarray(float(wca["r_guard"]) * 0.5, dtype=jnp.float32),
                "collapse_k": jnp.asarray(collapse["k_collapse"], dtype=jnp.float32),
                "collapse_r_guard": jnp.asarray(collapse["r_guard"], dtype=jnp.float32),
                "shell_p01": jnp.asarray(global_nb["p01"], dtype=jnp.float32),
                "shell_p99": jnp.asarray(global_nb["p99"], dtype=jnp.float32),
                "shell_k_repulsion": jnp.asarray(shell["k_repulsion"], dtype=jnp.float32),
                "shell_k_pull": jnp.asarray(shell["k_pull"], dtype=jnp.float32),
                "component_bond": jnp.asarray(flat_cfg.get("bond_scale", 1.0), dtype=jnp.float32),
                "component_angle": jnp.asarray(flat_cfg.get("angle_scale", 1.0), dtype=jnp.float32),
                "component_fragment": jnp.asarray(flat_cfg.get("fragment_scale", 1.0), dtype=jnp.float32),
                "component_wca": jnp.asarray(flat_cfg.get("wca_scale", 1.0), dtype=jnp.float32),
                "component_collapse": jnp.asarray(flat_cfg.get("collapse_scale", 1.0), dtype=jnp.float32),
                "component_pair_shell": jnp.asarray(flat_cfg.get("pair_shell_scale", 0.0), dtype=jnp.float32),
            }
        )
        model_logger.info("Loaded 5-particle flat-bottom prior from: %s", path)

    def _init_aa_integrated_baseline_prior(self, config) -> None:
        """Load a fixed local AA-integrated baseline energy artifact.

        This is deliberately separate from the flat-bottom safety prior.  It
        is an always-on conservative baseline fitted to mapped CG forces and
        is intended to be subtracted during residual force matching.
        """
        prior_cfg = config.get("model", "priors", default={}) or {}
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        baseline_cfg = prior_cfg.get("aa_integrated_baseline", {}) or {}
        if not isinstance(baseline_cfg, dict):
            baseline_cfg = {"enabled": bool(baseline_cfg)}

        self.aa_integrated_baseline_enabled = bool(baseline_cfg.get("enabled", False))
        self.aa_integrated_baseline_scale = jnp.asarray(
            float(baseline_cfg.get("energy_scale", 1.0)), dtype=jnp.float32
        )
        component_cfg = baseline_cfg.get("component_scales", {}) or {}
        self.aa_integrated_baseline_component_scales = {
            name: jnp.asarray(float(component_cfg.get(name, 1.0)), dtype=jnp.float32)
            for name in ("pair", "angle", "torsion", "density")
        }
        self.aa_integrated_baseline_spec: Dict[str, Any] = {}
        if not self.aa_integrated_baseline_enabled:
            return

        artifact_path = baseline_cfg.get("artifact_path", None)
        if artifact_path is None:
            raise ValueError(
                "model.priors.aa_integrated_baseline.enabled=true requires artifact_path."
            )
        path = Path(artifact_path)
        if not path.is_absolute():
            from_config = config.config_path.parent / path
            from_cwd = Path.cwd() / path
            path = from_config if from_config.exists() else from_cwd
        if not path.exists():
            raise FileNotFoundError(f"AA-integrated baseline artifact not found: {path}")

        self.aa_integrated_baseline_spec = load_aa_integrated_baseline_artifact(str(path))
        model_logger.info(
            "Loaded AA-integrated baseline prior from: %s (energy_scale=%.4g, components=%s)",
            path,
            float(self.aa_integrated_baseline_scale),
            {key: float(value) for key, value in self.aa_integrated_baseline_component_scales.items()},
        )

    def _init_ala2_feature_recovery_prior(self, config) -> None:
        """Load optional Ala2 feature restoring prior.

        This term is intentionally feature based rather than a direct FES prior:
        the torsion-support artifact only supplies a scalar activation
        complement, while force gradients come from local torsion and signed
        volume flat-bottom terms.
        """
        prior_cfg = config.get("model", "priors", default={}) or {}
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        cfg = prior_cfg.get("ala2_feature_recovery", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {"enabled": bool(cfg)}

        self.ala2_feature_recovery_enabled = bool(cfg.get("enabled", False))
        self.ala2_feature_recovery_stop_gradient_activation = bool(
            cfg.get("stop_gradient_activation", True)
        )
        self.ala2_feature_recovery_params: Dict[str, jax.Array] = {}
        if not self.ala2_feature_recovery_enabled:
            return

        json_path = cfg.get("json_path", None)
        if json_path is None:
            raise ValueError(
                "model.priors.ala2_feature_recovery.enabled=true requires json_path."
            )
        path = Path(json_path)
        if not path.is_absolute():
            from_config = config.config_path.parent / path
            from_cwd = Path.cwd() / path
            path = from_config if from_config.exists() else from_cwd
        if not path.exists():
            raise FileNotFoundError(f"Ala2 feature recovery prior JSON not found: {path}")

        with path.open("r") as f:
            data = json.load(f)

        torsions = data.get("torsions", [])
        volumes = data.get("volumes", [])
        support = data.get("support_gate", {})
        self.ala2_feature_recovery_params = {
            "torsion_indices": jnp.asarray(
                [item["idx"] for item in torsions], dtype=jnp.int32
            ).reshape((-1, 4)),
            "torsion_center_deg": jnp.asarray(
                [item["center_deg"] for item in torsions], dtype=jnp.float32
            ),
            "torsion_halfwidth_deg": jnp.asarray(
                [item["halfwidth_deg"] for item in torsions], dtype=jnp.float32
            ),
            "torsion_scale_deg": jnp.asarray(
                [item["scale_deg"] for item in torsions], dtype=jnp.float32
            ),
            "volume_indices": jnp.asarray(
                [item["idx"] for item in volumes], dtype=jnp.int32
            ).reshape((-1, 4)),
            "volume_low": jnp.asarray([item["low"] for item in volumes], dtype=jnp.float32),
            "volume_high": jnp.asarray([item["high"] for item in volumes], dtype=jnp.float32),
            "volume_scale": jnp.asarray([item["scale"] for item in volumes], dtype=jnp.float32),
            "k_torsion": jnp.asarray(
                cfg.get("k_torsion", data.get("k_torsion", 0.05)), dtype=jnp.float32
            ),
            "k_volume": jnp.asarray(
                cfg.get("k_volume", data.get("k_volume", 0.2)), dtype=jnp.float32
            ),
            "energy_scale": jnp.asarray(cfg.get("energy_scale", 1.0), dtype=jnp.float32),
            "activation_power": jnp.asarray(cfg.get("activation_power", 1.0), dtype=jnp.float32),
            "activation_floor": jnp.asarray(cfg.get("activation_floor", 0.0), dtype=jnp.float32),
            "support_phi_indices": jnp.asarray(support["phi_indices"], dtype=jnp.int32),
            "support_psi_indices": jnp.asarray(support["psi_indices"], dtype=jnp.int32),
            "support_reference_phi": jnp.asarray(support["reference_phi"], dtype=jnp.float32),
            "support_reference_psi": jnp.asarray(support["reference_psi"], dtype=jnp.float32),
            "support_k": jnp.asarray(support["k"], dtype=jnp.int32),
            "support_onset_score_deg": jnp.asarray(
                support["onset_score_deg"], dtype=jnp.float32
            ),
            "support_offset_score_deg": jnp.asarray(
                support["offset_score_deg"], dtype=jnp.float32
            ),
        }
        model_logger.info(
            "Loaded Ala2 feature recovery prior from: %s (torsions=%d volumes=%d)",
            path,
            len(torsions),
            len(volumes),
        )

    def _init_ala2_rama_recovery_prior(self, config) -> None:
        """Load optional Ala2 Ramachandran support restoring prior.

        This is intentionally molecule-specific and meant as a diagnostic
        "strong restorer" rather than a transferable safety prior.
        """
        prior_cfg = config.get("model", "priors", default={}) or {}
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        cfg = prior_cfg.get("ala2_rama_recovery", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {"enabled": bool(cfg)}

        self.ala2_rama_recovery_enabled = bool(cfg.get("enabled", False))
        self.ala2_rama_recovery_stop_gradient_activation = bool(
            cfg.get("stop_gradient_activation", True)
        )
        self.ala2_rama_recovery_params: Dict[str, jax.Array] = {}
        if not self.ala2_rama_recovery_enabled:
            return

        json_path = cfg.get("json_path", None)
        if json_path is None:
            raise ValueError(
                "model.priors.ala2_rama_recovery.enabled=true requires json_path."
            )
        path = Path(json_path)
        if not path.is_absolute():
            from_config = config.config_path.parent / path
            from_cwd = Path.cwd() / path
            path = from_config if from_config.exists() else from_cwd
        if not path.exists():
            raise FileNotFoundError(f"Ala2 Ramachandran recovery prior JSON not found: {path}")

        with path.open("r") as f:
            data = json.load(f)

        self.ala2_rama_recovery_params = {
            "phi_indices": jnp.asarray(data["phi_indices"], dtype=jnp.int32),
            "psi_indices": jnp.asarray(data["psi_indices"], dtype=jnp.int32),
            "reference_phi": jnp.asarray(data["reference_phi"], dtype=jnp.float32),
            "reference_psi": jnp.asarray(data["reference_psi"], dtype=jnp.float32),
            "k_nearest": jnp.asarray(data.get("k_nearest", 8), dtype=jnp.int32),
            "onset_score_deg": jnp.asarray(
                cfg.get("onset_score_deg", data.get("onset_score_deg", 12.0)),
                dtype=jnp.float32,
            ),
            "offset_score_deg": jnp.asarray(
                cfg.get("offset_score_deg", data.get("offset_score_deg", 35.0)),
                dtype=jnp.float32,
            ),
            "scale_deg": jnp.asarray(
                cfg.get("scale_deg", data.get("scale_deg", 15.0)),
                dtype=jnp.float32,
            ),
            "k_restore": jnp.asarray(
                cfg.get("k_restore", data.get("k_restore", 25.0)),
                dtype=jnp.float32,
            ),
            "activation_power": jnp.asarray(
                cfg.get("activation_power", data.get("activation_power", 1.0)),
                dtype=jnp.float32,
            ),
            "activation_floor": jnp.asarray(
                cfg.get("activation_floor", data.get("activation_floor", 0.0)),
                dtype=jnp.float32,
            ),
            "energy_cap": jnp.asarray(
                cfg.get("energy_cap", data.get("energy_cap", 1.0e6)),
                dtype=jnp.float32,
            ),
        }
        model_logger.info(
            "Loaded Ala2 Ramachandran recovery prior from: %s (reference=%d)",
            path,
            len(data["reference_phi"]),
        )

    def _init_ala2_geometry_support_recovery_prior(self, config) -> None:
        """Load optional transferable local-geometry support restoring prior."""
        prior_cfg = config.get("model", "priors", default={}) or {}
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        cfg = prior_cfg.get("ala2_geometry_support_recovery", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {"enabled": bool(cfg)}

        self.ala2_geometry_support_recovery_enabled = bool(cfg.get("enabled", False))
        self.ala2_geometry_support_recovery_params: Dict[str, jax.Array] = {}
        if not self.ala2_geometry_support_recovery_enabled:
            return

        artifact_path = cfg.get("artifact_path", cfg.get("npz_path", None))
        if artifact_path is None:
            raise ValueError(
                "model.priors.ala2_geometry_support_recovery.enabled=true requires artifact_path."
            )
        path = Path(artifact_path)
        if not path.is_absolute():
            from_config = config.config_path.parent / path
            from_cwd = Path.cwd() / path
            path = from_config if from_config.exists() else from_cwd
        if not path.exists():
            raise FileNotFoundError(f"Ala2 geometry support artifact not found: {path}")

        data = np.load(path, allow_pickle=False)
        self.ala2_geometry_support_recovery_params = {
            "pair_low": jnp.asarray(data["pair_low"], dtype=jnp.float32),
            "pair_high": jnp.asarray(data["pair_high"], dtype=jnp.float32),
            "pair_count": jnp.asarray(data["pair_count"], dtype=jnp.int32),
            "pair_margin_fraction": jnp.asarray(
                cfg.get("pair_margin_fraction", float(data["pair_margin_fraction"])),
                dtype=jnp.float32,
            ),
            "k_restore": jnp.asarray(cfg.get("k_restore", 0.25), dtype=jnp.float32),
            "score_onset": jnp.asarray(cfg.get("score_onset", 0.0), dtype=jnp.float32),
            "score_cap": jnp.asarray(cfg.get("score_cap", 4.0), dtype=jnp.float32),
            "min_sequence_separation": jnp.asarray(
                int(cfg.get("min_sequence_separation", 1)), dtype=jnp.int32
            ),
            "max_pairs": jnp.asarray(cfg.get("max_pairs", 0), dtype=jnp.int32),
        }
        model_logger.info(
            "Loaded Ala2 geometry support recovery prior from: %s (k=%.4g score_onset=%.4g)",
            path,
            float(self.ala2_geometry_support_recovery_params["k_restore"]),
            float(self.ala2_geometry_support_recovery_params["score_onset"]),
        )

    def _init_typed_interaction_metadata(
        self, config, id_to_aa: Optional[Dict[int, str]]
    ) -> None:
        """Initialize metadata and pair sets for typed prior terms."""
        prior_cfg = config.get("model", "priors", default={}) or {}
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}

        aa_typing_cfg = prior_cfg.get("aa_typing", {})
        if not isinstance(aa_typing_cfg, dict):
            aa_typing_cfg = {}

        self.aa_typing_source = str(aa_typing_cfg.get("source", "dataset_map")).strip().lower()
        self.his_charge = float(aa_typing_cfg.get("his_charge", 0.0))

        group_order = aa_typing_cfg.get("group_order", _DEFAULT_GROUP_ORDER)
        if not isinstance(group_order, (list, tuple)) or len(group_order) != 4:
            model_logger.warning(
                "model.priors.aa_typing.group_order is invalid; falling back to default order."
            )
            group_order = list(_DEFAULT_GROUP_ORDER)
        group_order = [str(x).strip().upper() for x in group_order]
        if set(group_order) != set(_DEFAULT_GROUP_ORDER):
            model_logger.warning(
                "model.priors.aa_typing.group_order does not contain the required 4 groups; "
                "falling back to default order."
            )
            group_order = list(_DEFAULT_GROUP_ORDER)
        self.group_order = group_order

        ref_group_name = str(
            aa_typing_cfg.get("stickiness_reference_group", "POLAR_UNCHARGED")
        ).strip().upper()
        if ref_group_name not in self.group_order:
            model_logger.warning(
                f"stickiness_reference_group='{ref_group_name}' not in group_order; "
                "falling back to POLAR_UNCHARGED."
            )
            ref_group_name = "POLAR_UNCHARGED"
        self.stick_reference_group_name = ref_group_name
        self.stick_reference_group_idx = int(self.group_order.index(ref_group_name))
        self.stick_nonref_group_indices = jnp.asarray(
            [i for i in range(4) if i != self.stick_reference_group_idx],
            dtype=jnp.int32,
        )

        dh_cfg = prior_cfg.get("dh", {})
        if not isinstance(dh_cfg, dict):
            dh_cfg = {}
        stick_cfg = prior_cfg.get("stickiness", {})
        if not isinstance(stick_cfg, dict):
            stick_cfg = {}
        sb_cfg = prior_cfg.get("salt_bridge", {})
        if not isinstance(sb_cfg, dict):
            sb_cfg = {}

        self.dh_enabled = bool(dh_cfg.get("enabled", False))
        self.dh_mode = str(dh_cfg.get("mode", "local_k")).strip().lower()
        self.dh_K = max(0, int(dh_cfg.get("K", 2)))

        self.stickiness_enabled = bool(stick_cfg.get("enabled", False))
        self.stickiness_min_seq_sep = max(1, int(stick_cfg.get("min_seq_sep", 3)))

        self.salt_bridge_enabled = bool(sb_cfg.get("enabled", False))
        self.salt_bridge_min_seq_sep = max(1, int(sb_cfg.get("min_seq_sep", 3)))

        self.dh_local_pairs = _build_local_sequence_pairs(
            self.topology.N_max, max_sep=self.dh_K, min_sep=1
        )
        self.dh_local_seq_sep = _pair_sequence_separations(self.dh_local_pairs)

        nb_ex = _filter_pairs_by_min_sep(self.excluded_vol_pairs, min_sep=3)
        self.nb_pairs_for_stick_sb = _concat_pairs(nb_ex, self.rep_pairs)
        self.nb_pairs_seq_sep = _pair_sequence_separations(self.nb_pairs_for_stick_sb)

        self.stickiness_pairs = _filter_pairs_by_min_sep(
            self.nb_pairs_for_stick_sb, self.stickiness_min_seq_sep
        )
        self.stickiness_pair_seq_sep = _pair_sequence_separations(self.stickiness_pairs)

        self.salt_bridge_pairs = _filter_pairs_by_min_sep(
            self.nb_pairs_for_stick_sb, self.salt_bridge_min_seq_sep
        )
        self.salt_bridge_pair_seq_sep = _pair_sequence_separations(self.salt_bridge_pairs)

        self.typed_terms_enabled = (
            self.dh_enabled or self.stickiness_enabled or self.salt_bridge_enabled
        )

        if self.typed_terms_enabled:
            if self.aa_typing_source != "dataset_map":
                raise ValueError(
                    "Only model.priors.aa_typing.source='dataset_map' is supported for typed priors."
                )
            if not id_to_aa:
                raise ValueError(
                    "Typed prior terms are enabled but dataset AA mapping (id_to_aa) is missing. "
                    "Disable typed terms or provide id_to_aa via DatasetLoader metadata."
                )
            self.charge_by_species, self.group_by_species = _build_charge_and_group_by_species(
                id_to_aa, his_charge=self.his_charge
            )
        else:
            self.charge_by_species = jnp.zeros((1,), dtype=jnp.float32)
            self.group_by_species = jnp.full((1,), _GROUP_INDEX["NONPOLAR"], dtype=jnp.int32)

    def _init_new_term_params(self, prior_params: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Initialize DH/stickiness/salt-bridge parameter block."""
        dh_cfg = prior_params.get("dh", {})
        if not isinstance(dh_cfg, dict):
            dh_cfg = {}
        stick_cfg = prior_params.get("stickiness", {})
        if not isinstance(stick_cfg, dict):
            stick_cfg = {}
        sb_cfg = prior_params.get("salt_bridge", {})
        if not isinstance(sb_cfg, dict):
            sb_cfg = {}

        w_by_sep = jnp.ravel(
            jnp.asarray(dh_cfg.get("w_by_sep", [0.0, 1.0, 0.1]), dtype=jnp.float32)
        )
        if w_by_sep.size == 0:
            w_by_sep = jnp.asarray([0.0], dtype=jnp.float32)

        s_free = jnp.ravel(
            jnp.asarray(stick_cfg.get("s_free_init", [0.0, 0.0, 0.0]), dtype=jnp.float32)
        )
        expected = int(self.stick_nonref_group_indices.shape[0])
        if s_free.shape[0] != expected:
            raise ValueError(
                f"model.priors.stickiness.s_free_init must have length {expected} "
                f"(one free parameter per non-reference group), got {s_free.shape[0]}."
            )

        return {
            "k_DH": jnp.asarray(dh_cfg.get("k_DH", 1.0), dtype=jnp.float32),
            "lambda_D": jnp.asarray(dh_cfg.get("lambda_D", 8.0), dtype=jnp.float32),
            "dh_w_by_sep": w_by_sep,
            "stick_r0": jnp.asarray(stick_cfg.get("r0", 3.8), dtype=jnp.float32),
            "stick_sigma": jnp.asarray(stick_cfg.get("sigma", 0.4), dtype=jnp.float32),
            "stick_s_free": s_free,
            "salt_delta": jnp.asarray(sb_cfg.get("delta", -0.5), dtype=jnp.float32),
            "salt_r0": jnp.asarray(sb_cfg.get("r0", 3.8), dtype=jnp.float32),
            "salt_sigma": jnp.asarray(sb_cfg.get("sigma", 0.3), dtype=jnp.float32),
        }

    def _init_lj_params(self, prior_params: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Initialize full Lennard-Jones prior parameters."""
        lj_cfg = prior_params.get("lj", {})
        if not isinstance(lj_cfg, dict):
            lj_cfg = {}
        return {
            "lj_epsilon": jnp.asarray(lj_cfg.get("epsilon", prior_params.get("lj_epsilon", 1.0)), dtype=jnp.float32),
            "lj_sigma": jnp.asarray(lj_cfg.get("sigma", prior_params.get("lj_sigma", 3.0)), dtype=jnp.float32),
        }

    def _init_wca_params(self, prior_params: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Initialize WCA clash-guard parameters."""
        wca_cfg = prior_params.get("wca", {})
        if not isinstance(wca_cfg, dict):
            wca_cfg = {}

        epsilon = float(wca_cfg.get("epsilon", prior_params.get("wca_epsilon", 1.0)))
        r_guard_raw = wca_cfg.get("r_guard", prior_params.get("wca_r_guard", None))
        sigma_raw = wca_cfg.get("sigma", prior_params.get("wca_sigma", None))

        if r_guard_raw is None and sigma_raw is None:
            r_guard = 3.2
            sigma = r_guard / (2.0 ** (1.0 / 6.0))
        elif r_guard_raw is not None:
            r_guard = float(r_guard_raw)
            sigma = float(sigma_raw) if sigma_raw is not None else r_guard / (2.0 ** (1.0 / 6.0))
        else:
            sigma = float(sigma_raw)
            r_guard = (2.0 ** (1.0 / 6.0)) * sigma

        r_floor = float(wca_cfg.get("r_floor", prior_params.get("wca_r_floor", 0.5)))

        return {
            "wca_epsilon": jnp.asarray(epsilon, dtype=jnp.float32),
            "wca_sigma": jnp.asarray(sigma, dtype=jnp.float32),
            "wca_r_cut": jnp.asarray(r_guard, dtype=jnp.float32),
            "wca_r_floor": jnp.asarray(r_floor, dtype=jnp.float32),
        }

    def _init_safety_prior_params(self, prior_params: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Initialize attractive safety prior parameters."""
        fene_cfg = prior_params.get("fene", {})
        if not isinstance(fene_cfg, dict):
            fene_cfg = {}
        leash_cfg = prior_params.get("leash", {})
        if not isinstance(leash_cfg, dict):
            leash_cfg = {}

        d_safe_raw = leash_cfg.get("d_safe", prior_params.get("leash_d_safe", 0.0))
        try:
            d_safe = float(d_safe_raw)
        except (TypeError, ValueError):
            d_safe = 0.0

        return {
            "fene_r0": jnp.asarray(fene_cfg.get("r0", 3.8), dtype=jnp.float32),
            "fene_R0": jnp.asarray(fene_cfg.get("R0", 1.5), dtype=jnp.float32),
            "fene_k": jnp.asarray(fene_cfg.get("k", 300.0), dtype=jnp.float32),
            "fene_wall_energy": jnp.asarray(
                fene_cfg.get("wall_energy", 1.0e6), dtype=jnp.float32
            ),
            "fene_eps": jnp.asarray(fene_cfg.get("eps", 1.0e-6), dtype=jnp.float32),
            "leash_d_safe": jnp.asarray(d_safe, dtype=jnp.float32),
            "leash_k_safe": jnp.asarray(leash_cfg.get("k_safe", 0.2), dtype=jnp.float32),
        }

    def _init_spline_priors(self, spline_path: str, config):
        """Initialize spline-based priors from NPZ file."""
        self.uses_splines = True

        # Resolve relative paths — try config directory, then CWD
        spline_path = Path(spline_path)
        if not spline_path.is_absolute():
            config_dir = config.config_path.parent
            from_config = config_dir / spline_path
            from_cwd = Path.cwd() / spline_path
            if from_config.exists():
                spline_path = from_config
            elif from_cwd.exists():
                model_logger.warning(
                    f"Spline file not found relative to config dir ({config_dir}); "
                    f"using CWD-relative path: {from_cwd}"
                )
                spline_path = from_cwd
            else:
                spline_path = from_config  # will fail below with a clear message

        if not spline_path.exists():
            raise FileNotFoundError(f"Spline prior file not found: {spline_path}")

        model_logger.info(f"Loading spline priors from: {spline_path}")
        spline_data = np.load(str(spline_path), allow_pickle=True)

        # Bond spline (global)
        self.bond_knots = jnp.asarray(spline_data["bond_knots"], dtype=jnp.float32)
        self.bond_coeffs = jnp.asarray(spline_data["bond_coeffs"], dtype=jnp.float32)

        # Angle spline (global fallback)
        self.angle_knots = jnp.asarray(spline_data["angle_knots"], dtype=jnp.float32)
        self.angle_coeffs = jnp.asarray(spline_data["angle_coeffs"], dtype=jnp.float32)

        # Angle splines (per-AA, if available and enabled by config)
        file_has_type_angles = bool(spline_data.get("residue_specific_angles", False))
        cfg_wants_type_angles = bool(
            config.get("model", "priors", "residue_specific_angles", default=file_has_type_angles)
        )
        self.residue_specific_angles = bool(file_has_type_angles and cfg_wants_type_angles)

        if cfg_wants_type_angles and not file_has_type_angles:
            model_logger.warning(
                "Config requests residue_specific_angles=true, but spline file has no angle_type_* arrays. "
                "Falling back to global angle spline."
            )

        if self.residue_specific_angles:
            self.angle_type_knots = jnp.asarray(spline_data["angle_type_knots"], dtype=jnp.float32)
            self.angle_type_coeffs = jnp.asarray(spline_data["angle_type_coeffs"], dtype=jnp.float32)
            self.angle_type_mask = jnp.asarray(spline_data["angle_type_mask"], dtype=jnp.float32)
            n_types = int(spline_data.get("angle_n_types", self.angle_type_mask.shape[0]))
            model_logger.info(f"Residue-specific angle priors: {n_types} types, "
                            f"{int(self.angle_type_mask.sum())} with own splines")
        else:
            model_logger.info("Global angle prior (no residue-specific typing)")

        # Dihedral spline (global)
        self.dih_knots = jnp.asarray(spline_data["dih_knots"], dtype=jnp.float32)
        self.dih_coeffs = jnp.asarray(spline_data["dih_coeffs"], dtype=jnp.float32)

        # Only repulsive params from YAML (still parametric)
        prior_params = config.get_prior_params()
        self.params = {
            "epsilon": jnp.asarray(prior_params.get("epsilon", 1.0), dtype=jnp.float32),
            "sigma": jnp.asarray(prior_params.get("sigma", 3.0), dtype=jnp.float32),
            # Excluded volume params (softer than long-range repulsion, for sep 2-5)
            "epsilon_ex": jnp.asarray(prior_params.get("epsilon_ex", 1.0), dtype=jnp.float32),
            "sigma_ex": jnp.asarray(prior_params.get("sigma_ex", 3.5), dtype=jnp.float32),
            # Hard repulsion params (n=12 steep repulsion)
            "epsilon_hard": jnp.asarray(prior_params.get("epsilon_hard", 1.0), dtype=jnp.float32),
            "sigma_hard": jnp.asarray(prior_params.get("sigma_hard", 3.0), dtype=jnp.float32),
        }
        self.params.update(self._init_lj_params(prior_params))
        self.params.update(self._init_new_term_params(prior_params))
        self.params.update(self._init_wca_params(prior_params))
        self.params.update(self._init_safety_prior_params(prior_params))

        model_logger.info("Spline priors loaded: bond, angle, dihedral (repulsive stays parametric)")

    def _init_parametric_priors(self, config):
        """Initialize parametric priors from config YAML."""
        self.uses_splines = False
        self.residue_specific_angles = False

        prior_params = config.get_prior_params()
        self.params = {
            "r0": jnp.asarray(prior_params.get("r0", 3.8), dtype=jnp.float32),
            "kr": jnp.asarray(prior_params.get("kr", 150.0), dtype=jnp.float32),
            "a": jnp.asarray(prior_params.get("a", [0.0]), dtype=jnp.float32),
            "b": jnp.asarray(prior_params.get("b", [0.0]), dtype=jnp.float32),
            "epsilon": jnp.asarray(prior_params.get("epsilon", 1.0), dtype=jnp.float32),
            "sigma": jnp.asarray(prior_params.get("sigma", 3.0), dtype=jnp.float32),
            "k_dih": jnp.asarray(prior_params.get("k_dih", [0.5]), dtype=jnp.float32),
            "gamma_dih": jnp.asarray(prior_params.get("gamma_dih", [0.0]), dtype=jnp.float32),
            # Excluded volume params (softer than long-range repulsion, for sep 2-5)
            "epsilon_ex": jnp.asarray(prior_params.get("epsilon_ex", 1.0), dtype=jnp.float32),
            "sigma_ex": jnp.asarray(prior_params.get("sigma_ex", 3.5), dtype=jnp.float32),
            # Hard repulsion params (n=12 steep repulsion)
            "epsilon_hard": jnp.asarray(prior_params.get("epsilon_hard", 1.0), dtype=jnp.float32),
            "sigma_hard": jnp.asarray(prior_params.get("sigma_hard", 3.0), dtype=jnp.float32),
        }
        self.params.update(self._init_lj_params(prior_params))
        self.params.update(self._init_new_term_params(prior_params))
        self.params.update(self._init_wca_params(prior_params))
        self.params.update(self._init_safety_prior_params(prior_params))

    def compute_bond_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute bond stretching energy.

        Spline mode: evaluates cubic spline PMF.
        Parametric mode: E_bond = 0.5 * kr * sum[ (r - r0)^2 ]

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total bond energy (scalar)
        """
        p = params if params is not None else self.params
        bi, bj = self.bonds[:, 0], self.bonds[:, 1]
        Ri, Rj = R[bi], R[bj]

        # Mask: both atoms must be valid
        bond_valid = (mask[bi] * mask[bj]) > 0
        bond_valid = bond_valid & _same_segment_mask(self.bonds, segment_id)

        # Compute distances (free-space: displacement is subtraction)
        dR = Ri - Rj
        r = _safe_norm(dR)

        if self.uses_splines:
            U_bond = evaluate_cubic_spline(r, self.bond_knots, self.bond_coeffs)
            E_bond = jnp.sum(jnp.where(bond_valid, U_bond, 0.0))
        else:
            # Harmonic energy with jnp.where for forward pass NaN prevention
            bond_energy = (r - p["r0"]) ** 2
            E_bond = 0.5 * p["kr"] * jnp.sum(jnp.where(bond_valid, bond_energy, 0.0))

        return E_bond

    def compute_angle_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute angle bending energy.

        Spline mode: evaluates cubic spline PMF (optionally residue-specific).
        Parametric mode: Fourier series.

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,). Required for residue-specific angles.
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total angle energy (scalar)
        """
        p = params if params is not None else self.params
        ia, ib, ic = self.angles[:, 0], self.angles[:, 1], self.angles[:, 2]

        # Mask: all three atoms must be valid
        angle_valid = (mask[ia] * mask[ib] * mask[ic]) > 0
        angle_valid = angle_valid & _same_segment_mask(self.angles, segment_id)

        # Compute angles
        theta = _compute_angles(R, self.angles, self.displacement)

        # CRITICAL FIX: Block gradients for invalid angles!
        # For padded atoms at same location: d(norm)/d(R) = v/||v|| is undefined when v=0.
        # Even though this is multiplied by 0 in the chain rule, 0 * NaN = NaN.
        # By applying stop_gradient to theta for invalid angles, we block NaN gradients.
        theta = jnp.where(angle_valid, theta, jax.lax.stop_gradient(theta))

        if self.uses_splines:
            if self.residue_specific_angles and species is not None:
                central_species = species[self.angles[:, 1]]
                central_species = jnp.clip(
                    central_species,
                    0,
                    self.angle_type_mask.shape[0] - 1
                ).astype(jnp.int32)
                U_angle = evaluate_cubic_spline_by_type(
                    theta, central_species,
                    self.angle_type_knots, self.angle_type_coeffs, self.angle_type_mask,
                    self.angle_knots, self.angle_coeffs,
                )
            else:
                U_angle = evaluate_cubic_spline(theta, self.angle_knots, self.angle_coeffs)
        else:
            # Fourier series energy
            U_angle = _angular_fourier_energy(theta, p["a"], p["b"])

        # Use jnp.where to avoid NaN propagation in forward pass
        E_angle = jnp.sum(jnp.where(angle_valid, U_angle, 0.0))

        return E_angle

    def compute_repulsive_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute soft-sphere repulsive energy for non-bonded pairs.

        Always parametric: E_rep = epsilon * sum[ (sigma / r)^4 ]

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total repulsive energy (scalar)
        """
        p = params if params is not None else self.params
        pi, pj = self.rep_pairs[:, 0], self.rep_pairs[:, 1]

        # Mask: both atoms must be valid
        rep_valid = (mask[pi] * mask[pj]) > 0
        rep_valid = rep_valid & _same_segment_mask(self.rep_pairs, segment_id)

        # Compute distances (free-space: displacement is subtraction)
        Rp_i, Rp_j = R[pi], R[pj]
        dR_rep = Rp_i - Rp_j
        r_rep = _safe_norm(dR_rep)

        # CRITICAL FIX: Block gradients for invalid pairs!
        # For padded atoms at same location: d(norm)/d(R) = v/||v|| is undefined when v=0.
        # Even with _safe_norm, we still want to block gradient flow for invalid pairs.
        r_rep = jnp.where(rep_valid, r_rep, jax.lax.stop_gradient(r_rep))

        # Avoid interactions with padded atoms (set large distance for forward pass)
        r_rep = jnp.where(rep_valid, r_rep, 1e6)

        # Avoid division by zero
        r_min = jnp.array(1e-3, dtype=R.dtype)
        r_safe = jnp.maximum(r_rep, r_min)

        # Soft-sphere repulsion: (sigma/r)^4
        rep_term = (p["sigma"] / r_safe) ** 4
        E_rep = p["epsilon"] * jnp.sum(jnp.where(rep_valid, rep_term, 0.0))

        return E_rep

    def compute_repulsive_hard_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute hard-sphere repulsive energy using (sigma/r)^12 potential.

        This is a much steeper, more short-range repulsion than compute_repulsive_energy.
        U(r) = epsilon * (sigma / r)^12

        Intended to create a hard barrier that beads cannot easily overcome.
        The steep n=12 power means the force grows very rapidly as r decreases.

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total hard repulsive energy (scalar)
        """
        p = params if params is not None else self.params
        pi, pj = self.rep_pairs[:, 0], self.rep_pairs[:, 1]

        rep_valid = (mask[pi] * mask[pj]) > 0
        rep_valid = rep_valid & _same_segment_mask(self.rep_pairs, segment_id)

        Rp_i, Rp_j = R[pi], R[pj]
        dR_rep = Rp_i - Rp_j
        r_rep = _safe_norm(dR_rep)

        r_rep = jnp.where(rep_valid, r_rep, jax.lax.stop_gradient(r_rep))
        r_rep = jnp.where(rep_valid, r_rep, 1e6)

        r_min = jnp.array(1e-3, dtype=R.dtype)
        r_safe = jnp.maximum(r_rep, r_min)

        epsilon_hard = p.get("epsilon_hard", p.get("epsilon", 1.0))
        sigma_hard = p.get("sigma_hard", p.get("sigma", 3.0))

        rep_term = (sigma_hard / r_safe) ** 12
        E_rep = epsilon_hard * jnp.sum(jnp.where(rep_valid, rep_term, 0.0))

        return E_rep

    def compute_excluded_volume_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute soft excluded volume for nearby residues (sequence separation 2-5).

        These residues are too close for the long-range repulsion (which starts
        at sep ≥ 6) but are not directly constrained by bonds/angles/dihedrals
        to prevent unphysical backbone self-intersection.

        Uses softer parameters than the long-range repulsion:
        - sigma_ex ~ 3.5 Å  (vs sigma=3.0 for long-range)
        - epsilon_ex ~ 1.0 kcal/mol (same scale, but can be tuned)

        E_ex = epsilon_ex * sum[ (sigma_ex / r)^4 ]  (for valid pairs with sep 2-5)

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total excluded volume energy (scalar)
        """
        p = params if params is not None else self.params
        pi, pj = self.excluded_vol_pairs[:, 0], self.excluded_vol_pairs[:, 1]

        # Mask: both atoms must be valid
        ex_valid = (mask[pi] * mask[pj]) > 0
        ex_valid = ex_valid & _same_segment_mask(self.excluded_vol_pairs, segment_id)

        Rp_i, Rp_j = R[pi], R[pj]
        dR_ex = Rp_i - Rp_j
        r_ex = _safe_norm(dR_ex)

        # Block gradients for invalid pairs
        r_ex = jnp.where(ex_valid, r_ex, jax.lax.stop_gradient(r_ex))

        # Set large distance for padded/invalid pairs (no energy contribution)
        r_ex = jnp.where(ex_valid, r_ex, 1e6)

        r_safe = jnp.maximum(r_ex, jnp.array(1e-3, dtype=R.dtype))

        ex_term = (p["sigma_ex"] / r_safe) ** 4
        E_ex = p["epsilon_ex"] * jnp.sum(jnp.where(ex_valid, ex_term, 0.0))

        return E_ex

    def compute_lj_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute full Lennard-Jones energy for configured nonbonded pairs."""
        p = params if params is not None else self.params
        if self.lj_pairs.shape[0] == 0:
            return jnp.array(0.0, dtype=R.dtype)

        pi, pj = self.lj_pairs[:, 0], self.lj_pairs[:, 1]
        valid = (mask[pi] * mask[pj]) > 0
        valid = valid & _same_segment_mask(self.lj_pairs, segment_id)

        dR = R[pi] - R[pj]
        r = _safe_norm(dR)
        r = jnp.where(valid, r, jax.lax.stop_gradient(r))
        r_eval = jnp.where(valid, r, 1e6)
        r_safe = jnp.maximum(r_eval, jnp.array(1e-3, dtype=R.dtype))

        sigma = jnp.asarray(p["lj_sigma"], dtype=R.dtype)
        epsilon = jnp.asarray(p["lj_epsilon"], dtype=R.dtype)
        sr6 = (sigma / r_safe) ** 6
        U = 4.0 * epsilon * (sr6 ** 2 - sr6)
        return jnp.sum(jnp.where(valid, U, 0.0))

    def compute_wca_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute a zero-above-cutoff WCA clash-guard energy.

        E_WCA = 4*epsilon*((sigma/r)^12 - (sigma/r)^6) + epsilon for
        r < r_cut, and 0 otherwise. By default r_cut is the requested guard
        distance and sigma = r_cut / 2^(1/6), making the energy and force zero
        at the guard boundary.
        """
        p = params if params is not None else self.params
        if self.wca_pairs.shape[0] == 0:
            return jnp.array(0.0, dtype=R.dtype)

        pi, pj = self.wca_pairs[:, 0], self.wca_pairs[:, 1]
        valid = (mask[pi] * mask[pj]) > 0
        valid = valid & _same_segment_mask(self.wca_pairs, segment_id)

        dR = R[pi] - R[pj]
        r = _safe_norm(dR)
        r = jnp.where(valid, r, jax.lax.stop_gradient(r))
        r_eval = jnp.where(valid, r, 1e6)

        r_floor = jnp.asarray(p["wca_r_floor"], dtype=R.dtype)
        r_safe = jnp.maximum(r_eval, r_floor)
        sigma = jnp.asarray(p["wca_sigma"], dtype=R.dtype)
        epsilon = jnp.asarray(p["wca_epsilon"], dtype=R.dtype)
        r_cut = jnp.asarray(p["wca_r_cut"], dtype=R.dtype)

        sr6 = (sigma / r_safe) ** 6
        U = 4.0 * epsilon * (sr6 ** 2 - sr6) + epsilon
        active = valid & (r_eval < r_cut)
        return jnp.sum(jnp.where(active, U, 0.0))

    def compute_local_in_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute explicit one-sided local i,i+n upper-wall energy."""
        if self.local_in_pairs.shape[0] == 0:
            return jnp.array(0.0, dtype=R.dtype)
        pi, pj = self.local_in_pairs[:, 0], self.local_in_pairs[:, 1]
        valid = (mask[pi] * mask[pj]) > 0
        valid = valid & _same_segment_mask(self.local_in_pairs, segment_id)
        r = _safe_norm(R[pi] - R[pj])
        r = jnp.where(valid, r, jax.lax.stop_gradient(r))
        stretch = jnp.maximum(
            r - (self.local_in_r0.astype(R.dtype) + self.local_in_margin.astype(R.dtype)),
            0.0,
        )
        E = 0.5 * self.local_in_k.astype(R.dtype) * stretch * stretch
        return jnp.sum(jnp.where(valid, E, 0.0))

    def compute_local_bond_in_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute explicit symmetric harmonic local i,i+n bond energy."""
        if self.local_bond_in_pairs.shape[0] == 0:
            return jnp.array(0.0, dtype=R.dtype)
        pi, pj = self.local_bond_in_pairs[:, 0], self.local_bond_in_pairs[:, 1]
        valid = (mask[pi] * mask[pj]) > 0
        valid = valid & _same_segment_mask(self.local_bond_in_pairs, segment_id)
        r = _safe_norm(R[pi] - R[pj])
        r = jnp.where(valid, r, jax.lax.stop_gradient(r))
        E = 0.5 * self.local_bond_in_k.astype(R.dtype) * (
            r - self.local_bond_in_r0.astype(R.dtype)
        ) ** 2
        return jnp.sum(jnp.where(valid, E, 0.0))

    def compute_local_angle_in_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute explicit local bending energy in invariant cos(theta)."""
        if self.local_angle_in_angles.shape[0] == 0:
            return jnp.array(0.0, dtype=R.dtype)
        i, j, k = (
            self.local_angle_in_angles[:, 0],
            self.local_angle_in_angles[:, 1],
            self.local_angle_in_angles[:, 2],
        )
        valid = (mask[i] * mask[j] * mask[k]) > 0
        valid = valid & _same_segment_mask(self.local_angle_in_angles, segment_id)
        u, v = R[i] - R[j], R[k] - R[j]
        cos_theta = jnp.sum(u * v, axis=-1) / (_safe_norm(u) * _safe_norm(v) + 1.0e-12)
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
        cos_theta = jnp.where(valid, cos_theta, jax.lax.stop_gradient(cos_theta))
        E = 0.5 * self.local_angle_in_k.astype(R.dtype) * (
            cos_theta - self.local_angle_in_cos0.astype(R.dtype)
        ) ** 2
        return jnp.sum(jnp.where(valid, E, 0.0))

    def compute_crowding_wall_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute local smooth-neighbor-count crowding wall energy."""
        if not self.crowding_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        n_atoms = R.shape[0]
        idx = jnp.arange(n_atoms)
        valid = (mask[:, None] * mask[None, :]) > 0
        valid = valid & (jnp.abs(idx[:, None] - idx[None, :]) > self.crowding_min_seq_sep)
        valid = valid & (idx[:, None] != idx[None, :])
        if segment_id is not None:
            same_segment = (segment_id[:, None] >= 0) & (segment_id[:, None] == segment_id[None, :])
            valid = valid & same_segment

        dR = R[:, None, :] - R[None, :, :]
        r = jnp.sqrt(jnp.sum(dR * dR, axis=-1) + jnp.asarray(1.0e-12, dtype=R.dtype))
        a = jnp.maximum(self.crowding_a.astype(R.dtype), jnp.asarray(1.0e-6, dtype=R.dtype))
        s = jax.nn.sigmoid((self.crowding_r0.astype(R.dtype) - r) / a)
        counts = jnp.sum(jnp.where(valid, s, 0.0), axis=1)
        excess = jnp.maximum(counts - self.crowding_N_max.astype(R.dtype), 0.0)
        active = mask > 0
        E = self.crowding_k.astype(R.dtype) * excess ** self.crowding_p.astype(R.dtype)
        return jnp.sum(jnp.where(active, E, 0.0))

    def compute_local_torsion_fourier_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute a bounded low-order periodic phi/psi prior.

        The expansion uses marginal sin/cos terms and their products.  It is a
        conservative force-matched potential representation, not a histogram
        or an FES lookup: all coefficients are obtained from coordinate-force
        regression in the external fitter.
        """
        if (
            self.local_torsion_order <= 0
            or self.local_torsion_phi_indices.shape[0] == 0
            or self.local_torsion_psi_indices.shape[0] == 0
        ):
            return jnp.array(0.0, dtype=R.dtype)

        def _dihedral(indices: jax.Array) -> jax.Array:
            p0, p1, p2, p3 = R[indices[0]], R[indices[1]], R[indices[2]], R[indices[3]]
            b0 = -(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            b1u = b1 / (_safe_norm(b1) + jnp.asarray(1.0e-12, dtype=R.dtype))
            v = b0 - jnp.sum(b0 * b1u) * b1u
            w = b2 - jnp.sum(b2 * b1u) * b1u
            return _safe_atan2(jnp.sum(jnp.cross(b1u, v) * w), jnp.sum(v * w))

        phi_idx = self.local_torsion_phi_indices
        psi_idx = self.local_torsion_psi_indices
        n_terms = min(phi_idx.shape[0], psi_idx.shape[0])
        phi_idx, psi_idx = phi_idx[:n_terms], psi_idx[:n_terms]
        valid = (jnp.prod(mask[phi_idx], axis=1) > 0) & (jnp.prod(mask[psi_idx], axis=1) > 0)
        valid = valid & _same_segment_mask(phi_idx, segment_id) & _same_segment_mask(psi_idx, segment_id)
        phi = jax.vmap(_dihedral)(phi_idx)
        psi = jax.vmap(_dihedral)(psi_idx)
        phi = jnp.where(valid, phi, jax.lax.stop_gradient(phi))
        psi = jnp.where(valid, psi, jax.lax.stop_gradient(psi))

        harmonic = jnp.arange(1, self.local_torsion_order + 1, dtype=R.dtype)
        cphi, sphi = jnp.cos(phi[:, None] * harmonic), jnp.sin(phi[:, None] * harmonic)
        cpsi, spsi = jnp.cos(psi[:, None] * harmonic), jnp.sin(psi[:, None] * harmonic)
        marginal_basis = jnp.concatenate([cphi, sphi, cpsi, spsi], axis=1)
        coupled_basis = jnp.concatenate(
            [
                (cphi[:, :, None] * cpsi[:, None, :]).reshape((n_terms, -1)),
                (cphi[:, :, None] * spsi[:, None, :]).reshape((n_terms, -1)),
                (sphi[:, :, None] * cpsi[:, None, :]).reshape((n_terms, -1)),
                (sphi[:, :, None] * spsi[:, None, :]).reshape((n_terms, -1)),
            ],
            axis=1,
        )
        e_marginal = jnp.sum(marginal_basis * self.local_torsion_marginal_coeff.astype(R.dtype), axis=1)
        e_coupled = jnp.sum(coupled_basis * self.local_torsion_coupled_coeff.astype(R.dtype), axis=1)
        energy = self.local_torsion_marginal_scale.astype(R.dtype) * e_marginal
        energy = energy + self.local_torsion_coupled_scale.astype(R.dtype) * e_coupled
        return jnp.sum(jnp.where(valid, energy, 0.0))

    def compute_five_particle_flat_bottom_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute fitted flat-bottom safety prior for the 5-particle toy system."""
        if not self.five_particle_flat_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        p = self.five_particle_flat_params

        def _pair_dist_energy(
            pairs: jax.Array,
            lower: jax.Array,
            upper: jax.Array,
            sigma: jax.Array,
            k: jax.Array,
        ) -> jax.Array:
            if pairs.shape[0] == 0:
                return jnp.array(0.0, dtype=R.dtype)
            pi, pj = pairs[:, 0], pairs[:, 1]
            valid = (mask[pi] * mask[pj]) > 0
            valid = valid & _same_segment_mask(pairs, segment_id)
            r = _safe_norm(R[pi] - R[pj])
            r = jnp.where(valid, r, jax.lax.stop_gradient(r))
            z2 = _flat_bottom_quadratic(
                r,
                lower.astype(R.dtype),
                upper.astype(R.dtype),
                sigma.astype(R.dtype),
            )
            return 0.5 * k.astype(R.dtype) * jnp.sum(jnp.where(valid, z2, 0.0))

        E_bond = _pair_dist_energy(
            self.five_particle_flat_bonds,
            p["bond_p01"],
            p["bond_p99"],
            p["bond_sigma"],
            p["bond_k"],
        )

        angle_idx = self.five_particle_flat_angles
        if angle_idx.shape[0] == 0:
            E_angle = jnp.array(0.0, dtype=R.dtype)
        else:
            ia, ib, ic = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
            valid = (mask[ia] * mask[ib] * mask[ic]) > 0
            valid = valid & _same_segment_mask(angle_idx, segment_id)
            v1 = R[ia] - R[ib]
            v2 = R[ic] - R[ib]
            cos_theta = jnp.sum(v1 * v2, axis=1) / (
                _safe_norm(v1) * _safe_norm(v2) + jnp.asarray(1.0e-12, dtype=R.dtype)
            )
            cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
            cos_theta = jnp.where(valid, cos_theta, jax.lax.stop_gradient(cos_theta))
            z2 = _flat_bottom_quadratic(
                cos_theta,
                p["angle_cos_p01"].astype(R.dtype),
                p["angle_cos_p99"].astype(R.dtype),
                p["angle_sigma"].astype(R.dtype),
            )
            E_angle = 0.5 * p["angle_k"].astype(R.dtype) * jnp.sum(jnp.where(valid, z2, 0.0))

        E_fragment = jnp.array(0.0, dtype=R.dtype)
        for frag in self.five_particle_flat_fragments.values():
            k_eff = p["fragment_k_base"] * frag["weight"]
            E_fragment = E_fragment + _pair_dist_energy(
                frag["pairs"],
                frag["p01"],
                frag["p99"],
                frag["sigma"],
                k_eff,
            )

        pairs = self.five_particle_flat_pairs
        if pairs.shape[0] == 0:
            E_wca = E_collapse = E_shell = jnp.array(0.0, dtype=R.dtype)
        else:
            pi, pj = pairs[:, 0], pairs[:, 1]
            valid = (mask[pi] * mask[pj]) > 0
            valid = valid & _same_segment_mask(pairs, segment_id)
            r = _safe_norm(R[pi] - R[pj])
            r = jnp.where(valid, r, jax.lax.stop_gradient(r))
            r_eval = jnp.where(valid, r, jnp.asarray(1.0e6, dtype=R.dtype))

            r_floor = p["wca_r_floor"].astype(R.dtype)
            r_safe = jnp.maximum(r_eval, r_floor)
            sigma = p["wca_sigma"].astype(R.dtype)
            epsilon = p["wca_epsilon"].astype(R.dtype)
            sr6 = (sigma / r_safe) ** 6
            U_wca = 4.0 * epsilon * (sr6 ** 2 - sr6) + epsilon
            E_wca = jnp.sum(jnp.where(valid & (r_eval < p["wca_r_cut"].astype(R.dtype)), U_wca, 0.0))

            collapse = jnp.maximum(p["collapse_r_guard"].astype(R.dtype) - r_eval, 0.0)
            E_collapse = 0.5 * p["collapse_k"].astype(R.dtype) * jnp.sum(
                jnp.where(valid, collapse ** 2, 0.0)
            )

            shell_low = jnp.maximum(p["shell_p01"].astype(R.dtype) - r_eval, 0.0)
            shell_high = jnp.maximum(r_eval - p["shell_p99"].astype(R.dtype), 0.0)
            E_shell = 0.5 * p["shell_k_repulsion"].astype(R.dtype) * jnp.sum(
                jnp.where(valid, shell_low ** 2, 0.0)
            )
            E_shell = E_shell + 0.5 * p["shell_k_pull"].astype(R.dtype) * jnp.sum(
                jnp.where(valid, shell_high ** 2, 0.0)
            )

        return (
            p["component_bond"].astype(R.dtype) * E_bond
            + p["component_angle"].astype(R.dtype) * E_angle
            + p["component_fragment"].astype(R.dtype) * E_fragment
            + p["component_wca"].astype(R.dtype) * E_wca
            + p["component_collapse"].astype(R.dtype) * E_collapse
            + p["component_pair_shell"].astype(R.dtype) * E_shell
        )

    def compute_aa_integrated_baseline_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Evaluate the frozen conservative AA-integrated residual baseline."""
        del segment_id
        if not self.aa_integrated_baseline_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        return self.aa_integrated_baseline_scale.astype(R.dtype) * aa_integrated_baseline_energy(
            R,
            mask,
            self.aa_integrated_baseline_spec,
            self.aa_integrated_baseline_component_scales,
        )

    def compute_ala2_feature_recovery_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute complement-gated Ala2 local feature restoring prior."""
        if not self.ala2_feature_recovery_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        p = self.ala2_feature_recovery_params

        def _standard_dihedral_deg(indices: jax.Array) -> jax.Array:
            p0 = R[indices[0]]
            p1 = R[indices[1]]
            p2 = R[indices[2]]
            p3 = R[indices[3]]
            b0 = -(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            b1u = b1 / (_safe_norm(b1) + jnp.asarray(1.0e-12, dtype=R.dtype))
            v = b0 - jnp.sum(b0 * b1u) * b1u
            w = b2 - jnp.sum(b2 * b1u) * b1u
            x = jnp.sum(v * w)
            y = jnp.sum(jnp.cross(b1u, v) * w)
            angle = jnp.degrees(_safe_atan2(y, x))
            return jnp.mod(angle + 180.0, 360.0) - 180.0

        def _signed_volume(indices: jax.Array) -> jax.Array:
            p0 = R[indices[0]]
            a = R[indices[1]] - p0
            b = R[indices[2]] - p0
            c = R[indices[3]] - p0
            denom = _safe_norm(a) * _safe_norm(b) * _safe_norm(c) + jnp.asarray(1.0e-12, dtype=R.dtype)
            return jnp.sum(jnp.cross(a, b) * c) / denom

        E_torsion = jnp.array(0.0, dtype=R.dtype)
        torsion_idx = p["torsion_indices"]
        if torsion_idx.shape[0] > 0:
            valid = (jnp.prod(mask[torsion_idx], axis=1) > 0)
            valid = valid & _same_segment_mask(torsion_idx, segment_id)
            phi = jax.vmap(_standard_dihedral_deg)(torsion_idx).astype(R.dtype)
            center = p["torsion_center_deg"].astype(R.dtype)
            delta = jnp.abs(jnp.mod(phi - center + 180.0, 360.0) - 180.0)
            excess = jnp.maximum(delta - p["torsion_halfwidth_deg"].astype(R.dtype), 0.0)
            z2 = (excess / (p["torsion_scale_deg"].astype(R.dtype) + 1.0e-8)) ** 2
            E_torsion = 0.5 * p["k_torsion"].astype(R.dtype) * jnp.sum(
                jnp.where(valid, z2, 0.0)
            )

        E_volume = jnp.array(0.0, dtype=R.dtype)
        volume_idx = p["volume_indices"]
        if volume_idx.shape[0] > 0:
            valid = (jnp.prod(mask[volume_idx], axis=1) > 0)
            valid = valid & _same_segment_mask(volume_idx, segment_id)
            vol = jax.vmap(_signed_volume)(volume_idx).astype(R.dtype)
            low = p["volume_low"].astype(R.dtype)
            high = p["volume_high"].astype(R.dtype)
            excess = jnp.maximum(low - vol, 0.0) + jnp.maximum(vol - high, 0.0)
            z2 = (excess / (p["volume_scale"].astype(R.dtype) + 1.0e-8)) ** 2
            E_volume = 0.5 * p["k_volume"].astype(R.dtype) * jnp.sum(
                jnp.where(valid, z2, 0.0)
            )

        # Complement of the Ala2 torsion-support gate. Stop-gradient by default:
        # the activation decides when the restorer acts, but does not itself
        # introduce a Ramachandran-density force.
        phi_support = _standard_dihedral_deg(p["support_phi_indices"]).astype(R.dtype)
        psi_support = _standard_dihedral_deg(p["support_psi_indices"]).astype(R.dtype)
        dphi = jnp.mod(
            phi_support - p["support_reference_phi"].astype(R.dtype) + 180.0,
            360.0,
        ) - 180.0
        dpsi = jnp.mod(
            psi_support - p["support_reference_psi"].astype(R.dtype) + 180.0,
            360.0,
        ) - 180.0
        dist = jnp.sqrt(dphi * dphi + dpsi * dpsi)
        kth = max(0, min(int(p["support_k"]), int(dist.shape[0])) - 1)
        score = jnp.partition(dist, kth)[kth]
        width = jnp.maximum(
            p["support_offset_score_deg"].astype(R.dtype)
            - p["support_onset_score_deg"].astype(R.dtype),
            jnp.asarray(1.0e-6, dtype=R.dtype),
        )
        x = jnp.clip(
            (score - p["support_onset_score_deg"].astype(R.dtype)) / width,
            0.0,
            1.0,
        )
        activation = x * x * (3.0 - 2.0 * x)
        activation = p["activation_floor"].astype(R.dtype) + (
            1.0 - p["activation_floor"].astype(R.dtype)
        ) * activation ** p["activation_power"].astype(R.dtype)
        if self.ala2_feature_recovery_stop_gradient_activation:
            activation = jax.lax.stop_gradient(activation)
        return p["energy_scale"].astype(R.dtype) * activation * (E_torsion + E_volume)

    def compute_ala2_rama_recovery_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute strong Ala2 phi/psi support restoring potential."""
        if not self.ala2_rama_recovery_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        p = self.ala2_rama_recovery_params

        def _standard_dihedral_deg(indices: jax.Array) -> jax.Array:
            p0 = R[indices[0]]
            p1 = R[indices[1]]
            p2 = R[indices[2]]
            p3 = R[indices[3]]
            b0 = -(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            b1u = b1 / (_safe_norm(b1) + jnp.asarray(1.0e-12, dtype=R.dtype))
            v = b0 - jnp.sum(b0 * b1u) * b1u
            w = b2 - jnp.sum(b2 * b1u) * b1u
            x = jnp.sum(v * w)
            y = jnp.sum(jnp.cross(b1u, v) * w)
            angle = jnp.degrees(_safe_atan2(y, x))
            return jnp.mod(angle + 180.0, 360.0) - 180.0

        phi_idx = p["phi_indices"]
        psi_idx = p["psi_indices"]
        valid_phi = (jnp.prod(mask[phi_idx]) > 0) & _same_segment_mask(phi_idx[None, :], segment_id)[0]
        valid_psi = (jnp.prod(mask[psi_idx]) > 0) & _same_segment_mask(psi_idx[None, :], segment_id)[0]
        valid = valid_phi & valid_psi

        phi = _standard_dihedral_deg(phi_idx).astype(R.dtype)
        psi = _standard_dihedral_deg(psi_idx).astype(R.dtype)

        dphi = jnp.mod(phi - p["reference_phi"].astype(R.dtype) + 180.0, 360.0) - 180.0
        dpsi = jnp.mod(psi - p["reference_psi"].astype(R.dtype) + 180.0, 360.0) - 180.0
        dist = jnp.sqrt(dphi * dphi + dpsi * dpsi)
        kth = max(0, min(int(p["k_nearest"]), int(dist.shape[0])) - 1)
        score = jnp.partition(dist, kth)[kth]

        onset = p["onset_score_deg"].astype(R.dtype)
        offset = p["offset_score_deg"].astype(R.dtype)
        width = jnp.maximum(offset - onset, jnp.asarray(1.0e-6, dtype=R.dtype))
        x = jnp.clip((score - onset) / width, 0.0, 1.0)
        activation = x * x * (3.0 - 2.0 * x)
        activation = p["activation_floor"].astype(R.dtype) + (
            1.0 - p["activation_floor"].astype(R.dtype)
        ) * activation ** p["activation_power"].astype(R.dtype)
        if self.ala2_rama_recovery_stop_gradient_activation:
            activation = jax.lax.stop_gradient(activation)

        excess = jnp.maximum(score - onset, 0.0)
        z2 = (excess / (p["scale_deg"].astype(R.dtype) + 1.0e-8)) ** 2
        energy = 0.5 * p["k_restore"].astype(R.dtype) * activation * z2
        energy = jnp.minimum(energy, p["energy_cap"].astype(R.dtype))
        return jnp.where(valid, energy, 0.0)

    def compute_ala2_geometry_support_recovery_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute local pair/type geometry support restoring energy."""
        if not self.ala2_geometry_support_recovery_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        if species is None:
            return jnp.array(0.0, dtype=R.dtype)
        p = self.ala2_geometry_support_recovery_params
        n = R.shape[0]
        idx_i, idx_j = jnp.triu_indices(n, k=1)
        pairs = jnp.stack([idx_i, idx_j], axis=1)
        seq_sep = jnp.abs(idx_i - idx_j)
        valid = (mask[idx_i] * mask[idx_j]) > 0
        valid = valid & (seq_sep >= p["min_sequence_separation"])
        valid = valid & _same_segment_mask(pairs, segment_id)

        n_species = int(p["pair_count"].shape[0])
        si = jnp.clip(species[idx_i], 0, n_species - 1)
        sj = jnp.clip(species[idx_j], 0, n_species - 1)
        seen = p["pair_count"][si, sj] > 0
        lo = p["pair_low"][si, sj].astype(R.dtype)
        hi = p["pair_high"][si, sj].astype(R.dtype)
        width = jnp.maximum(hi - lo, jnp.asarray(1.0e-6, dtype=R.dtype))
        margin = jnp.maximum(
            width * p["pair_margin_fraction"].astype(R.dtype),
            jnp.asarray(1.0e-6, dtype=R.dtype),
        )

        r = _safe_norm(R[idx_i] - R[idx_j])
        r_eval = jnp.where(valid & seen, r, jax.lax.stop_gradient(r))
        lower = jnp.maximum((lo - r_eval) / margin, 0.0)
        upper = jnp.maximum((r_eval - hi) / margin, 0.0)
        score = jnp.maximum(lower, upper)
        excess = jnp.maximum(score - p["score_onset"].astype(R.dtype), 0.0)
        excess = jnp.minimum(excess, p["score_cap"].astype(R.dtype))
        term = 0.5 * p["k_restore"].astype(R.dtype) * excess ** 2
        return jnp.sum(jnp.where(valid & seen, term, 0.0))

    def compute_fene_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute FENE safety bonds for consecutive residues."""
        p = params if params is not None else self.params
        return compute_fene_energy(
            R=R,
            mask=mask,
            bonds=self.fene_pairs,
            r0=p["fene_r0"],
            R0=p["fene_R0"],
            k=p["fene_k"],
            wall_energy=p["fene_wall_energy"],
            eps=p["fene_eps"],
            segment_id=segment_id,
        )

    def compute_leash_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute flat-bottom pair-distance leash energy."""
        p = params if params is not None else self.params
        return compute_leash_energy(
            R=R,
            mask=mask,
            pairs=self.leash_pairs,
            d_safe=p["leash_d_safe"],
            k_safe=p["leash_k_safe"],
            segment_id=segment_id,
        )

    def compute_dh_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute Debye-Huckel energy for configured local sequence pairs."""
        if not self.dh_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        if species is None:
            raise ValueError("DH prior term is enabled but species was not provided.")
        if self.dh_mode != "local_k":
            raise ValueError(
                f"Unsupported model.priors.dh.mode='{self.dh_mode}'. "
                "Only 'local_k' is currently implemented."
            )

        p = params if params is not None else self.params
        return compute_dh_energy(
            R=R,
            mask=mask,
            species=species,
            pairs=self.dh_local_pairs,
            seq_sep=self.dh_local_seq_sep,
            charge_by_species=self.charge_by_species,
            k_dh=p["k_DH"],
            lambda_d=p["lambda_D"],
            w_by_sep=p["dh_w_by_sep"],
            segment_id=segment_id,
        )

    def compute_stickiness_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute typed nonbonded stickiness energy."""
        if not self.stickiness_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        if species is None:
            raise ValueError("Stickiness prior term is enabled but species was not provided.")

        p = params if params is not None else self.params
        alpha = _stickiness_alpha_from_free(
            p["stick_s_free"],
            nonref_group_indices=self.stick_nonref_group_indices,
            reference_group_idx=self.stick_reference_group_idx,
            n_groups=4,
        )
        return compute_stickiness_energy(
            R=R,
            mask=mask,
            species=species,
            pairs=self.stickiness_pairs,
            group_by_species=self.group_by_species,
            alpha=alpha,
            r0=p["stick_r0"],
            sigma=p["stick_sigma"],
            segment_id=segment_id,
        )

    def compute_salt_bridge_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute short-range salt-bridge correction energy."""
        if not self.salt_bridge_enabled:
            return jnp.array(0.0, dtype=R.dtype)
        if species is None:
            raise ValueError("Salt-bridge prior term is enabled but species was not provided.")

        p = params if params is not None else self.params
        return compute_salt_bridge_energy(
            R=R,
            mask=mask,
            species=species,
            pairs=self.salt_bridge_pairs,
            charge_by_species=self.charge_by_species,
            delta_sb=p["salt_delta"],
            r0_sb=p["salt_r0"],
            sigma_sb=p["salt_sigma"],
            segment_id=segment_id,
        )

    def compute_dihedral_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute dihedral torsion energy.

        Spline mode: evaluates periodic cubic spline PMF.
        Parametric mode: periodic cosine.

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total dihedral energy (scalar)
        """
        p = params if params is not None else self.params
        i, j, k, l = self.dihedrals[:, 0], self.dihedrals[:, 1], self.dihedrals[:, 2], self.dihedrals[:, 3]

        # Mask: all four atoms must be valid
        dih_valid = (mask[i] * mask[j] * mask[k] * mask[l]) > 0
        dih_valid = dih_valid & _same_segment_mask(self.dihedrals, segment_id)

        # Compute dihedral angles
        phi = _compute_dihedrals(R, self.dihedrals, self.displacement)

        # CRITICAL FIX: Block gradients for invalid dihedrals!
        # For padded atoms at same location: atan2(0, 0) has UNDEFINED gradients.
        # jnp.where alone doesn't prevent NaN gradients because the gradient of
        # atan2 is computed before being multiplied by the mask (0 * NaN = NaN).
        # By applying stop_gradient to phi for invalid dihedrals, we block NaN
        # gradient propagation at the source.
        phi = jnp.where(dih_valid, phi, jax.lax.stop_gradient(phi))

        if self.uses_splines:
            U_dih = evaluate_cubic_spline_periodic(phi, self.dih_knots, self.dih_coeffs)
        else:
            # Periodic energy
            U_dih = _dihedral_periodic_energy(phi, p["k_dih"], p["gamma_dih"])

        # Use jnp.where to avoid NaN propagation in forward pass
        E_dih = jnp.sum(jnp.where(dih_valid, U_dih, 0.0))

        return E_dih

    def compute_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        """
        Compute all energy components.

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,). Needed for residue-specific angles.
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Dictionary with energy components:
                - E_bond: Bond stretching energy (WEIGHTED)
                - E_angle: Angle bending energy (WEIGHTED)
                - E_repulsive: Repulsive interaction energy (WEIGHTED)
                - E_dihedral: Dihedral torsion energy (WEIGHTED)
                - E_excluded_volume: Nearby-separation excluded volume (WEIGHTED)
                - E_wca: WCA clash-guard term (WEIGHTED)
                - E_fene: FENE chain-safety term (WEIGHTED)
                - E_leash: Flat-bottom global leash term (WEIGHTED)
                - E_dh: Debye-Huckel term (WEIGHTED)
                - E_stickiness: Typed stickiness term (WEIGHTED)
                - E_salt_bridge: Salt-bridge correction term (WEIGHTED)
                - E_total: Sum of all weighted components
        """
        # Compute raw energies
        p = params if params is not None else self.params
        E_bond_raw = self.compute_bond_energy(R, mask, params=p, segment_id=segment_id)
        E_angle_raw = self.compute_angle_energy(
            R, mask, species=species, params=p, segment_id=segment_id
        )
        E_rep_raw = self.compute_repulsive_energy(R, mask, params=p, segment_id=segment_id)
        E_dih_raw = self.compute_dihedral_energy(R, mask, params=p, segment_id=segment_id)
        E_ex_raw = self.compute_excluded_volume_energy(
            R, mask, params=p, segment_id=segment_id
        )
        E_lj_weight = self.weights.get("lj", 0.0)
        E_lj_raw = (
            self.compute_lj_energy(R, mask, params=p, segment_id=segment_id)
            if E_lj_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_wca_weight = self.weights.get("wca", 0.0)
        E_wca_raw = (
            self.compute_wca_energy(R, mask, params=p, segment_id=segment_id)
            if E_wca_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_fene_weight = self.weights.get("fene", 0.0)
        E_fene_raw = (
            self.compute_fene_energy(R, mask, params=p, segment_id=segment_id)
            if E_fene_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_leash_weight = self.weights.get("leash", 0.0)
        E_leash_raw = (
            self.compute_leash_energy(R, mask, params=p, segment_id=segment_id)
            if E_leash_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_dh_raw = self.compute_dh_energy(
            R, mask, species=species, params=p, segment_id=segment_id
        )
        E_stick_raw = self.compute_stickiness_energy(
            R, mask, species=species, params=p, segment_id=segment_id
        )
        E_sb_raw = self.compute_salt_bridge_energy(
            R, mask, species=species, params=p, segment_id=segment_id
        )
        E_local_in_weight = self.weights.get("local_in", 0.0)
        E_local_in_raw = (
            self.compute_local_in_energy(R, mask, segment_id=segment_id)
            if E_local_in_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_local_bond_in_weight = self.weights.get("local_bond_in", 0.0)
        E_local_bond_in_raw = (
            self.compute_local_bond_in_energy(R, mask, segment_id=segment_id)
            if E_local_bond_in_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_local_angle_in_weight = self.weights.get("local_angle_in", 0.0)
        E_local_angle_in_raw = (
            self.compute_local_angle_in_energy(R, mask, segment_id=segment_id)
            if E_local_angle_in_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_crowding_weight = self.weights.get("crowding_wall", 0.0)
        E_crowding_raw = (
            self.compute_crowding_wall_energy(R, mask, segment_id=segment_id)
            if E_crowding_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_local_torsion_weight = self.weights.get("local_torsion_fourier", 0.0)
        E_local_torsion_raw = (
            self.compute_local_torsion_fourier_energy(R, mask, segment_id=segment_id)
            if E_local_torsion_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_rep_hard_weight = self.weights.get("repulsive_hard", 0.0)
        E_rep_hard_raw = (
            self.compute_repulsive_hard_energy(R, mask, params=p, segment_id=segment_id)
            if E_rep_hard_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_5p_flat_weight = self.weights.get("five_particle_flat_bottom", 0.0)
        E_5p_flat_raw = (
            self.compute_five_particle_flat_bottom_energy(R, mask, segment_id=segment_id)
            if E_5p_flat_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_aa_baseline_weight = self.weights.get("aa_integrated_baseline", 0.0)
        E_aa_baseline_raw = (
            self.compute_aa_integrated_baseline_energy(R, mask, segment_id=segment_id)
            if E_aa_baseline_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_ala2_feature_weight = self.weights.get("ala2_feature_recovery", 0.0)
        E_ala2_feature_raw = (
            self.compute_ala2_feature_recovery_energy(R, mask, segment_id=segment_id)
            if E_ala2_feature_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_ala2_rama_weight = self.weights.get("ala2_rama_recovery", 0.0)
        E_ala2_rama_raw = (
            self.compute_ala2_rama_recovery_energy(R, mask, segment_id=segment_id)
            if E_ala2_rama_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_ala2_geom_weight = self.weights.get("ala2_geometry_support_recovery", 0.0)
        E_ala2_geom_raw = (
            self.compute_ala2_geometry_support_recovery_energy(
                R, mask, species=species, segment_id=segment_id
            )
            if E_ala2_geom_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )

        # Apply weights
        E_bond = self.weights["bond"] * E_bond_raw
        E_angle = self.weights["angle"] * E_angle_raw
        E_rep = self.weights["repulsive"] * E_rep_raw
        E_dih = self.weights["dihedral"] * E_dih_raw
        E_ex = self.weights.get("excluded_volume", 1.0) * E_ex_raw
        E_lj = E_lj_weight * E_lj_raw
        E_wca = E_wca_weight * E_wca_raw
        E_fene = E_fene_weight * E_fene_raw
        E_leash = E_leash_weight * E_leash_raw
        E_dh = self.weights.get("dh", 0.0) * E_dh_raw
        E_stick = self.weights.get("stickiness", 0.0) * E_stick_raw
        E_sb = self.weights.get("salt_bridge", 0.0) * E_sb_raw
        E_local_in = E_local_in_weight * E_local_in_raw
        E_local_bond_in = E_local_bond_in_weight * E_local_bond_in_raw
        E_local_angle_in = E_local_angle_in_weight * E_local_angle_in_raw
        E_crowding_wall = E_crowding_weight * E_crowding_raw
        E_local_torsion = E_local_torsion_weight * E_local_torsion_raw
        E_rep_hard = E_rep_hard_weight * E_rep_hard_raw
        E_5p_flat = E_5p_flat_weight * E_5p_flat_raw
        E_aa_baseline = E_aa_baseline_weight * E_aa_baseline_raw
        E_ala2_feature = E_ala2_feature_weight * E_ala2_feature_raw
        E_ala2_rama = E_ala2_rama_weight * E_ala2_rama_raw
        E_ala2_geom = E_ala2_geom_weight * E_ala2_geom_raw

        E_total = (
            E_bond
            + E_angle
            + E_rep
            + E_dih
            + E_ex
            + E_lj
            + E_wca
            + E_fene
            + E_leash
            + E_dh
            + E_stick
            + E_sb
            + E_local_in
            + E_local_bond_in
            + E_local_angle_in
            + E_crowding_wall
            + E_local_torsion
            + E_rep_hard
            + E_5p_flat
            + E_aa_baseline
            + E_ala2_feature
            + E_ala2_rama
            + E_ala2_geom
        )

        return {
            "E_bond": E_bond,
            "E_angle": E_angle,
            "E_repulsive": E_rep,
            "E_repulsive_hard": E_rep_hard,
            "E_dihedral": E_dih,
            "E_excluded_volume": E_ex,
            "E_lj": E_lj,
            "E_wca": E_wca,
            "E_fene": E_fene,
            "E_leash": E_leash,
            "E_dh": E_dh,
            "E_stickiness": E_stick,
            "E_salt_bridge": E_sb,
            "E_local_in": E_local_in,
            "E_local_bond_in": E_local_bond_in,
            "E_local_angle_in": E_local_angle_in,
            "E_crowding_wall": E_crowding_wall,
            "E_local_torsion_fourier": E_local_torsion,
            "E_five_particle_flat_bottom": E_5p_flat,
            "E_aa_integrated_baseline": E_aa_baseline,
            "E_ala2_feature_recovery": E_ala2_feature,
            "E_ala2_rama_recovery": E_ala2_rama,
            "E_ala2_geometry_support_recovery": E_ala2_geom,
            "E_total": E_total,
        }

    def compute_total_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute total prior energy (weighted sum of all terms).

        Args:
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,). Needed for residue-specific angles.
            params: Optional prior params dict (for train_priors mode)

        Returns:
            Total energy (scalar)
        """
        return self.compute_energy(
            R, mask, species=species, params=params, segment_id=segment_id
        )["E_total"]

    def compute_total_energy_from_params(
        self,
        params: Dict[str, jax.Array],
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute total prior energy with given parameters.

        Used for LBFGS optimization where params are being updated.

        Args:
            params: Prior parameters dict
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)

        Returns:
            Total energy (scalar)
        """
        return self.compute_total_energy(
            R, mask, species=species, params=params, segment_id=segment_id
        )

    def __repr__(self) -> str:
        mode = "spline" if self.uses_splines else "parametric"
        return f"PriorEnergy(N_max={self.topology.N_max}, mode={mode}, weights={self.weights})"
