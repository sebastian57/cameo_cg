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
) -> jax.Array:
    """Debye-Huckel energy over a pair set."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0

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
) -> jax.Array:
    """Typed nonbonded stickiness energy over a pair set."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0

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
) -> jax.Array:
    """Short-range salt-bridge correction for opposite-charge pairs."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0

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
) -> jax.Array:
    """FENE energy for consecutive sequence bonds."""
    if bonds.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    bi, bj = bonds[:, 0], bonds[:, 1]
    valid = (mask[bi] * mask[bj]) > 0

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
) -> jax.Array:
    """Flat-bottom pair-distance leash energy."""
    if pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (mask[pi] * mask[pj]) > 0

    dR = R[pi] - R[pj]
    r = _safe_norm(dR)
    r = jnp.where(valid, r, jax.lax.stop_gradient(r))
    r_eval = jnp.where(valid, r, 0.0)

    d_safe = jnp.asarray(d_safe, dtype=R.dtype)
    k_safe = jnp.asarray(k_safe, dtype=R.dtype)
    dr = jnp.maximum(r_eval - d_safe, 0.0)
    U = 0.5 * k_safe * dr**2
    return jnp.sum(jnp.where(valid, U, 0.0))


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
        }
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
        }
        self.params.update(self._init_new_term_params(prior_params))
        self.params.update(self._init_wca_params(prior_params))
        self.params.update(self._init_safety_prior_params(prior_params))

    def compute_bond_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None
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
        params: Optional[Dict[str, jax.Array]] = None
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
        params: Optional[Dict[str, jax.Array]] = None
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

    def compute_excluded_volume_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None
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

    def compute_wca_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None
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

    def compute_fene_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
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
        )

    def compute_leash_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None,
    ) -> jax.Array:
        """Compute flat-bottom pair-distance leash energy."""
        p = params if params is not None else self.params
        return compute_leash_energy(
            R=R,
            mask=mask,
            pairs=self.leash_pairs,
            d_safe=p["leash_d_safe"],
            k_safe=p["leash_k_safe"],
        )

    def compute_dh_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
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
        )

    def compute_stickiness_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
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
        )

    def compute_salt_bridge_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None,
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
        )

    def compute_dihedral_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        params: Optional[Dict[str, jax.Array]] = None
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
        params: Optional[Dict[str, jax.Array]] = None
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
        E_bond_raw = self.compute_bond_energy(R, mask, params=p)
        E_angle_raw = self.compute_angle_energy(R, mask, species=species, params=p)
        E_rep_raw = self.compute_repulsive_energy(R, mask, params=p)
        E_dih_raw = self.compute_dihedral_energy(R, mask, params=p)
        E_ex_raw = self.compute_excluded_volume_energy(R, mask, params=p)
        E_wca_weight = self.weights.get("wca", 0.0)
        E_wca_raw = (
            self.compute_wca_energy(R, mask, params=p)
            if E_wca_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_fene_weight = self.weights.get("fene", 0.0)
        E_fene_raw = (
            self.compute_fene_energy(R, mask, params=p)
            if E_fene_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_leash_weight = self.weights.get("leash", 0.0)
        E_leash_raw = (
            self.compute_leash_energy(R, mask, params=p)
            if E_leash_weight != 0.0
            else jnp.array(0.0, dtype=R.dtype)
        )
        E_dh_raw = self.compute_dh_energy(R, mask, species=species, params=p)
        E_stick_raw = self.compute_stickiness_energy(R, mask, species=species, params=p)
        E_sb_raw = self.compute_salt_bridge_energy(R, mask, species=species, params=p)

        # Apply weights
        E_bond = self.weights["bond"] * E_bond_raw
        E_angle = self.weights["angle"] * E_angle_raw
        E_rep = self.weights["repulsive"] * E_rep_raw
        E_dih = self.weights["dihedral"] * E_dih_raw
        E_ex = self.weights.get("excluded_volume", 1.0) * E_ex_raw
        E_wca = E_wca_weight * E_wca_raw
        E_fene = E_fene_weight * E_fene_raw
        E_leash = E_leash_weight * E_leash_raw
        E_dh = self.weights.get("dh", 0.0) * E_dh_raw
        E_stick = self.weights.get("stickiness", 0.0) * E_stick_raw
        E_sb = self.weights.get("salt_bridge", 0.0) * E_sb_raw

        E_total = (
            E_bond
            + E_angle
            + E_rep
            + E_dih
            + E_ex
            + E_wca
            + E_fene
            + E_leash
            + E_dh
            + E_stick
            + E_sb
        )

        return {
            "E_bond": E_bond,
            "E_angle": E_angle,
            "E_repulsive": E_rep,
            "E_dihedral": E_dih,
            "E_excluded_volume": E_ex,
            "E_wca": E_wca,
            "E_fene": E_fene,
            "E_leash": E_leash,
            "E_dh": E_dh,
            "E_stickiness": E_stick,
            "E_salt_bridge": E_sb,
            "E_total": E_total,
        }

    def compute_total_energy(
        self,
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None,
        params: Optional[Dict[str, jax.Array]] = None
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
        return self.compute_energy(R, mask, species=species, params=params)["E_total"]

    def compute_total_energy_from_params(
        self,
        params: Dict[str, jax.Array],
        R: jax.Array,
        mask: jax.Array,
        species: Optional[jax.Array] = None
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
        return self.compute_total_energy(R, mask, species=species, params=params)

    def __repr__(self) -> str:
        mode = "spline" if self.uses_splines else "parametric"
        return f"PriorEnergy(N_max={self.topology.N_max}, mode={mode}, weights={self.weights})"
