"""
Prior Energy Terms for Coarse-Grained Proteins

Implements physics-based energy terms:
- Bonds: Harmonic stretching between consecutive beads
- Angles: Fourier series bending potential
- Dihedrals: Periodic torsion potential
- Repulsive: Soft-sphere non-bonded interactions

Supports two evaluation modes:
- Parametric (legacy): harmonic bond, Fourier angle, periodic dihedral
- Spline (new): cubic spline PMF from KDE + Boltzmann inversion

Consolidated from:
- allegro_energyfn_multiple_proteins.py
- prior_energyfn.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from .topology import TopologyBuilder
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
        displacement: JAX-MD displacement function

    Returns:
        Angles in radians, shape (n_angles,)
    """
    ia, ib, ic = angles[:, 0], angles[:, 1], angles[:, 2]
    Ra, Rb, Rc = R[ia], R[ib], R[ic]

    # Vectors from central atom
    v_ba = jax.vmap(displacement)(Rb, Ra)  # b -> a
    v_bc = jax.vmap(displacement)(Rb, Rc)  # b -> c

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
        displacement: JAX-MD displacement function

    Returns:
        Dihedral angles in radians, shape (n_dihedrals,)
    """
    i, j, k, l = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]
    Ri, Rj, Rk, Rl = R[i], R[j], R[k], R[l]

    # Bond vectors
    b1 = jax.vmap(displacement)(Rj, Ri)  # i -> j
    b2 = jax.vmap(displacement)(Rk, Rj)  # j -> k
    b3 = jax.vmap(displacement)(Rl, Rk)  # k -> l

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

    def __init__(self, config, topology: TopologyBuilder, displacement):
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

        # Load energy term weights from config
        self.weights = config.get("model", "priors", "weights", default={
            "bond": 0.5,
            "angle": 0.1,
            "repulsive": 0.25,
            "dihedral": 0.15,
        })

        # Get topology
        self.bonds, self.angles = topology.get_bonds_and_angles()
        self.dihedrals = topology.get_dihedrals()
        self.rep_pairs = topology.get_repulsive_pairs()

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

    def _init_spline_priors(self, spline_path: str, config):
        """Initialize spline-based priors from NPZ file."""
        self.uses_splines = True

        # Resolve relative paths
        spline_path = Path(spline_path)
        if not spline_path.is_absolute():
            # Resolve relative to config file location
            config_dir = config.config_path.parent
            spline_path = config_dir / spline_path

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
        }

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
        }

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

        # Compute distances using _safe_norm for well-defined gradients at zero
        dR = jax.vmap(self.displacement)(Ri, Rj)
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

        # Compute distances using _safe_norm for well-defined gradients at zero
        Rp_i, Rp_j = R[pi], R[pj]
        dR_rep = jax.vmap(self.displacement)(Rp_i, Rp_j)
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

    # ========================================================================
    # SCIENTIFIC FIX: Excluded Volume for Nearby Residues (OPTIONAL)
    # ========================================================================
    # ISSUE: No repulsion for sequence separation 2-5 → backbone self-intersection
    # FIX: Add soft excluded volume energy term
    #
    # TO ENABLE:
    # 1. Uncomment this method
    # 2. Add excluded_vol_pairs to __init__ (get from topology)
    # 3. Add epsilon_ex, sigma_ex to config (softer than regular repulsion)
    # 4. Add to compute_energy() and compute_total_energy()
    # ========================================================================
    # def compute_excluded_volume_energy(self, R: jax.Array, mask: jax.Array) -> jax.Array:
    #     """
    #     Compute soft excluded volume for nearby residues (sequence separation 2-5).
    #
    #     E_ex = epsilon_ex * sum[ (sigma_ex / r)^4 ]  (for valid pairs)
    #
    #     Args:
    #         R: Coordinates, shape (n_atoms, 3)
    #         mask: Validity mask, shape (n_atoms,)
    #
    #     Returns:
    #         Total excluded volume energy (scalar)
    #
    #     Note:
    #         Use SOFTER parameters than regular repulsion:
    #         epsilon_ex ~ 1-2 kcal/mol (vs 5 kcal/mol for regular)
    #         sigma_ex ~ 3.5 Å (vs 4 Å for regular)
    #     """
    #     # Requires: self.excluded_vol_pairs = topology.get_excluded_volume_pairs()
    #     pi, pj = self.excluded_vol_pairs[:, 0], self.excluded_vol_pairs[:, 1]
    #
    #     # Mask: both atoms must be valid
    #     ex_valid = (mask[pi] * mask[pj]) > 0
    #
    #     # Compute distances using _safe_norm for well-defined gradients at zero
    #     Rp_i, Rp_j = R[pi], R[pj]
    #     dR_ex = jax.vmap(self.displacement)(Rp_i, Rp_j)
    #     r_ex = _safe_norm(dR_ex)
    #
    #     # Block gradients for invalid pairs
    #     r_ex = jnp.where(ex_valid, r_ex, jax.lax.stop_gradient(r_ex))
    #
    #     # Avoid interactions with padded atoms (set large distance for forward pass)
    #     r_ex = jnp.where(ex_valid, r_ex, 1e6)
    #
    #     # Avoid division by zero
    #     r_min = jnp.array(1e-3, dtype=R.dtype)
    #     r_safe = jnp.maximum(r_ex, r_min)
    #
    #     # Soft-sphere excluded volume (SOFTER than regular repulsion)
    #     ex_term = (self.params["sigma_ex"] / r_safe) ** 4
    #     E_ex = self.params["epsilon_ex"] * jnp.sum(jnp.where(ex_valid, ex_term, 0.0))
    #
    #     return E_ex
    #
    # TO ENABLE IN __init__:
    # self.excluded_vol_pairs = topology.get_excluded_volume_pairs(min_sep=2, max_sep=5)
    # self.params["epsilon_ex"] = jnp.asarray(prior_params.get("epsilon_ex", 1.5), dtype=jnp.float32)
    # self.params["sigma_ex"] = jnp.asarray(prior_params.get("sigma_ex", 3.5), dtype=jnp.float32)
    #
    # TO ENABLE IN compute_energy():
    # E_ex_raw = self.compute_excluded_volume_energy(R, mask)
    # E_ex = self.weights.get("excluded_volume", 1.0) * E_ex_raw  # Usually weight=1.0
    # Add E_ex to the returned dict and to E_total
    # ========================================================================

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
                - E_total: Sum of all weighted components
        """
        # Compute raw energies
        p = params if params is not None else self.params
        E_bond_raw = self.compute_bond_energy(R, mask, params=p)
        E_angle_raw = self.compute_angle_energy(R, mask, species=species, params=p)
        E_rep_raw = self.compute_repulsive_energy(R, mask, params=p)
        E_dih_raw = self.compute_dihedral_energy(R, mask, params=p)

        # Apply weights (matching original code behavior)
        E_bond = self.weights["bond"] * E_bond_raw
        E_angle = self.weights["angle"] * E_angle_raw
        E_rep = self.weights["repulsive"] * E_rep_raw
        E_dih = self.weights["dihedral"] * E_dih_raw

        # Total is sum of weighted components
        # NOTE: Dihedral is NOW INCLUDED after fixing the gradient NaN issue.
        # The fix in compute_dihedral_energy() applies stop_gradient to phi for
        # invalid dihedrals, which blocks NaN gradients from atan2(0,0).
        E_total = E_bond + E_angle + E_rep + E_dih

        return {
            "E_bond": E_bond,
            "E_angle": E_angle,
            "E_repulsive": E_rep,
            "E_dihedral": E_dih,
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
