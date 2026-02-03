"""
Topology Generation for Coarse-Grained Protein Models

Generates bond, angle, dihedral, and repulsive pair indices for
linear chain topology (1-bead-per-residue proteins).

Consolidated from:
- allegro_energyfn_multiple_proteins.py
- prior_energyfn.py
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from jax_md import partition


def precompute_chain_topology(N_max: int) -> Tuple[jax.Array, jax.Array]:
    """
    Generate bonds and angles for a linear chain topology.

    For 1-bead-per-residue proteins, consecutive beads are bonded,
    and every triplet forms an angle.

    Args:
        N_max: Maximum number of beads in the chain

    Returns:
        bonds: Array of shape (N_max-1, 2) with bond pairs
        angles: Array of shape (N_max-2, 3) with angle triplets

    Example:
        >>> bonds, angles = precompute_chain_topology(5)
        >>> bonds.shape
        (4, 2)
        >>> angles.shape
        (3, 3)
    """
    i = jnp.arange(N_max, dtype=jnp.int32)

    # Bonds: consecutive pairs (i, i+1)
    bonds = jnp.stack([i[:-1], i[1:]], axis=1)  # (N_max-1, 2)

    # Angles: triplets (i, i+1, i+2)
    angles = jnp.stack([i[:-2], i[1:-1], i[2:]], axis=1)  # (N_max-2, 3)

    return bonds, angles


def precompute_dihedrals(N_max: int) -> jax.Array:
    """
    Generate dihedral indices for a linear chain topology.

    Dihedrals are defined by four consecutive beads (i, i+1, i+2, i+3).

    Args:
        N_max: Maximum number of beads in the chain

    Returns:
        dihedrals: Array of shape (N_max-3, 4) with dihedral quadruplets

    Example:
        >>> dihedrals = precompute_dihedrals(6)
        >>> dihedrals.shape
        (3, 4)
    """
    i = jnp.arange(N_max, dtype=jnp.int32)

    # Dihedrals: quadruplets (i, i+1, i+2, i+3)
    dihedrals = jnp.stack([i[:-3], i[1:-2], i[2:-1], i[3:]], axis=1)  # (N_max-3, 4)

    return dihedrals


def precompute_repulsive_pairs(N_max: int, min_sep: int = 6) -> jax.Array:
    """
    Generate non-bonded repulsive pairs for a linear chain.

    Includes all pairs (i, j) where j > i and (j - i) >= min_sep.
    This excludes nearby residues that are already handled by
    bonds, angles, and dihedrals.

    Args:
        N_max: Maximum number of beads in the chain
        min_sep: Minimum sequence separation (default: 6 residues)

    Returns:
        rep_pairs: Array of shape (N_rep, 2) with repulsive pairs

    Example:
        >>> pairs = precompute_repulsive_pairs(10, min_sep=6)
        >>> # All pairs (i,j) with j-i >= 6
        >>> assert pairs[0, 1] - pairs[0, 0] >= 6
    """
    idx = jnp.arange(N_max, dtype=jnp.int32)
    ii, jj = jnp.meshgrid(idx, idx, indexing="ij")

    # Keep pairs where j > i and separation >= min_sep
    keep = (jj > ii) & ((jj - ii) >= min_sep)
    rep_pairs = jnp.stack([ii[keep], jj[keep]], axis=1).astype(jnp.int32)

    return rep_pairs


def filter_neighbors_by_mask(nbrs: partition.NeighborList, mask: jax.Array) -> partition.NeighborList:
    """
    Filter neighbor list to exclude padded (masked) atoms.

    For padded systems with variable protein sizes, this filters the
    neighbor list so that interactions only occur between real (masked=1) atoms.
    Interactions involving padded atoms (masked=0) are removed.

    Args:
        nbrs: JAX-MD NeighborList object
        mask: Boolean/float mask array of shape (N_atoms,), where 1=real, 0=padded

    Returns:
        Filtered NeighborList with masked neighbors set to -1

    Example:
        >>> mask = jnp.array([1, 1, 1, 0, 0])  # 3 real atoms, 2 padded
        >>> nbrs_filtered = filter_neighbors_by_mask(nbrs, mask)
    """
    idx = nbrs.idx

    # Central atom mask: shape (N_atoms, 1)
    central_valid = mask[:, None]

    # Neighbor atom mask: shape (N_atoms, n_neighbors)
    # Use safe indexing to avoid out-of-bounds access
    idx_safe = jnp.where(idx >= 0, idx, 0)
    neighbor_valid = mask[idx_safe]

    # Both central and neighbor must be valid
    both_valid = central_valid * neighbor_valid

    # Only keep valid pairs (others set to -1)
    both_valid = jnp.where(idx >= 0, both_valid, 0)
    filtered_idx = jnp.where(both_valid > 0, idx, -1)

    return nbrs._replace(idx=filtered_idx)


class TopologyBuilder:
    """
    Builder class for protein chain topology.

    Provides a convenient interface for generating and caching
    topology information (bonds, angles, dihedrals, repulsive pairs).

    Example:
        >>> topology = TopologyBuilder(N_max=100, min_repulsive_sep=6)
        >>> bonds, angles = topology.get_bonds_and_angles()
        >>> dihedrals = topology.get_dihedrals()
        >>> rep_pairs = topology.get_repulsive_pairs()
    """

    def __init__(self, N_max: int, min_repulsive_sep: int = 6):
        """
        Initialize topology builder.

        Args:
            N_max: Maximum number of beads in the chain
            min_repulsive_sep: Minimum sequence separation for repulsive pairs
        """
        self.N_max = N_max
        self.min_repulsive_sep = min_repulsive_sep

        # Cache topology on first access
        self._bonds = None
        self._angles = None
        self._dihedrals = None
        self._repulsive_pairs = None

    def get_bonds_and_angles(self) -> Tuple[jax.Array, jax.Array]:
        """Get bonds and angles (computed and cached on first call)."""
        if self._bonds is None or self._angles is None:
            self._bonds, self._angles = precompute_chain_topology(self.N_max)
        return self._bonds, self._angles

    def get_bonds(self) -> jax.Array:
        """Get bonds array."""
        if self._bonds is None:
            self.get_bonds_and_angles()
        return self._bonds

    def get_angles(self) -> jax.Array:
        """Get angles array."""
        if self._angles is None:
            self.get_bonds_and_angles()
        return self._angles

    def get_dihedrals(self) -> jax.Array:
        """Get dihedrals (computed and cached on first call)."""
        if self._dihedrals is None:
            self._dihedrals = precompute_dihedrals(self.N_max)
        return self._dihedrals

    def get_repulsive_pairs(self) -> jax.Array:
        """Get repulsive pairs (computed and cached on first call)."""
        if self._repulsive_pairs is None:
            self._repulsive_pairs = precompute_repulsive_pairs(
                self.N_max, self.min_repulsive_sep
            )
        return self._repulsive_pairs

    def get_all(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Get all topology components.

        Returns:
            bonds, angles, dihedrals, repulsive_pairs
        """
        return (
            self.get_bonds(),
            self.get_angles(),
            self.get_dihedrals(),
            self.get_repulsive_pairs()
        )

    # ========================================================================
    # SCIENTIFIC FIX: Excluded Volume for Nearby Residues
    # ========================================================================
    # ISSUE: No repulsion for sequence separation 2-5 → backbone can self-intersect
    # FIX: Add soft excluded volume for nearby residues
    #
    # USAGE: Uncomment the code in prior_energy.py to enable this feature
    # ========================================================================
    def get_excluded_volume_pairs(self, min_sep: int = 2, max_sep: int = 5) -> jax.Array:
        """
        Get atom pairs for excluded volume (soft repulsion for nearby residues).

        These pairs prevent backbone self-intersection for residues that are
        close in sequence but not directly bonded.

        Args:
            min_sep: Minimum sequence separation (default: 2)
            max_sep: Maximum sequence separation (default: 5)

        Returns:
            Array of pair indices, shape (n_pairs, 2)

        Note:
            This is SEPARATE from regular repulsive pairs (which handle sep ≥6).
            Use softer epsilon/sigma for these pairs to avoid over-constraining.
        """
        pairs = []
        for i in range(self.N_max):
            for j in range(i + min_sep, min(i + max_sep + 1, self.N_max)):
                pairs.append([i, j])

        if not pairs:
            return jnp.array([], dtype=jnp.int32).reshape(0, 2)

        return jnp.array(pairs, dtype=jnp.int32)
    # ========================================================================

    def __repr__(self) -> str:
        return f"TopologyBuilder(N_max={self.N_max}, min_repulsive_sep={self.min_repulsive_sep})"
