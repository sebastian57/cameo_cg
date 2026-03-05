"""Neighbor-list format helpers shared across model wrappers."""

from typing import Any, Tuple

import jax.numpy as jnp
from jax_md import partition


def resolve_neighbor_list_format(format_name: str) -> Tuple[str, Any]:
    """
    Resolve validated config string to JAX-MD neighbor-list format enum.

    Args:
        format_name: One of "dense" or "sparse" (case-insensitive).

    Returns:
        Tuple of normalized name and matching partition format enum.
    """
    normalized = str(format_name).strip().lower().replace("-", "_")
    mapping = {
        "dense": partition.Dense,
        "sparse": partition.Sparse,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported neighbor_list_format='{format_name}'. "
            "Expected one of: dense, sparse."
        )
    return normalized, mapping[normalized]


def compute_avg_num_neighbors(nbrs: Any, n_atoms: int) -> float:
    """
    Compute average neighbors per atom from a dense or sparse neighbor list.

    Args:
        nbrs: JAX-MD NeighborList
        n_atoms: Number of atoms in the structure

    Returns:
        Average number of neighbors per atom.
    """
    if n_atoms <= 0:
        return 0.0

    idx = nbrs.idx

    # Sparse format stores edge list as idx=(senders, receivers) with shape (2, E).
    if idx.ndim == 2 and idx.shape[0] == 2:
        senders = idx[0]
        valid_edges = (senders >= 0) & (senders < n_atoms)
        n_edges = jnp.sum(valid_edges.astype(jnp.float32))
        return float(n_edges / float(n_atoms))

    # Dense format stores neighbor slots as shape (N, M).
    valid_slots = (idx >= 0) & (idx < n_atoms)
    return float(jnp.mean(jnp.sum(valid_slots, axis=-1).astype(jnp.float32)))
