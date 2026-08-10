"""Static per-segment neighbor graphs for tiled force matching.

Builds the directed sparse radius graph of a packed tile once, per molecular
segment, so training reuses fixed connectivity instead of calling JAX-MD
`neighbor.update()` for every sample. Design note:
`KNOWLEDGE_BASE/P_cameo_cg/DESIGN/STATIC_TILED_NEIGHBOR_LISTS.md`.

Scope: neighbor-candidate construction and repeated neighbor updates. This does
not reduce Allegro force-gradient activation memory, which is a separate limit.

Two interchangeable backends produce the same undirected pair set:

* :func:`segment_pairs_kdtree` -- SciPy ``cKDTree.query_pairs``; default.
* :func:`segment_pairs_chunked` -- dependency-minimal reference oracle that
  materializes at most a ``block_size x block_size`` squared-distance block, so
  temporary memory is bounded at ``O(block_size**2)`` regardless of segment size.

JAX-MD conventions reproduced here (verified against the installed JAX-MD,
``jax_md/partition.py::prune_neighbor_list_sparse``):

* candidate radius is ``r_cutoff + dr_threshold``;
* the comparator is **strict**: ``squared_distance < cutoff**2``;
* ``Sparse`` index layout is ``stack((receiver_idx, sender_idx))``, i.e. row 0
  holds receivers and row 1 holds senders;
* the invalid/padding sentinel is the tile atom capacity ``N``.

Note: ``custom_partition.mask_neighbor_list`` unpacks the same array as
``senders, receivers = idx``, a naming inversion relative to JAX-MD. It is
harmless there because the mask is applied symmetrically to both rows, but it is
not evidence of the row order.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from utils.logging import data_logger

DEFAULT_BLOCK_SIZE = 1024

# Dtype the radius comparison runs in. Must match the model compute dtype, or
# boundary membership diverges from the graph JAX-MD would have built.
_COMPARE_DTYPE = np.float32

# Relative inflation applied to the KD-tree query radius so the float64 tree
# search is a strict superset of the float32 comparison performed after it.
_KDTREE_RADIUS_SLACK = 1e-5


class StaticGraphIncompatibleError(RuntimeError):
    """Raised when a config enables a mechanism that invalidates static graphs."""


# ---------------------------------------------------------------------------
#  Compatibility guard
# ---------------------------------------------------------------------------

# (label, predicate, reason). Each predicate takes the cameo_cg config manager.
_INCOMPATIBLE = [
    (
        "training.dsm.enabled",
        lambda c: bool(c.get("training", "dsm", "enabled", default=False)),
        "DSM evaluates the model at R + sigma * eps inside the loss "
        "(training/dsm.py::make_dsm_quantity), so pairs entering the cutoff "
        "under the perturbation are missing from a graph built at R. Gaussian "
        "noise has no finite displacement bound, so no enlarged candidate "
        "radius fixes this without a documented truncation policy.",
    ),
    (
        "training.noised_residual_training.enabled",
        lambda c: bool(
            (c.get("training", "noised_residual_training", default={}) or {}).get(
                "enabled", False
            )
        ),
        "Noised residual training regenerates perturbed coordinates, which "
        "changes connectivity relative to the graph built at the stored R.",
    ),
    (
        "training.relative_entropy.enabled",
        lambda c: bool(c.get("training", "relative_entropy", "enabled", default=False)),
        "Relative entropy matching samples fresh configurations by MD; "
        "coordinates change every step, so no fixed graph is valid.",
    ),
    (
        "model.pbc",
        lambda c: bool(c.get("model", "pbc", default=False)),
        "Static graphs are currently built for nonperiodic packed tiles only. "
        "Periodic minimum-image pairs are not reproduced by the segment-local "
        "search backends.",
    ),
]

# Mechanisms deliberately NOT rejected, with the reason each one is safe:
#
# training.hvp.enabled
#     HVP targets are jax.grad of (F . probe) evaluated at state.position
#     (training/hvp_matching.py). A directional derivative at R, not a
#     displacement of R, so a graph built at R stays exact.
#
# data.noise_decoys.every_n
#     Decoy frames are appended to the source dataset before build_tiled_dataset
#     runs (scripts/train.py), so decoys are ordinary tile rows whose graphs are
#     built from their own stored coordinates.
#
# data.tile_rebuild_each_epoch
#     Graphs are rebuilt inside build_tiled_dataset, so every repacking path
#     (epoch rebuild, DSM refresh) regenerates connectivity with the tiles.
#
# training.msam / training.swa / support and extrapolation gates
#     These change parameters or scale energies; none of them moves coordinates.


def assert_static_graph_compatible(config: Any) -> None:
    """Raise if the config enables a mechanism a static graph cannot support.

    This raises rather than warns because a stale graph still yields finite
    forces and a converging run: the failure is invisible without a reference.

    Args:
        config: `config.manager.ConfigManager` instance.

    Raises:
        StaticGraphIncompatibleError: if batch mode is not `tiled`, or if any
            graph-invalidating mechanism is enabled.
    """
    batch_mode = str(config.get("data", "batch_mode", default="standard")).strip().lower()
    if batch_mode != "tiled":
        raise StaticGraphIncompatibleError(
            f"data.static_neighbors.enabled requires data.batch_mode='tiled', "
            f"got '{batch_mode}'. Segment decomposition is only defined for "
            "packed tiles; use the dynamic JAX-MD path for standard batching."
        )

    found = [(k, reason) for k, predicate, reason in _INCOMPATIBLE if predicate(config)]
    if found:
        details = "\n".join(f"  - {key}: {reason}" for key, reason in found)
        raise StaticGraphIncompatibleError(
            "data.static_neighbors.enabled is incompatible with the following "
            f"enabled mechanisms:\n{details}\n"
            "Disable them, or set data.static_neighbors.enabled: false to use "
            "the dynamic JAX-MD neighbor path."
        )


# ---------------------------------------------------------------------------
#  Radius search backends
# ---------------------------------------------------------------------------


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distances via explicit differences, matching JAX-MD reduction order.

    The expanded ``|a|^2 + |b|^2 - 2 a.b`` form is deliberately avoided: it loses
    the cancellation accuracy that decides membership for pairs sitting on the
    cutoff, which is exactly where equality to the JAX-MD graph is fragile.
    """
    d = a - b
    return d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2


def segment_pairs_chunked(
    X: np.ndarray,
    r_list: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """Exact upper-triangular radius pairs of one segment via chunked blocks.

    Args:
        X: `(n, 3)` segment-local coordinates.
        r_list: candidate radius (`r_cutoff + dr_threshold`).
        block_size: side length of the largest materialized distance block.

    Returns:
        `(n_pairs, 2)` int64 local indices with `i < j`.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}.")

    X = np.ascontiguousarray(X, dtype=_COMPARE_DTYPE)
    n = int(X.shape[0])
    r2 = _COMPARE_DTYPE(r_list) ** 2
    found = []

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        Xi = X[i0:i1]
        # j0 starts at i0, so off-diagonal blocks are already strictly ordered
        # (j0 > i0 => every j index exceeds every i index) and only the diagonal
        # block needs an explicit upper-triangular mask.
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            d2 = _squared_distances(Xi[:, None, :], X[j0:j1][None, :, :])
            hit = d2 < r2
            if i0 == j0:
                hit &= np.triu(np.ones(hit.shape, dtype=bool), k=1)
            ii, jj = np.nonzero(hit)
            if ii.size:
                found.append(np.stack((ii + i0, jj + j0), axis=1))

    if not found:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(found, axis=0)


def segment_pairs_kdtree(X: np.ndarray, r_list: float) -> np.ndarray:
    """Exact upper-triangular radius pairs of one segment via SciPy cKDTree.

    `query_pairs` is inclusive (`d <= r`) and searches in float64, while JAX-MD
    compares `d2 < r2` in the compute dtype. The tree is therefore queried at a
    slightly inflated radius to guarantee a superset, and the result is
    re-filtered with the exact strict comparison.

    Args:
        X: `(n, 3)` segment-local coordinates.
        r_list: candidate radius (`r_cutoff + dr_threshold`).

    Returns:
        `(n_pairs, 2)` int64 local indices with `i < j`.
    """
    from scipy.spatial import cKDTree

    X = np.ascontiguousarray(X, dtype=_COMPARE_DTYPE)
    if X.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)

    tree = cKDTree(np.asarray(X, dtype=np.float64))
    pairs = tree.query_pairs(
        float(r_list) * (1.0 + _KDTREE_RADIUS_SLACK), output_type="ndarray"
    )
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    d2 = _squared_distances(X[pairs[:, 0]], X[pairs[:, 1]])
    return np.asarray(pairs[d2 < _COMPARE_DTYPE(r_list) ** 2], dtype=np.int64)


_BACKENDS = {
    "chunked": segment_pairs_chunked,
    "kdtree": segment_pairs_kdtree,
}


# ---------------------------------------------------------------------------
#  Tile assembly
# ---------------------------------------------------------------------------


def build_tile_graph(
    R: np.ndarray,
    mask: np.ndarray,
    segment_id: np.ndarray,
    r_list: float,
    backend: str = "kdtree",
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Tuple[np.ndarray, int]:
    """Build the directed sparse graph of one packed tile.

    Pairs are searched independently inside each `segment_id`, so cross-segment
    edges are impossible by construction and peak search memory scales with the
    largest individual segment rather than the tile.

    Args:
        R: `(N, 3)` tile coordinates, padded rows included.
        mask: `(N,)` validity mask; `> 0` marks a real bead.
        segment_id: `(N,)` segment membership; `< 0` marks padding.
        r_list: candidate radius (`r_cutoff + dr_threshold`).
        backend: `"kdtree"` or `"chunked"`.
        block_size: block side length for the chunked backend.

    Returns:
        `(idx, n_edges)` where `idx` is `(2, n_edges)` int32 with row 0 receivers
        and row 1 senders, sorted lexicographically by sender then receiver.
        Padding to a static capacity is applied by :func:`build_static_graphs`.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown static neighbor backend '{backend}'. "
            f"Expected one of: {sorted(_BACKENDS)}."
        )
    pair_fn = _BACKENDS[backend]

    R = np.asarray(R)
    mask = np.asarray(mask)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    n_atoms = int(R.shape[0])
    if mask.shape[0] != n_atoms or segment_id.shape[0] != n_atoms:
        raise ValueError(
            "R, mask and segment_id must share the leading tile dimension, got "
            f"{R.shape[0]}, {mask.shape[0]}, {segment_id.shape[0]}."
        )

    valid = (mask > 0) & (segment_id >= 0)
    collected: List[np.ndarray] = []

    for seg in np.unique(segment_id[valid]):
        local_to_global = np.nonzero(valid & (segment_id == seg))[0]
        if local_to_global.size < 2:
            continue
        kwargs = {"block_size": block_size} if backend == "chunked" else {}
        local_pairs = pair_fn(R[local_to_global], r_list, **kwargs)
        if local_pairs.size:
            collected.append(local_to_global[local_pairs])

    if not collected:
        return np.empty((2, 0), dtype=np.int32), 0

    undirected = np.concatenate(collected, axis=0)
    # Expand each undirected pair into both orientations: Allegro consumes a full
    # directed graph and JAX-MD `Sparse` stores one entry per direction.
    directed = np.concatenate((undirected, undirected[:, ::-1]), axis=0)

    receivers = directed[:, 1]
    senders = directed[:, 0]
    order = np.lexsort((receivers, senders))
    idx = np.stack((receivers[order], senders[order]), axis=0)
    return np.asarray(idx, dtype=np.int32), int(idx.shape[1])


def build_static_graphs(
    R: np.ndarray,
    mask: np.ndarray,
    segment_id: np.ndarray,
    r_list: float,
    backend: str = "kdtree",
    block_size: int = DEFAULT_BLOCK_SIZE,
    capacity_multiplier: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build padded static graphs for every tile of a tiled dataset.

    Args:
        R: `(n_tiles, N, 3)` tile coordinates.
        mask: `(n_tiles, N)` validity masks.
        segment_id: `(n_tiles, N)` segment membership.
        r_list: candidate radius (`r_cutoff + dr_threshold`).
        backend: `"kdtree"` or `"chunked"`.
        block_size: block side length for the chunked backend.
        capacity_multiplier: reserve applied to the measured maximum edge count.

    Returns:
        Dict with `neighbor_idx` `(n_tiles, 2, capacity)` int32 padded with the
        sentinel `N`, `neighbor_n_edges` `(n_tiles,)` int32, and
        `neighbor_capacity` `(n_tiles,)` int32.
    """
    if capacity_multiplier < 1.0:
        raise ValueError(
            f"capacity_multiplier must be >= 1.0, got {capacity_multiplier}."
        )

    R = np.asarray(R)
    mask = np.asarray(mask)
    segment_id = np.asarray(segment_id)
    if R.ndim != 3:
        raise ValueError(f"R must have shape (n_tiles, N, 3), got {R.shape}.")
    n_tiles, n_atoms = int(R.shape[0]), int(R.shape[1])

    graphs: List[np.ndarray] = []
    n_edges = np.zeros((n_tiles,), dtype=np.int32)
    for t in range(n_tiles):
        idx, count = build_tile_graph(
            R[t],
            mask[t],
            segment_id[t],
            r_list,
            backend=backend,
            block_size=block_size,
        )
        graphs.append(idx)
        n_edges[t] = count

    measured_max = int(n_edges.max()) if n_tiles else 0
    # A zero-width edge axis would make the graph unusable downstream; keep at
    # least one padded slot so the sentinel-only tile is still well-formed.
    capacity = max(int(np.ceil(measured_max * capacity_multiplier)), 1)

    # Sentinel N matches the JAX-MD/custom_partition padding convention: an index
    # equal to the atom capacity marks an invalid edge endpoint.
    neighbor_idx = np.full((n_tiles, 2, capacity), n_atoms, dtype=np.int32)
    for t, idx in enumerate(graphs):
        neighbor_idx[t, :, : idx.shape[1]] = idx

    occupancy = float(n_edges.mean() / capacity) if capacity and n_tiles else 0.0
    data_logger.info(
        "[StaticNeighbors] backend=%s r_list=%.3f tiles=%d beads/tile=%d "
        "edges: max=%d mean=%.1f capacity=%d occupancy=%.2f indices=%.1f MB",
        backend,
        r_list,
        n_tiles,
        n_atoms,
        measured_max,
        float(n_edges.mean()) if n_tiles else 0.0,
        capacity,
        occupancy,
        neighbor_idx.nbytes / 1e6,
    )

    return {
        "neighbor_idx": neighbor_idx,
        "neighbor_n_edges": n_edges,
        "neighbor_capacity": np.full((n_tiles,), capacity, dtype=np.int32),
    }
