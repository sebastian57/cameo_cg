"""Fixed-capacity neighbor lists for orthorhombic boxes that vary by frame.

The standard JAX-MD cell-list implementation treats the box used during
allocation as static when positions are supplied in Cartesian coordinates.
This module keeps the cell grid static in fractional coordinates instead.  A
frame box is used only by the displacement function, while the grid is sized
from a caller-supplied lower bound on the box lengths.  Consequently the
update path has static array shapes and remains suitable for ``jax.jit``.

The list is deliberately small in scope: orthorhombic periodic boxes,
dense/sparse JAX-MD formats, and a boolean particle mask.  Allocation is
host-side, as with JAX-MD neighbor lists; updates are fixed-capacity and
JIT-compatible.  A box change forces a rebuild because the list is intended
for one frame at a time.
"""

from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax_md import dataclasses, partition, space


def _mask_dense(idx, mask=None):
    """Mask self edges, sentinels, and padded particles in dense candidates."""
    n_atoms = idx.shape[0]
    valid = (idx >= 0) & (idx < n_atoms)
    safe_idx = jnp.where(valid, idx, 0)
    invalid = (idx == jnp.arange(n_atoms)[:, None]) | jnp.logical_not(valid)
    if mask is not None:
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        invalid = invalid | jnp.logical_not(mask[:, None])
        invalid = invalid | jnp.logical_not(mask[safe_idx])
    return jnp.where(invalid, n_atoms, idx)


def _fractional_positions(position, box):
    """Map Cartesian positions into the unit fractional box."""
    box = jnp.asarray(box, dtype=position.dtype)
    return jnp.mod(position / box[None, :], 1.0)


def _metric_sq(displacement_or_metric, Ra, Rb, **kwargs):
    value = displacement_or_metric(Ra, Rb, **kwargs)
    return value * value if value.ndim == 0 else space.square_distance(value)


def dynamic_neighbor_list(
    displacement_or_metric,
    box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    format=partition.NeighborListFormat.Dense,
    box_min=None,
):
    """Build a fixed-capacity neighbor-list function accepting ``box=...``.

    Args mirror :func:`jax_md.partition.neighbor_list`.  ``box`` is the
    reference box used for allocation.  ``box_min`` is an optional static
    lower bound on each box length; it determines the fractional grid.  If
    omitted, the reference box is used.  Every runtime box must be at least
    this bound.  ``mask`` is accepted by ``allocate`` and ``update`` and marks
    valid particles.
    """
    if format not in (
        partition.NeighborListFormat.Dense,
        partition.NeighborListFormat.Sparse,
    ):
        raise ValueError("dynamic_neighbor_list supports Dense and Sparse formats")

    reference_box = np.asarray(box, dtype=np.float32)
    lower_box = reference_box if box_min is None else np.asarray(box_min, dtype=np.float32)
    if reference_box.shape != (3,) or lower_box.shape != (3,):
        raise ValueError("dynamic_neighbor_list currently requires orthorhombic (3,) boxes")
    if np.any(reference_box <= 0.0) or np.any(lower_box <= 0.0):
        raise ValueError("dynamic_neighbor_list requires positive box lengths")
    if np.any(reference_box < lower_box):
        raise ValueError("reference box must be no smaller than box_min")

    cutoff = float(r_cutoff) + float(dr_threshold)
    cutoff_sq = cutoff * cutoff
    reference_box_jax = jnp.asarray(reference_box, dtype=jnp.float32)
    # The fractional cell size is chosen from the smallest supported physical
    # box.  JAX-MD's cell_list rounds this down to a fixed >=3-cell grid, so
    # each cell remains at least ``cutoff`` wide for every runtime box.
    minimum_fractional_cell = cutoff / lower_box
    if np.any(np.floor(1.0 / minimum_fractional_cell) < 3):
        raise ValueError(
            "dynamic neighbor lists require each lower-bound box length to be "
            "at least 3 * (cutoff + dr_threshold)"
        )
    cell_fns = None
    if not disable_cell_list:
        cell_fns = partition.cell_list(
            np.ones(3, dtype=np.float32),
            minimum_fractional_cell,
            buffer_size_multiplier=float(capacity_multiplier),
        )

    # This is the same fixed-shape 27-cell candidate construction used by
    # JAX-MD, but it is applied to fractional positions and our fixed grid.
    @partial(jax.jit, static_argnums=1)
    def cell_candidates(id_buffer, position_shape):
        n_atoms, _ = position_shape
        cell_idx = [id_buffer]
        for dindex in partition._neighboring_cells(3):
            if np.all(dindex == 0):
                continue
            cell_idx.append(partition.shift_array(id_buffer, dindex))
        cell_idx = jnp.concatenate(cell_idx, axis=-2)
        cell_idx = cell_idx[..., None, :, :]
        cell_idx = jnp.broadcast_to(
            cell_idx, id_buffer.shape[:-1] + cell_idx.shape[-2:]
        )

        def copy_values(value, cell_value, cell_id):
            scatter_indices = jnp.reshape(cell_id, (-1,))
            cell_value = jnp.reshape(cell_value, (-1,) + cell_value.shape[-2:])
            return value.at[scatter_indices].set(cell_value)

        out = jnp.zeros((n_atoms + 1,) + cell_idx.shape[-2:], dtype=jnp.int32)
        out = copy_values(out, cell_idx, id_buffer)
        return out[:-1, :, 0]

    def all_candidates(n_atoms):
        candidates = jnp.arange(n_atoms, dtype=jnp.int32)
        return jnp.broadcast_to(candidates[None, :], (n_atoms, n_atoms))

    def prune_dense(position, candidates, runtime_box):
        n_atoms = position.shape[0]
        safe = jnp.where((candidates >= 0) & (candidates < n_atoms), candidates, 0)
        metric = partial(_metric_sq, displacement_or_metric, box=runtime_box)
        metric = space.map_neighbor(metric)
        distances_sq = metric(position, position[safe])
        valid = (distances_sq < cutoff_sq) & (candidates < n_atoms)
        out = n_atoms * jnp.ones(candidates.shape, dtype=jnp.int32)
        cumsum = jnp.cumsum(valid, axis=1)
        slot = jnp.where(valid, cumsum - 1, candidates.shape[1] - 1)
        rows = jnp.arange(n_atoms)[:, None]
        out = out.at[rows, slot].set(candidates)
        return out, jnp.max(cumsum[:, -1])

    def prune_sparse(position, candidates, runtime_box):
        n_atoms = position.shape[0]
        safe = jnp.where((candidates >= 0) & (candidates < n_atoms), candidates, 0)
        senders = jnp.broadcast_to(jnp.arange(n_atoms)[:, None], candidates.shape)
        metric = partial(_metric_sq, displacement_or_metric, box=runtime_box)
        metric = space.map_bond(metric)
        distances_sq = metric(position[senders.reshape(-1)], position[safe.reshape(-1)])
        valid = (distances_sq < cutoff_sq) & (candidates.reshape(-1) < n_atoms)
        out = n_atoms * jnp.ones((valid.shape[0],), dtype=jnp.int32)
        cumsum = jnp.cumsum(valid)
        slot = jnp.where(valid, cumsum - 1, valid.shape[0] - 1)
        out = out.at[slot].set(safe.reshape(-1))
        out_senders = n_atoms * jnp.ones((valid.shape[0],), dtype=jnp.int32)
        out_senders = out_senders.at[slot].set(senders.reshape(-1))
        return jnp.stack((out, out_senders)), cumsum[-1]

    def build(position, runtime_box, *, mask=None, neighbors=None, extra_capacity=0):
        n_atoms = position.shape[0]
        safe_position = position
        if mask is not None:
            safe_position = jnp.where(jnp.asarray(mask, dtype=jnp.bool_)[:, None], safe_position, 0.0)
        fractional = _fractional_positions(safe_position, runtime_box)

        if cell_fns is None:
            candidates = all_candidates(n_atoms)
            cell_capacity = None
            cell_overflow = jnp.asarray(False)
            cell_size = None
        elif neighbors is None:
            cell = cell_fns.allocate(fractional, extra_capacity=int(extra_capacity))
            candidates = cell_candidates(cell.id_buffer, fractional.shape)
            cell_capacity = cell.cell_capacity
            cell_overflow = cell.did_buffer_overflow
            cell_size = cell.cell_size
        else:
            cell = cell_fns.update(fractional, neighbors.cell_list_capacity)
            candidates = cell_candidates(cell.id_buffer, fractional.shape)
            cell_capacity = neighbors.cell_list_capacity
            cell_overflow = cell.did_buffer_overflow
            cell_size = neighbors.cell_size

        candidates = _mask_dense(candidates, mask=mask)
        if format is partition.NeighborListFormat.Dense:
            idx, occupancy = prune_dense(position, candidates, runtime_box)
            if neighbors is None:
                capacity = int(float(jax.device_get(occupancy)) * float(capacity_multiplier)) + int(extra_capacity)
                capacity = min(capacity, idx.shape[1], n_atoms - 1)
            else:
                capacity = neighbors.max_occupancy
        else:
            idx, occupancy = prune_sparse(position, candidates, runtime_box)
            if neighbors is None:
                capacity = int(float(jax.device_get(occupancy)) * float(capacity_multiplier)) + n_atoms * int(extra_capacity)
                capacity = min(capacity, idx.shape[1], n_atoms * (n_atoms - 1))
            else:
                capacity = neighbors.max_occupancy

        capacity = max(int(capacity), 1)
        idx = idx[..., :capacity]
        error = partition.PartitionError(jnp.zeros((), dtype=jnp.uint8))
        error = error.update(partition.PEC.CELL_LIST_OVERFLOW, cell_overflow)
        error = error.update(partition.PEC.NEIGHBOR_LIST_OVERFLOW, occupancy > capacity)
        return partition.NeighborList(
            idx,
            position,
            error,
            cell_capacity,
            capacity,
            format,
            cell_size,
            cell_fns,
            update_neighbor_fn,
        )

    def allocate_neighbor_fn(position, extra_capacity=0, mask=None, box=None, **kwargs):
        runtime_box = reference_box_jax if box is None else jnp.asarray(box, dtype=position.dtype)
        return build(position, runtime_box, mask=mask, extra_capacity=extra_capacity)

    @jax.jit
    def update_neighbor_fn(position, neighbors, mask=None, box=None, **kwargs):
        runtime_box = reference_box_jax if box is None else jnp.asarray(box, dtype=position.dtype)
        return build(position, runtime_box, mask=mask, neighbors=neighbors)

    return partition.NeighborListFns(allocate_neighbor_fn, update_neighbor_fn)

