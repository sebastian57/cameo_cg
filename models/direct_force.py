"""Backend-independent physics kernel for central direct-force heads."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scatter_central_pair_forces(
    coefficient: jax.Array,
    vectors: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    valid_edges: jax.Array,
    num_nodes: int,
    edge_scale: jax.Array | None = None,
) -> jax.Array:
    """Scatter central directed-edge coefficients with exact action/reaction.

    ``vectors[e]`` must point from receiver to sender.  ``edge_scale`` is used
    for the cutoff envelope.  The factor one half implements the mean of the
    two directional coefficients when the graph contains both orientations.
    """
    coefficient = jnp.asarray(coefficient)
    vectors = jnp.asarray(vectors, dtype=coefficient.dtype)
    distance = jnp.linalg.norm(vectors, axis=-1)
    direction = vectors / jnp.maximum(distance[:, None], 1.0e-12)
    if edge_scale is None:
        edge_scale = jnp.ones_like(coefficient)
    mask = jnp.asarray(valid_edges, dtype=coefficient.dtype)
    edge_force = (
        mask[:, None]
        * jnp.asarray(edge_scale, dtype=coefficient.dtype)[:, None]
        * coefficient[:, None]
        * direction
    )
    sender_force = jax.ops.segment_sum(edge_force, senders, num_segments=num_nodes)
    receiver_force = jax.ops.segment_sum(edge_force, receivers, num_segments=num_nodes)
    return 0.5 * (sender_force - receiver_force)

