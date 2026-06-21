"""Training-data support gates for energy-level ML residual throttling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SupportGateBank:
    centers: jax.Array
    sigma: float
    descriptor_mean: jax.Array
    descriptor_std: jax.Array
    n_atoms: int
    floor: float = 0.0
    stop_gradient: bool = False


def support_gate_enabled(config: Any) -> bool:
    return bool(config.get("training", "support_gate", "enabled", default=False))


def support_gate_config(config: Any) -> Dict[str, Any]:
    cfg = config.get("training", "support_gate", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {"enabled": bool(cfg)}
    return cfg


def support_gate_scope(config: Any) -> str:
    cfg = support_gate_config(config)
    return str(cfg.get("scope", "segment")).strip().lower()


def build_pairwise_distance_descriptors(R: jax.Array, mask: jax.Array) -> jax.Array:
    """Return flattened upper-triangle pair-distance descriptors.

    The descriptor is translation/rotation invariant and keeps a fixed length for
    padded batches by zeroing invalid bead-pair distances.
    """
    R = jnp.asarray(R)
    mask = jnp.asarray(mask)
    squeeze = False
    if R.ndim == 2:
        R = R[None, ...]
        mask = mask[None, ...]
        squeeze = True

    n_atoms = R.shape[1]
    i_idx, j_idx = jnp.triu_indices(n_atoms, k=1)
    disp = R[:, i_idx, :] - R[:, j_idx, :]
    distances = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + 1.0e-12)
    pair_mask = (mask[:, i_idx] > 0) & (mask[:, j_idx] > 0)
    descriptors = jnp.where(pair_mask, distances, 0.0)
    return descriptors[0] if squeeze else descriptors


def _standardize_descriptors(descriptors: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean = jnp.mean(descriptors, axis=0)
    std = jnp.std(descriptors, axis=0)
    std = jnp.where(std > 1.0e-6, std, 1.0)
    return (descriptors - mean) / std, mean, std


def _estimate_sigma(centers: np.ndarray, sigma_multiplier: float) -> float:
    if centers.shape[0] <= 1:
        return float(max(sigma_multiplier, 1.0e-6))
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float64))
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    finite = nn[np.isfinite(nn) & (nn > 1.0e-8)]
    if finite.size == 0:
        base = 1.0
    else:
        base = float(np.median(finite))
    return float(max(base * sigma_multiplier, 1.0e-6))


def build_support_gate_bank(
    *,
    R: np.ndarray,
    mask: np.ndarray,
    max_centers: int = 512,
    sigma_multiplier: float = 1.0,
    seed: int = 0,
    floor: float = 0.0,
    stop_gradient: bool = False,
) -> SupportGateBank:
    """Fit a small RBF support bank from training structures."""
    if max_centers <= 0:
        raise ValueError("max_centers must be positive")
    if sigma_multiplier <= 0.0:
        raise ValueError("sigma_multiplier must be positive")

    descriptors = build_pairwise_distance_descriptors(jnp.asarray(R), jnp.asarray(mask))
    standardized, mean, std = _standardize_descriptors(descriptors)
    standardized_np = np.asarray(jax.device_get(standardized), dtype=np.float32)
    n_frames = int(standardized_np.shape[0])
    if n_frames == 0:
        raise ValueError("cannot build support gate bank from an empty training set")

    n_centers = min(int(max_centers), n_frames)
    rng = np.random.RandomState(seed)
    if n_centers == n_frames:
        center_idx = np.arange(n_frames, dtype=np.int32)
    else:
        center_idx = np.sort(rng.choice(n_frames, size=n_centers, replace=False).astype(np.int32))
    centers_np = standardized_np[center_idx]
    sigma = _estimate_sigma(centers_np, sigma_multiplier)
    return SupportGateBank(
        centers=jnp.asarray(centers_np, dtype=jnp.float32),
        sigma=sigma,
        descriptor_mean=jnp.asarray(mean, dtype=jnp.float32),
        descriptor_std=jnp.asarray(std, dtype=jnp.float32),
        n_atoms=int(R.shape[1]),
        floor=float(floor),
        stop_gradient=bool(stop_gradient),
    )


def rbf_structure_support(R: jax.Array, mask: jax.Array, bank: SupportGateBank) -> jax.Array:
    descriptor = build_pairwise_distance_descriptors(R, mask)
    standardized = (descriptor - bank.descriptor_mean) / bank.descriptor_std
    diff = bank.centers - standardized[None, :]
    sq_dist = jnp.sum(diff * diff, axis=-1)
    support = jnp.max(jnp.exp(-0.5 * sq_dist / (bank.sigma**2)))
    support = bank.floor + (1.0 - bank.floor) * support
    support = jnp.clip(support, bank.floor, 1.0)
    if bank.stop_gradient:
        support = jax.lax.stop_gradient(support)
    return support


def rbf_segment_supports(
    R: jax.Array,
    mask: jax.Array,
    segment_id: jax.Array,
    bank: SupportGateBank,
    *,
    num_segments: Optional[int] = None,
) -> jax.Array:
    """Return one RBF support value per packed segment.

    Segments are compacted to the bank's original structure size before pairwise
    descriptors are computed. This lets tiled batches use a support bank fitted
    on untiled structures. Empty segment slots receive support 0.
    """
    R = jnp.asarray(R)
    mask = jnp.asarray(mask)
    segment_id = jnp.asarray(segment_id, dtype=jnp.int32)
    if R.ndim != 2:
        raise ValueError("rbf_segment_supports expects a single packed structure with shape (N, 3)")
    n_nodes = int(R.shape[0])
    n_atoms = int(bank.n_atoms)
    n_segments = int(num_segments) if num_segments is not None else n_nodes
    slots = jnp.arange(n_segments, dtype=jnp.int32)
    atom_slots = jnp.arange(n_atoms, dtype=jnp.int32)

    def compact_one(seg):
        in_seg = (segment_id == seg) & (mask > 0)
        ordinals = jnp.cumsum(in_seg.astype(jnp.int32)) - 1
        take = in_seg & (ordinals < n_atoms)
        weights = take[:, None] & (ordinals[:, None] == atom_slots[None, :])
        coords = jnp.sum(jnp.where(weights[:, :, None], R[:, None, :], 0.0), axis=0)
        seg_mask = jnp.sum(weights.astype(jnp.float32), axis=0)
        return coords, seg_mask

    def process_one(_, seg):
        coords, seg_mask = compact_one(seg)
        support = rbf_structure_support(coords, seg_mask, bank)
        nonempty = jnp.sum(seg_mask) > 0
        support = jnp.where(nonempty, support, 0.0)
        return None, support

    _, supports = jax.lax.scan(process_one, None, slots)
    if bank.stop_gradient:
        supports = jax.lax.stop_gradient(supports)
    return supports
