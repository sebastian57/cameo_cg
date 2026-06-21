"""Directed amino-acid pair edge-distance gates for Allegro edge energies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class EdgeDistanceGateStats:
    min_distance: np.ndarray
    max_distance: np.ndarray
    count: np.ndarray
    cutoff: float
    n_species: int
    falloff_percent_default: float = 0.05
    dataset_path: str = ""
    config_path: str = ""


@dataclass(frozen=True)
class EdgeDistanceGateBank:
    min_distance: jax.Array
    max_distance: jax.Array
    count: jax.Array
    falloff_percent: float = 0.05
    onset_percent: float = 0.0
    offset_percent: Optional[float] = None
    floor: float = 0.0
    stop_gradient: bool = True

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        falloff_percent: Optional[float] = None,
        onset_percent: float = 0.0,
        offset_percent: Optional[float] = None,
        floor: float = 0.0,
        stop_gradient: bool = True,
    ) -> "EdgeDistanceGateBank":
        data = np.load(Path(path), allow_pickle=False)
        default_percent = float(data.get("falloff_percent_default", np.asarray(0.05)))
        return cls(
            min_distance=jnp.asarray(data["min_distance"], dtype=jnp.float32),
            max_distance=jnp.asarray(data["max_distance"], dtype=jnp.float32),
            count=jnp.asarray(data["count"], dtype=jnp.int32),
            falloff_percent=(
                default_percent if falloff_percent is None else float(falloff_percent)
            ),
            onset_percent=float(onset_percent),
            offset_percent=(None if offset_percent is None else float(offset_percent)),
            floor=float(floor),
            stop_gradient=bool(stop_gradient),
        )


def select_clean_training_frames(
    *,
    R: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce train.py's clean frame shuffle/split before augmentation."""
    R = np.asarray(R, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    species = np.asarray(species, dtype=np.int32)
    if R.ndim != 3:
        raise ValueError(f"R must have shape (n_frames, n_atoms, 3), got {R.shape}")
    n_frames = int(R.shape[0])
    order = np.arange(n_frames, dtype=np.int32)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(order)
    n_train = int(np.round(n_frames * (1.0 - float(val_fraction))))
    train_idx = order[:n_train]
    return R[train_idx], mask[train_idx], species[train_idx]


def build_edge_distance_gate_stats(
    *,
    R: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    cutoff: float,
    n_species: Optional[int] = None,
    falloff_percent_default: float = 0.05,
    dataset_path: str = "",
    config_path: str = "",
) -> EdgeDistanceGateStats:
    """Collect directed per-type min/max distances for all valid edges."""
    R = np.asarray(R, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    species = np.asarray(species, dtype=np.int32)
    if species.ndim == 1:
        species = np.broadcast_to(species[None, :], mask.shape)
    if R.shape[:2] != mask.shape or mask.shape != species.shape:
        raise ValueError(
            "R, mask, and species frame/atom axes must match; "
            f"got R={R.shape}, mask={mask.shape}, species={species.shape}"
        )

    observed_max_species = int(np.max(species[mask > 0])) + 1 if np.any(mask > 0) else 0
    n_types = int(n_species) if n_species is not None else observed_max_species
    n_types = max(n_types, observed_max_species)
    min_distance = np.full((n_types, n_types), np.inf, dtype=np.float32)
    max_distance = np.full((n_types, n_types), -np.inf, dtype=np.float32)
    count = np.zeros((n_types, n_types), dtype=np.int32)
    cutoff = float(cutoff)

    for frame_R, frame_mask, frame_species in zip(R, mask, species):
        valid = np.nonzero(frame_mask > 0)[0]
        for offset, i in enumerate(valid):
            si = int(frame_species[i])
            for j in valid[offset + 1 :]:
                sj = int(frame_species[j])
                dist = float(np.linalg.norm(frame_R[i] - frame_R[j]))
                if dist > cutoff:
                    continue
                if dist < min_distance[si, sj]:
                    min_distance[si, sj] = dist
                if dist > max_distance[si, sj]:
                    max_distance[si, sj] = dist
                if dist < min_distance[sj, si]:
                    min_distance[sj, si] = dist
                if dist > max_distance[sj, si]:
                    max_distance[sj, si] = dist
                count[si, sj] += 1
                count[sj, si] += 1

    min_distance = np.where(count > 0, min_distance, 0.0).astype(np.float32)
    max_distance = np.where(count > 0, max_distance, 0.0).astype(np.float32)
    return EdgeDistanceGateStats(
        min_distance=min_distance,
        max_distance=max_distance,
        count=count,
        cutoff=cutoff,
        n_species=n_types,
        falloff_percent_default=float(falloff_percent_default),
        dataset_path=str(dataset_path),
        config_path=str(config_path),
    )


def save_edge_distance_gate_stats(stats: EdgeDistanceGateStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        min_distance=np.asarray(stats.min_distance, dtype=np.float32),
        max_distance=np.asarray(stats.max_distance, dtype=np.float32),
        count=np.asarray(stats.count, dtype=np.int32),
        cutoff=np.asarray(stats.cutoff, dtype=np.float32),
        n_species=np.asarray(stats.n_species, dtype=np.int32),
        falloff_percent_default=np.asarray(
            stats.falloff_percent_default, dtype=np.float32
        ),
        dataset_path=np.asarray(stats.dataset_path),
        config_path=np.asarray(stats.config_path),
    )


def edge_distance_gate_enabled(config: Any) -> bool:
    cfg = config.get("model", "edge_distance_gate", default={}) or {}
    return bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else bool(cfg)


def edge_distance_gate_config(config: Any) -> dict[str, Any]:
    cfg = config.get("model", "edge_distance_gate", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {"enabled": bool(cfg)}
    return dict(cfg)


def compute_edge_distance_gate(
    *,
    distances: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    species: jax.Array,
    valid_edges: jax.Array,
    bank: EdgeDistanceGateBank,
) -> jax.Array:
    """Return one smooth support weight per directed edge."""
    distances = jnp.asarray(distances, dtype=jnp.float32)
    senders = jnp.asarray(senders, dtype=jnp.int32)
    receivers = jnp.asarray(receivers, dtype=jnp.int32)
    species = jnp.asarray(species, dtype=jnp.int32)
    valid_edges = jnp.asarray(valid_edges, dtype=jnp.bool_)

    n_species = int(bank.count.shape[0])
    sender_species = jnp.clip(species[jnp.clip(senders, 0, species.shape[0] - 1)], 0, n_species - 1)
    receiver_species = jnp.clip(
        species[jnp.clip(receivers, 0, species.shape[0] - 1)], 0, n_species - 1
    )
    min_r = bank.min_distance[sender_species, receiver_species]
    max_r = bank.max_distance[sender_species, receiver_species]
    seen = bank.count[sender_species, receiver_species] > 0

    offset_percent = (
        float(bank.falloff_percent)
        if bank.offset_percent is None
        else float(bank.offset_percent)
    )
    onset_percent = float(bank.onset_percent)
    lower_onset = min_r * (1.0 + onset_percent)
    lower_offset = min_r * jnp.maximum(1.0 - offset_percent, 0.0)
    upper_onset = max_r * jnp.maximum(1.0 - onset_percent, 0.0)
    upper_offset = max_r * (1.0 + offset_percent)

    midpoint = 0.5 * (min_r + max_r)
    lower_onset = jnp.minimum(lower_onset, midpoint)
    upper_onset = jnp.maximum(upper_onset, midpoint)

    lower_width = jnp.maximum(lower_onset - lower_offset, 1.0e-6)
    upper_width = jnp.maximum(upper_offset - upper_onset, 1.0e-6)
    lower_x = jnp.clip((lower_onset - distances) / lower_width, 0.0, 1.0)
    upper_x = jnp.clip((distances - upper_onset) / upper_width, 0.0, 1.0)
    x = jnp.maximum(lower_x, upper_x)
    smooth = x * x * (3.0 - 2.0 * x)
    alpha = 1.0 - smooth
    alpha = float(bank.floor) + (1.0 - float(bank.floor)) * alpha
    alpha = jnp.where(valid_edges & seen, alpha, float(bank.floor))
    alpha = jnp.clip(alpha, float(bank.floor), 1.0)
    if bank.stop_gradient:
        alpha = jax.lax.stop_gradient(alpha)
    return alpha.astype(jnp.float32)
