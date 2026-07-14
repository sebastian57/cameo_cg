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
    alpha_power: float = 1.0
    stop_gradient: bool = True
    fragment_torsion_reference_phi: Optional[jax.Array] = None
    fragment_torsion_reference_psi: Optional[jax.Array] = None
    fragment_torsion_k: int = 0
    fragment_torsion_onset_score_deg: float = 0.0
    fragment_torsion_offset_score_deg: float = 1.0
    fragment_torsion_phi_indices: Optional[jax.Array] = None
    fragment_torsion_psi_indices: Optional[jax.Array] = None
    fragment_torsion_fragment_beads: Optional[jax.Array] = None
    ala2_pair_low: Optional[jax.Array] = None
    ala2_pair_high: Optional[jax.Array] = None
    ala2_pair_count: Optional[jax.Array] = None
    ala2_pair_margin_fraction: float = 0.2
    ala2_distance_mean: Optional[jax.Array] = None
    ala2_distance_inv_cov: Optional[jax.Array] = None
    ala2_distance_count: Optional[jax.Array] = None
    ala2_distance_onset: float = 0.0
    ala2_distance_offset: float = 1.0
    ala2_angular_center_low: Optional[jax.Array] = None
    ala2_angular_center_high: Optional[jax.Array] = None
    ala2_angular_center_count: Optional[jax.Array] = None
    ala2_angular_global_low: Optional[jax.Array] = None
    ala2_angular_global_high: Optional[jax.Array] = None
    ala2_angular_margin_fraction: float = 0.25
    ala2_latent_mean: Optional[jax.Array] = None
    ala2_latent_inv_cov: Optional[jax.Array] = None
    ala2_latent_count: Optional[jax.Array] = None
    ala2_latent_onset: float = 0.0
    ala2_latent_offset: float = 1.0
    ala2_latent_feature_key: str = "tensor_norm_features_enveloped"
    ala2_combined_components: tuple[str, ...] = ()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        falloff_percent: Optional[float] = None,
        onset_percent: float = 0.0,
        offset_percent: Optional[float] = None,
        floor: float = 0.0,
        alpha_power: float = 1.0,
        stop_gradient: bool = True,
        fragment_torsion_gate_path: str | Path | None = None,
        ala2_combined_gate_path: str | Path | None = None,
    ) -> "EdgeDistanceGateBank":
        data = np.load(Path(path), allow_pickle=False)
        default_percent = float(data.get("falloff_percent_default", np.asarray(0.05)))
        torsion_kwargs: dict[str, Any] = {}
        if fragment_torsion_gate_path:
            torsion_data = np.load(Path(fragment_torsion_gate_path), allow_pickle=False)
            torsion_kwargs = {
                "fragment_torsion_reference_phi": jnp.asarray(
                    torsion_data["reference_phi"], dtype=jnp.float32
                ),
                "fragment_torsion_reference_psi": jnp.asarray(
                    torsion_data["reference_psi"], dtype=jnp.float32
                ),
                "fragment_torsion_k": int(torsion_data["k"]),
                "fragment_torsion_onset_score_deg": float(
                    torsion_data["onset_score_deg"]
                ),
                "fragment_torsion_offset_score_deg": float(
                    torsion_data["offset_score_deg"]
                ),
                "fragment_torsion_phi_indices": jnp.asarray(
                    torsion_data["phi_indices"], dtype=jnp.int32
                ),
                "fragment_torsion_psi_indices": jnp.asarray(
                    torsion_data["psi_indices"], dtype=jnp.int32
                ),
                "fragment_torsion_fragment_beads": jnp.asarray(
                    torsion_data["fragment_beads"], dtype=jnp.int32
                ),
            }
        ala2_kwargs: dict[str, Any] = {}
        if ala2_combined_gate_path:
            ala2_data = np.load(Path(ala2_combined_gate_path), allow_pickle=False)
            comps = tuple(str(x) for x in np.asarray(ala2_data["combined_components"]).tolist())
            ala2_kwargs = {
                "ala2_pair_low": jnp.asarray(ala2_data["pair_low"], dtype=jnp.float32),
                "ala2_pair_high": jnp.asarray(ala2_data["pair_high"], dtype=jnp.float32),
                "ala2_pair_count": jnp.asarray(ala2_data["pair_count"], dtype=jnp.int32),
                "ala2_pair_margin_fraction": float(ala2_data["pair_margin_fraction"]),
                "ala2_distance_mean": jnp.asarray(ala2_data["distance_matrix_mean"], dtype=jnp.float32),
                "ala2_distance_inv_cov": jnp.asarray(ala2_data["distance_matrix_inv_cov"], dtype=jnp.float32),
                "ala2_distance_count": jnp.asarray(ala2_data["distance_matrix_count"], dtype=jnp.int32),
                "ala2_distance_onset": float(ala2_data["distance_matrix_onset"]),
                "ala2_distance_offset": float(ala2_data["distance_matrix_offset"]),
                "ala2_angular_center_low": jnp.asarray(ala2_data["angular_center_low"], dtype=jnp.float32),
                "ala2_angular_center_high": jnp.asarray(ala2_data["angular_center_high"], dtype=jnp.float32),
                "ala2_angular_center_count": jnp.asarray(ala2_data["angular_center_count"], dtype=jnp.int32),
                "ala2_angular_global_low": jnp.asarray(ala2_data["angular_global_low"], dtype=jnp.float32),
                "ala2_angular_global_high": jnp.asarray(ala2_data["angular_global_high"], dtype=jnp.float32),
                "ala2_angular_margin_fraction": float(ala2_data["angular_margin_fraction"]),
                "ala2_combined_components": comps,
            }
            if "latent_mean" in ala2_data.files:
                ala2_kwargs.update(
                    {
                        "ala2_latent_mean": jnp.asarray(ala2_data["latent_mean"], dtype=jnp.float32),
                        "ala2_latent_inv_cov": jnp.asarray(ala2_data["latent_inv_cov"], dtype=jnp.float32),
                        "ala2_latent_count": jnp.asarray(ala2_data["latent_count"], dtype=jnp.int32),
                        "ala2_latent_onset": float(ala2_data["latent_onset"]),
                        "ala2_latent_offset": float(ala2_data["latent_offset"]),
                        "ala2_latent_feature_key": str(
                            np.asarray(
                                ala2_data["latent_feature_key"]
                                if "latent_feature_key" in ala2_data.files
                                else np.asarray("tensor_norm_features_enveloped")
                            ).item()
                        ),
                    }
                )
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
            alpha_power=max(float(alpha_power), 1.0e-6),
            stop_gradient=bool(stop_gradient),
            **torsion_kwargs,
            **ala2_kwargs,
        )

    @property
    def has_fragment_torsion_gate(self) -> bool:
        return (
            self.fragment_torsion_reference_phi is not None
            and self.fragment_torsion_reference_psi is not None
            and self.fragment_torsion_phi_indices is not None
            and self.fragment_torsion_psi_indices is not None
            and self.fragment_torsion_fragment_beads is not None
            and int(self.fragment_torsion_k) > 0
        )

    @property
    def has_ala2_combined_gate(self) -> bool:
        return (
            self.ala2_pair_low is not None
            and self.ala2_pair_high is not None
            and self.ala2_pair_count is not None
            and bool(self.ala2_combined_components)
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


def _wrap_degrees(angle: jax.Array) -> jax.Array:
    return jnp.mod(angle + 180.0, 360.0) - 180.0


def _dihedral_degrees(positions: jax.Array, indices: jax.Array) -> jax.Array:
    p0 = positions[indices[0]]
    p1 = positions[indices[1]]
    p2 = positions[indices[2]]
    p3 = positions[indices[3]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = jnp.linalg.norm(b1)
    b1_unit = b1 / jnp.maximum(b1_norm, 1.0e-12)
    v = b0 - jnp.sum(b0 * b1_unit) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit) * b1_unit
    x = jnp.sum(v * w)
    y = jnp.sum(jnp.cross(b1_unit, v) * w)
    angle = jnp.degrees(jnp.arctan2(y, x))
    return jnp.where(b1_norm > 1.0e-12, _wrap_degrees(angle), 0.0)


def _apply_floor_and_power(alpha: jax.Array, *, floor: float, alpha_power: float) -> jax.Array:
    floor_f = float(floor)
    alpha = floor_f + (1.0 - floor_f) * alpha
    alpha = floor_f + (1.0 - floor_f) * (
        (alpha - floor_f) / max(1.0 - floor_f, 1.0e-6)
    ) ** float(alpha_power)
    return jnp.clip(alpha, floor_f, 1.0)


def _fragment_torsion_alpha(positions: jax.Array, bank: EdgeDistanceGateBank) -> jax.Array:
    phi = _dihedral_degrees(positions, bank.fragment_torsion_phi_indices)
    psi = _dihedral_degrees(positions, bank.fragment_torsion_psi_indices)
    dphi = _wrap_degrees(phi - bank.fragment_torsion_reference_phi)
    dpsi = _wrap_degrees(psi - bank.fragment_torsion_reference_psi)
    dist = jnp.sqrt(dphi * dphi + dpsi * dpsi)
    kth = max(0, min(int(bank.fragment_torsion_k), int(dist.shape[0])) - 1)
    score = jnp.partition(dist, kth)[kth]
    width = max(
        float(bank.fragment_torsion_offset_score_deg)
        - float(bank.fragment_torsion_onset_score_deg),
        1.0e-6,
    )
    x = jnp.clip(
        (score - float(bank.fragment_torsion_onset_score_deg)) / width,
        0.0,
        1.0,
    )
    smooth = x * x * (3.0 - 2.0 * x)
    alpha = 1.0 - smooth
    return _apply_floor_and_power(
        alpha,
        floor=float(bank.floor),
        alpha_power=float(bank.alpha_power),
    )


def _ala2_pair_quantile_alpha(
    distances: jax.Array,
    sender_species: jax.Array,
    receiver_species: jax.Array,
    valid_edges: jax.Array,
    bank: EdgeDistanceGateBank,
) -> jax.Array:
    low = bank.ala2_pair_low[sender_species, receiver_species]
    high = bank.ala2_pair_high[sender_species, receiver_species]
    seen = bank.ala2_pair_count[sender_species, receiver_species] > 0
    width = jnp.maximum(high - low, 1.0e-6)
    margin = jnp.maximum(width * float(bank.ala2_pair_margin_fraction), 1.0e-6)
    lower = jnp.maximum((low - distances) / margin, 0.0)
    upper = jnp.maximum((distances - high) / margin, 0.0)
    score = jnp.clip(jnp.maximum(lower, upper), 0.0, 1.0)
    alpha = 1.0 - score * score * (3.0 - 2.0 * score)
    alpha = jnp.where(valid_edges & seen, alpha, float(bank.floor))
    return _apply_floor_and_power(alpha, floor=float(bank.floor), alpha_power=float(bank.alpha_power))


def _ala2_local_distance_descriptor(positions: jax.Array, mask: jax.Array, center: int) -> jax.Array:
    n = positions.shape[0]
    idx = jnp.arange(n)
    d = jnp.linalg.norm(positions - positions[center], axis=1)
    valid = (mask > 0.5) & (idx != center)
    d_sort = jnp.sort(jnp.where(valid, d, jnp.asarray(1.0e6, dtype=positions.dtype)))
    center_d = d_sort[:6]
    neigh = jnp.argsort(jnp.where(valid, d, jnp.asarray(1.0e6, dtype=positions.dtype)))[:6]
    neigh_pos = positions[neigh]
    pd = jnp.linalg.norm(neigh_pos[:, None, :] - neigh_pos[None, :, :], axis=-1)
    tri = pd[jnp.triu_indices(6, k=1)]
    return jnp.concatenate([center_d, jnp.sort(tri)], axis=0).astype(jnp.float32)


def _ala2_distance_matrix_alpha(positions: jax.Array, mask: jax.Array, bank: EdgeDistanceGateBank) -> jax.Array:
    n_centers = int(bank.ala2_distance_count.shape[0])

    def score_center(c):
        desc = _ala2_local_distance_descriptor(positions, mask, c)
        delta = desc - bank.ala2_distance_mean[c]
        q = delta @ bank.ala2_distance_inv_cov[c] @ delta
        score = jnp.sqrt(jnp.maximum(q, 0.0))
        valid = (mask[c] > 0.5) & (bank.ala2_distance_count[c] > 0)
        return jnp.where(valid, score, 0.0)

    scores = jax.vmap(score_center)(jnp.arange(n_centers))
    score = jnp.max(scores)
    return _apply_floor_and_power(
        1.0 - _smoothstep01((score - float(bank.ala2_distance_onset)) / max(float(bank.ala2_distance_offset) - float(bank.ala2_distance_onset), 1.0e-6)),
        floor=float(bank.floor),
        alpha_power=float(bank.alpha_power),
    )


def _safe_cos(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    v1 = a - b
    v2 = c - b
    return jnp.clip(
        jnp.sum(v1 * v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1.0e-8),
        -1.0,
        1.0,
    )


def _dihedral_radians(a: jax.Array, b: jax.Array, c: jax.Array, d: jax.Array) -> jax.Array:
    b0 = b - a
    b1 = c - b
    b2 = d - c
    n0 = jnp.cross(b0, b1)
    n1 = jnp.cross(b1, b2)
    n0 = n0 / (jnp.linalg.norm(n0) + 1.0e-8)
    n1 = n1 / (jnp.linalg.norm(n1) + 1.0e-8)
    m1 = jnp.cross(n0, b1 / (jnp.linalg.norm(b1) + 1.0e-8))
    return jnp.arctan2(jnp.sum(m1 * n1), jnp.sum(n0 * n1))


def _ala2_angular_features(positions: jax.Array, center: jax.Array) -> jax.Array:
    start = jnp.clip(center - 2, 0, positions.shape[0] - 5)
    p = jax.lax.dynamic_slice(positions, (start, 0), (5, 3))
    distances = jnp.asarray(
        [
            jnp.linalg.norm(p[1] - p[0]),
            jnp.linalg.norm(p[2] - p[1]),
            jnp.linalg.norm(p[3] - p[2]),
            jnp.linalg.norm(p[4] - p[3]),
            jnp.linalg.norm(p[2] - p[0]),
            jnp.linalg.norm(p[3] - p[1]),
            jnp.linalg.norm(p[4] - p[2]),
            jnp.linalg.norm(p[3] - p[0]),
            jnp.linalg.norm(p[4] - p[1]),
            jnp.linalg.norm(p[4] - p[0]),
        ],
        dtype=positions.dtype,
    )
    phi_left = _dihedral_radians(p[0], p[1], p[2], p[3])
    phi_right = _dihedral_radians(p[1], p[2], p[3], p[4])
    chirality = jnp.sum(jnp.cross(p[1] - p[0], p[2] - p[1]) * (p[3] - p[2]))
    return jnp.concatenate(
        [
            distances,
            jnp.asarray(
                [
                    _safe_cos(p[1], p[2], p[3]),
                    _safe_cos(p[0], p[1], p[2]),
                    _safe_cos(p[2], p[3], p[4]),
                    jnp.sin(phi_left),
                    jnp.cos(phi_left),
                    jnp.sin(phi_right),
                    jnp.cos(phi_right),
                    chirality,
                ],
                dtype=positions.dtype,
            ),
        ],
        axis=0,
    ).astype(jnp.float32)


def _ala2_angular_alpha(positions: jax.Array, mask: jax.Array, bank: EdgeDistanceGateBank) -> jax.Array:
    n_centers = int(bank.ala2_angular_center_count.shape[0])
    centers = jnp.arange(n_centers)

    def center_score(c):
        has_window = (c >= 2) & (c + 2 < n_centers)
        lo = jnp.where(
            bank.ala2_angular_center_count[c] > 0,
            bank.ala2_angular_center_low[c],
            bank.ala2_angular_global_low,
        )
        hi = jnp.where(
            bank.ala2_angular_center_count[c] > 0,
            bank.ala2_angular_center_high[c],
            bank.ala2_angular_global_high,
        )
        feat = jax.lax.cond(
            has_window,
            lambda _: _ala2_angular_features(positions, c),
            lambda _: jnp.zeros_like(bank.ala2_angular_global_low),
            operand=None,
        )
        width = jnp.maximum(hi - lo, 1.0e-6)
        margin = jnp.maximum(width * float(bank.ala2_angular_margin_fraction), 1.0e-6)
        score = jnp.max(jnp.maximum((lo - feat) / margin, (feat - hi) / margin))
        valid = has_window & (mask[c] > 0.5)
        return jnp.where(valid, jnp.clip(score, 0.0, 1.0), 0.0)

    score = jnp.max(jax.vmap(center_score)(centers))
    return _apply_floor_and_power(
        1.0 - score * score * (3.0 - 2.0 * score),
        floor=float(bank.floor),
        alpha_power=float(bank.alpha_power),
    )


def _smoothstep01(x: jax.Array) -> jax.Array:
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ala2_latent_alpha(
    edge_latent_features: jax.Array,
    sender_species: jax.Array,
    receiver_species: jax.Array,
    valid_edges: jax.Array,
    bank: EdgeDistanceGateBank,
) -> jax.Array:
    if (
        bank.ala2_latent_mean is None
        or bank.ala2_latent_inv_cov is None
        or bank.ala2_latent_count is None
    ):
        return jnp.where(valid_edges, jnp.asarray(1.0, dtype=jnp.float32), float(bank.floor))
    n_species = int(bank.ala2_latent_count.shape[0])
    si = jnp.clip(sender_species, 0, n_species - 1)
    sj = jnp.clip(receiver_species, 0, n_species - 1)
    mean = bank.ala2_latent_mean[si, sj]
    inv_cov = bank.ala2_latent_inv_cov[si, sj]
    seen = bank.ala2_latent_count[si, sj] > 0
    delta = jnp.asarray(edge_latent_features, dtype=jnp.float32) - mean
    q = jnp.einsum("ed,edd,ed->e", delta, inv_cov, delta)
    score = jnp.sqrt(jnp.maximum(q, 0.0))
    width = max(float(bank.ala2_latent_offset) - float(bank.ala2_latent_onset), 1.0e-6)
    x = (score - float(bank.ala2_latent_onset)) / width
    alpha = 1.0 - _smoothstep01(x)
    alpha = _apply_floor_and_power(
        alpha,
        floor=float(bank.floor),
        alpha_power=float(bank.alpha_power),
    )
    return jnp.where(valid_edges & seen, alpha, float(bank.floor))


def compute_ala2_combined_gate_diagnostics(
    positions: jax.Array,
    mask: jax.Array,
    bank: EdgeDistanceGateBank,
) -> dict[str, jax.Array]:
    torsion = (
        _fragment_torsion_alpha(positions, bank)
        if bank.has_fragment_torsion_gate
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    distance = (
        _ala2_distance_matrix_alpha(positions, mask, bank)
        if bank.has_ala2_combined_gate and bank.ala2_distance_mean is not None
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    angular = (
        _ala2_angular_alpha(positions, mask, bank)
        if bank.has_ala2_combined_gate and bank.ala2_angular_center_low is not None
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    combined = jnp.asarray(1.0, dtype=jnp.float32)
    if (not bank.has_ala2_combined_gate) or ("torsion" in bank.ala2_combined_components):
        combined = jnp.minimum(combined, torsion)
    if bank.has_ala2_combined_gate and ("distance_matrix" in bank.ala2_combined_components):
        combined = jnp.minimum(combined, distance)
    if bank.has_ala2_combined_gate and ("angular" in bank.ala2_combined_components):
        combined = jnp.minimum(combined, angular)
    return {
        "torsion_alpha": torsion,
        "distance_matrix_alpha": distance,
        "angular_alpha": angular,
        "combined_structure_alpha": combined,
    }


def compute_edge_distance_gate(
    *,
    distances: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    species: jax.Array,
    valid_edges: jax.Array,
    bank: EdgeDistanceGateBank,
    positions: Optional[jax.Array] = None,
    edge_latent_features: Optional[jax.Array] = None,
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
    base_pair_alpha = 1.0 - smooth
    base_pair_alpha = _apply_floor_and_power(
        base_pair_alpha,
        floor=float(bank.floor),
        alpha_power=float(bank.alpha_power),
    )
    base_pair_alpha = jnp.where(valid_edges & seen, base_pair_alpha, float(bank.floor))
    alpha = base_pair_alpha
    if bank.has_ala2_combined_gate:
        alpha = jnp.where(valid_edges, jnp.asarray(1.0, dtype=jnp.float32), float(bank.floor))
        if "base_pair" in bank.ala2_combined_components:
            alpha = jnp.minimum(alpha, base_pair_alpha)
        if "pair" in bank.ala2_combined_components:
            alpha = jnp.minimum(
                alpha,
                _ala2_pair_quantile_alpha(
                    distances,
                    sender_species,
                    receiver_species,
                    valid_edges,
                    bank,
                ),
            )
    else:
        alpha = base_pair_alpha
    if positions is not None and bank.has_fragment_torsion_gate:
        torsion_alpha = _fragment_torsion_alpha(
            jnp.asarray(positions, dtype=jnp.float32),
            bank,
        )
        fragment = bank.fragment_torsion_fragment_beads
        sender_touches = jnp.any(senders[:, None] == fragment[None, :], axis=1)
        receiver_touches = jnp.any(receivers[:, None] == fragment[None, :], axis=1)
        touches_fragment = sender_touches | receiver_touches
        if (not bank.has_ala2_combined_gate) or ("torsion" in bank.ala2_combined_components):
            alpha = jnp.where(touches_fragment, jnp.minimum(alpha, torsion_alpha), alpha)
    if positions is not None and bank.has_ala2_combined_gate:
        diag = compute_ala2_combined_gate_diagnostics(
            jnp.asarray(positions, dtype=jnp.float32),
            jnp.ones((positions.shape[0],), dtype=jnp.float32),
            bank,
        )
        scalar_alpha = diag["combined_structure_alpha"]
        alpha = jnp.minimum(alpha, scalar_alpha)
    if (
        edge_latent_features is not None
        and bank.has_ala2_combined_gate
        and "latent" in bank.ala2_combined_components
    ):
        alpha = jnp.minimum(
            alpha,
            _ala2_latent_alpha(
                edge_latent_features,
                sender_species,
                receiver_species,
                valid_edges,
                bank,
            ),
        )
    alpha = jnp.clip(alpha, float(bank.floor), 1.0)
    if bank.stop_gradient:
        alpha = jax.lax.stop_gradient(alpha)
    return alpha.astype(jnp.float32)
