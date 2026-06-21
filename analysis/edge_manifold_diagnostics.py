#!/usr/bin/env python3
"""Offline edge-manifold diagnostics for Allegro CG models.

This runner compares radial type-pair support, Allegro edge latent support, and
center-local descriptor support on clean and generated OOD structures. It is an
offline analysis tool only; it does not alter model or MD behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

# Keep login-node analysis on CPU unless the caller explicitly asks otherwise.
if "JAX_PLATFORMS" not in os.environ and not os.environ.get("SLURM_JOB_ID"):
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in tiny unit tests.
    plt = None

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - pure unit tests do not need JAX import success.
    jax = None
    jnp = None


@dataclass(frozen=True)
class OODBatch:
    group: str
    R: np.ndarray
    mask: np.ndarray
    species: np.ndarray
    label: int
    severity: float
    touched_mask: np.ndarray


@dataclass(frozen=True)
class LatentTypePairStats:
    mean: np.ndarray
    inv_cov: np.ndarray
    count: np.ndarray
    regularization: float


@dataclass(frozen=True)
class CenterSupportStats:
    mean: np.ndarray
    inv_cov: np.ndarray
    regularization: float


def _valid_center(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask[..., None].astype(np.float32)
    denom = np.maximum(np.sum(weights, axis=1, keepdims=True), 1.0)
    return np.sum(R * weights, axis=1, keepdims=True) / denom


def _pair_min(radial_min: np.ndarray, species_i: int, species_j: int, fallback: float) -> float:
    if radial_min is None:
        return float(fallback)
    if species_i < radial_min.shape[0] and species_j < radial_min.shape[1]:
        value = float(radial_min[species_i, species_j])
        if np.isfinite(value) and value > 0.0:
            return value
    return float(fallback)


def generate_ood_batches(
    R: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    *,
    radial_min: Optional[np.ndarray] = None,
    seed: int = 0,
    noise_stds: Iterable[float] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
    clash_factors: Iterable[float] = (0.9, 0.7, 0.5),
    stretch_scales: Iterable[float] = (1.2, 1.5, 2.0),
    clash_edges_per_frame: int = 8,
) -> Dict[str, OODBatch]:
    """Create deterministic clean and OOD structure batches.

    The corruptions are intentionally simple and graded. They preserve array
    shapes and masks and return a touched mask for locality attribution.
    """
    R = np.asarray(R, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    species = np.asarray(species, dtype=np.int32)
    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError("R must have shape (frames, atoms, 3)")
    if mask.shape != R.shape[:2] or species.shape != R.shape[:2]:
        raise ValueError("mask and species must have shape R.shape[:2]")

    rng = np.random.default_rng(seed)
    out: Dict[str, OODBatch] = {}
    zero_touch = np.zeros(mask.shape, dtype=bool)
    out["clean"] = OODBatch("clean", R.copy(), mask.copy(), species.copy(), 0, 0.0, zero_touch)

    for std in noise_stds:
        noisy = R + float(std) * rng.normal(size=R.shape).astype(np.float32) * mask[..., None]
        key = f"noise_{float(std):g}"
        out[key] = OODBatch(key, np.where(mask[..., None] > 0, noisy, 0.0).astype(np.float32), mask.copy(), species.copy(), 1, float(std), mask > 0)

    center = _valid_center(R, mask)
    rel = R - center
    for scale in stretch_scales:
        key = f"stretch_{float(scale):g}x"
        stretched = center + float(scale) * rel
        out[key] = OODBatch(key, np.where(mask[..., None] > 0, stretched, 0.0).astype(np.float32), mask.copy(), species.copy(), 1, float(scale), mask > 0)

    # Type-aware clash: move several valid neighbors toward the first bead in each frame.
    # Multiple touched edges make local-event metrics less dominated by untouched edges.
    for factor in clash_factors:
        clashed = R.copy()
        touched = np.zeros(mask.shape, dtype=bool)
        for b in range(R.shape[0]):
            valid = np.flatnonzero(mask[b] > 0.5)
            if valid.size < 2:
                continue
            i = int(valid[0])
            candidates = valid[1:]
            n_pick = min(max(int(clash_edges_per_frame), 1), int(candidates.size))
            picked = rng.choice(candidates, size=n_pick, replace=False)
            for j_raw in np.atleast_1d(picked):
                j = int(j_raw)
                vec = R[b, j] - R[b, i]
                dist = float(np.linalg.norm(vec))
                if dist < 1.0e-8:
                    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    dist = 1.0
                else:
                    direction = (vec / dist).astype(np.float32)
                target = _pair_min(radial_min, int(species[b, i]), int(species[b, j]), dist) * float(factor)
                clashed[b, j] = R[b, i] + direction * target
                touched[b, i] = True
                touched[b, j] = True
        key = f"clash_{float(factor):g}xmin"
        out[key] = OODBatch(key, np.where(mask[..., None] > 0, clashed, 0.0).astype(np.float32), mask.copy(), species.copy(), 1, float(factor), touched)

    # Density collapse: put all neighbors close to the first neighbor shell direction.
    density = R.copy()
    touched = np.zeros(mask.shape, dtype=bool)
    for b in range(R.shape[0]):
        valid = np.flatnonzero(mask[b] > 0.5)
        if valid.size < 3:
            continue
        i = int(valid[0])
        anchor = R[b, int(valid[1])] - R[b, i]
        norm = float(np.linalg.norm(anchor))
        if norm < 1.0e-8:
            anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction = anchor / norm
        for pos, j in enumerate(valid[1:]):
            jitter = 0.03 * rng.normal(size=3).astype(np.float32)
            density[b, int(j)] = R[b, i] + direction * (norm + 0.02 * pos) + jitter
            touched[b, int(j)] = True
        touched[b, i] = True
    out["density_collapse"] = OODBatch("density_collapse", np.where(mask[..., None] > 0, density, 0.0).astype(np.float32), mask.copy(), species.copy(), 1, 1.0, touched)

    # Angular shell shuffle: preserve each center-neighbor radius but roll directions.
    angular = R.copy()
    touched = np.zeros(mask.shape, dtype=bool)
    for b in range(R.shape[0]):
        valid = np.flatnonzero(mask[b] > 0.5)
        if valid.size < 3:
            continue
        i = int(valid[0])
        neigh = valid[1:]
        rel_n = R[b, neigh] - R[b, i]
        dist = np.linalg.norm(rel_n, axis=1)
        unit = rel_n / np.maximum(dist[:, None], 1.0e-8)
        rolled = np.roll(unit, 1, axis=0)
        angular[b, neigh] = R[b, i] + rolled * dist[:, None]
        touched[b, i] = True
        touched[b, neigh] = True
    out["angular_shell_shuffle"] = OODBatch("angular_shell_shuffle", np.where(mask[..., None] > 0, angular, 0.0).astype(np.float32), mask.copy(), species.copy(), 1, 1.0, touched)

    # Sequence-bio OOD: stretch odd adjacent bonds and compress even adjacent bonds.
    seq = R.copy()
    touched = np.zeros(mask.shape, dtype=bool)
    for b in range(R.shape[0]):
        valid = np.flatnonzero(mask[b] > 0.5)
        for idx in range(len(valid) - 1):
            i, j = int(valid[idx]), int(valid[idx + 1])
            vec = seq[b, j] - seq[b, i]
            dist = float(np.linalg.norm(vec))
            if dist < 1.0e-8:
                continue
            scale = 1.45 if idx % 2 == 0 else 0.65
            seq[b, j] = seq[b, i] + (vec / dist) * dist * scale
            touched[b, i] = True
            touched[b, j] = True
    out["sequence_bio_distort"] = OODBatch("sequence_bio_distort", np.where(mask[..., None] > 0, seq, 0.0).astype(np.float32), mask.copy(), species.copy(), 1, 1.0, touched)
    return out


def compute_radial_alpha(
    distances: np.ndarray,
    sender_types: np.ndarray,
    receiver_types: np.ndarray,
    min_distance: np.ndarray,
    max_distance: np.ndarray,
    count: np.ndarray,
    *,
    onset_percent: float = 0.0,
    offset_percent: float = 0.05,
    floor: float = 0.0,
) -> np.ndarray:
    """Numpy version of the smooth directed type-pair radial support alpha."""
    distances = np.asarray(distances, dtype=np.float32)
    sender_types = np.asarray(sender_types, dtype=np.int32)
    receiver_types = np.asarray(receiver_types, dtype=np.int32)
    min_distance = np.asarray(min_distance, dtype=np.float32)
    max_distance = np.asarray(max_distance, dtype=np.float32)
    count = np.asarray(count, dtype=np.int32)
    st = np.clip(sender_types, 0, min_distance.shape[0] - 1)
    rt = np.clip(receiver_types, 0, min_distance.shape[1] - 1)
    min_r = min_distance[st, rt]
    max_r = max_distance[st, rt]
    seen = count[st, rt] > 0

    lower_onset = min_r * (1.0 + float(onset_percent))
    lower_offset = min_r * max(1.0 - float(offset_percent), 0.0)
    upper_onset = max_r * max(1.0 - float(onset_percent), 0.0)
    upper_offset = max_r * (1.0 + float(offset_percent))
    midpoint = 0.5 * (min_r + max_r)
    lower_onset = np.minimum(lower_onset, midpoint)
    upper_onset = np.maximum(upper_onset, midpoint)
    lower_width = np.maximum(lower_onset - lower_offset, 1.0e-6)
    upper_width = np.maximum(upper_offset - upper_onset, 1.0e-6)
    lower_x = np.clip((lower_onset - distances) / lower_width, 0.0, 1.0)
    upper_x = np.clip((distances - upper_onset) / upper_width, 0.0, 1.0)
    x = np.maximum(lower_x, upper_x)
    smooth = x * x * (3.0 - 2.0 * x)
    alpha = 1.0 - smooth
    alpha = float(floor) + (1.0 - float(floor)) * alpha
    alpha = np.where(seen, alpha, float(floor))
    return np.clip(alpha, float(floor), 1.0).astype(np.float32)


def fit_latent_type_pair_stats(
    features: np.ndarray,
    sender_types: np.ndarray,
    receiver_types: np.ndarray,
    valid: np.ndarray,
    *,
    n_species: Optional[int] = None,
    regularization: float = 1.0e-2,
    min_count: int = 2,
) -> LatentTypePairStats:
    features = np.asarray(features, dtype=np.float32)
    sender_types = np.asarray(sender_types, dtype=np.int32)
    receiver_types = np.asarray(receiver_types, dtype=np.int32)
    valid = np.asarray(valid, dtype=bool)
    if features.ndim != 2:
        raise ValueError("features must be (n_edges, feature_dim)")
    if n_species is None:
        observed = np.concatenate([sender_types[valid], receiver_types[valid]]) if np.any(valid) else np.array([0])
        n_species = int(np.max(observed)) + 1
    d = features.shape[1]
    mean = np.zeros((n_species, n_species, d), dtype=np.float32)
    inv_cov = np.zeros((n_species, n_species, d, d), dtype=np.float32)
    count = np.zeros((n_species, n_species), dtype=np.int32)
    eye = np.eye(d, dtype=np.float64)
    for si in range(n_species):
        for sj in range(n_species):
            sel = valid & (sender_types == si) & (receiver_types == sj)
            x = features[sel].astype(np.float64)
            count[si, sj] = int(x.shape[0])
            if x.shape[0] < int(min_count):
                continue
            mu = x.mean(axis=0)
            centered = x - mu
            if x.shape[0] > 1:
                cov = centered.T @ centered / max(x.shape[0] - 1, 1)
            else:
                cov = np.zeros((d, d), dtype=np.float64)
            cov = cov + float(regularization) * eye
            mean[si, sj] = mu.astype(np.float32)
            inv_cov[si, sj] = np.linalg.pinv(cov).astype(np.float32)
    return LatentTypePairStats(mean=mean, inv_cov=inv_cov, count=count, regularization=float(regularization))


def compute_latent_mahalanobis_scores(
    features: np.ndarray,
    sender_types: np.ndarray,
    receiver_types: np.ndarray,
    valid: np.ndarray,
    stats: LatentTypePairStats,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    sender_types = np.asarray(sender_types, dtype=np.int32)
    receiver_types = np.asarray(receiver_types, dtype=np.int32)
    valid = np.asarray(valid, dtype=bool)
    out = np.full((features.shape[0],), np.inf, dtype=np.float32)
    n_species = stats.count.shape[0]
    in_range = valid & (sender_types >= 0) & (receiver_types >= 0) & (sender_types < n_species) & (receiver_types < n_species)
    for si in range(n_species):
        for sj in range(n_species):
            sel = in_range & (sender_types == si) & (receiver_types == sj) & (stats.count[si, sj] > 0)
            if not np.any(sel):
                continue
            delta = features[sel] - stats.mean[si, sj]
            q = np.einsum("nd,dd,nd->n", delta, stats.inv_cov[si, sj], delta)
            out[sel] = np.sqrt(np.maximum(q, 0.0)).astype(np.float32)
    return out


def smooth_alpha_from_scores(scores: np.ndarray, *, onset: float, offset: float, floor: float = 0.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    width = max(float(offset) - float(onset), 1.0e-6)
    t = np.clip((scores - float(onset)) / width, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    alpha = 1.0 - smooth
    return (float(floor) + (1.0 - float(floor)) * alpha).astype(np.float32)


def fit_center_support_stats(desc: np.ndarray, *, regularization: float = 1.0e-3) -> CenterSupportStats:
    desc = np.asarray(desc, dtype=np.float32)
    mu = desc.mean(axis=0)
    centered = desc.astype(np.float64) - mu.astype(np.float64)
    cov = centered.T @ centered / max(desc.shape[0] - 1, 1)
    cov = cov + float(regularization) * np.eye(desc.shape[1], dtype=np.float64)
    return CenterSupportStats(mu.astype(np.float32), np.linalg.pinv(cov).astype(np.float32), float(regularization))


def compute_center_scores(desc: np.ndarray, stats: CenterSupportStats) -> np.ndarray:
    delta = np.asarray(desc, dtype=np.float32) - stats.mean
    q = np.einsum("nd,dd,nd->n", delta, stats.inv_cov, delta)
    return np.sqrt(np.maximum(q, 0.0)).astype(np.float32)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Average ties.
    unique, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    rank_sum_pos = np.sum(ranks[labels == 1])
    return float((rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if np.sum(labels == 1) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    recall = tp / max(np.sum(labels == 1), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def _summarize_alpha(alpha: np.ndarray, label: np.ndarray) -> dict[str, float]:
    alpha = np.asarray(alpha, dtype=np.float32)
    label = np.asarray(label, dtype=np.int32)
    score = 1.0 - alpha
    return {
        "alpha_mean": float(np.nanmean(alpha)),
        "alpha_min": float(np.nanmin(alpha)),
        "frac_alpha_lt_0.99": float(np.mean(alpha < 0.99)),
        "frac_alpha_lt_0.95": float(np.mean(alpha < 0.95)),
        "frac_alpha_lt_0.5": float(np.mean(alpha < 0.5)),
        "auroc_vs_clean": _binary_auc(label, score),
        "auprc_vs_clean": _binary_auprc(label, score),
    }


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, float(q)))


def summarize_outlier_scores(
    prefix: str,
    *,
    alpha: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    touched: Optional[np.ndarray] = None,
) -> dict[str, float]:
    alpha = np.asarray(alpha, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    finite_valid = valid & np.isfinite(alpha) & np.isfinite(score)
    out = {
        f"{prefix}_alpha_p01": _quantile(alpha[finite_valid], 0.01),
        f"{prefix}_alpha_p05": _quantile(alpha[finite_valid], 0.05),
        f"{prefix}_score_p95": _quantile(score[finite_valid], 0.95),
        f"{prefix}_score_p99": _quantile(score[finite_valid], 0.99),
        f"{prefix}_score_max": _quantile(score[finite_valid], 1.0),
    }
    if touched is None:
        touched_valid = np.zeros_like(finite_valid, dtype=bool)
    else:
        touched_valid = finite_valid & np.asarray(touched, dtype=bool)
    out[f"{prefix}_n_touched_valid"] = int(np.sum(touched_valid))
    if np.any(touched_valid):
        out.update({
            f"{prefix}_touched_alpha_mean": float(np.mean(alpha[touched_valid])),
            f"{prefix}_touched_alpha_min": float(np.min(alpha[touched_valid])),
            f"{prefix}_touched_frac_alpha_lt_0.99": float(np.mean(alpha[touched_valid] < 0.99)),
            f"{prefix}_touched_frac_alpha_lt_0.95": float(np.mean(alpha[touched_valid] < 0.95)),
            f"{prefix}_touched_frac_alpha_lt_0.5": float(np.mean(alpha[touched_valid] < 0.5)),
            f"{prefix}_touched_score_p95": _quantile(score[touched_valid], 0.95),
            f"{prefix}_touched_score_p99": _quantile(score[touched_valid], 0.99),
            f"{prefix}_touched_score_max": _quantile(score[touched_valid], 1.0),
        })
    else:
        out.update({
            f"{prefix}_touched_alpha_mean": float("nan"),
            f"{prefix}_touched_alpha_min": float("nan"),
            f"{prefix}_touched_frac_alpha_lt_0.99": float("nan"),
            f"{prefix}_touched_frac_alpha_lt_0.95": float("nan"),
            f"{prefix}_touched_frac_alpha_lt_0.5": float("nan"),
            f"{prefix}_touched_score_p95": float("nan"),
            f"{prefix}_touched_score_p99": float("nan"),
            f"{prefix}_touched_score_max": float("nan"),
        })
    return out


def _load_npz_arrays(dataset_path: Path, max_frames: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    R = data["R"].astype(np.float32) if "R" in data else data["coords"].astype(np.float32)
    mask = data["mask"].astype(np.float32) if "mask" in data else np.ones(R.shape[:2], dtype=np.float32)
    species = data["species"].astype(np.int32) if "species" in data else np.zeros(R.shape[:2], dtype=np.int32)
    if max_frames is not None:
        R = R[: int(max_frames)]
        mask = mask[: int(max_frames)]
        species = species[: int(max_frames)]
    return R, mask, species


def _split_indices(n: int, *, seed: int, val_fraction: float, holdout_fraction: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_val = int(round(n * float(val_fraction)))
    n_holdout = int(round(n * float(holdout_fraction)))
    val = order[:n_val]
    holdout = order[n_val:n_val + n_holdout]
    train = order[n_val + n_holdout:]
    if train.size == 0:
        train = order[n_val:]
        holdout = np.asarray([], dtype=np.int64)
    return {"train": train, "val": val, "holdout": holdout}


def split_indices_with_optional_protein_holdout(
    n: int,
    *,
    seed: int,
    val_fraction: float,
    protein_id: Optional[np.ndarray] = None,
    holdout_protein_id: Optional[int] = None,
    holdout_fraction: float = 0.1,
) -> dict[str, np.ndarray | str]:
    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    if protein_id is not None and holdout_protein_id is not None and int(holdout_protein_id) >= 0:
        protein_id = np.asarray(protein_id, dtype=np.int32)[:n]
        holdout = all_idx[protein_id == int(holdout_protein_id)]
        pool = all_idx[protein_id != int(holdout_protein_id)]
        if holdout.size > 0 and pool.size > 1:
            order = pool.copy()
            rng.shuffle(order)
            n_val = max(1, int(round(order.size * float(val_fraction))))
            val = order[:n_val]
            train = order[n_val:]
            return {
                "train": train,
                "val": val,
                "holdout": holdout,
                "holdout_source": f"protein_id={int(holdout_protein_id)}",
            }

    splits = _split_indices(n, seed=seed, val_fraction=val_fraction, holdout_fraction=holdout_fraction)
    splits["holdout_source"] = "random_fraction"
    return splits


def _load_protein_id(dataset_path: Path, max_frames: Optional[int] = None) -> Optional[np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    if "protein_id" not in data:
        return None
    protein_id = np.asarray(data["protein_id"], dtype=np.int32)
    if max_frames is not None:
        protein_id = protein_id[: int(max_frames)]
    return protein_id


def _fit_radial_stats(R: np.ndarray, mask: np.ndarray, species: np.ndarray, *, cutoff: float, n_species: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_d = np.full((n_species, n_species), np.inf, dtype=np.float32)
    max_d = np.zeros((n_species, n_species), dtype=np.float32)
    count = np.zeros((n_species, n_species), dtype=np.int32)
    for b in range(R.shape[0]):
        valid = np.flatnonzero(mask[b] > 0.5)
        for ai in valid:
            for aj in valid:
                if ai == aj:
                    continue
                d = float(np.linalg.norm(R[b, aj] - R[b, ai]))
                if d <= float(cutoff):
                    si, sj = int(species[b, ai]), int(species[b, aj])
                    min_d[si, sj] = min(min_d[si, sj], d)
                    max_d[si, sj] = max(max_d[si, sj], d)
                    count[si, sj] += 1
    min_d = np.where(np.isfinite(min_d), min_d, 0.0).astype(np.float32)
    return min_d, max_d, count


def load_radial_stats_from_artifact(path: str | Path, *, n_species: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    min_distance = np.asarray(data["min_distance"], dtype=np.float32)
    max_distance = np.asarray(data["max_distance"], dtype=np.float32)
    count = np.asarray(data["count"], dtype=np.int32)
    if n_species is None or int(n_species) <= min_distance.shape[0]:
        return min_distance, max_distance, count
    target = int(n_species)
    min_out = np.zeros((target, target), dtype=np.float32)
    max_out = np.zeros((target, target), dtype=np.float32)
    count_out = np.zeros((target, target), dtype=np.int32)
    n0, n1 = min_distance.shape
    min_out[:n0, :n1] = min_distance
    max_out[:n0, :n1] = max_distance
    count_out[:n0, :n1] = count
    return min_out, max_out, count_out


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _load_model(training_config_path: Path, params_path: Path, dataset_path: Path, R0: np.ndarray, mask0: np.ndarray, species0: np.ndarray):
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.jax_setup import apply_jax_compat_shims
    apply_jax_compat_shims()
    from config.manager import ConfigManager
    from data.loader import DatasetLoader
    from data.preprocessor import CoordinatePreprocessor
    from models.combined_model import CombinedModel

    cfg = ConfigManager(str(training_config_path))
    loader = DatasetLoader(str(dataset_path))
    cutoff = float(cfg.get_cutoff())
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=cfg.get_buffer_multiplier(),
        park_multiplier=cfg.get_park_multiplier(),
    )
    box, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    R0p = preprocessor.center_and_park(R0[None], mask0[None], box, R_shift)[0]
    data_n_species = int(np.max(loader.species)) + 1
    config_n_species = cfg.get("model", "allegro", "num_types", default=None)
    n_species = max(data_n_species, int(config_n_species or 0))
    model = CombinedModel(
        config=cfg,
        R0=jnp.asarray(R0p, dtype=jnp.float32),
        box=box,
        species=jnp.asarray(species0, dtype=jnp.int32),
        N_max=loader.N_max,
        n_species_override=n_species,
    )
    with params_path.open("rb") as fh:
        params = pickle.load(fh)
    if isinstance(params, dict):
        if isinstance(params.get("params"), dict):
            params = params["params"]
        elif isinstance(params.get("best_params"), dict):
            params = params["best_params"]
    if not isinstance(params, dict) or "ml" not in params:
        params = {"ml": params}
    return cfg, model, params, preprocessor, box, R_shift


def _extract_edge_features_for_batch(model, params: Mapping, preprocessor, box, R_shift, batch: OODBatch) -> list[dict[str, np.ndarray]]:
    rows = []
    for b in range(batch.R.shape[0]):
        Rp = preprocessor.center_and_park(batch.R[b:b + 1], batch.mask[b:b + 1], box, R_shift)[0]
        aux = model.ml_model.compute_al_features(
            params["ml"],
            jnp.asarray(Rp, dtype=jnp.float32),
            jnp.asarray(batch.mask[b], dtype=jnp.float32),
            jnp.asarray(batch.species[b], dtype=jnp.int32),
        )
        item = {k: np.asarray(jax.device_get(v)) for k, v in aux.items()}
        rows.append(item)
    return rows


def _edge_table(edge_outputs: list[dict[str, np.ndarray]], batch: OODBatch) -> dict[str, np.ndarray]:
    features = []
    features_un = []
    distances = []
    energies = []
    senders = []
    receivers = []
    frame = []
    valid = []
    touched_edge = []
    for b, aux in enumerate(edge_outputs):
        v = aux["valid_edges"].astype(bool)
        s = aux["senders"].astype(np.int32)
        r = aux["receivers"].astype(np.int32)
        features.append(aux["edge_features"])
        features_un.append(aux["edge_features_unenveloped"])
        distances.append(aux["distances"].astype(np.float32))
        energies.append(aux["per_edge_energy"].astype(np.float32))
        senders.append(s)
        receivers.append(r)
        frame.append(np.full(s.shape, b, dtype=np.int32))
        valid.append(v)
        touched = batch.touched_mask[b]
        touched_edge.append(touched[np.clip(s, 0, touched.shape[0] - 1)] | touched[np.clip(r, 0, touched.shape[0] - 1)])
    senders_a = np.concatenate(senders)
    receivers_a = np.concatenate(receivers)
    frame_a = np.concatenate(frame)
    return {
        "edge_features": np.concatenate(features).astype(np.float32),
        "edge_features_unenveloped": np.concatenate(features_un).astype(np.float32),
        "distances": np.concatenate(distances).astype(np.float32),
        "per_edge_energy": np.concatenate(energies).astype(np.float32),
        "senders": senders_a,
        "receivers": receivers_a,
        "frame": frame_a,
        "valid_edges": np.concatenate(valid).astype(bool),
        "touched_edge": np.concatenate(touched_edge).astype(bool),
        "sender_types": batch.species[frame_a, np.clip(senders_a, 0, batch.species.shape[1] - 1)],
        "receiver_types": batch.species[frame_a, np.clip(receivers_a, 0, batch.species.shape[1] - 1)],
    }


def _center_descriptor(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.local_extrapolation_gate import build_jax_geometric_bio_descriptor
    descs = []
    for b in range(R.shape[0]):
        descs.append(np.asarray(build_jax_geometric_bio_descriptor(R[b], mask[b], cutoff=10.0), dtype=np.float32))
    return np.concatenate(descs, axis=0)


def run_diagnostics(args: argparse.Namespace) -> Path:
    root = Path(args.project_root).resolve()
    dataset_path = _resolve(args.dataset, root)
    training_config_path = _resolve(args.training_config, root)
    params_path = _resolve(args.params, root)
    out_dir = _resolve(args.output_root, root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    R, mask, species = _load_npz_arrays(dataset_path, max_frames=args.max_frames)
    protein_id = _load_protein_id(dataset_path, max_frames=args.max_frames)
    splits = split_indices_with_optional_protein_holdout(
        R.shape[0],
        seed=args.seed,
        val_fraction=args.val_fraction,
        protein_id=protein_id,
        holdout_protein_id=args.holdout_protein_id,
    )
    train_idx = np.asarray(splits["train"], dtype=np.int64)[: args.max_train_frames]
    eval_idx = np.asarray(splits["val"], dtype=np.int64)[: args.max_eval_frames]
    holdout_idx = np.asarray(splits["holdout"], dtype=np.int64)[: args.max_eval_frames]
    holdout_source = str(splits.get("holdout_source", "unknown"))
    n_species = int(np.max(species)) + 1

    # Load model once using the first clean eval frame for neighbor-list shape setup.
    ref_idx = int(eval_idx[0] if eval_idx.size else train_idx[0])
    cfg, model, params, preprocessor, box, R_shift = _load_model(
        training_config_path, params_path, dataset_path, R[ref_idx], mask[ref_idx], species[ref_idx]
    )
    cutoff = float(cfg.get_cutoff())

    radial_source = "computed_from_selected_train_frames"
    radial_artifact = getattr(args, "radial_artifact", None)
    if radial_artifact is not None and str(radial_artifact).strip().lower() not in ("", "none", "null"):
        radial_path = _resolve(radial_artifact, root)
        if radial_path.exists():
            radial_min, radial_max, radial_count = load_radial_stats_from_artifact(radial_path, n_species=n_species)
            radial_source = str(radial_path)
        else:
            radial_min, radial_max, radial_count = _fit_radial_stats(R[train_idx], mask[train_idx], species[train_idx], cutoff=cutoff, n_species=n_species)
            radial_source = f"missing_artifact_fallback:{radial_path}"
    else:
        radial_min, radial_max, radial_count = _fit_radial_stats(R[train_idx], mask[train_idx], species[train_idx], cutoff=cutoff, n_species=n_species)

    train_batch = OODBatch("train", R[train_idx], mask[train_idx], species[train_idx], 0, 0.0, np.zeros(mask[train_idx].shape, dtype=bool))
    train_edges = _edge_table(_extract_edge_features_for_batch(model, params, preprocessor, box, R_shift, train_batch), train_batch)
    latent_stats = fit_latent_type_pair_stats(
        train_edges["edge_features"], train_edges["sender_types"], train_edges["receiver_types"], train_edges["valid_edges"], n_species=n_species
    )
    train_latent_scores = compute_latent_mahalanobis_scores(
        train_edges["edge_features"], train_edges["sender_types"], train_edges["receiver_types"], train_edges["valid_edges"], latent_stats
    )
    finite_train_latent = train_latent_scores[np.isfinite(train_latent_scores)]
    latent_onset = float(np.quantile(finite_train_latent, args.latent_onset_quantile)) if finite_train_latent.size else 1.0
    latent_offset = float(np.quantile(finite_train_latent, args.latent_offset_quantile)) if finite_train_latent.size else 2.0
    if latent_offset <= latent_onset:
        latent_offset = latent_onset + 1.0

    train_center_desc = _center_descriptor(R[train_idx], mask[train_idx])
    center_stats = fit_center_support_stats(train_center_desc)
    train_center_scores = compute_center_scores(train_center_desc, center_stats)
    center_onset = float(np.quantile(train_center_scores, args.center_onset_quantile))
    center_offset = float(np.quantile(train_center_scores, args.center_offset_quantile))
    if center_offset <= center_onset:
        center_offset = center_onset + 1.0

    eval_batches = generate_ood_batches(
        R[eval_idx],
        mask[eval_idx],
        species[eval_idx],
        radial_min=radial_min,
        seed=args.seed + 100,
        clash_edges_per_frame=args.clash_edges_per_frame,
    )
    if holdout_idx.size:
        eval_batches["protein_holdout_clean"] = OODBatch(
            "protein_holdout_clean", R[holdout_idx], mask[holdout_idx], species[holdout_idx], 0, 0.0, np.zeros(mask[holdout_idx].shape, dtype=bool)
        )

    group_rows = []
    frame_rows = []
    all_metric_arrays = {"label": [], "radial_score": [], "latent_score": [], "center_score": []}
    hist_data = {}
    scatter_dist = []
    scatter_latent = []
    scatter_label = []

    for group, batch in eval_batches.items():
        edge_table = _edge_table(_extract_edge_features_for_batch(model, params, preprocessor, box, R_shift, batch), batch)
        valid = edge_table["valid_edges"]
        radial_alpha = compute_radial_alpha(
            edge_table["distances"], edge_table["sender_types"], edge_table["receiver_types"], radial_min, radial_max, radial_count,
            onset_percent=args.radial_onset_percent, offset_percent=args.radial_offset_percent, floor=0.0,
        )
        latent_score = compute_latent_mahalanobis_scores(
            edge_table["edge_features"], edge_table["sender_types"], edge_table["receiver_types"], valid, latent_stats
        )
        latent_alpha = smooth_alpha_from_scores(latent_score, onset=latent_onset, offset=latent_offset, floor=0.0)
        valid_idx = valid & np.isfinite(latent_score)
        label_edges = np.full((valid_idx.sum(),), int(batch.label), dtype=np.int32)

        center_desc = _center_descriptor(batch.R, batch.mask)
        center_score = compute_center_scores(center_desc, center_stats)
        center_alpha = smooth_alpha_from_scores(center_score, onset=center_onset, offset=center_offset, floor=0.0)

        metrics = {
            "group": group,
            "label": int(batch.label),
            "severity": float(batch.severity),
            "n_frames": int(batch.R.shape[0]),
            "n_valid_edges": int(np.sum(valid)),
            "n_touched_edges": int(np.sum(edge_table["touched_edge"] & valid)),
            "radial_abs_energy_corr": float(np.corrcoef(1.0 - radial_alpha[valid], np.abs(edge_table["per_edge_energy"][valid]))[0, 1]) if np.sum(valid) > 2 else float("nan"),
            "latent_abs_energy_corr": float(np.corrcoef(latent_score[valid_idx], np.abs(edge_table["per_edge_energy"][valid_idx]))[0, 1]) if np.sum(valid_idx) > 2 else float("nan"),
        }
        center_valid = batch.mask.reshape(-1) > 0.5
        center_touched = batch.touched_mask.reshape(-1)
        metrics.update({f"radial_{k}": v for k, v in _summarize_alpha(radial_alpha[valid], np.full(np.sum(valid), batch.label)).items()})
        metrics.update({f"latent_{k}": v for k, v in _summarize_alpha(latent_alpha[valid_idx], label_edges).items()})
        metrics.update({f"center_{k}": v for k, v in _summarize_alpha(center_alpha[center_valid], np.full(np.sum(center_valid), batch.label)).items()})
        metrics.update(summarize_outlier_scores("radial", alpha=radial_alpha, score=1.0 - radial_alpha, valid=valid, touched=edge_table["touched_edge"]))
        metrics.update(summarize_outlier_scores("latent", alpha=latent_alpha, score=latent_score, valid=valid_idx, touched=edge_table["touched_edge"]))
        metrics.update(summarize_outlier_scores("center", alpha=center_alpha, score=center_score, valid=center_valid, touched=center_touched))
        group_rows.append(metrics)

        # Per-frame aggregate rows.
        for b in range(batch.R.shape[0]):
            e_sel = valid & (edge_table["frame"] == b)
            l_sel = valid_idx & (edge_table["frame"] == b)
            center_slice = slice(b * batch.R.shape[1], (b + 1) * batch.R.shape[1])
            frame_rows.append({
                "group": group,
                "frame_local": b,
                "label": int(batch.label),
                "severity": float(batch.severity),
                "radial_alpha_mean": float(np.mean(radial_alpha[e_sel])) if np.any(e_sel) else float("nan"),
                "radial_alpha_min": float(np.min(radial_alpha[e_sel])) if np.any(e_sel) else float("nan"),
                "radial_score_p99": _quantile((1.0 - radial_alpha)[e_sel], 0.99),
                "latent_score_mean": float(np.mean(latent_score[l_sel])) if np.any(l_sel) else float("nan"),
                "latent_score_p99": _quantile(latent_score[l_sel], 0.99),
                "latent_score_max": float(np.max(latent_score[l_sel])) if np.any(l_sel) else float("nan"),
                "latent_alpha_mean": float(np.mean(latent_alpha[l_sel])) if np.any(l_sel) else float("nan"),
                "latent_alpha_min": float(np.min(latent_alpha[l_sel])) if np.any(l_sel) else float("nan"),
                "center_score_mean": float(np.mean(center_score[center_slice])),
                "center_score_p99": _quantile(center_score[center_slice], 0.99),
                "center_alpha_mean": float(np.mean(center_alpha[center_slice])),
                "center_alpha_min": float(np.min(center_alpha[center_slice])),
            })

        hist_data[group] = {
            "radial_alpha": radial_alpha[valid].tolist(),
            "latent_alpha": latent_alpha[valid_idx].tolist(),
            "center_alpha": center_alpha.tolist(),
        }
        all_metric_arrays["label"].append(np.full(np.sum(valid), batch.label, dtype=np.int32))
        all_metric_arrays["radial_score"].append(1.0 - radial_alpha[valid])
        all_metric_arrays["latent_score"].append(np.where(np.isfinite(latent_score[valid]), latent_score[valid], np.nan))
        # Center scores have different cardinality; keep separate in summary below.
        scatter_dist.append(edge_table["distances"][valid_idx])
        scatter_latent.append(latent_score[valid_idx])
        scatter_label.append(np.full(np.sum(valid_idx), batch.label, dtype=np.int32))

    _add_group_vs_clean_detection_metrics(group_rows, hist_data)

    _write_csv(out_dir / "per_group_metrics.csv", group_rows)
    _write_csv(out_dir / "per_frame_metrics.csv", frame_rows)

    labels_all = np.concatenate(all_metric_arrays["label"]) if all_metric_arrays["label"] else np.array([], dtype=np.int32)
    radial_scores_all = np.concatenate(all_metric_arrays["radial_score"]) if all_metric_arrays["radial_score"] else np.array([], dtype=np.float32)
    latent_scores_all = np.concatenate(all_metric_arrays["latent_score"]) if all_metric_arrays["latent_score"] else np.array([], dtype=np.float32)
    summary = {
        "run_name": args.run_name,
        "dataset": str(dataset_path),
        "training_config": str(training_config_path),
        "params": str(params_path),
        "n_train_frames": int(len(train_idx)),
        "n_eval_frames": int(len(eval_idx)),
        "n_holdout_frames": int(len(holdout_idx)),
        "holdout_source": holdout_source,
        "holdout_protein_id": None if args.holdout_protein_id is None or int(args.holdout_protein_id) < 0 else int(args.holdout_protein_id),
        "radial_source": radial_source,
        "cutoff": cutoff,
        "n_species": n_species,
        "latent_onset": latent_onset,
        "latent_offset": latent_offset,
        "center_onset": center_onset,
        "center_offset": center_offset,
        "overall": {
            "radial_auroc": _binary_auc(labels_all, radial_scores_all),
            "radial_auprc": _binary_auprc(labels_all, radial_scores_all),
            "latent_auroc": _binary_auc(labels_all, latent_scores_all),
            "latent_auprc": _binary_auprc(labels_all, latent_scores_all),
        },
        "groups": group_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "hist_data.json").write_text(json.dumps(hist_data) + "\n")
    _write_plots(out_dir, group_rows, hist_data, scatter_dist, scatter_latent, scatter_label)
    return out_dir


def _add_group_vs_clean_detection_metrics(group_rows: list[dict], hist_data: Mapping[str, Mapping[str, list]]) -> None:
    clean_names = [name for name in ("clean", "protein_holdout_clean") if name in hist_data]
    if not clean_names:
        return
    for metric in ("radial_alpha", "latent_alpha", "center_alpha"):
        clean_alpha = np.concatenate([np.asarray(hist_data[name][metric], dtype=np.float32) for name in clean_names])
        clean_score = 1.0 - clean_alpha
        for row in group_rows:
            group = str(row["group"])
            if int(row.get("label", 0)) == 0 or group not in hist_data:
                continue
            group_alpha = np.asarray(hist_data[group][metric], dtype=np.float32)
            group_score = 1.0 - group_alpha
            labels = np.concatenate([np.zeros(clean_score.shape, dtype=np.int32), np.ones(group_score.shape, dtype=np.int32)])
            scores = np.concatenate([clean_score, group_score])
            prefix = metric.replace("_alpha", "")
            row[f"{prefix}_auroc_vs_clean"] = _binary_auc(labels, scores)
            row[f"{prefix}_auprc_vs_clean"] = _binary_auprc(labels, scores)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_plots(out_dir: Path, group_rows, hist_data, scatter_dist, scatter_latent, scatter_label) -> None:
    if plt is None:
        return
    clean_groups = {"clean", "protein_holdout_clean"}
    for metric in ("radial_alpha", "latent_alpha", "center_alpha"):
        plt.figure(figsize=(8, 5))
        for group, data in hist_data.items():
            values = np.asarray(data[metric], dtype=np.float32)
            if values.size == 0:
                continue
            alpha = 0.85 if group in clean_groups else 0.35
            plt.hist(values, bins=40, range=(0, 1), density=True, histtype="step", alpha=alpha, label=group)
        plt.xlabel(metric)
        plt.ylabel("density")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_hist.png", dpi=160)
        plt.close()

    groups = [r["group"] for r in group_rows]
    x = np.arange(len(groups))
    plt.figure(figsize=(max(8, len(groups) * 0.55), 5))
    plt.bar(x - 0.25, [r.get("radial_auroc_vs_clean", np.nan) for r in group_rows], width=0.25, label="radial")
    plt.bar(x, [r.get("latent_auroc_vs_clean", np.nan) for r in group_rows], width=0.25, label="latent")
    plt.bar(x + 0.25, [r.get("center_auroc_vs_clean", np.nan) for r in group_rows], width=0.25, label="center")
    plt.xticks(x, groups, rotation=70, ha="right", fontsize=8)
    plt.ylabel("AUROC vs clean")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "auroc_by_group.png", dpi=160)
    plt.close()

    if scatter_dist and scatter_latent:
        d = np.concatenate(scatter_dist)
        s = np.concatenate(scatter_latent)
        lab = np.concatenate(scatter_label)
        finite = np.isfinite(s)
        if np.any(finite):
            plt.figure(figsize=(7, 5))
            plt.scatter(d[finite & (lab == 0)], s[finite & (lab == 0)], s=4, alpha=0.25, label="clean")
            plt.scatter(d[finite & (lab == 1)], s[finite & (lab == 1)], s=4, alpha=0.25, label="OOD")
            plt.xlabel("edge distance")
            plt.ylabel("latent Mahalanobis score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "latent_score_vs_distance.png", dpi=160)
            plt.close()


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Offline Allegro edge-manifold diagnostic runner.")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--dataset", type=Path, default=root / "data_prep/datasets/dataset_2605_5pro_320_aggforce_1bead_mlonly/combined_dataset.npz")
    parser.add_argument("--training-config", type=Path, default=root / "local_work/outputs/20260609_5pro_aggforce_fm_residual_smaller_tiles/config_runtime_670425.yaml")
    parser.add_argument("--params", type=Path, default=root / "local_work/outputs/20260609_5pro_aggforce_fm_residual_smaller_tiles/checkpoints/epoch00090.pkl")
    parser.add_argument("--output-root", type=Path, default=root / "local_work/edge_manifold_diagnostics")
    parser.add_argument("--run-name", default="20260611_5pro_epoch90_edge_latent_diagnostics")
    parser.add_argument("--radial-artifact", type=Path, default=root / "local_work/edge_distance_gate_artifacts/20260610_5pro_epoch50_pairdist_type_edge_gate_falloff5pct.npz")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-train-frames", type=int, default=128)
    parser.add_argument("--max-eval-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-protein-id", type=int, default=4, help="Protein id to reserve as true clean holdout; set -1 for random holdout fallback.")
    parser.add_argument("--clash-edges-per-frame", type=int, default=8)
    parser.add_argument("--radial-onset-percent", type=float, default=0.10)
    parser.add_argument("--radial-offset-percent", type=float, default=0.10)
    parser.add_argument("--latent-onset-quantile", type=float, default=0.99)
    parser.add_argument("--latent-offset-quantile", type=float, default=0.999)
    parser.add_argument("--center-onset-quantile", type=float, default=0.99)
    parser.add_argument("--center-offset-quantile", type=float, default=0.999)
    return parser


def main(argv: Optional[list[str]] = None) -> Path:
    args = build_parser().parse_args(argv)
    out = run_diagnostics(args)
    print(f"Wrote edge manifold diagnostics to {out}")
    return out


if __name__ == "__main__":
    main()
