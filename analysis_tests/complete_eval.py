#!/usr/bin/env python3
"""
Complete offline force-field diagnostic suite.

Implements seven diagnostic modules for held-out validation data:
  1. Error vs force magnitude  (binned RMSE / MAE / cosine / norm bias)
  2. Predicted-vs-target calibration  (scatter, linear fit, amplitude compression)
  3. Strain-conditioned analysis  (prior energy as strain proxy)
  4. Local environment analysis  (neighbor count, local density, min nonbonded dist)
  5. Per-bead / residue-type analysis  (RMSE / cosine / norm bias per species)
  6. Top-k tail analysis  (error on top-1% / 5% / 10% highest-force beads)
  7. Local smoothness test  (small perturbations → force continuity)

Usage:
    python analysis_tests/complete_eval.py <config> <params> <output_dir> [options]

Designed to be invoked as a subprocess from analyze_suite.py (--complete-eval).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.manager import ConfigManager


# ---------------------------------------------------------------------------
# Shared helpers (mirrors detailed_force_eval.py conventions)
# ---------------------------------------------------------------------------

def _resolve_data_path(config: ConfigManager) -> Path:
    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return data_path


def _split_indices(
    n_frames: int, *, val_fraction: float, min_val_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_frames, dtype=np.int32)
    if n_frames > 1:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_frames)
    n_train = int(np.round(n_frames * (1.0 - val_fraction)))
    n_val = n_frames - n_train
    if n_val < min_val_samples:
        n_train = n_frames
        n_val = 0
    return indices[:n_train], indices[n_train : n_train + n_val]


def _resolve_spline_path_if_needed(config: ConfigManager) -> None:
    if not config.use_spline_priors_enabled():
        return
    spline_path = Path(config.get_spline_file_path())
    if spline_path.is_absolute():
        resolved = spline_path
    else:
        candidates = [
            Path.cwd() / spline_path,
            PROJECT_ROOT / spline_path,
            config.config_path.parent / spline_path,
        ]
        resolved = spline_path
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
    config.set("model", "priors", "spline_file", str(resolved.resolve()))


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _cosine_per_bead(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-bead cosine similarity.  Shape (N,)."""
    dots = np.sum(y_true * y_pred, axis=-1)
    norms = np.linalg.norm(y_true, axis=-1) * np.linalg.norm(y_pred, axis=-1)
    out = np.where(norms > 1e-12, dots / norms, np.nan)
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    d = np.linalg.norm(xc) * np.linalg.norm(yc)
    if d < 1e-12:
        return float("nan")
    return float(np.dot(xc.ravel(), yc.ravel()) / d)


def _bin_stats(
    values: np.ndarray, group: np.ndarray, n_bins: int = 10
) -> List[Dict[str, Any]]:
    """Bin *group* into equal-count quantiles and compute mean/std of *values* per bin."""
    edges = np.percentile(group, np.linspace(0, 100, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (group >= lo) & (group < hi)
        if not np.any(mask):
            continue
        rows.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(np.sum(mask)),
            "mean": float(np.nanmean(values[mask])),
            "std": float(np.nanstd(values[mask])),
        })
    if rows:
        rows[-1]["bin_hi"] = float(np.max(group))
    return rows


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _savefig(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ===================================================================
# Module 1: Error vs force magnitude
# ===================================================================

def _module_error_vs_force_magnitude(
    F_true: np.ndarray,
    F_pred: np.ndarray,
    out_dir: Path,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Bin by target force norm, compute RMSE/MAE/cosine/norm-bias per bin."""
    plt = _get_plt()

    norm_true = np.linalg.norm(F_true, axis=-1)
    norm_pred = np.linalg.norm(F_pred, axis=-1)
    error_norm = np.linalg.norm(F_pred - F_true, axis=-1)
    cosine = _cosine_per_bead(F_true, F_pred)
    norm_bias = norm_pred - norm_true

    edges = np.percentile(norm_true, np.linspace(0, 100, n_bins + 1))
    bin_data = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (norm_true >= lo) & (norm_true < hi)
        if not np.any(m):
            continue
        bin_data.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "count": int(np.sum(m)),
            "rmse": float(np.sqrt(np.mean(error_norm[m] ** 2))),
            "mae": float(np.mean(error_norm[m])),
            "mean_cosine": float(np.nanmean(cosine[m])),
            "mean_norm_bias": float(np.mean(norm_bias[m])),
            "mean_pred_norm": float(np.mean(norm_pred[m])),
            "mean_true_norm": float(np.mean(norm_true[m])),
        })
    if bin_data:
        bin_data[-1]["bin_hi"] = float(np.max(norm_true))

    # --- plot: binned metrics ---
    centers = [0.5 * (b["bin_lo"] + b["bin_hi"]) for b in bin_data]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, key, label in [
        (axes[0, 0], "rmse", "RMSE"),
        (axes[0, 1], "mae", "MAE"),
        (axes[1, 0], "mean_cosine", "Mean cosine sim"),
        (axes[1, 1], "mean_norm_bias", "Mean norm bias"),
    ]:
        vals = [b[key] for b in bin_data]
        ax.bar(range(len(centers)), vals, color="#2a9d8f", alpha=0.85, edgecolor="black")
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels([f"{c:.2f}" for c in centers], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Target force norm (bin center)")
        ax.set_ylabel(label)
        ax.set_title(label + " by target force norm")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Error vs Force Magnitude", fontsize=14)
    _savefig(fig, out_dir / "error_vs_force_magnitude.png")

    # --- plot: ||F_pred|| vs ||F_true|| density scatter ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(norm_true, norm_pred, gridsize=60, cmap="viridis", mincnt=1)
    lim = max(np.max(norm_true), np.max(norm_pred)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel(r"$\|F^{\mathrm{true}}\|$")
    ax.set_ylabel(r"$\|F^{\mathrm{pred}}\|$")
    ax.set_title("Predicted vs Target Force Norm")
    ax.legend()
    ax.set_aspect("equal")
    plt.colorbar(ax.collections[0], ax=ax, label="count")
    _savefig(fig, out_dir / "pred_vs_true_force_norm.png")

    return {"error_vs_force_magnitude_bins": bin_data}


# ===================================================================
# Module 2: Calibration / amplitude compression
# ===================================================================

def _module_calibration(
    F_true: np.ndarray,
    F_pred: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """Linear fit ||F_pred|| = a * ||F_true|| + b.  Check amplitude compression."""
    plt = _get_plt()

    norm_true = np.linalg.norm(F_true, axis=-1).astype(np.float64)
    norm_pred = np.linalg.norm(F_pred, axis=-1).astype(np.float64)

    valid = np.isfinite(norm_true) & np.isfinite(norm_pred)
    nt, np_ = norm_true[valid], norm_pred[valid]

    if nt.size < 2:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}

    slope, intercept = np.polyfit(nt, np_, 1)
    pearson = _pearson(nt, np_)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(nt, np_, gridsize=60, cmap="viridis", mincnt=1)
    lim = max(np.max(nt), np.max(np_)) * 1.05
    xs = np.linspace(0, lim, 200)
    ax.plot(xs, xs, "r--", linewidth=1.5, label="y = x")
    ax.plot(xs, slope * xs + intercept, "b-", linewidth=1.5,
            label=f"fit: a={slope:.4f}, b={intercept:.4f}")
    ax.set_xlabel(r"$\|F^{\mathrm{true}}\|$")
    ax.set_ylabel(r"$\|F^{\mathrm{pred}}\|$")
    ax.set_title(f"Force Norm Calibration (Pearson r = {pearson:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    plt.colorbar(ax.collections[0], ax=ax, label="count")
    _savefig(fig, out_dir / "calibration_fit.png")

    return {
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_pearson": float(pearson),
    }


# ===================================================================
# Module 3: Strain-conditioned analysis (prior energy as proxy)
# ===================================================================

def _module_strain_conditioned(
    F_true: np.ndarray,
    F_pred: np.ndarray,
    per_frame_prior_energy: Optional[np.ndarray],
    out_dir: Path,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Split frames by prior energy (strain proxy) and report per-group metrics.

    If prior energy is unavailable, falls back to per-frame mean target force norm.
    """
    plt = _get_plt()

    n_frames = F_true.shape[0]
    if per_frame_prior_energy is not None and per_frame_prior_energy.size == n_frames:
        strain_proxy = per_frame_prior_energy.astype(np.float64)
        proxy_label = "Prior energy"
    else:
        strain_proxy = np.mean(np.linalg.norm(F_true, axis=-1), axis=-1).astype(np.float64)
        proxy_label = "Mean target force norm"

    edges = np.percentile(strain_proxy, np.linspace(0, 100, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (strain_proxy >= lo) & (strain_proxy < hi)
        if not np.any(mask):
            continue
        ft = F_true[mask].reshape(-1, 3)
        fp = F_pred[mask].reshape(-1, 3)
        cosine = _cosine_per_bead(ft, fp)
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "n_frames": int(np.sum(mask)),
            "rmse": _rmse(ft, fp),
            "mae": _mae(ft, fp),
            "mean_cosine": float(np.nanmean(cosine)),
            "norm_bias": float(np.mean(np.linalg.norm(fp, axis=-1) - np.linalg.norm(ft, axis=-1))),
        })
    if rows:
        rows[-1]["bin_hi"] = float(np.max(strain_proxy))

    centers = [0.5 * (r["bin_lo"] + r["bin_hi"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, label in [
        (axes[0], "rmse", "RMSE"),
        (axes[1], "mean_cosine", "Mean cosine"),
        (axes[2], "norm_bias", "Norm bias"),
    ]:
        vals = [r[key] for r in rows]
        ax.bar(range(len(centers)), vals, color="#e76f51", alpha=0.85, edgecolor="black")
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels([f"{c:.1f}" for c in centers], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{proxy_label} (bin center)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by strain level")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Strain-Conditioned Analysis", fontsize=14)
    _savefig(fig, out_dir / "strain_conditioned.png")

    return {"strain_conditioned_bins": rows, "strain_proxy": proxy_label}


# ===================================================================
# Module 4: Local environment analysis
# ===================================================================

def _compute_neighbor_counts(R: np.ndarray, mask: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Count neighbors within cutoff for each valid bead across all frames.

    Returns flat array aligned with the flattened valid-bead array.
    """
    from scipy.spatial import cKDTree

    counts_flat = []
    for f_idx in range(R.shape[0]):
        valid = mask[f_idx] > 0
        coords = R[f_idx][valid]
        tree = cKDTree(coords)
        # -1 to exclude self
        n_neighbors = np.array([len(tree.query_ball_point(c, cutoff)) - 1 for c in coords],
                               dtype=np.int32)
        counts_flat.append(n_neighbors)
    return np.concatenate(counts_flat)


def _compute_min_nonbonded_dist(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Minimum distance to any other valid bead, per bead per frame (flat)."""
    from scipy.spatial import cKDTree

    min_dists = []
    for f_idx in range(R.shape[0]):
        valid = mask[f_idx] > 0
        coords = R[f_idx][valid]
        if coords.shape[0] < 2:
            min_dists.append(np.full(coords.shape[0], np.inf))
            continue
        tree = cKDTree(coords)
        dd, _ = tree.query(coords, k=2)
        min_dists.append(dd[:, 1].astype(np.float64))
    return np.concatenate(min_dists)


def _module_local_environment(
    F_true_flat: np.ndarray,
    F_pred_flat: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    cutoff: float,
    out_dir: Path,
    n_bins: int = 8,
) -> Dict[str, Any]:
    """Analyse error conditioned on local environment descriptors."""
    plt = _get_plt()

    neighbor_counts = _compute_neighbor_counts(R, mask, cutoff)
    min_nb_dist = _compute_min_nonbonded_dist(R, mask)

    error_norm = np.linalg.norm(F_pred_flat - F_true_flat, axis=-1)
    cosine = _cosine_per_bead(F_true_flat, F_pred_flat)

    results: Dict[str, Any] = {}
    for desc_name, desc_values in [
        ("neighbor_count", neighbor_counts),
        ("min_nonbonded_dist", min_nb_dist),
    ]:
        rmse_bins = _bin_stats(error_norm, desc_values, n_bins)
        cos_bins = _bin_stats(cosine, desc_values, n_bins)
        results[f"{desc_name}_rmse_bins"] = rmse_bins
        results[f"{desc_name}_cosine_bins"] = cos_bins

        centers_rmse = [0.5 * (b["bin_lo"] + b["bin_hi"]) for b in rmse_bins]
        centers_cos = [0.5 * (b["bin_lo"] + b["bin_hi"]) for b in cos_bins]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(range(len(centers_rmse)), [b["mean"] for b in rmse_bins],
                     color="#457b9d", alpha=0.85, edgecolor="black")
        axes[0].set_xticks(range(len(centers_rmse)))
        axes[0].set_xticklabels([f"{c:.2f}" for c in centers_rmse], rotation=45, ha="right", fontsize=8)
        axes[0].set_xlabel(desc_name)
        axes[0].set_ylabel("Error norm")
        axes[0].set_title(f"Error norm by {desc_name}")
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(len(centers_cos)), [b["mean"] for b in cos_bins],
                     color="#264653", alpha=0.85, edgecolor="black")
        axes[1].set_xticks(range(len(centers_cos)))
        axes[1].set_xticklabels([f"{c:.2f}" for c in centers_cos], rotation=45, ha="right", fontsize=8)
        axes[1].set_xlabel(desc_name)
        axes[1].set_ylabel("Cosine similarity")
        axes[1].set_title(f"Cosine by {desc_name}")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"Error vs {desc_name}", fontsize=14)
        _savefig(fig, out_dir / f"env_{desc_name}.png")

    return results


# ===================================================================
# Module 5: Per-bead / residue-type analysis
# ===================================================================

def _module_per_bead_type(
    F_true_flat: np.ndarray,
    F_pred_flat: np.ndarray,
    species_flat: np.ndarray,
    id_to_aa: Optional[Dict[int, str]],
    out_dir: Path,
) -> Dict[str, Any]:
    """RMSE / cosine / norm bias per species (residue type)."""
    plt = _get_plt()

    unique_species = np.unique(species_flat)
    rows = []
    for sid in unique_species:
        m = species_flat == sid
        ft = F_true_flat[m]
        fp = F_pred_flat[m]
        cosine = _cosine_per_bead(ft, fp)
        label = id_to_aa.get(int(sid), str(sid)) if id_to_aa else str(sid)
        rows.append({
            "species_id": int(sid),
            "label": label,
            "count": int(np.sum(m)),
            "rmse": _rmse(ft, fp),
            "mae": _mae(ft, fp),
            "mean_cosine": float(np.nanmean(cosine)),
            "norm_bias": float(np.mean(np.linalg.norm(fp, axis=-1) - np.linalg.norm(ft, axis=-1))),
        })
    rows.sort(key=lambda r: r["rmse"], reverse=True)

    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(labels) * 0.7), 5))
    for ax, key, title in [
        (axes[0], "rmse", "RMSE by bead type"),
        (axes[1], "mean_cosine", "Cosine by bead type"),
        (axes[2], "norm_bias", "Norm bias by bead type"),
    ]:
        vals = [r[key] for r in rows]
        colors = ["#e76f51" if key == "rmse" else "#2a9d8f" if key == "mean_cosine" else "#457b9d"]
        ax.barh(range(len(labels)), vals, color=colors[0], alpha=0.85, edgecolor="black")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Per Bead/Residue Type Analysis", fontsize=14)
    _savefig(fig, out_dir / "per_bead_type.png")

    return {"per_bead_type": rows}


# ===================================================================
# Module 6: Top-k tail analysis
# ===================================================================

def _module_topk_tail(
    F_true_flat: np.ndarray,
    F_pred_flat: np.ndarray,
    out_dir: Path,
    percentiles: Tuple[float, ...] = (1.0, 5.0, 10.0, 25.0),
) -> Dict[str, Any]:
    """Compute error metrics on the top-k% highest-force beads."""
    plt = _get_plt()

    norm_true = np.linalg.norm(F_true_flat, axis=-1)
    error_norm = np.linalg.norm(F_pred_flat - F_true_flat, axis=-1)
    cosine = _cosine_per_bead(F_true_flat, F_pred_flat)
    norm_pred = np.linalg.norm(F_pred_flat, axis=-1)

    rows = []
    for pct in percentiles:
        thresh = np.percentile(norm_true, 100.0 - pct)
        m = norm_true >= thresh
        if not np.any(m):
            continue
        rows.append({
            "top_pct": float(pct),
            "threshold": float(thresh),
            "count": int(np.sum(m)),
            "rmse": float(np.sqrt(np.mean(error_norm[m] ** 2))),
            "mae": float(np.mean(error_norm[m])),
            "mean_cosine": float(np.nanmean(cosine[m])),
            "mean_norm_bias": float(np.mean(norm_pred[m] - norm_true[m])),
        })

    # Also add the "all beads" row for context
    rows.append({
        "top_pct": 100.0,
        "threshold": 0.0,
        "count": int(norm_true.size),
        "rmse": float(np.sqrt(np.mean(error_norm ** 2))),
        "mae": float(np.mean(error_norm)),
        "mean_cosine": float(np.nanmean(cosine)),
        "mean_norm_bias": float(np.mean(norm_pred - norm_true)),
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pcts = [r["top_pct"] for r in rows]
    xlabels = [f"top {p:.0f}%" if p < 100 else "all" for p in pcts]
    for ax, key, label in [
        (axes[0], "rmse", "RMSE"),
        (axes[1], "mean_cosine", "Mean cosine"),
        (axes[2], "mean_norm_bias", "Norm bias"),
    ]:
        vals = [r[key] for r in rows]
        ax.bar(range(len(pcts)), vals, color="#d62828", alpha=0.85, edgecolor="black")
        ax.set_xticks(range(len(pcts)))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} on highest-force subsets")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Top-k Tail Analysis", fontsize=14)
    _savefig(fig, out_dir / "topk_tail.png")

    return {"topk_tail": rows}


# ===================================================================
# Module 7: Local smoothness test
# ===================================================================

def _module_smoothness(
    model,
    params,
    R_val,
    mask_val,
    species_val,
    out_dir: Path,
    *,
    n_test_frames: int = 5,
    n_perturbations: int = 20,
    sigma: float = 0.01,
    seed: int = 99,
) -> Dict[str, Any]:
    """
    Perturb coordinates by small Gaussian noise and measure force response continuity.

    Reports ||F(x+d) - F(x)|| / ||d|| (sensitivity) across perturbations.
    """
    import jax
    import jax.numpy as jnp

    plt = _get_plt()
    rng = np.random.RandomState(seed)

    n_frames = min(n_test_frames, R_val.shape[0])
    frame_indices = rng.choice(R_val.shape[0], size=n_frames, replace=False)

    def _forces_single(R, mask, species):
        def energy_fn(R_):
            return model.compute_energy(params, R_, mask, species, None)
        return -jax.grad(energy_fn)(R)

    forces_fn = jax.jit(_forces_single)

    sensitivities = []
    for f_idx in frame_indices:
        R0 = jnp.asarray(R_val[f_idx], dtype=jnp.float32)
        m = jnp.asarray(mask_val[f_idx], dtype=jnp.float32)
        sp = jnp.asarray(species_val[f_idx], dtype=jnp.int32)
        F0 = np.asarray(forces_fn(R0, m, sp), dtype=np.float64)
        valid = np.asarray(mask_val[f_idx]) > 0

        for _ in range(n_perturbations):
            delta = rng.normal(0.0, sigma, size=R_val.shape[1:]).astype(np.float32)
            delta[~valid] = 0.0
            R_pert = R0 + jnp.asarray(delta)
            F_pert = np.asarray(forces_fn(R_pert, m, sp), dtype=np.float64)
            dF = np.linalg.norm((F_pert - F0)[valid], axis=-1)
            dR = np.linalg.norm(delta[valid], axis=-1)
            safe_dR = np.maximum(dR, 1e-12)
            sensitivities.extend((dF / safe_dR).tolist())

    sensitivities = np.array(sensitivities, dtype=np.float64)
    metrics = {
        "smoothness_mean_sensitivity": float(np.mean(sensitivities)),
        "smoothness_median_sensitivity": float(np.median(sensitivities)),
        "smoothness_p95_sensitivity": float(np.percentile(sensitivities, 95)),
        "smoothness_p99_sensitivity": float(np.percentile(sensitivities, 99)),
        "smoothness_max_sensitivity": float(np.max(sensitivities)),
        "smoothness_sigma": float(sigma),
        "smoothness_n_frames": int(n_frames),
        "smoothness_n_perturbations": int(n_perturbations),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sensitivities, bins=60, alpha=0.75, color="#457b9d", edgecolor="black")
    ax.axvline(metrics["smoothness_mean_sensitivity"], color="red", linestyle="--",
               linewidth=2, label=f"mean = {metrics['smoothness_mean_sensitivity']:.2f}")
    ax.axvline(metrics["smoothness_p95_sensitivity"], color="orange", linestyle="--",
               linewidth=2, label=f"p95 = {metrics['smoothness_p95_sensitivity']:.2f}")
    ax.set_xlabel(r"$\|F(x+\delta) - F(x)\| / \|\delta\|$")
    ax.set_ylabel("Count")
    ax.set_title(f"Local Smoothness Test (sigma={sigma:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / "smoothness_sensitivity.png")

    return metrics


# ===================================================================
# Main orchestrator
# ===================================================================

def compute_complete_eval(
    config_path: Path,
    params_path: Path,
    output_dir: Path,
    *,
    devices_per_run: int = 4,
    max_val_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
    smoothness_frames: int = 5,
    smoothness_perturbations: int = 20,
    smoothness_sigma: float = 0.01,
) -> Dict[str, Any]:
    """Run all seven diagnostic modules and return combined metrics."""
    import jax
    import jax.numpy as jnp

    from utils.jax_setup import apply_jax_compat_shims
    apply_jax_compat_shims()

    from data.loader import DatasetLoader
    from data.preprocessor import CoordinatePreprocessor
    from models.combined_model import CombinedModel
    import pickle

    config = ConfigManager(str(config_path))
    _resolve_spline_path_if_needed(config)

    # --- load data ---
    data_path = _resolve_data_path(config)
    loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(), seed=config.get_seed())
    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, r_shift)
    box = np.asarray(extent, dtype=np.float32)

    train_idx, val_idx = _split_indices(
        dataset["R"].shape[0],
        val_fraction=float(config.get_val_fraction()),
        min_val_samples=int(config.get_batch_per_device()) * int(devices_per_run),
        seed=int(config.get_seed()),
    )
    if max_val_frames is not None:
        val_idx = val_idx[: int(max_val_frames)]
    if val_idx.size == 0:
        raise ValueError("No held-out validation samples for complete eval.")

    # --- build model ---
    config.set("model", "use_priors", bool(config.export_combined_ml_priors_enabled()))
    config.set("model", "train_priors", False)

    with open(params_path, "rb") as f:
        payload = pickle.load(f)

    params = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("params"), dict):
            params = payload["params"]
        elif isinstance(payload.get("trainer_state"), dict) and isinstance(payload["trainer_state"].get("params"), dict):
            params = payload["trainer_state"]["params"]

    if isinstance(params, dict) and "ml" not in params and "allegro" in params:
        params = dict(params)
        params["ml"] = params["allegro"]

    ref_idx = train_idx[0] if train_idx.size else val_idx[0]
    model = CombinedModel(
        config=config,
        R0=np.asarray(dataset["R"][ref_idx], dtype=np.float32),
        box=box,
        species=np.asarray(dataset["species"][ref_idx], dtype=np.int32),
        N_max=loader.N_max,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    # --- predict forces on validation set ---
    R_val = np.asarray(dataset["R"][val_idx], dtype=np.float32)
    F_val_true = np.asarray(dataset["F"][val_idx], dtype=np.float32)
    mask_val = np.asarray(dataset["mask"][val_idx], dtype=np.float32)
    species_val = np.asarray(dataset["species"][val_idx], dtype=np.int32)

    R_val_jax = jnp.asarray(R_val)
    mask_val_jax = jnp.asarray(mask_val)
    species_val_jax = jnp.asarray(species_val)

    def _forces_single(R, mask, species):
        def energy_fn(R_):
            return model.compute_energy(params, R_, mask, species, None)
        return -jax.grad(energy_fn)(R)

    batched_forces_fn = jax.jit(jax.vmap(_forces_single))

    n_val = val_idx.size
    chunk_size = n_val if batch_size is None else int(batch_size)
    chunks = []
    for start in range(0, n_val, chunk_size):
        end = min(start + chunk_size, n_val)
        F_chunk = batched_forces_fn(
            R_val_jax[start:end], mask_val_jax[start:end], species_val_jax[start:end],
        )
        chunks.append(np.asarray(F_chunk, dtype=np.float32))
        print(f"  [complete_eval] predicted forces {end}/{n_val} frames", flush=True)
    F_val_pred = np.concatenate(chunks, axis=0)

    # --- flatten to valid beads ---
    valid_mask = mask_val > 0
    F_true_flat = F_val_true[valid_mask]
    F_pred_flat = F_val_pred[valid_mask]
    species_flat = species_val[valid_mask]

    # --- compute prior energy per frame as strain proxy ---
    per_frame_prior = None
    if model.use_priors and model.prior is not None:
        print("  [complete_eval] computing per-frame prior energy...", flush=True)
        prior_energies = []
        for i in range(n_val):
            R_i = jnp.asarray(R_val[i])
            m_i = jnp.asarray(mask_val[i])
            sp_i = jnp.asarray(species_val[i])
            mask_3d = m_i[:, None]
            R_masked = jnp.where(mask_3d > 0, R_i, jax.lax.stop_gradient(R_i))
            E_prior = model.prior.compute_total_energy(R_masked, m_i, species=sp_i)
            prior_energies.append(float(E_prior))
        per_frame_prior = np.array(prior_energies, dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Any] = {
        "n_val_frames": int(n_val),
        "n_valid_beads": int(F_true_flat.shape[0]),
    }

    # --- Module 1: Error vs force magnitude ---
    print("  [complete_eval] module 1: error vs force magnitude", flush=True)
    all_metrics.update(_module_error_vs_force_magnitude(F_true_flat, F_pred_flat, output_dir))

    # --- Module 2: Calibration ---
    print("  [complete_eval] module 2: calibration", flush=True)
    all_metrics.update(_module_calibration(F_true_flat, F_pred_flat, output_dir))

    # --- Module 3: Strain-conditioned ---
    print("  [complete_eval] module 3: strain-conditioned analysis", flush=True)
    all_metrics.update(_module_strain_conditioned(
        F_val_true, F_val_pred, per_frame_prior, output_dir,
    ))

    # --- Module 4: Local environment ---
    print("  [complete_eval] module 4: local environment analysis", flush=True)
    all_metrics.update(_module_local_environment(
        F_true_flat, F_pred_flat, R_val, mask_val,
        cutoff=float(config.get_cutoff()), out_dir=output_dir,
    ))

    # --- Module 5: Per bead type ---
    print("  [complete_eval] module 5: per bead type", flush=True)
    all_metrics.update(_module_per_bead_type(
        F_true_flat, F_pred_flat, species_flat, loader.id_to_aa, output_dir,
    ))

    # --- Module 6: Top-k tail ---
    print("  [complete_eval] module 6: top-k tail analysis", flush=True)
    all_metrics.update(_module_topk_tail(F_true_flat, F_pred_flat, output_dir))

    # --- Module 7: Smoothness ---
    print("  [complete_eval] module 7: local smoothness test", flush=True)
    all_metrics.update(_module_smoothness(
        model, params, R_val, mask_val, species_val, output_dir,
        n_test_frames=smoothness_frames,
        n_perturbations=smoothness_perturbations,
        sigma=smoothness_sigma,
    ))

    # --- save ---
    metrics_path = output_dir / "complete_eval_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, default=str))

    # Save flat arrays for downstream re-analysis
    np.savez_compressed(
        str(output_dir / "complete_eval_arrays.npz"),
        F_true_flat=F_true_flat,
        F_pred_flat=F_pred_flat,
        species_flat=species_flat,
        per_frame_prior_energy=per_frame_prior if per_frame_prior is not None else np.array([]),
    )

    all_metrics["metrics_json_path"] = str(metrics_path)
    all_metrics["arrays_npz_path"] = str(output_dir / "complete_eval_arrays.npz")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Complete offline force-field diagnostic suite")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("params", type=str, help="Path to params.pkl")
    parser.add_argument("output_dir", type=str, help="Output directory for plots and metrics")
    parser.add_argument("--devices-per-run", type=int, default=4)
    parser.add_argument("--max-val-frames", type=int, default=None,
                        help="Cap on held-out validation frames")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Frames per vmap chunk (default: all at once)")
    parser.add_argument("--smoothness-frames", type=int, default=5,
                        help="Frames for smoothness test")
    parser.add_argument("--smoothness-perturbations", type=int, default=20,
                        help="Perturbations per frame for smoothness test")
    parser.add_argument("--smoothness-sigma", type=float, default=0.01,
                        help="Gaussian noise std for smoothness test")
    args = parser.parse_args()

    metrics = compute_complete_eval(
        Path(args.config).resolve(),
        Path(args.params).resolve(),
        Path(args.output_dir).resolve(),
        devices_per_run=args.devices_per_run,
        max_val_frames=args.max_val_frames,
        batch_size=args.batch_size,
        smoothness_frames=args.smoothness_frames,
        smoothness_perturbations=args.smoothness_perturbations,
        smoothness_sigma=args.smoothness_sigma,
    )
    print(json.dumps(
        {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        indent=2,
    ))


if __name__ == "__main__":
    main()
