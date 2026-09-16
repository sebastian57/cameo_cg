"""Pure numerical primitives for the Ala2 MD-less landscape experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


BASINS = ("beta", "alphaR", "alphaL", "other")


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def assign_basins(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Assign the project-standard non-overlapping bb6 Ramachandran labels."""
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.shape != psi.shape:
        raise ValueError(f"phi and psi must have the same shape, got {phi.shape} and {psi.shape}")
    phi, psi = wrap_degrees(phi), wrap_degrees(psi)
    region = np.full(phi.shape, "other", dtype="U6")
    region[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
    region[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
    region[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
    return region


def evenly_spaced_indices(n_total: int, n_select: int) -> np.ndarray:
    if n_total <= 0 or n_select <= 0 or n_select > n_total:
        raise ValueError(f"require 0 < n_select <= n_total, got {n_select} and {n_total}")
    return np.linspace(0, n_total - 1, n_select).astype(np.int64)


def stratified_indices(regions: np.ndarray, per_region: int, seed: int) -> np.ndarray:
    regions = np.asarray(regions).astype(str)
    if per_region <= 0:
        raise ValueError(f"per_region must be positive, got {per_region}")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for name in BASINS:
        candidates = np.flatnonzero(regions == name)
        if len(candidates) < per_region:
            raise ValueError(
                f"region {name} has {len(candidates)} frames, fewer than requested {per_region}"
            )
        selected.append(rng.choice(candidates, size=per_region, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def _bin_indices(phi: np.ndarray, psi: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    width = 360.0 / bins
    ix = np.floor((wrap_degrees(phi) + 180.0) / width).astype(np.int64)
    iy = np.floor((wrap_degrees(psi) + 180.0) / width).astype(np.int64)
    return np.clip(ix, 0, bins - 1), np.clip(iy, 0, bins - 1)


def histogram2d_periodic(
    phi: np.ndarray, psi: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.shape != psi.shape:
        raise ValueError("phi and psi must have the same shape")
    edges = np.linspace(-180.0, 180.0, bins + 1)
    counts, _, _ = np.histogram2d(wrap_degrees(phi), wrap_degrees(psi), bins=(edges, edges))
    return counts.astype(np.float64), edges


def reference_zero_cell(phi: np.ndarray, psi: np.ndarray, bins: int) -> tuple[int, int]:
    regions = assign_basins(phi, psi)
    beta = regions == "beta"
    if not beta.any():
        raise ValueError("reference contains no beta frames")
    ix, iy = _bin_indices(np.asarray(phi)[beta], np.asarray(psi)[beta], bins)
    counts = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(counts, (ix, iy), 1)
    return tuple(int(v) for v in np.unravel_index(np.argmax(counts), counts.shape))


def free_energy(counts: np.ndarray, kT: float, zero_cell: tuple[int, int]) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or kT <= 0:
        raise ValueError("counts must be 2D and kT must be positive")
    zero_count = float(counts[zero_cell])
    if zero_count <= 0:
        raise ValueError(f"shared zero cell {zero_cell} has zero probability")
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    supported = counts > 0
    out[supported] = -kT * np.log(counts[supported] / zero_count)
    return out


def aggregate_static(
    phi: np.ndarray,
    psi: np.ndarray,
    energies: np.ndarray,
    bins: int,
    min_count: int,
    zero_cell: tuple[int, int],
) -> dict[str, np.ndarray]:
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    if phi.shape != psi.shape or phi.shape != energies.shape:
        raise ValueError("phi, psi, and energies must have the same shape")
    if min_count <= 0:
        raise ValueError(f"min_count must be positive, got {min_count}")
    ix, iy = _bin_indices(phi, psi, bins)
    count = np.zeros((bins, bins), dtype=np.int64)
    mean = np.full((bins, bins), np.nan)
    median = np.full((bins, bins), np.nan)
    std = np.full((bins, bins), np.nan)
    sem = np.full((bins, bins), np.nan)
    for i in range(bins):
        for j in range(bins):
            values = energies[(ix == i) & (iy == j)]
            count[i, j] = len(values)
            if len(values) < min_count:
                continue
            mean[i, j] = float(values.mean())
            median[i, j] = float(np.median(values))
            if len(values) > 1:
                std[i, j] = float(values.std(ddof=1))
                sem[i, j] = std[i, j] / np.sqrt(len(values))
    if not np.isfinite(mean[zero_cell]):
        raise ValueError(
            f"shared zero cell {zero_cell} has {count[zero_cell]} frames, below min_count {min_count}"
        )
    mean -= mean[zero_cell]
    if np.isfinite(median[zero_cell]):
        median -= median[zero_cell]
    return {"count": count, "mean": mean, "median": median, "std": std, "sem": sem}


def basin_thermodynamics(
    reference_regions: np.ndarray,
    energies: np.ndarray,
    md_regions: np.ndarray,
    kT: float,
) -> list[dict[str, float | int | str]]:
    reference_regions = np.asarray(reference_regions).astype(str)
    energies = np.asarray(energies, dtype=np.float64)
    md_regions = np.asarray(md_regions).astype(str)
    if reference_regions.shape != energies.shape:
        raise ValueError("reference_regions and energies must have the same shape")
    for name in ("beta", "alphaR", "alphaL"):
        if not np.any(reference_regions == name):
            raise ValueError(f"reference contains no {name} frames")
        if not np.any(md_regions == name):
            raise ValueError(f"MD ensemble contains no {name} frames")
    beta_u = float(energies[reference_regions == "beta"].mean())
    beta_md = int(np.sum(md_regions == "beta"))
    rows: list[dict[str, float | int | str]] = []
    for name in ("beta", "alphaR", "alphaL"):
        ref_sel = reference_regions == name
        md_count = int(np.sum(md_regions == name))
        rows.append(
            {
                "basin": name,
                "reference_count": int(ref_sel.sum()),
                "md_count": md_count,
                "mean_U": float(energies[ref_sel].mean()),
                "dU_vs_beta": float(energies[ref_sel].mean() - beta_u),
                "dF_md_vs_beta": float(-kT * np.log(md_count / beta_md)),
            }
        )
    return rows


def fit_landscape(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("x, y, and mask must have the same shape")
    use = mask & np.isfinite(x) & np.isfinite(y)
    if int(use.sum()) < 10:
        raise ValueError(f"fit requires at least 10 supported cells, got {int(use.sum())}")
    xv, yv = x[use], y[use]
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        raise ValueError("fit requires variation in both supported landscapes")
    design = np.column_stack([xv, np.ones_like(xv)])
    slope, intercept = np.linalg.lstsq(design, yv, rcond=None)[0]
    predicted = slope * xv + intercept
    residual_values = yv - predicted
    total = float(np.sum((yv - yv.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual_values**2) / total) if total > 0 else float("nan")
    residual = np.full(x.shape, np.nan, dtype=np.float64)
    residual[use] = residual_values
    return {
        "n_cells": int(use.sum()),
        "pearson_r": float(stats.pearsonr(xv, yv).statistic),
        "spearman_r": float(stats.spearmanr(xv, yv).statistic),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "mask": use,
        "residual": residual,
    }
