"""Project stored Cartesian forces and trajectory drift into two-dimensional CVs.

The module is intentionally model-agnostic: it consumes the ``R`` and ``F``
arrays already written by CAMEO CG datasets or MD trajectories.  Two CV
families are supported:

* two periodic dihedrals (the Ala2 Ramachandran convention by default), and
* a reference-fitted TICA model on pair-distance features.

For a CV vector ``q(R)`` with Jacobian ``J = dq/dR``, the plotted force
response is ``J M^-1 F``.  This is the force-induced part of the CV
acceleration, not a thermodynamic mean force.  Empirical ``dq/dt`` from
consecutive stored frames is computed and plotted separately.

Run with::

    python -m md.projected_force_analysis path/to/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import gaussian_filter
import yaml

from md.analyze_traj import build_features, choose_pairs, fit_tica


KB_KCAL_MOL_K = 1.98720425864083e-3


@dataclass(frozen=True)
class DihedralCVSpec:
    """Indices and display shift for one periodic dihedral CV."""

    indices: tuple[int, int, int, int]
    shift_deg: float = 180.0


@dataclass
class SourceData:
    """Coordinates, forces, and trajectory segmentation for one ensemble."""

    name: str
    label: str
    coordinates: np.ndarray
    forces: np.ndarray
    segment_lengths: list[int]
    frame_dt_ps: float | None
    paths: list[str]
    discard_fraction: float


@dataclass
class VectorFieldGrid:
    """Conditional vector statistics on a common rectangular grid."""

    xedges: np.ndarray
    yedges: np.ndarray
    count: np.ndarray
    mean: np.ndarray
    standard_error: np.ndarray
    coherence: np.ndarray
    mean_sample_magnitude: np.ndarray
    excluded_samples: int

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.mean, axis=-1)


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-180, 180)``."""

    values = np.asarray(values, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def periodic_difference(values: np.ndarray) -> np.ndarray:
    """Return the shortest signed difference between periodic degree values."""

    return wrap_degrees(np.diff(np.asarray(values, dtype=np.float64), axis=0))


def _wrap_radians(value: jax.Array) -> jax.Array:
    return jnp.mod(value + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def _dihedral_radians(position: jax.Array, indices: jax.Array) -> jax.Array:
    p0, p1, p2, p3 = (position[indices[i]] for i in range(4))
    # Keep the same signed convention as charron_fes_analysis.compute_dihedral.
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = jnp.linalg.norm(b1)
    b1_unit = b1 / jnp.maximum(b1_norm, 1.0e-12)
    v = b0 - jnp.sum(b0 * b1_unit) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit) * b1_unit
    angle = jnp.arctan2(jnp.sum(jnp.cross(b1_unit, v) * w), jnp.sum(v * w))
    return jnp.where(b1_norm > 1.0e-12, _wrap_radians(angle), 0.0)


def ramachandran_values_and_jacobians(
    coordinates: np.ndarray,
    specs: Sequence[DihedralCVSpec] = (
        DihedralCVSpec((0, 1, 2, 3)),
        DihedralCVSpec((1, 2, 3, 4)),
    ),
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shifted dihedral values and Cartesian Jacobians in degrees.

    Returns:
        values: ``(frames, 2)`` in wrapped degrees.
        jacobians: ``(frames, 2, atoms, 3)`` in degrees / Angstrom.
    """

    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"Expected coordinates shaped (frames, atoms, 3), got {coordinates.shape}.")
    if len(specs) != 2:
        raise ValueError("Exactly two dihedral CV specifications are required.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    max_index = max(max(spec.indices) for spec in specs)
    if max_index >= coordinates.shape[1]:
        raise ValueError(f"Dihedral index {max_index} is invalid for {coordinates.shape[1]} atoms.")

    indices = tuple(jnp.asarray(spec.indices, dtype=jnp.int32) for spec in specs)
    shifts = jnp.deg2rad(jnp.asarray([spec.shift_deg for spec in specs], dtype=jnp.float32))

    def cv_one(position: jax.Array) -> jax.Array:
        values = jnp.stack([_dihedral_radians(position, idx) for idx in indices])
        return _wrap_radians(values + shifts)

    jacobian_one = jax.jacrev(cv_one)
    evaluate = jax.jit(jax.vmap(lambda position: (cv_one(position), jacobian_one(position))))
    value_parts: list[np.ndarray] = []
    jacobian_parts: list[np.ndarray] = []
    conversion = 180.0 / np.pi
    for start in range(0, coordinates.shape[0], batch_size):
        stop = min(start + batch_size, coordinates.shape[0])
        values, jacobians = evaluate(jnp.asarray(coordinates[start:stop]))
        value_parts.append(np.asarray(values, dtype=np.float64) * conversion)
        jacobian_parts.append(np.asarray(jacobians, dtype=np.float64) * conversion)
    return wrap_degrees(np.concatenate(value_parts)), np.concatenate(jacobian_parts)


def pair_distance_jacobians(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Analytic Jacobian of linear pair-distance CVs.

    ``coefficients`` has shape ``(n_pairs, n_cvs)`` and maps pair distances
    to the CVs.  Centering constants do not affect the derivative.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (frames, atoms, 3).")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_pairs, 2).")
    if coefficients.shape[0] != pairs.shape[0]:
        raise ValueError("One coefficient row is required per pair.")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= coordinates.shape[1]):
        raise ValueError("Pair index is outside the coordinate array.")

    differences = coordinates[:, pairs[:, 0], :] - coordinates[:, pairs[:, 1], :]
    distances = np.linalg.norm(differences, axis=-1)
    units = differences / np.maximum(distances[..., None], 1.0e-12)
    jacobians = np.zeros(
        (coordinates.shape[0], coefficients.shape[1], coordinates.shape[1], 3),
        dtype=np.float64,
    )
    for pair_index, (atom_i, atom_j) in enumerate(pairs):
        contribution = coefficients[pair_index][None, :, None] * units[:, pair_index, None, :]
        jacobians[:, :, atom_i, :] += contribution
        jacobians[:, :, atom_j, :] -= contribution
    return jacobians


def remove_center_of_mass_force(forces: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Remove net translation while preserving all internal force components."""

    forces = np.asarray(forces, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise ValueError("forces must have shape (frames, atoms, 3).")
    if masses.shape != (forces.shape[1],) or np.any(masses <= 0.0):
        raise ValueError("masses must contain one positive value per atom.")
    mass_fraction = masses / masses.sum()
    return forces - mass_fraction[None, :, None] * forces.sum(axis=1, keepdims=True)


def project_force_response(
    jacobians: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    *,
    remove_net_force: bool = True,
) -> np.ndarray:
    """Compute ``J M^-1 F`` for every frame."""

    jacobians = np.asarray(jacobians, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if jacobians.ndim != 4 or jacobians.shape[0] != forces.shape[0]:
        raise ValueError("jacobians must have shape (frames, cvs, atoms, 3).")
    if jacobians.shape[2:] != forces.shape[1:]:
        raise ValueError("Jacobian atom dimensions do not match forces.")
    adjusted = remove_center_of_mass_force(forces, masses) if remove_net_force else forces
    acceleration = adjusted / masses[None, :, None]
    return np.einsum("fcan,fan->fc", jacobians, acceleration, optimize=True)


def aggregate_vector_field(
    cv_values: np.ndarray,
    vectors: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
) -> VectorFieldGrid:
    """Aggregate conditional means, errors, and directional coherence by bin."""

    cv_values = np.asarray(cv_values, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    xedges = np.asarray(xedges, dtype=np.float64)
    yedges = np.asarray(yedges, dtype=np.float64)
    if cv_values.shape != vectors.shape or cv_values.ndim != 2 or cv_values.shape[1] != 2:
        raise ValueError("cv_values and vectors must both have shape (samples, 2).")
    nx, ny = len(xedges) - 1, len(yedges) - 1
    ix = np.searchsorted(xedges, cv_values[:, 0], side="right") - 1
    iy = np.searchsorted(yedges, cv_values[:, 1], side="right") - 1
    valid = (
        np.isfinite(cv_values).all(axis=1)
        & np.isfinite(vectors).all(axis=1)
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )
    flat = ix[valid] * ny + iy[valid]
    count = np.bincount(flat, minlength=nx * ny).reshape(nx, ny).astype(np.int64)
    sums = np.zeros((nx * ny, 2), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    sum_magnitude = np.zeros((nx * ny,), dtype=np.float64)
    np.add.at(sums, flat, vectors[valid])
    np.add.at(sums_sq, flat, vectors[valid] ** 2)
    np.add.at(sum_magnitude, flat, np.linalg.norm(vectors[valid], axis=1))
    sums = sums.reshape(nx, ny, 2)
    sums_sq = sums_sq.reshape(nx, ny, 2)
    sum_magnitude = sum_magnitude.reshape(nx, ny)
    denominator = np.maximum(count[..., None], 1)
    mean = sums / denominator
    variance = np.maximum(sums_sq / denominator - mean**2, 0.0)
    standard_error = np.sqrt(variance / denominator)
    mean_sample_magnitude = sum_magnitude / np.maximum(count, 1)
    coherence = np.linalg.norm(mean, axis=-1) / np.maximum(mean_sample_magnitude, 1.0e-30)
    coherence[count == 0] = np.nan
    mean[count == 0] = np.nan
    standard_error[count == 0] = np.nan
    mean_sample_magnitude[count == 0] = np.nan
    return VectorFieldGrid(
        xedges=xedges,
        yedges=yedges,
        count=count,
        mean=mean,
        standard_error=standard_error,
        coherence=coherence,
        mean_sample_magnitude=mean_sample_magnitude,
        excluded_samples=int(cv_values.shape[0] - np.count_nonzero(valid)),
    )


def empirical_drift_samples(
    cv_values: np.ndarray,
    segment_lengths: Sequence[int],
    frame_dt_ps: float,
    *,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return starting CVs and consecutive-frame drift without crossing files."""

    if frame_dt_ps <= 0.0:
        raise ValueError("frame_dt_ps must be positive.")
    cv_values = np.asarray(cv_values, dtype=np.float64)
    if sum(segment_lengths) != cv_values.shape[0]:
        raise ValueError("segment_lengths do not sum to the number of CV frames.")
    starts: list[np.ndarray] = []
    drifts: list[np.ndarray] = []
    offset = 0
    for length in segment_lengths:
        segment = cv_values[offset : offset + length]
        offset += length
        if length < 2:
            continue
        delta = periodic_difference(segment) if periodic else np.diff(segment, axis=0)
        starts.append(segment[:-1])
        drifts.append(delta / frame_dt_ps)
    if not starts:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.concatenate(starts), np.concatenate(drifts)


def _resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (root / value).resolve()


def _expand_source_paths(value: str | Sequence[str], root: Path) -> list[Path]:
    raw_values = [value] if isinstance(value, str) else list(value)
    paths: list[Path] = []
    for raw in raw_values:
        path = _resolve_path(raw, root)
        if path.is_dir():
            found = sorted(path.glob("trajectory_rep[0-9][0-9].npz"))
            if not found:
                found = sorted(path.glob("*.npz"))
            paths.extend(found)
        else:
            paths.append(path)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Could not resolve source NPZ files from {value!r}.")
    return paths


def load_source(source_cfg: Mapping[str, Any], root: Path) -> SourceData:
    """Load and concatenate one reference or model ensemble."""

    name = str(source_cfg["name"])
    label = str(source_cfg.get("label", name))
    paths = _expand_source_paths(source_cfg["path"], root)
    discard_fraction = float(source_cfg.get("discard_fraction", 0.0))
    stride = int(source_cfg.get("stride", 1))
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError(f"discard_fraction for {name} must be in [0, 1).")
    if stride <= 0:
        raise ValueError(f"stride for {name} must be positive.")
    coordinate_parts: list[np.ndarray] = []
    force_parts: list[np.ndarray] = []
    segment_lengths: list[int] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "R" not in data.files or "F" not in data.files:
                raise KeyError(f"{path} must contain R and F arrays.")
            coordinates = np.asarray(data["R"], dtype=np.float64)
            forces = np.asarray(data["F"], dtype=np.float64)
            if coordinates.shape != forces.shape or coordinates.ndim != 3:
                raise ValueError(f"R/F shape mismatch in {path}: {coordinates.shape} vs {forces.shape}.")
            if "mask" in data.files:
                mask = np.asarray(data["mask"])
                valid = (mask[0] if mask.ndim == 2 else mask) > 0
                coordinates = coordinates[:, valid]
                forces = forces[:, valid]
        first = int(np.floor(discard_fraction * coordinates.shape[0]))
        coordinates = coordinates[first::stride]
        forces = forces[first::stride]
        if not coordinates.shape[0]:
            raise ValueError(f"Discard/stride removed all frames from {path}.")
        if not np.isfinite(coordinates).all() or not np.isfinite(forces).all():
            raise ValueError(f"Non-finite coordinates or forces in {path}.")
        coordinate_parts.append(coordinates)
        force_parts.append(forces)
        segment_lengths.append(int(coordinates.shape[0]))
    atom_counts = {part.shape[1] for part in coordinate_parts}
    if len(atom_counts) != 1:
        raise ValueError(f"Atom counts differ within source {name}: {sorted(atom_counts)}.")
    frame_dt = source_cfg.get("frame_dt_ps")
    return SourceData(
        name=name,
        label=label,
        coordinates=np.concatenate(coordinate_parts),
        forces=np.concatenate(force_parts),
        segment_lengths=segment_lengths,
        frame_dt_ps=None if frame_dt is None else float(frame_dt) * stride,
        paths=[str(path.resolve()) for path in paths],
        discard_fraction=discard_fraction,
    )


def _fes_from_values(
    values: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    *,
    temperature: float,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.histogram2d(values[:, 0], values[:, 1], bins=(xedges, yedges))[0]
    smooth = gaussian_filter(counts, sigma=0.75, mode="wrap" if periodic else "nearest")
    probability = smooth / max(float(smooth.sum()), 1.0)
    fes = -KB_KCAL_MOL_K * temperature * np.log(np.maximum(probability, 1.0e-300))
    support = counts > 0
    if np.any(support):
        fes -= np.min(fes[support])
    fes[~support] = np.nan
    return fes, counts


def _common_magnitude_limits(fields: Iterable[VectorFieldGrid], min_count: int) -> tuple[float, float]:
    values = []
    for field in fields:
        mask = (field.count >= min_count) & np.isfinite(field.magnitude) & (field.magnitude > 0.0)
        values.extend(field.magnitude[mask].tolist())
    if not values:
        return 1.0, 1.0
    array = np.asarray(values)
    return float(np.percentile(array, 5.0)), float(np.percentile(array, 95.0))


def plot_vector_fields(
    sources: Sequence[SourceData],
    cv_values: Mapping[str, np.ndarray],
    fields: Mapping[str, VectorFieldGrid],
    *,
    xedges_fes: np.ndarray,
    yedges_fes: np.ndarray,
    periodic: bool,
    temperature: float,
    fes_cap: float,
    min_count: int,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    """Plot one common-scale vector-field panel per ensemble."""

    ncols = len(sources)
    fig, axes = plt.subplots(1, ncols, figsize=(4.35 * ncols, 4.25), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    low, high = _common_magnitude_limits(fields.values(), min_count)
    low_log = np.log10(max(low, 1.0e-30))
    high_log = np.log10(max(high, low * 1.001, 1.0e-29))
    color_norm = Normalize(low_log, high_log)
    # Keep the FES subdued so vector direction remains legible.  A shared
    # magnitude normalization and shared display-length mapping still make
    # relative differences between panels directly visible.
    arrow_cmap = plt.colormaps["turbo"]
    background_cmap = plt.colormaps["Greys_r"].copy()
    background_cmap.set_bad("#eeeeee")
    background_mesh = None
    for ax, source in zip(axes, sources):
        fes, _ = _fes_from_values(
            cv_values[source.name],
            xedges_fes,
            yedges_fes,
            temperature=temperature,
            periodic=periodic,
        )
        background_mesh = ax.pcolormesh(
            xedges_fes,
            yedges_fes,
            np.ma.masked_invalid(fes).T,
            shading="flat",
            cmap=background_cmap,
            vmin=0.0,
            vmax=fes_cap,
        )
        field = fields[source.name]
        xc = 0.5 * (field.xedges[:-1] + field.xedges[1:])
        yc = 0.5 * (field.yedges[:-1] + field.yedges[1:])
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        magnitude = field.magnitude
        mask = (field.count >= min_count) & np.isfinite(magnitude) & (magnitude > 0.0)
        if np.any(mask):
            direction = field.mean[mask] / magnitude[mask, None]
            robust = max(high, 1.0e-30)
            relative = np.clip(magnitude[mask] / robust, 0.0, 1.5)
            spacing = min(np.median(np.diff(field.xedges)), np.median(np.diff(field.yedges)))
            display_length = spacing * (0.52 + 0.82 * relative)
            colors = arrow_cmap(color_norm(np.log10(np.maximum(magnitude[mask], 1.0e-30))))
            colors[:, 3] = 0.68 + 0.32 * np.clip(field.coherence[mask], 0.0, 1.0)
            quiver_kwargs = {
                "angles": "xy",
                "scale_units": "xy",
                "scale": 1.0,
                "headwidth": 4.2,
                "headlength": 5.2,
                "headaxislength": 4.6,
                "pivot": "middle",
            }
            # A thin dark underlay prevents arrows from disappearing against
            # either low- or high-free-energy background regions.
            ax.quiver(
                xx[mask],
                yy[mask],
                direction[:, 0] * display_length,
                direction[:, 1] * display_length,
                color=(0.04, 0.04, 0.04, 0.82),
                width=0.0105,
                **quiver_kwargs,
            )
            ax.quiver(
                xx[mask],
                yy[mask],
                direction[:, 0] * display_length,
                direction[:, 1] * display_length,
                color=colors,
                width=0.0075,
                **quiver_kwargs,
            )
        ax.set_title(source.label)
        ax.set_xlabel(xlabel)
        ax.set_xlim(xedges_fes[0], xedges_fes[-1])
        ax.set_ylim(yedges_fes[0], yedges_fes[-1])
        if periodic:
            ax.set_aspect("equal")
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel(ylabel)
    if background_mesh is not None:
        fig.colorbar(
            background_mesh,
            ax=axes.tolist(),
            pad=0.015,
            fraction=0.025,
            label="Delta F [kcal/mol]",
        )
    scalar = ScalarMappable(norm=color_norm, cmap=arrow_cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes.tolist(), pad=0.055, fraction=0.025)
    colorbar.set_label("log10 projected-vector magnitude")
    fig.suptitle(title, y=1.02)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_vector_field_csv(path: Path, field: VectorFieldGrid) -> None:
    xc = 0.5 * (field.xedges[:-1] + field.xedges[1:])
    yc = 0.5 * (field.yedges[:-1] + field.yedges[1:])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bin_x",
                "bin_y",
                "count",
                "mean_x",
                "mean_y",
                "magnitude",
                "standard_error_x",
                "standard_error_y",
                "coherence",
                "mean_sample_magnitude",
            ]
        )
        for i, x in enumerate(xc):
            for j, y in enumerate(yc):
                writer.writerow(
                    [
                        float(x),
                        float(y),
                        int(field.count[i, j]),
                        float(field.mean[i, j, 0]),
                        float(field.mean[i, j, 1]),
                        float(field.magnitude[i, j]),
                        float(field.standard_error[i, j, 0]),
                        float(field.standard_error[i, j, 1]),
                        float(field.coherence[i, j]),
                        float(field.mean_sample_magnitude[i, j]),
                    ]
                )


def _field_region_metrics(
    field: VectorFieldGrid,
    reference_counts: np.ndarray,
    *,
    min_count: int,
) -> dict[str, Any]:
    occupied_reference = reference_counts[reference_counts > 0]
    low = float(np.percentile(occupied_reference, 25.0)) if occupied_reference.size else 0.0
    high = float(np.percentile(occupied_reference, 75.0)) if occupied_reference.size else 0.0
    regions = {
        "reference_core": reference_counts >= high,
        "reference_outskirts": (reference_counts > 0) & (reference_counts <= low),
        "outside_reference_support": reference_counts == 0,
    }
    valid = (field.count >= min_count) & np.isfinite(field.magnitude)
    result: dict[str, Any] = {}
    for name, region in regions.items():
        selected = valid & region
        result[name] = {
            "bins": int(np.count_nonzero(selected)),
            "median_magnitude": (
                float(np.median(field.magnitude[selected])) if np.any(selected) else None
            ),
            "mean_coherence": (
                float(np.mean(field.coherence[selected])) if np.any(selected) else None
            ),
        }
    return result


def _alignment_with_reference_downhill(
    field: VectorFieldGrid,
    reference_counts: np.ndarray,
    *,
    min_count: int,
    periodic: bool,
) -> float | None:
    smooth = gaussian_filter(reference_counts.astype(float), sigma=1.0, mode="wrap" if periodic else "nearest")
    free_energy = -np.log(np.maximum(smooth / max(float(smooth.sum()), 1.0), 1.0e-12))
    dx = float(np.median(np.diff(field.xedges)))
    dy = float(np.median(np.diff(field.yedges)))
    if periodic:
        gradient_x = (
            np.roll(free_energy, -1, axis=0) - np.roll(free_energy, 1, axis=0)
        ) / (2.0 * dx)
        gradient_y = (
            np.roll(free_energy, -1, axis=1) - np.roll(free_energy, 1, axis=1)
        ) / (2.0 * dy)
    else:
        gradient_x, gradient_y = np.gradient(free_energy, dx, dy)
    downhill = -np.stack([gradient_x, gradient_y], axis=-1)
    downhill_norm = np.linalg.norm(downhill, axis=-1)
    magnitude = field.magnitude
    valid = (
        (field.count >= min_count)
        & (reference_counts > 0)
        & np.isfinite(magnitude)
        & (magnitude > 0.0)
        & (downhill_norm > 1.0e-12)
    )
    if not np.any(valid):
        return None
    cosine = np.sum(field.mean[valid] * downhill[valid], axis=-1) / (
        magnitude[valid] * downhill_norm[valid]
    )
    weights = field.count[valid].astype(float)
    return float(np.average(cosine, weights=weights))


def _projection_edges(
    values: Mapping[str, np.ndarray],
    *,
    periodic: bool,
    bins: int,
    quantiles: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if periodic:
        return np.linspace(-180.0, 180.0, bins + 1), np.linspace(-180.0, 180.0, bins + 1)
    combined = np.concatenate(list(values.values()))
    low = np.quantile(combined, quantiles[0], axis=0)
    high = np.quantile(combined, quantiles[1], axis=0)
    margin = 0.04 * np.maximum(high - low, 1.0e-12)
    return (
        np.linspace(low[0] - margin[0], high[0] + margin[0], bins + 1),
        np.linspace(low[1] - margin[1], high[1] + margin[1], bins + 1),
    )


def run_analysis(config_path: str | Path) -> dict[str, Any]:
    """Run the configured projected-force and empirical-drift analysis."""

    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    analysis = config.get("analysis", config)
    root = _resolve_path(analysis.get("project_root", config_path.parent), config_path.parent)
    output_dir = _resolve_path(analysis["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [load_source(entry, root) for entry in analysis["sources"]]
    if len(sources) < 2:
        raise ValueError("At least a reference and one model source are required.")
    reference_name = str(analysis.get("reference_name", sources[0].name))
    source_by_name = {source.name: source for source in sources}
    if len(source_by_name) != len(sources) or reference_name not in source_by_name:
        raise ValueError("Source names must be unique and reference_name must match one source.")
    atom_counts = {source.coordinates.shape[1] for source in sources}
    if len(atom_counts) != 1:
        raise ValueError(f"Atom counts differ across sources: {sorted(atom_counts)}.")
    masses = np.asarray(analysis["masses"], dtype=np.float64)
    if masses.shape != (next(iter(atom_counts)),):
        raise ValueError("analysis.masses must provide one mass per retained atom.")

    temperature = float(analysis.get("temperature_K", 300.0))
    min_count = int(analysis.get("min_bin_count", 10))
    arrow_bins = int(analysis.get("arrow_bins", 18))
    fes_bins = int(analysis.get("fes_bins", 72))
    fes_cap = float(analysis.get("fes_cap_kcal_mol", 8.0))
    remove_net = bool(analysis.get("remove_net_force", True))
    batch_size = int(analysis.get("jacobian_batch_size", 4096))
    tica_quantiles_raw = analysis.get("tica_range_quantiles", [0.001, 0.999])
    tica_quantiles = (float(tica_quantiles_raw[0]), float(tica_quantiles_raw[1]))

    rama_cfg = analysis.get("ramachandran", {}) or {}
    specs = tuple(
        DihedralCVSpec(tuple(int(i) for i in item["indices"]), float(item.get("shift_deg", 180.0)))
        for item in rama_cfg.get(
            "dihedrals",
            [
                {"indices": [0, 1, 2, 3], "shift_deg": 180.0},
                {"indices": [1, 2, 3, 4], "shift_deg": 180.0},
            ],
        )
    )

    rama_values: dict[str, np.ndarray] = {}
    rama_force: dict[str, np.ndarray] = {}
    for source in sources:
        values, jacobians = ramachandran_values_and_jacobians(
            source.coordinates, specs, batch_size=batch_size
        )
        rama_values[source.name] = values
        rama_force[source.name] = project_force_response(
            jacobians, source.forces, masses, remove_net_force=remove_net
        )
    rama_arrow_edges = _projection_edges(
        rama_values, periodic=True, bins=arrow_bins, quantiles=tica_quantiles
    )
    rama_fes_edges = _projection_edges(
        rama_values, periodic=True, bins=fes_bins, quantiles=tica_quantiles
    )
    rama_force_fields = {
        source.name: aggregate_vector_field(
            rama_values[source.name], rama_force[source.name], *rama_arrow_edges
        )
        for source in sources
    }

    reference = source_by_name[reference_name]
    tica_cfg = analysis.get("tica", {}) or {}
    pairs = choose_pairs(
        n_atoms=reference.coordinates.shape[1],
        n_pairs=int(tica_cfg.get("n_pairs", 200)),
        mode=str(tica_cfg.get("pair_mode", "random")),
        seed=int(tica_cfg.get("pair_seed", 42)),
    )
    reference_features = build_features(reference.coordinates, pairs)
    tica_model, reference_projection = fit_tica(
        reference_features, lagtime=int(tica_cfg.get("lag_frames", 10))
    )
    coefficients = np.asarray(tica_model.instantaneous_coefficients, dtype=np.float64)[:, :2]
    tica_values: dict[str, np.ndarray] = {}
    tica_force: dict[str, np.ndarray] = {}
    for source in sources:
        features = build_features(source.coordinates, pairs)
        values = (
            np.asarray(reference_projection, dtype=np.float64)
            if source.name == reference_name
            else np.asarray(tica_model.transform(features), dtype=np.float64)
        )
        jacobians = pair_distance_jacobians(source.coordinates, pairs, coefficients)
        tica_values[source.name] = values[:, :2]
        tica_force[source.name] = project_force_response(
            jacobians, source.forces, masses, remove_net_force=remove_net
        )
    tica_arrow_edges = _projection_edges(
        tica_values, periodic=False, bins=arrow_bins, quantiles=tica_quantiles
    )
    tica_fes_edges = _projection_edges(
        tica_values, periodic=False, bins=fes_bins, quantiles=tica_quantiles
    )
    tica_force_fields = {
        source.name: aggregate_vector_field(
            tica_values[source.name], tica_force[source.name], *tica_arrow_edges
        )
        for source in sources
    }

    drift_fields: dict[str, dict[str, VectorFieldGrid]] = {"ramachandran": {}, "tica": {}}
    for source in sources:
        if source.frame_dt_ps is None:
            continue
        starts, drift = empirical_drift_samples(
            rama_values[source.name],
            source.segment_lengths,
            source.frame_dt_ps,
            periodic=True,
        )
        drift_fields["ramachandran"][source.name] = aggregate_vector_field(
            starts, drift, *rama_arrow_edges
        )
        starts, drift = empirical_drift_samples(
            tica_values[source.name],
            source.segment_lengths,
            source.frame_dt_ps,
            periodic=False,
        )
        drift_fields["tica"][source.name] = aggregate_vector_field(
            starts, drift, *tica_arrow_edges
        )

    plots = {
        "ramachandran_force": output_dir / "ramachandran_projected_force_vectors.png",
        "tica_force": output_dir / "tica_projected_force_vectors.png",
        "ramachandran_drift": output_dir / "ramachandran_empirical_drift_vectors.png",
        "tica_drift": output_dir / "tica_empirical_drift_vectors.png",
    }
    plot_vector_fields(
        sources,
        rama_values,
        rama_force_fields,
        xedges_fes=rama_fes_edges[0],
        yedges_fes=rama_fes_edges[1],
        periodic=True,
        temperature=temperature,
        fes_cap=fes_cap,
        min_count=min_count,
        xlabel="phi [deg], Charron shifted",
        ylabel="psi [deg], Charron shifted",
        title="Ala2 conditional mass-weighted force response in Ramachandran space",
        output=plots["ramachandran_force"],
    )
    plot_vector_fields(
        sources,
        tica_values,
        tica_force_fields,
        xedges_fes=tica_fes_edges[0],
        yedges_fes=tica_fes_edges[1],
        periodic=False,
        temperature=temperature,
        fes_cap=fes_cap,
        min_count=min_count,
        xlabel="TIC 1",
        ylabel="TIC 2",
        title="Ala2 conditional mass-weighted force response in reference-fitted TICA space",
        output=plots["tica_force"],
    )
    drift_sources = [source for source in sources if source.name in drift_fields["ramachandran"]]
    if drift_sources:
        plot_vector_fields(
            drift_sources,
            rama_values,
            drift_fields["ramachandran"],
            xedges_fes=rama_fes_edges[0],
            yedges_fes=rama_fes_edges[1],
            periodic=True,
            temperature=temperature,
            fes_cap=fes_cap,
            min_count=min_count,
            xlabel="phi [deg], Charron shifted",
            ylabel="psi [deg], Charron shifted",
            title="Ala2 empirical consecutive-frame drift in Ramachandran space",
            output=plots["ramachandran_drift"],
        )
        plot_vector_fields(
            drift_sources,
            tica_values,
            drift_fields["tica"],
            xedges_fes=tica_fes_edges[0],
            yedges_fes=tica_fes_edges[1],
            periodic=False,
            temperature=temperature,
            fes_cap=fes_cap,
            min_count=min_count,
            xlabel="TIC 1",
            ylabel="TIC 2",
            title="Ala2 empirical consecutive-frame drift in reference-fitted TICA space",
            output=plots["tica_drift"],
        )

    field_groups = {
        "ramachandran_force": rama_force_fields,
        "tica_force": tica_force_fields,
        "ramachandran_drift": drift_fields["ramachandran"],
        "tica_drift": drift_fields["tica"],
    }
    for group_name, group in field_groups.items():
        for source_name, field in group.items():
            write_vector_field_csv(output_dir / f"{group_name}_{source_name}.csv", field)

    reference_rama_counts = rama_force_fields[reference_name].count
    reference_tica_counts = tica_force_fields[reference_name].count
    metrics: dict[str, Any] = {}
    for source in sources:
        metrics[source.name] = {
            "frames": int(source.coordinates.shape[0]),
            "segments": source.segment_lengths,
            "frame_dt_ps": source.frame_dt_ps,
            "ramachandran": {
                "regions": _field_region_metrics(
                    rama_force_fields[source.name], reference_rama_counts, min_count=min_count
                ),
                "alignment_with_reference_downhill": _alignment_with_reference_downhill(
                    rama_force_fields[source.name],
                    reference_rama_counts,
                    min_count=min_count,
                    periodic=True,
                ),
                "excluded_force_samples": rama_force_fields[source.name].excluded_samples,
            },
            "tica": {
                "regions": _field_region_metrics(
                    tica_force_fields[source.name], reference_tica_counts, min_count=min_count
                ),
                "alignment_with_reference_downhill": _alignment_with_reference_downhill(
                    tica_force_fields[source.name],
                    reference_tica_counts,
                    min_count=min_count,
                    periodic=False,
                ),
                "excluded_force_samples": tica_force_fields[source.name].excluded_samples,
            },
        }

    with (output_dir / "tica_reference_model.pkl").open("wb") as handle:
        pickle.dump(tica_model, handle)
    np.savez_compressed(
        output_dir / "cv_values_and_force_response.npz",
        pairs=pairs,
        tica_coefficients=coefficients,
        **{f"ramachandran_{name}": value for name, value in rama_values.items()},
        **{f"ramachandran_force_{name}": value for name, value in rama_force.items()},
        **{f"tica_{name}": value for name, value in tica_values.items()},
        **{f"tica_force_{name}": value for name, value in tica_force.items()},
    )
    summary = {
        "method": {
            "force_projection": "J(q) M^-1 F after mass-weighted net-force removal",
            "empirical_drift": "shortest periodic delta(q) / saved-frame interval; file boundaries excluded",
            "tica_fit": "mapped-AA reference only, pair-distance features",
            "tica_lag_frames": int(tica_cfg.get("lag_frames", 10)),
            "remove_net_force": remove_net,
            "arrow_bins": arrow_bins,
            "minimum_samples_per_arrow": min_count,
        },
        "reference_name": reference_name,
        "sources": {
            source.name: {
                "label": source.label,
                "paths": source.paths,
                "discard_fraction": source.discard_fraction,
                "frames": int(source.coordinates.shape[0]),
                "frame_dt_ps": source.frame_dt_ps,
            }
            for source in sources
        },
        "metrics": metrics,
        "plots": {name: str(path) for name, path in plots.items() if path.exists()},
        "limitations": [
            "Projected J M^-1 F is a force-induced CV response, not a thermodynamic mean force.",
            "Conditional averages combine different hidden coordinates that share the same two CV values.",
            "Empirical drift includes inertia, thermostat noise, and finite output-stride effects.",
            "The mapped-AA reference uses 5 ps frame spacing, much coarser than the CG output spacing.",
            "Forces are evaluated on each ensemble's own sampled structures; this is not a same-structure model comparison.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML analysis configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(args.config)
    print(json.dumps({"output": summary["plots"], "sources": summary["sources"]}, indent=2))


if __name__ == "__main__":
    main()
