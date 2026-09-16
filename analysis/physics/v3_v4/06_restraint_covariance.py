#!/usr/bin/env python3
"""Reference/v3 restraint-deconvolved covariance diagnostic.

This is a read-only proof-of-concept.  It estimates local internal fluctuation modes from
raw coordinate frames, then removes the known finite restraint through
``H ~= kT * pinv(C_K) - K``.  The script deliberately refuses to infer a covariance from
state means or collected mean-force labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_KT = 0.5921868690749673


def _flatten_displacements(displacements: np.ndarray) -> np.ndarray:
    values = np.asarray(displacements, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 3:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("displacements must have shape (n_frames, n_dof) with n_frames >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("displacements contain non-finite values")
    return values


def _as_coordinates(frames: np.ndarray, name: str = "frames") -> np.ndarray:
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[1] < 3:
        raise ValueError(f"{name} must have shape (n_frames, n_beads, 3), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contain non-finite values")
    return values


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T)


def _internal_basis(anchor: np.ndarray, tolerance: float = 1.0e-10) -> np.ndarray:
    """Return an orthonormal basis for the non-rigid Cartesian subspace."""
    projector = rigid_projector(anchor)
    values, vectors = np.linalg.eigh(projector)
    keep = values > tolerance
    if not np.any(keep):
        raise ValueError("anchor has no internal Cartesian degrees of freedom")
    return vectors[:, keep]


def rigid_projector(anchor: np.ndarray) -> np.ndarray:
    """Project Cartesian bead displacements away from translations and rotations."""
    reference = np.asarray(anchor, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[-1] != 3 or len(reference) < 3:
        raise ValueError("anchor must have shape (n_beads >= 3, 3)")
    if not np.all(np.isfinite(reference)):
        raise ValueError("anchor contains non-finite values")

    centered = reference - reference.mean(axis=0, keepdims=True)
    columns = []
    for axis in np.eye(3):
        columns.append(np.tile(axis, (len(reference), 1)).reshape(-1))
    for axis in np.eye(3):
        columns.append(np.cross(axis[None, :], centered).reshape(-1))
    tangent = np.column_stack(columns)
    u, singular_values, _ = np.linalg.svd(tangent, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] <= 0.0:
        raise ValueError("anchor does not define a rigid-body tangent space")
    keep = singular_values > singular_values[0] * 1.0e-12
    q = u[:, keep]
    identity = np.eye(reference.size)
    return _symmetrize(identity - q @ q.T)


def _kabsch_align(frame: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, float, float]:
    frame_center = frame.mean(axis=0)
    anchor_center = anchor.mean(axis=0)
    frame_centered = frame - frame_center
    anchor_centered = anchor - anchor_center
    covariance = frame_centered.T @ anchor_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = frame_centered @ rotation + anchor_center
    translation_rms = float(np.sqrt(np.mean((frame_center - anchor_center) ** 2)))
    rotation_rms = float(np.sqrt(np.mean((frame_centered - aligned + anchor_center) ** 2)))
    return aligned, translation_rms, rotation_rms


def align_internal_displacements(
    frames: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Rigidly align frames to an anchor and return flattened internal displacements."""
    values = _as_coordinates(frames)
    reference = np.asarray(anchor, dtype=np.float64)
    if reference.shape != values.shape[1:]:
        raise ValueError(f"anchor shape {reference.shape} does not match frames {values.shape[1:]}")

    displacements = np.empty((len(values), reference.size), dtype=np.float64)
    translations = []
    rotations = []
    for index, frame in enumerate(values):
        aligned, translation_rms, rotation_rms = _kabsch_align(frame, reference)
        displacements[index] = (aligned - reference).reshape(-1)
        translations.append(translation_rms)
        rotations.append(rotation_rms)
    projector = rigid_projector(reference)
    rigid_component = displacements - displacements @ projector.T
    return displacements, {
        "translation_rms_A": float(np.sqrt(np.mean(np.square(translations)))),
        "rotation_rms_A": float(np.sqrt(np.mean(np.square(rotations)))),
        "internal_rms_A": float(np.sqrt(np.mean(np.square(displacements)))),
        "rigid_leakage_rms_A": float(np.sqrt(np.mean(np.square(rigid_component)))),
    }


def estimate_covariance(
    displacements: np.ndarray, shrinkage: float = 0.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate a centered covariance and report its numerical support."""
    values = _flatten_displacements(displacements)
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must satisfy 0 <= shrinkage < 1")
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = _symmetrize(centered.T @ centered / (len(values) - 1))
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(eigenvalues)), 1.0)
    positive = eigenvalues > scale * 1.0e-12
    if shrinkage:
        target = float(np.trace(covariance) / covariance.shape[0])
        covariance = _symmetrize((1.0 - shrinkage) * covariance + shrinkage * target * np.eye(covariance.shape[0]))
        eigenvalues = np.linalg.eigvalsh(covariance)
        positive = eigenvalues > max(float(np.max(eigenvalues)), 1.0) * 1.0e-12
    positive_values = eigenvalues[positive]
    condition = float(np.max(positive_values) / np.min(positive_values)) if len(positive_values) else float("inf")
    variances = np.diag(covariance)
    gaussian_ratio = np.divide(
        np.mean(centered**4, axis=0),
        3.0 * np.maximum(variances, 1.0e-30) ** 2,
    )
    return covariance, {
        "n_frames": int(len(values)),
        "n_dof": int(values.shape[1]),
        "effective_rank": int(np.count_nonzero(positive)),
        "condition_number": condition,
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "shrinkage": float(shrinkage),
        "maximum_gaussian_ratio": float(np.max(gaussian_ratio)),
        "mean_gaussian_ratio": float(np.mean(gaussian_ratio)),
    }


def deconvolve_covariance(
    covariance: np.ndarray,
    restraint_matrix: np.ndarray,
    kT: float,
    eigen_floor: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Remove a finite harmonic restraint from an internal covariance."""
    observed = _symmetrize(np.asarray(covariance, dtype=np.float64))
    restraint = _symmetrize(np.asarray(restraint_matrix, dtype=np.float64))
    if observed.ndim != 2 or observed.shape[0] != observed.shape[1]:
        raise ValueError("covariance must be square")
    if restraint.shape != observed.shape:
        raise ValueError("restraint_matrix must have the same shape as covariance")
    if not np.isfinite(kT) or kT <= 0.0:
        raise ValueError("kT must be positive and finite")
    if not np.isfinite(eigen_floor) or eigen_floor <= 0.0:
        raise ValueError("eigen_floor must be positive and finite")

    eigenvalues, vectors = np.linalg.eigh(observed)
    if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0.0:
        raise ValueError("covariance must be positive definite before deconvolution")
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    regularized = np.maximum(eigenvalues, eigen_floor)
    inverse = vectors @ np.diag(1.0 / regularized) @ vectors.T
    hessian = _symmetrize(float(kT) * inverse - restraint)
    curvature = np.linalg.eigvalsh(hessian)
    return hessian, {
        "condition_number": condition,
        "eigen_floor": float(eigen_floor),
        "n_regularized": int(np.count_nonzero(eigenvalues < eigen_floor)),
        "minimum_curvature": float(np.min(curvature)),
        "maximum_curvature": float(np.max(curvature)),
        "n_negative_curvatures": int(np.count_nonzero(curvature < 0.0)),
        "valid": bool(np.all(np.isfinite(hessian))),
    }


def _block_indices(n_frames: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length < 1 or block_length > n_frames:
        raise ValueError("block_length must be between 1 and n_frames")
    starts = rng.integers(0, n_frames, size=math.ceil(n_frames / block_length))
    indices = np.concatenate(
        [(start + np.arange(block_length, dtype=np.int64)) % n_frames for start in starts]
    )
    return indices[:n_frames]


def bootstrap_covariance(
    displacements: np.ndarray, block_length: int, n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """Block-bootstrap covariance eigenvalues and sampled frame indices."""
    values = _flatten_displacements(displacements)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, len(values)), dtype=np.int64)
    eigenvalues = np.empty((n_boot, values.shape[1]), dtype=np.float64)
    covariances = np.empty((n_boot, values.shape[1], values.shape[1]), dtype=np.float64)
    for replicate in range(n_boot):
        indices = _block_indices(len(values), block_length, rng)
        samples[replicate] = indices
        covariance, _ = estimate_covariance(values[indices])
        covariances[replicate] = covariance
        eigenvalues[replicate] = np.linalg.eigvalsh(covariance)
    return {"samples": samples, "eigenvalues": eigenvalues, "covariances": covariances}


def compare_mode_sets(
    reference: Mapping[str, np.ndarray], restrained: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Compare mode overlaps and pair each reference mode to its closest restrained mode."""
    ref_vectors = np.asarray(reference["eigenvectors"], dtype=np.float64)
    rest_vectors = np.asarray(restrained["eigenvectors"], dtype=np.float64)
    if ref_vectors.ndim != 2 or rest_vectors.ndim != 2 or ref_vectors.shape[0] != rest_vectors.shape[0]:
        raise ValueError("mode eigenvectors must be 2-D with the same Cartesian dimension")
    overlap = np.abs(ref_vectors.T @ rest_vectors)
    best = np.argmax(overlap, axis=1)
    principal = np.arccos(np.clip(np.max(overlap, axis=1), 0.0, 1.0))
    return {
        "overlap_matrix": overlap,
        "best_restrained_mode": best.astype(np.int64),
        "best_overlap": np.max(overlap, axis=1),
        "principal_angle_rad": principal,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def _mode_spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _npz_array(payload: Mapping[str, Any], key: str, default: np.ndarray | None = None) -> np.ndarray | None:
    if key not in payload:
        return default
    return np.asarray(payload[key])


def load_reference_frames(path: Path) -> dict[str, np.ndarray]:
    """Load mapped reference coordinates and optional metadata from an NPZ."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as raw:
        if "R" not in raw.files:
            raise KeyError(f"{path}: missing required key R; found {raw.files}")
        frames = _as_coordinates(raw["R"], "reference R")
        result: dict[str, np.ndarray] = {"R": frames}
        for key in ("time_ps", "anchor", "anchor_index", "mask", "state"):
            if key in raw.files:
                result[key] = np.asarray(raw[key])
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must contain a JSON object")
    return value


def _state_id(entry: Mapping[str, Any]) -> int:
    for key in ("state", "state_id", "id"):
        if key in entry:
            return int(entry[key])
    raise ValueError("manifest state entry is missing state/state_id/id")


def _state_dir(campaign: Path, state: int, entry: Mapping[str, Any]) -> Path:
    for key in ("directory", "state_dir", "path", "case_dir"):
        if key in entry:
            candidate = Path(str(entry[key]))
            return candidate if candidate.is_absolute() else campaign / candidate
    candidates = [campaign / f"state_{state:06d}", campaign / f"state_{state:05d}", campaign / f"state_{state}"]
    candidates.extend([campaign / f"case_{state:06d}", campaign / f"case_{state:05d}", campaign / f"case_{state}"])
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _coord_path(state_dir: Path, entry: Mapping[str, Any]) -> Path | None:
    for key in ("coordinates", "coords", "coordinate_file", "raw_coordinates"):
        if key in entry:
            candidate = Path(str(entry[key]))
            candidate = candidate if candidate.is_absolute() else state_dir / candidate
            return candidate
    for name in ("cg_coords.npz", "cg_frames.npz", "coordinates.npz", "raw_coords.npz"):
        candidate = state_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_coordinate_file(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    with np.load(path, allow_pickle=False) as raw:
        key = next((candidate for candidate in ("R", "coords", "coordinates") if candidate in raw.files), None)
        if key is None:
            raise KeyError(f"{path}: no coordinate key R/coords/coordinates; found {raw.files}")
        frames = _as_coordinates(raw[key], f"{path}:{key}")
        times = np.asarray(raw["time_ps"], dtype=np.float64) if "time_ps" in raw.files else None
    if times is not None and len(times) != len(frames):
        raise ValueError(f"{path}: time_ps length {len(times)} does not match {len(frames)} frames")
    return frames, times, key


def _kappa_from_entry(entry: Mapping[str, Any], manifest: Mapping[str, Any], kT: float) -> tuple[float | None, str | None]:
    for source in (entry, manifest):
        for key in ("kappa_kcal_mol_A2", "kappa", "restraint_kappa"):
            if key in source and source[key] is not None:
                kappa = float(source[key])
                if np.isfinite(kappa) and kappa > 0.0:
                    return kappa, key
                return None, f"invalid {key}={source[key]!r}"
        for key in ("restraint_width_A", "restraint_width", "width_A"):
            if key in source and source[key] is not None:
                width = float(source[key])
                if np.isfinite(width) and width > 0.0:
                    return float(kT / width**2), f"{key}->kappa"
                return None, f"invalid {key}={source[key]!r}"
    return None, "missing finite restraint stiffness"


def _is_frozen(campaign: Path, state_dir: Path, entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    if any(bool(entry.get(key, False)) for key in ("frozen", "freeze", "fixed")):
        return True
    if any(bool(manifest.get(key, False)) for key in ("frozen", "freeze", "fixed")):
        return True
    if (campaign / "beads.ndx").exists() and not any(
        key in entry for key in ("kappa_kcal_mol_A2", "kappa", "restraint_kappa", "restraint_width_A")
    ):
        return True
    mdp = state_dir / "production.mdp"
    if mdp.exists():
        text = mdp.read_text()
        if any(line.strip().startswith("freezegrps") for line in text.splitlines()):
            return True
    return False


def load_v3_windows(
    campaign: Path,
    manifest: Path | None = None,
    max_frames: int | None = None,
    discard_ps: float = 0.0,
    kT: float = DEFAULT_KT,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load raw v3 state frames and return usable windows plus an explicit audit."""
    campaign = Path(campaign)
    manifest_path = Path(manifest) if manifest is not None else campaign / "manifest.json"
    audit: dict[str, object] = {
        "campaign": str(campaign),
        "manifest": str(manifest_path),
        "usable_windows": 0,
        "unusable_windows": 0,
        "entries": [],
        "aggregated_only": False,
    }
    if not manifest_path.exists():
        aggregated = [
            campaign / name
            for name in ("meanforce_dataset.npz", "stencil_states.npz", "cg_coords_all.npz")
        ]
        present = [path for path in aggregated if path.exists()]
        if present:
            audit["aggregated_only"] = True
            _write_audit_entry(
                audit,
                {
                    "status": "audit_only",
                    "reason": "missing manifest; available aggregated files do not contain raw covariance frames",
                    "files": [str(path) for path in present],
                },
            )
        else:
            _write_audit_entry(audit, {"status": "error", "reason": "missing manifest.json"})
        return [], audit
    data = _read_json(manifest_path)
    entries = data.get("states", data.get("windows", []))
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_path}: states/windows must be a list")
    seen: set[int] = set()
    windows: list[dict[str, object]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{manifest_path}: state entry is not an object")
        state = _state_id(raw_entry)
        if state in seen:
            raise ValueError(f"{manifest_path}: duplicate state ID {state}")
        seen.add(state)
        state_dir = _state_dir(campaign, state, raw_entry)
        frozen = _is_frozen(campaign, state_dir, raw_entry, data)
        kappa, kappa_source = _kappa_from_entry(raw_entry, data, kT)
        coord_path = _coord_path(state_dir, raw_entry)
        record: dict[str, object] = {
            "state": state,
            "target": np.asarray(raw_entry.get("target", []), dtype=np.float64),
            "reference_index": raw_entry.get("reference_index", raw_entry.get("anchor_index")),
            "state_dir": str(state_dir),
            "source": str(coord_path) if coord_path else None,
            "kappa_kcal_mol_A2": kappa,
            "kappa_source": kappa_source,
            "frozen": frozen,
        }
        try:
            if frozen:
                raise RuntimeError("frozen window has no finite covariance restraint")
            if kappa is None:
                raise RuntimeError(str(kappa_source))
            if coord_path is None or not coord_path.exists():
                raise FileNotFoundError("raw coordinate file not found")
            frames, times, coord_key = _load_coordinate_file(coord_path)
            keep = np.ones(len(frames), dtype=bool)
            if times is not None and discard_ps > 0.0:
                keep = times >= float(times[0] + discard_ps) - 1.0e-12
            frames = frames[keep]
            kept_times = times[keep] if times is not None else None
            if max_frames is not None and max_frames > 0 and len(frames) > max_frames:
                select = np.linspace(0, len(frames) - 1, max_frames, dtype=np.int64)
                frames = frames[select]
                kept_times = kept_times[select] if kept_times is not None else None
            if len(frames) < 2:
                raise RuntimeError(f"only {len(frames)} usable frames after discard")
            record.update({"frames": frames, "time_ps": kept_times, "coordinate_key": coord_key})
            windows.append(record)
            record["status"] = "usable"
            audit["usable_windows"] = int(audit["usable_windows"]) + 1
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            record["status"] = "unusable"
            record["reason"] = str(exc)
            audit["unusable_windows"] = int(audit["unusable_windows"]) + 1
        _write_audit_entry(
            audit,
            {key: value for key, value in record.items() if key not in ("frames", "time_ps")},
        )

    if not windows:
        aggregated = [campaign / name for name in ("meanforce_dataset.npz", "stencil_states.npz", "cg_coords_all.npz")]
        if any(path.exists() for path in aggregated):
            audit["aggregated_only"] = True
            _write_audit_entry(audit, {
                "status": "audit_only",
                "reason": "aggregated state/mean-force files do not contain raw covariance frames",
                "files": [str(path) for path in aggregated if path.exists()],
            })
    return windows, audit


def _write_audit_entry(audit: dict[str, object], entry: Mapping[str, object]) -> None:
    entries = audit.setdefault("entries", [])
    if isinstance(entries, list):
        entries.append(_jsonable(dict(entry)))


def _nearest_reference_index(reference: np.ndarray, target: np.ndarray) -> int:
    if target.size == 0:
        raise ValueError("state has neither reference_index nor target coordinates")
    target = np.asarray(target, dtype=np.float64)
    if target.shape != reference.shape[1:]:
        raise ValueError(f"target shape {target.shape} does not match reference frame {reference.shape[1:]}")
    distances = []
    for frame in reference:
        aligned, _, _ = _kabsch_align(frame, target)
        distances.append(float(np.sqrt(np.mean((aligned - target) ** 2))))
    return int(np.argmin(distances))


def _local_reference_frames(
    reference: dict[str, np.ndarray], index: int, radius_A: float
) -> np.ndarray:
    """Select a local reference basin by aligned RMSD from the matched anchor."""
    frames = reference["R"]
    anchor = frames[index]
    distances = np.empty(len(frames), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        aligned, _, _ = _kabsch_align(frame, anchor)
        distances[frame_index] = np.sqrt(np.mean((aligned - anchor) ** 2))
    selected = frames[distances <= float(radius_A)]
    if len(selected) == 0:
        selected = frames[np.argsort(distances)[:1]]
    return selected


def _bootstrap_interval(values: np.ndarray, lower: float = 2.5, upper: float = 97.5) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    return [float(np.percentile(values, lower)), float(np.percentile(values, upper))]


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("\n")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _analyze_anchor(
    reference_frames: np.ndarray,
    v3_frames: np.ndarray,
    kappa: float,
    kT: float,
    shrinkage: float,
    eigen_floor: float,
    block_length: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    anchor = reference_frames[len(reference_frames) // 2]
    reference_displacements, ref_align = align_internal_displacements(reference_frames, anchor)
    v3_displacements, v3_align = align_internal_displacements(v3_frames, anchor)
    basis = _internal_basis(anchor)
    ref_internal = reference_displacements @ basis
    v3_internal = v3_displacements @ basis
    ref_covariance, ref_info = estimate_covariance(ref_internal, shrinkage=shrinkage)
    v3_covariance, v3_info = estimate_covariance(v3_internal, shrinkage=shrinkage)
    restraint = np.eye(basis.shape[1], dtype=np.float64) * float(kappa)
    v3_hessian, deconv_info = deconvolve_covariance(v3_covariance, restraint, kT, eigen_floor)
    ref_curvature, ref_deconv_info = deconvolve_covariance(
        ref_covariance, np.zeros_like(restraint), kT, eigen_floor
    )
    ref_eigenvalues, ref_vectors = _mode_spectrum(ref_curvature)
    v3_eigenvalues, v3_vectors = _mode_spectrum(v3_hessian)
    comparison = compare_mode_sets(
        {"eigenvalues": ref_eigenvalues, "eigenvectors": ref_vectors},
        {"eigenvalues": v3_eigenvalues, "eigenvectors": v3_vectors},
    )
    ref_boot = bootstrap_covariance(ref_internal, block_length, n_boot, seed)
    v3_boot = bootstrap_covariance(v3_internal, block_length, n_boot, seed + 1)
    ref_boot_curvature = np.asarray(
        [deconvolve_covariance(cov, np.zeros_like(restraint), kT, eigen_floor)[0] for cov in ref_boot["covariances"]]
    )
    v3_boot_curvature = np.asarray(
        [deconvolve_covariance(cov, restraint, kT, eigen_floor)[0] for cov in v3_boot["covariances"]]
    )
    ref_boot_eigenvalues = np.asarray([np.linalg.eigvalsh(matrix) for matrix in ref_boot_curvature])
    v3_boot_eigenvalues = np.asarray([np.linalg.eigvalsh(matrix) for matrix in v3_boot_curvature])
    v3_boot_interval = np.percentile(v3_boot_eigenvalues, [2.5, 97.5], axis=0)
    v3_boot_median = np.median(v3_boot_eigenvalues, axis=0)
    bootstrap_relative_width = float(
        np.max((v3_boot_interval[1] - v3_boot_interval[0]) / np.maximum(np.abs(v3_boot_median), eigen_floor))
    )
    flags: list[str] = []
    if ref_info["condition_number"] > 1.0e8 or v3_info["condition_number"] > 1.0e8:
        flags.append("ill_conditioned_covariance")
    if ref_align["internal_rms_A"] > 5.0 or v3_align["internal_rms_A"] > 5.0:
        flags.append("large_internal_fluctuations")
    if max(ref_align["rigid_leakage_rms_A"], v3_align["rigid_leakage_rms_A"]) > 1.0e-8:
        flags.append("rigid_motion_leakage")
    if max(ref_info["maximum_gaussian_ratio"], v3_info["maximum_gaussian_ratio"]) > 3.0:
        flags.append("non_gaussian_fluctuations")
    if bootstrap_relative_width > 1.0:
        flags.append("bootstrap_mode_instability")
    if np.any(v3_eigenvalues < 0.0):
        flags.append("negative_deconvolved_curvature")
    return {
        "anchor": anchor,
        "basis": basis,
        "reference_covariance": ref_covariance,
        "v3_covariance": v3_covariance,
        "reference_curvature": ref_curvature,
        "v3_curvature": v3_hessian,
        "reference_eigenvalues": ref_eigenvalues,
        "v3_eigenvalues": v3_eigenvalues,
        "reference_eigenvectors": ref_vectors,
        "v3_eigenvectors": v3_vectors,
        "overlap_matrix": comparison["overlap_matrix"],
        "best_overlap": comparison["best_overlap"],
        "principal_angle_rad": comparison["principal_angle_rad"],
        "reference_bootstrap_eigenvalues": ref_boot_eigenvalues,
        "v3_bootstrap_eigenvalues": v3_boot_eigenvalues,
        "v3_bootstrap_interval": v3_boot_interval,
        "bootstrap_relative_width": bootstrap_relative_width,
        "reference_alignment": ref_align,
        "v3_alignment": v3_align,
        "reference_covariance_info": ref_info,
        "v3_covariance_info": v3_info,
        "deconvolution_info": deconv_info,
        "reference_deconvolution_info": ref_deconv_info,
        "flags": flags,
    }


def _write_plots(results: list[dict[str, Any]], outdir: Path) -> None:
    if not results:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    reference = np.asarray([result["reference_eigenvalues"] for result in results])
    v3 = np.asarray([result["v3_eigenvalues"] for result in results])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(reference.shape[1]), np.median(reference, axis=0), "o-", label="reference K=0")
    axis.plot(np.arange(v3.shape[1]), np.median(v3, axis=0), "o-", label="v3 deconvolved")
    axis.set_xlabel("local mode index (ascending curvature)")
    axis.set_ylabel("curvature [kcal/mol/A^2]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / "soft_spectrum.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    overlap = np.mean(np.asarray([result["overlap_matrix"] for result in results]), axis=0)
    image = axis.imshow(overlap, vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    axis.set_xlabel("v3 mode")
    axis.set_ylabel("reference mode")
    figure.colorbar(image, ax=axis, label="mean absolute overlap")
    figure.tight_layout()
    figure.savefig(outdir / "mode_overlap.png", dpi=160)
    plt.close(figure)


def _write_audit_plot(audit: Mapping[str, Any], outdir: Path) -> None:
    """Plot the input availability when no physical covariance comparison is possible."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    reference_frames = int(audit.get("reference_frames", 0) or 0)
    usable_windows = int(audit.get("usable_windows", 0) or 0)
    aggregated_only = bool(audit.get("aggregated_only", False))
    labels = ["reference\nraw frames", "finite-restraint\nv3 windows", "v3 aggregated\nstate file"]
    available = [reference_frames > 0, usable_windows > 0, aggregated_only]
    colors = ["#2ca02c" if available[0] else "#d62728",
              "#2ca02c" if available[1] else "#d62728",
              "#ff7f0e" if available[2] else "#d62728"]
    annotations = [
        f"{reference_frames:,} frames",
        f"{usable_windows:,} usable windows",
        "present, not covariance-bearing" if aggregated_only else "not found",
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(np.arange(len(labels)), [int(value) for value in available], color=colors)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0, 1.35)
    axis.set_yticks([0, 1], ["missing", "available"])
    axis.set_title("Restraint-deconvolved covariance input audit")
    for bar, annotation in zip(bars, annotations):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(bar.get_height(), 0.03) + 0.06,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    reason = next(
        (str(entry.get("reason", "")) for entry in audit.get("entries", [])
         if isinstance(entry, Mapping) and entry.get("reason")),
        "no usable finite-restraint v3 windows",
    )
    figure.text(0.5, 0.01, reason, ha="center", va="bottom", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(outdir / "audit_status.png", dpi=160)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--v3-campaign", type=Path, required=True)
    parser.add_argument("--v3-manifest", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-anchors", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--discard-ps", type=float, default=20.0)
    parser.add_argument("--min-reference-frames", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--kT", type=float, default=DEFAULT_KT)
    parser.add_argument(
        "--reference-radius-A",
        type=float,
        default=0.5,
        help="local reference-basin aligned RMSD radius in Angstrom",
    )
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--eigen-floor", type=float, default=1.0e-10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_anchors < 1 or args.n_boot < 1 or args.min_reference_frames < 2:
        raise SystemExit("n-anchors, n-boot, and min-reference-frames must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = load_reference_frames(args.reference)
    windows, audit = load_v3_windows(
        args.v3_campaign, args.v3_manifest, args.max_frames, args.discard_ps, args.kT
    )
    audit.update({"reference": str(args.reference), "reference_frames": int(len(reference["R"]))})
    _write_json(args.outdir / "input_audit.json", audit)
    if not windows:
        _write_audit_plot(audit, args.outdir)
        _write_json(args.outdir / "covariance_summary.json", {
            "status": "audit_only",
            "reason": "no usable finite-restraint raw v3 windows",
            "reference_frames": int(len(reference["R"])),
        })
        _write_csv(args.outdir / "per_anchor.csv", [])
        return 0

    results: list[dict[str, Any]] = []
    rows: list[dict[str, object]] = []
    raw_payload: dict[str, Any] = {}
    for window in windows[: args.n_anchors]:
        target = np.asarray(window["target"], dtype=np.float64)
        reference_index = window.get("reference_index")
        if reference_index is None:
            try:
                reference_index = _nearest_reference_index(reference["R"], target)
            except ValueError as exc:
                _write_audit_entry(audit, {"state": window["state"], "status": "unusable", "reason": str(exc)})
                continue
        reference_index = int(reference_index)
        if not 0 <= reference_index < len(reference["R"]):
            _write_audit_entry(audit, {"state": window["state"], "status": "unusable", "reason": "reference_index out of bounds"})
            continue
        local_reference = _local_reference_frames(
            reference, reference_index, args.reference_radius_A
        )
        if len(local_reference) < args.min_reference_frames:
            _write_audit_entry(audit, {"state": window["state"], "status": "unusable", "reason": f"reference neighborhood has {len(local_reference)} frames"})
            continue
        v3_frames = np.asarray(window["frames"], dtype=np.float64)
        block_length = args.block_length or max(1, min(len(local_reference) // 10, 50))
        result = _analyze_anchor(
            local_reference,
            v3_frames,
            float(window["kappa_kcal_mol_A2"]),
            args.kT,
            args.shrinkage,
            args.eigen_floor,
            min(block_length, len(local_reference), len(v3_frames)),
            args.n_boot,
            args.seed + int(window["state"]),
        )
        result["state"] = int(window["state"])
        result["reference_index"] = reference_index
        result["kappa_kcal_mol_A2"] = float(window["kappa_kcal_mol_A2"])
        results.append(result)
        rows.append({
            "state": result["state"],
            "reference_index": reference_index,
            "kappa_kcal_mol_A2": result["kappa_kcal_mol_A2"],
            "reference_frames": len(local_reference),
            "v3_frames": len(v3_frames),
            "reference_condition_number": result["reference_covariance_info"]["condition_number"],
            "v3_condition_number": result["v3_covariance_info"]["condition_number"],
            "minimum_v3_curvature": result["deconvolution_info"]["minimum_curvature"],
            "mean_best_mode_overlap": float(np.mean(result["best_overlap"])),
            "max_principal_angle_deg": float(np.degrees(np.max(result["principal_angle_rad"]))),
            "rigid_leakage_rms_A": max(
                result["reference_alignment"]["rigid_leakage_rms_A"],
                result["v3_alignment"]["rigid_leakage_rms_A"],
            ),
            "max_gaussian_ratio": max(
                result["reference_covariance_info"]["maximum_gaussian_ratio"],
                result["v3_covariance_info"]["maximum_gaussian_ratio"],
            ),
            "bootstrap_relative_width": result["bootstrap_relative_width"],
            "flags": ";".join(result["flags"]),
        })
        prefix = f"state_{result['state']}"
        raw_payload[f"{prefix}_reference_covariance"] = result["reference_covariance"]
        raw_payload[f"{prefix}_v3_covariance"] = result["v3_covariance"]
        raw_payload[f"{prefix}_reference_curvature"] = result["reference_curvature"]
        raw_payload[f"{prefix}_v3_curvature"] = result["v3_curvature"]
        raw_payload[f"{prefix}_reference_eigenvalues"] = result["reference_eigenvalues"]
        raw_payload[f"{prefix}_v3_eigenvalues"] = result["v3_eigenvalues"]
        raw_payload[f"{prefix}_overlap_matrix"] = result["overlap_matrix"]

    _write_json(args.outdir / "input_audit.json", audit)
    _write_csv(args.outdir / "per_anchor.csv", rows)
    summary = {
        "status": "complete" if results else "audit_only",
        "n_usable_windows": len(windows),
        "n_analyzed_anchors": len(results),
        "mean_best_mode_overlap": float(np.mean([np.mean(r["best_overlap"]) for r in results])) if results else float("nan"),
        "mean_minimum_v3_curvature": float(np.mean([r["deconvolution_info"]["minimum_curvature"] for r in results])) if results else float("nan"),
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed},
        "settings": vars(args),
        "quality_flags": {flag: sum(flag in r["flags"] for r in results) for flag in sorted({f for r in results for f in r["flags"]})},
    }
    _write_json(args.outdir / "covariance_summary.json", summary)
    np.savez_compressed(args.outdir / "covariance_raw.npz", **raw_payload)
    _write_plots(results, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
