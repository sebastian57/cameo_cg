"""Shared, dependency-light helpers for the v3/v4 physics probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _leading_length(values: Mapping[str, np.ndarray]) -> int:
    lengths = {len(np.asarray(value)) for value in values.values() if np.asarray(value).ndim}
    if len(lengths) > 1:
        raise ValueError(f"array leading lengths disagree: {sorted(lengths)}")
    return next(iter(lengths), 0)


def load_npz_required(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load an NPZ and fail clearly when a required field is absent."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as raw:
        missing = [key for key in keys if key not in raw.files]
        if missing:
            raise KeyError(f"{path}: missing required keys {missing}; found {raw.files}")
        out = {key: np.asarray(raw[key]) for key in raw.files}
    _leading_length(out)
    return out


def load_state_table(path: Path) -> dict[str, np.ndarray]:
    """Load a state file; v4 state files use row number as the state ID."""
    out = load_npz_required(Path(path), ("R", "anchor", "direction"))
    n = len(out["R"])
    out.setdefault("state", np.arange(n, dtype=np.int64))
    for key in ("state", "anchor", "direction"):
        if len(out[key]) != n:
            raise ValueError(f"{path}: {key} has length {len(out[key])}, expected {n}")
    return out


def load_label_table(path: Path) -> dict[str, np.ndarray]:
    """Load collected mean-force labels, which must preserve state IDs."""
    out = load_npz_required(Path(path), ("state", "R", "F"))
    n = len(out["state"])
    for key in ("R", "F"):
        if len(out[key]) != n:
            raise ValueError(f"{path}: {key} has length {len(out[key])}, expected {n}")
    if len(np.unique(out["state"])) != n:
        raise ValueError(f"{path}: duplicate state IDs in labels")
    return out


def join_labels_by_state(
    states: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return label arrays in state-table order; require a complete exact join."""
    state_ids = np.asarray(states["state"], dtype=np.int64)
    label_ids = np.asarray(labels["state"], dtype=np.int64)
    if len(np.unique(label_ids)) != len(label_ids):
        raise ValueError("duplicate label state IDs")
    positions = {int(state): i for i, state in enumerate(label_ids)}
    missing = [int(state) for state in state_ids if int(state) not in positions]
    if missing:
        raise ValueError(f"missing label state IDs: {missing[:8]}")
    order = np.asarray([positions[int(state)] for state in state_ids], dtype=np.int64)
    out = dict(states)
    for key, value in labels.items():
        if key == "state":
            continue
        arr = np.asarray(value)
        out[key if key not in out else f"label_{key}"] = arr[order]
    return out


def available_label_rows(
    states: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Align only available labels and return a boolean availability mask."""
    state_ids = np.asarray(states["state"], dtype=np.int64)
    label_ids = np.asarray(labels["state"], dtype=np.int64)
    if len(np.unique(label_ids)) != len(label_ids):
        raise ValueError("duplicate label state IDs")
    positions = {int(state): i for i, state in enumerate(label_ids)}
    available = np.asarray([int(state) in positions for state in state_ids], dtype=bool)
    order = np.asarray([positions[int(state)] for state in state_ids[available]], dtype=np.int64)
    out = {key: np.asarray(value)[available] for key, value in states.items()}
    for key, value in labels.items():
        if key != "state":
            out[f"label_{key}"] = np.asarray(value)[order]
    return out, available


def signed_path_cost(R: np.ndarray, F: np.ndarray) -> float:
    """Estimate ``A(R[-1])-A(R[0])`` from ``F=-grad(A)`` by trapezoid integration."""
    R = np.asarray(R, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    if R.shape != F.shape or R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"R and F must both have shape (n, beads, 3), got {R.shape}, {F.shape}")
    if len(R) < 2:
        return 0.0
    delta = R[1:] - R[:-1]
    return float(np.sum(-0.5 * (F[1:] + F[:-1]) * delta))


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap interval for one-dimensional values."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[samples].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit-normalized axis."""
    axis = np.asarray(axis, dtype=np.float64).ravel()
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis cannot be zero")
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotate_about_centroid(R: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    center = R.mean(axis=0)
    return (R - center) @ rotation_matrix(axis, angle).T + center


def rigid_tangent(R: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """First derivative at zero of a rotation about the bead centroid."""
    R = np.asarray(R, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64).ravel()
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis cannot be zero")
    axis = axis / norm
    centered = R - R.mean(axis=0)
    return np.cross(axis[None, :], centered)


def jsonable_summary(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Convert NumPy/path values in a result mapping into JSON-safe values."""
    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    return convert(mapping)


def write_json(path: Path, mapping: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(jsonable_summary(mapping), indent=2) + "\n")
