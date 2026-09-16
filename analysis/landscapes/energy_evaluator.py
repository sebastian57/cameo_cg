"""Checkpoint evaluation and regression against the established 20k energy cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from analysis.landscapes.experiment_config import ModelSpec
from analysis.landscapes.landscape import evenly_spaced_indices, wrap_degrees


ATOL = 5.0e-4
RTOL = 1.0e-6


def batched_energy_values(
    batch_energy,
    positions: np.ndarray,
    masks: np.ndarray,
    species: np.ndarray,
    frame_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Evaluate fixed-size padded batches and return only requested-frame values."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("frame_indices must be one-dimensional")
    if len(indices) == 0:
        return np.empty(0, dtype=np.float64)
    values = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        requested = indices[start : start + batch_size]
        if len(requested) < batch_size:
            requested = np.pad(requested, (0, batch_size - len(requested)), mode="edge")
        batch = np.asarray(
            batch_energy(positions[requested], masks[requested], species[requested]),
            dtype=np.float64,
        )
        count = min(batch_size, len(indices) - start)
        values[start : start + count] = batch[:count]
    return values


def extract_params(checkpoint: Any, key: str) -> dict[str, Any]:
    if key == "__root__":
        if not isinstance(checkpoint, dict):
            raise ValueError("root checkpoint must be a parameter mapping")
        return checkpoint
    if not isinstance(checkpoint, dict) or key not in checkpoint:
        available = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(f"missing checkpoint key {key!r}; available: {available}")
    params = checkpoint[key]
    if not isinstance(params, dict):
        raise ValueError(f"checkpoint key {key!r} must contain a parameter mapping")
    return params


def cache_reference_indices(n_reference: int, n_cache: int) -> np.ndarray:
    return evenly_spaced_indices(n_reference, n_cache)


def compare_to_cache(
    direct: np.ndarray, cached: np.ndarray, atol: float = ATOL, rtol: float = RTOL
) -> dict[str, float | bool | int]:
    direct = np.asarray(direct, dtype=np.float64)
    cached = np.asarray(cached, dtype=np.float64)
    if direct.shape != cached.shape:
        raise ValueError(f"direct/cache shape mismatch: {direct.shape} vs {cached.shape}")
    if not np.isfinite(direct).all() or not np.isfinite(cached).all():
        raise ValueError("direct/cache comparison contains non-finite energies")
    abs_error = np.abs(direct - cached)
    scale = np.maximum(np.abs(cached), np.finfo(np.float64).tiny)
    rel_error = abs_error / scale
    report: dict[str, float | bool | int] = {
        "passed": bool(np.allclose(direct, cached, atol=atol, rtol=rtol)),
        "n": int(direct.size),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "max_rel_error": float(rel_error.max(initial=0.0)),
    }
    if not report["passed"]:
        raise ValueError(
            "direct energies disagree with cache: "
            f"max_abs_error={report['max_abs_error']:.6g}, "
            f"max_rel_error={report['max_rel_error']:.6g}, atol={atol}, rtol={rtol}"
        )
    return report


def _project_imports(cameo_root: Path):
    from sampling.mapping import get_mapping
    return get_mapping


def validate_cache_alignment(
    reference_path: Path,
    cache_path: Path,
    cameo_root: Path,
    mapping_name: str,
) -> dict[str, float | int]:
    get_mapping = _project_imports(cameo_root)
    mapping = get_mapping(mapping_name)
    with np.load(reference_path, allow_pickle=False) as data:
        reference = np.asarray(data["R"], dtype=np.float64)
    with np.load(cache_path, allow_pickle=False) as data:
        cache_phi = np.asarray(data["phi"], dtype=np.float64)
        cache_psi = np.asarray(data["psi"], dtype=np.float64)
    indices = cache_reference_indices(len(reference), len(cache_phi))
    phi = mapping.cvs["phi"].evaluate(reference[indices])
    psi = mapping.cvs["psi"].evaluate(reference[indices])
    phi_error = np.abs(wrap_degrees(phi - cache_phi))
    psi_error = np.abs(wrap_degrees(psi - cache_psi))
    report = {
        "n_reference_frames": int(len(reference)),
        "n_cache_frames": int(len(cache_phi)),
        "max_phi_periodic_error_deg": float(phi_error.max(initial=0.0)),
        "max_psi_periodic_error_deg": float(psi_error.max(initial=0.0)),
    }
    if report["max_phi_periodic_error_deg"] >= 1.0e-8 or report["max_psi_periodic_error_deg"] >= 1.0e-8:
        raise ValueError(f"energy cache coordinates do not align with reference: {report}")
    return report


def evaluate_model(
    spec: ModelSpec,
    reference_path: Path,
    frame_indices: np.ndarray,
    cameo_root: Path,
    batch_size: int = 1,
) -> np.ndarray:
    """Evaluate one checkpoint on independent bb6 reference frames."""
    import pickle
    import tempfile

    import yaml

    from utils.jax_setup import apply_jax_compat_shims

    apply_jax_compat_shims()
    import jax
    import jax.numpy as jnp
    from config.manager import ConfigManager
    from data.preprocessor import CoordinatePreprocessor
    from models.combined_model import CombinedModel

    if os.environ.get("JAX_PLATFORMS") == "cpu" and jax.default_backend() != "cpu":
        raise RuntimeError(f"smoke requested CPU but JAX selected {jax.default_backend()}")
    cfg_dict = yaml.safe_load(spec.config.read_text()) or {}
    cfg_dict.setdefault("data", {})["path"] = str(reference_path)
    cfg_dict.setdefault("model", {})["neighbor_disable_cell_list"] = True
    runtime_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(cfg_dict, handle)
            runtime_path = Path(handle.name)
        cfg = ConfigManager(str(runtime_path))
        with np.load(reference_path, allow_pickle=False) as data:
            R = np.asarray(data["R"], dtype=np.float32)
            species = np.asarray(data["species"], dtype=np.int32)
            mask = (np.asarray(data["mask"], dtype=np.float32) if "mask" in data
                    else np.ones(R.shape[:2], dtype=np.float32))
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices.ndim != 1 or np.any(frame_indices < 0) or np.any(frame_indices >= len(R)):
            raise ValueError("frame_indices must be a valid 1D reference index array")
        preprocessor = CoordinatePreprocessor(
            cutoff=cfg.get_cutoff(),
            buffer_multiplier=cfg.get_buffer_multiplier(),
            park_multiplier=cfg.get_park_multiplier(),
        )
        box, shift = preprocessor.compute_box_extent(R, mask)
        R0 = preprocessor.center_and_park(R[:1], mask[:1], box, shift)[0]
        mask0 = jnp.asarray(mask[0])
        species0 = jnp.asarray(species[0])
        n_species = max(
            int(species.max()) + 1,
            int(cfg.get("model", "allegro", "num_types", default=0) or 0),
        )
        model = CombinedModel(
            config=cfg,
            R0=jnp.asarray(R0),
            box=box,
            species=species0,
            N_max=int(R.shape[1]),
            prior_only=cfg.prior_only_enabled(),
            n_species_override=n_species,
        )
        with spec.checkpoint.open("rb") as handle:
            params = extract_params(pickle.load(handle), spec.checkpoint_key)

        @jax.jit
        def energy_batch(position, batch_mask, batch_species):
            return jax.vmap(
                lambda r, m, s: model.compute_energy(params, r, m, s)
            )(position, batch_mask, batch_species)

        values = batched_energy_values(
            energy_batch, R, mask, species, frame_indices, batch_size
        )
        if not np.isfinite(values).all():
            raise ValueError(f"model {spec.label} produced non-finite energies")
        return values
    finally:
        if runtime_path is not None:
            runtime_path.unlink(missing_ok=True)
