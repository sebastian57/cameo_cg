#!/usr/bin/env python3
"""Orchestrate direct checkpoint validation and MD-less landscape analysis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from analyze_landscapes import analyze_arrays
from analysis.landscapes.experiment_config import ExperimentConfig, load_experiment, validate_input_paths
from analysis.landscapes.energy_evaluator import (
    cache_reference_indices,
    compare_to_cache,
    evaluate_model,
    validate_cache_alignment,
)
from analysis.landscapes.landscape import evenly_spaced_indices, stratified_indices


LOG = logging.getLogger("md_less_energy_landscape")


def prepare_indices(
    config: ExperimentConfig, cache_regions: np.ndarray, n_reference: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return direct cache indices, direct reference indices, and map cache indices."""
    cache_regions = np.asarray(cache_regions).astype(str)
    reference_indices = cache_reference_indices(n_reference, len(cache_regions))
    if config.profile.name == "smoke":
        direct_cache = stratified_indices(
            cache_regions, per_region=getattr(config.profile, "cache_validation_frames", config.profile.direct_frames) // 4, seed=config.seed
        )
    else:
        validation_frames = getattr(config.profile, "cache_validation_frames", config.profile.direct_frames)
        if validation_frames != len(cache_regions):
            raise ValueError(
                "production direct_frames must equal the established cache frame count "
                f"({len(cache_regions)}), got {validation_frames}"
            )
        direct_cache = np.arange(len(cache_regions), dtype=np.int64)
    map_cache = evenly_spaced_indices(len(cache_regions), config.profile.map_frames)
    return direct_cache, reference_indices[direct_cache], map_cache


def scientific_reference_indices(n_reference: int, reference_frames: int | None) -> np.ndarray:
    """Select the independent reference panel; ``None`` means every frame."""
    if reference_frames is None:
        return np.arange(n_reference, dtype=np.int64)
    return evenly_spaced_indices(n_reference, reference_frames)


def evaluation_plan(
    direct_reference: np.ndarray, map_reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate evaluation frames and retain both requested output orders."""
    direct = np.asarray(direct_reference, dtype=np.int64)
    mapped = np.asarray(map_reference, dtype=np.int64)
    if direct.ndim != 1 or mapped.ndim != 1:
        raise ValueError("evaluation index arrays must be one-dimensional")
    evaluate, inverse = np.unique(np.concatenate((direct, mapped)), return_inverse=True)
    split = len(direct)
    return evaluate, inverse[:split], inverse[split:]


def ensure_output_allowed(output_dir: Path, profile_name: str, overwrite: bool) -> None:
    completed = output_dir / "summary.json"
    if completed.exists() and profile_name == "production" and not overwrite:
        raise FileExistsError(
            f"completed production result exists at {output_dir}; pass --overwrite explicitly"
        )


def validate_run_mode(
    profile_name: str, use_cache_only: bool, models: Sequence[Any] = ()
) -> None:
    if profile_name == "production" and use_cache_only:
        raise ValueError("cache-only mode is a smoke/diagnostic path and is forbidden for production")
    uncached = [model.label for model in models if model.cache_key is None]
    if use_cache_only and uncached:
        raise ValueError(
            "cache-only mode cannot evaluate uncached model(s): " + ", ".join(uncached)
        )



def validate_backend(requested: str, actual: str) -> str:
    """Reject a runtime backend that differs from the scientific profile."""
    if actual != requested:
        raise RuntimeError(f"requested backend {requested}, but JAX selected {actual}")
    return actual

def _project_mapping(cameo_root: Path, name: str):
    from sampling.mapping import get_mapping
    return get_mapping(name)


def _path_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime": stat.st_mtime}



def load_md_ensemble(model: Any, n_beads: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load positions and validate contributor-level cleaning metadata when present."""
    metadata_keys = (
        "source_replica_ids",
        "source_files",
        "raw_frames",
        "equil_start_frames",
        "cut_frames",
        "kept_frames",
        "dissociated",
    )
    with np.load(model.md_ensemble, allow_pickle=False) as data:
        positions = np.asarray(data["R"], dtype=np.float64)
        available = [key for key in metadata_keys if key in data]
        if available and len(available) != len(metadata_keys):
            missing = sorted(set(metadata_keys) - set(available))
            raise ValueError(f"{model.label} MD metadata is incomplete; missing {missing}")
        arrays = {key: np.asarray(data[key]) for key in metadata_keys} if available else {}
    if positions.ndim != 3 or positions.shape[1:] != (n_beads, 3):
        raise ValueError(
            f"{model.label} MD shape mismatch: {positions.shape}; expected (*, {n_beads}, 3)"
        )
    if not np.isfinite(positions).all():
        raise ValueError(f"{model.label} MD coordinates contain non-finite values")
    if not arrays:
        return positions, {
            "ensemble_path": str(Path(model.md_ensemble).resolve()),
            "metadata_available": False,
            "eligible_for_md_rank": False,
            "declared_replicas": int(model.md_replicas),
            "observed_frames": int(positions.shape[0]),
            "observed_beads": int(positions.shape[1]),
        }
    lengths = {key: len(value) for key, value in arrays.items()}
    observed_replicas = lengths["source_replica_ids"]
    if any(length != observed_replicas for length in lengths.values()):
        raise ValueError(f"{model.label} MD metadata length mismatch: {lengths}")
    if observed_replicas != model.md_replicas:
        raise ValueError(
            f"{model.label} declares {model.md_replicas} MD replicas, "
            f"but metadata contains {observed_replicas}"
        )
    if int(arrays["kept_frames"].sum()) != len(positions):
        raise ValueError(f"{model.label} MD kept-frame metadata does not match positions")
    metadata: dict[str, Any] = {
        "ensemble_path": str(Path(model.md_ensemble).resolve()),
        "metadata_available": True,
        "eligible_for_md_rank": True,
        "declared_replicas": int(model.md_replicas),
        "observed_frames": int(positions.shape[0]),
        "observed_beads": int(positions.shape[1]),
    }
    metadata.update({key: value.tolist() for key, value in arrays.items()})
    return positions, metadata

def _effective_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "manifest": str(config.manifest),
        "cameo_root": str(config.cameo_root),
        "reference": str(config.reference),
        "energy_cache": str(config.energy_cache),
        "mapping": config.mapping,
        "temperature_K": config.temperature,
        "seed": config.seed,
        "profile": vars(config.profile),
        "results_dir": str(config.results_dir),
        "pairs": [vars(pair) for pair in getattr(config, "pairs", [])],
        "models": [
            {**vars(model), **{key: str(value) for key, value in vars(model).items()
                              if isinstance(value, Path)}}
            for model in config.models
        ],
    }


def _configure_logging(path: Path) -> None:
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def run(config: ExperimentConfig, *, overwrite: bool, use_cache_only: bool) -> Path:
    validate_run_mode(config.profile.name, use_cache_only, config.models)
    validate_input_paths(config)
    ensure_output_allowed(config.results_dir, config.profile.name, overwrite)
    config.results_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{config.profile.name}-", dir=config.results_dir.parent))
    _configure_logging(staging / "run.log")
    LOG.info("profile=%s scientific_result=%s", config.profile.name,
             config.profile.scientific_result)
    LOG.info("requested backend=%s JAX_PLATFORMS=%s", config.profile.backend,
             os.environ.get("JAX_PLATFORMS", "<unset>"))
    (staging / "effective_config.yaml").write_text(
        yaml.safe_dump(_effective_config(config), sort_keys=False)
    )
    actual_jax_backend = None
    if not use_cache_only:
        import jax

        actual_jax_backend = validate_backend(config.profile.backend, jax.default_backend())

    alignment = validate_cache_alignment(
        config.reference, config.energy_cache, config.cameo_root, config.mapping
    )
    mapping = _project_mapping(config.cameo_root, config.mapping)
    with np.load(config.reference, allow_pickle=False) as reference_data:
        reference_positions = np.asarray(reference_data["R"], dtype=np.float64)
    n_reference = len(reference_positions)
    with np.load(config.energy_cache, allow_pickle=False) as cache:
        cache_phi = np.asarray(cache["phi"], dtype=np.float64)
        cache_psi = np.asarray(cache["psi"], dtype=np.float64)
        cache_regions = np.asarray(cache["region"]).astype(str)
        cache_energies = {
            model.label: np.asarray(cache[f"{model.cache_key}__U"], dtype=np.float64)
            for model in config.models
            if model.cache_key is not None
        }
    direct_cache, direct_reference, map_cache = prepare_indices(
        config, cache_regions, n_reference
    )
    cache_reference = cache_reference_indices(n_reference, len(cache_regions))
    if use_cache_only:
        scientific_cache = map_cache
        scientific_reference = cache_reference[scientific_cache]
        scientific_phi = cache_phi[scientific_cache]
        scientific_psi = cache_psi[scientific_cache]
    else:
        requested_frames = getattr(config.profile, "reference_frames", config.profile.map_frames)
        scientific_reference = scientific_reference_indices(n_reference, requested_frames)
        scientific_positions = reference_positions[scientific_reference]
        scientific_phi = mapping.cvs["phi"].evaluate(scientific_positions)
        scientific_psi = mapping.cvs["psi"].evaluate(scientific_positions)
    np.save(staging / "selected_indices.npy", scientific_reference)
    comparisons: dict[str, dict[str, Any]] = {}
    direct_energies: dict[str, np.ndarray] = {}
    scientific_energies: dict[str, np.ndarray] = {}
    for model in config.models:
        if use_cache_only:
            direct = cache_energies[model.label][direct_cache]
            scientific = cache_energies[model.label][scientific_cache]
            comparisons[model.label] = {
                "passed": True, "skipped": True, "reason": "--use-cache-only diagnostic"
            }
            LOG.info("%s direct evaluation skipped (--use-cache-only)", model.label)
        else:
            evaluate_indices, direct_lookup, scientific_lookup = evaluation_plan(
                direct_reference, scientific_reference
            )
            LOG.info(
                "evaluating %s on %d unique validation/scientific frames in batches of %d",
                model.label, len(evaluate_indices), config.profile.energy_batch_size,
            )
            evaluated = evaluate_model(
                model, config.reference, evaluate_indices, config.cameo_root,
                batch_size=config.profile.energy_batch_size,
            )
            direct = evaluated[direct_lookup]
            scientific = evaluated[scientific_lookup]
            if model.cache_key is None:
                comparisons[model.label] = {
                    "passed": True,
                    "skipped": True,
                    "reason": "no established cache entry",
                }
                LOG.info("%s cache regression skipped (no established cache entry)", model.label)
            else:
                comparisons[model.label] = compare_to_cache(
                    direct, cache_energies[model.label][direct_cache],
                    atol=config.profile.cache_atol,
                    rtol=config.profile.cache_rtol,
                )
                LOG.info("%s cache regression max_abs_error=%.3g", model.label,
                         comparisons[model.label]["max_abs_error"])
        direct_energies[model.label] = direct
        scientific_energies[model.label] = scientific
    energy_payload: dict[str, np.ndarray] = {
        "direct_cache_indices": direct_cache,
        "direct_reference_indices": direct_reference,
        "scientific_reference_indices": scientific_reference,
        "map_cache_indices": map_cache,
        "map_phi": scientific_phi,
        "map_psi": scientific_psi,
    }
    for label in scientific_energies:
        energy_payload[f"{label}__direct_U"] = direct_energies[label]
        energy_payload[f"{label}__map_U"] = scientific_energies[label]
    np.savez_compressed(staging / "energies.npz", **energy_payload)
    md_ensemble_summary: dict[str, dict[str, Any]] = {}
    md_cvs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in config.models:
        if model.md_ensemble is None:
            md_ensemble_summary[model.label] = {
                "available": False, "reason": "no MD ensemble supplied"
            }
            continue
        positions, md_ensemble_summary[model.label] = load_md_ensemble(
            model, mapping.n_beads
        )
        if md_ensemble_summary[model.label]["eligible_for_md_rank"]:
            md_cvs[model.label] = (
                mapping.cvs["phi"].evaluate(positions),
                mapping.cvs["psi"].evaluate(positions),
            )
        else:
            md_ensemble_summary[model.label]["reason"] = (
                "ensemble lacks contributor-level cleaning provenance"
            )
    run_timestamp = datetime.now(timezone.utc)
    summary = analyze_arrays(
        config,
        scientific_phi,
        scientific_psi,
        scientific_energies,
        md_cvs,
        staging,
        md_provenance=md_ensemble_summary,
        evaluation_timestamp_utc=run_timestamp.isoformat(),
    )
    provenance = {
        "timestamp_utc": run_timestamp.isoformat(),
        "command": shlex.join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_backend": config.profile.backend,
        "actual_jax_backend": actual_jax_backend,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cache_alignment": alignment,
        "direct_cache_comparison": comparisons,
        "md_ensembles": md_ensemble_summary,
        "inputs": {
            "reference": _path_record(config.reference),
            "energy_cache": _path_record(config.energy_cache),
            **{f"{model.label}.config": _path_record(model.config) for model in config.models},
            **{f"{model.label}.checkpoint": _path_record(model.checkpoint) for model in config.models},
            **{f"{model.label}.md_ensemble": _path_record(model.md_ensemble)
               for model in config.models if model.md_ensemble is not None},
        },
    }
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2))
    summary.update({
        "status": "ok",
        "direct_cache_comparison": comparisons,
        "cache_alignment": alignment,
        "direct_reference_indices": direct_reference.tolist(),
        "map_cache_indices": map_cache.tolist(),
        "scientific_reference_indices": scientific_reference.tolist(),
    })
    (staging / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))

    if config.results_dir.exists():
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = config.results_dir.with_name(f"{config.results_dir.name}.previous-{suffix}")
        config.results_dir.rename(backup)
        LOG.info("preserved previous result at %s", backup)
    staging.rename(config.results_dir)
    LOG.info("completed result: %s", config.results_dir)
    return config.results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.yaml"))
    parser.add_argument("--profile", choices=("smoke", "production"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-cache-only", action="store_true")
    args = parser.parse_args()
    config = load_experiment(args.config, args.profile)
    result = run(config, overwrite=args.overwrite, use_cache_only=args.use_cache_only)
    print(result)


if __name__ == "__main__":
    main()
