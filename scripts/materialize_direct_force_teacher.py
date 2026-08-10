#!/usr/bin/env python3
"""Materialize out-of-fold direct-force ensemble labels into an NPZ dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
import numpy as np

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, parent: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (parent / value).resolve()


def _extract_params(payload: Any):
    if isinstance(payload, dict):
        if "ml" in payload:
            return payload
        for key in ("best_params", "params", "energy_params"):
            if key in payload:
                found = _extract_params(payload[key])
                if found is not None:
                    return found
        state = payload.get("trainer_state")
        if state is not None:
            found = _extract_params(state)
            if found is not None:
                return found
    if hasattr(payload, "params"):
        return _extract_params(payload.params)
    return None


def _load_params(path: Path, source: str = "auto"):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    source = str(source).strip().lower()
    if source == "auto":
        params = _extract_params(payload)
    elif source == "trainer_state":
        if not isinstance(payload, dict) or "trainer_state" not in payload:
            raise ValueError(f"No trainer_state parameter source in {path}.")
        params = _extract_params(payload["trainer_state"])
    elif source == "best_params":
        if not isinstance(payload, dict) or "best_params" not in payload:
            raise ValueError(f"No best_params parameter source in {path}.")
        params = _extract_params(payload["best_params"])
    else:
        raise ValueError(
            f"Unknown params_source={source!r}; expected auto, trainer_state, or best_params."
        )
    if params is None:
        raise ValueError(f"Could not find a CombinedModel parameter tree in {path}.")
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves or not all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise ValueError(f"Parameters in {path} are empty or non-finite.")
    return params


def _predict_member(
    config_path: Path,
    params_path: Path,
    R: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    box: jax.Array,
    indices: np.ndarray,
    batch_size: int,
    disable_cell_list: bool = True,
    params_source: str = "auto",
) -> np.ndarray:
    config = ConfigManager(config_path)
    if config.get_model_output_mode() != "direct_force":
        raise ValueError(f"Teacher config is not direct-force mode: {config_path}")
    # Materialization evaluates independent, very small structures. Reusing a
    # cell-list allocation whose reference coordinates came from one arbitrary
    # frame can make the sparse edge list reference-frame dependent. The
    # all-pairs candidate builder is both cheap for these systems and invariant
    # to the frame used to reconstruct the trained model.
    if disable_cell_list:
        config.set("model", "neighbor_disable_cell_list", True)
    first = int(indices[0])
    n_species = int(np.max(species)) + 1
    model = CombinedModel(
        config=config,
        R0=jnp.asarray(R[first], dtype=jnp.float32),
        box=box,
        species=jnp.asarray(species[first], dtype=jnp.int32),
        N_max=int(R.shape[1]),
        init_mask=jnp.asarray(mask[first], dtype=jnp.float32),
        n_species_override=n_species,
    )
    params = _load_params(params_path, source=params_source)

    def one(R_i, mask_i, species_i):
        return model.compute_direct_force(
            params,
            R_i,
            mask_i,
            species_i,
            neighbor=None,
            segment_id=None,
        )

    predict = jax.jit(jax.vmap(one, in_axes=(0, 0, 0)))
    output = np.zeros((indices.size, R.shape[1], 3), dtype=np.float32)
    for start in range(0, indices.size, batch_size):
        stop = min(start + batch_size, indices.size)
        selected = indices[start:stop]
        value = predict(
            jnp.asarray(R[selected], dtype=jnp.float32),
            jnp.asarray(mask[selected], dtype=jnp.float32),
            jnp.asarray(species[selected], dtype=jnp.int32),
        )
        output[start:stop] = np.asarray(value, dtype=np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("crossfit_manifest", type=Path)
    parser.add_argument("ensemble_spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    spec_parent = args.ensemble_spec.resolve().parent
    spec = json.loads(args.ensemble_spec.read_text())
    members = list(spec.get("members", []))
    if not members:
        raise ValueError("Ensemble spec must contain a non-empty members list.")

    loader = DatasetLoader(args.dataset)
    preprocessor = CoordinatePreprocessor(
        cutoff=float(spec.get("cutoff", 5.5)),
        buffer_multiplier=float(spec.get("buffer_multiplier", 4.0)),
        park_multiplier=float(spec.get("park_multiplier", 0.95)),
    )
    box, shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    R_model = np.asarray(
        preprocessor.center_and_park(loader.R, loader.mask, box, shift),
        dtype=np.float32,
    )
    mask = np.asarray(loader.mask, dtype=np.float32)
    species = np.asarray(loader.species, dtype=np.int32)
    n_frames = int(R_model.shape[0])

    with np.load(args.crossfit_manifest, allow_pickle=False) as manifest:
        if int(np.asarray(manifest["n_frames"]).item()) != n_frames:
            raise ValueError("Cross-fit manifest and dataset frame counts differ.")
        fold_holdouts = {
            fold: np.asarray(manifest[f"holdout_indices_{fold}"], dtype=np.int64)
            for fold in sorted({int(member["fold"]) for member in members})
        }

    sums = np.zeros_like(R_model, dtype=np.float64)
    sumsq = np.zeros_like(R_model, dtype=np.float64)
    counts = np.zeros((n_frames,), dtype=np.int32)
    member_records = []
    for member_idx, member in enumerate(members):
        fold = int(member["fold"])
        indices = fold_holdouts[fold]
        config_path = _resolve(member["config"], spec_parent)
        params_path = _resolve(member["params"], spec_parent)
        print(
            f"member {member_idx + 1}/{len(members)} fold={fold} "
            f"frames={indices.size} config={config_path.name}",
            flush=True,
        )
        prediction = _predict_member(
            config_path,
            params_path,
            R_model,
            mask,
            species,
            box,
            indices,
            args.batch_size,
            disable_cell_list=bool(spec.get("single_structure_disable_cell_list", True)),
            params_source=str(member.get("params_source", "auto")),
        )
        if not np.isfinite(prediction).all():
            raise ValueError(f"Teacher prediction from {params_path} contains NaN/Inf.")
        sums[indices] += prediction
        sumsq[indices] += np.square(prediction, dtype=np.float64)
        counts[indices] += 1
        member_records.append(
            {
                "fold": fold,
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "params": str(params_path),
                "params_sha256": _sha256(params_path),
                "params_source": str(member.get("params_source", "auto")),
            }
        )

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)
        raise ValueError(f"No out-of-fold teacher prediction for {missing.size} frames.")
    mean = sums / counts[:, None, None]
    variance = np.maximum(sumsq / counts[:, None, None] - mean * mean, 0.0)
    std = np.sqrt(variance)
    mean = np.asarray(mean, dtype=np.float32) * mask[..., None]
    std = np.asarray(std, dtype=np.float32) * mask[..., None]

    with np.load(args.dataset, allow_pickle=True) as source:
        output = {key: np.asarray(source[key]) for key in source.files}
    # Some legacy datasets rely on DatasetLoader to synthesize these arrays.
    # Materialized teacher datasets should be self-contained and unambiguous.
    output.setdefault("mask", mask)
    output.setdefault("species", species)
    output["TeacherForce"] = mean
    output["teacher_force_std"] = std
    output["teacher_force_count"] = counts
    output["teacher_force_mask"] = mask
    metadata = {
        "version": 1,
        "materialization_mode": spec.get("materialization_mode", "unspecified"),
        "scientifically_out_of_fold": spec.get("scientifically_out_of_fold"),
        "description": spec.get("description", spec.get("purpose", "")),
        "single_structure_disable_cell_list": bool(
            spec.get("single_structure_disable_cell_list", True)
        ),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "crossfit_manifest": str(args.crossfit_manifest.resolve()),
        "crossfit_manifest_sha256": _sha256(args.crossfit_manifest),
        "ensemble_spec": str(args.ensemble_spec.resolve()),
        "members": member_records,
        "prediction_count_min": int(counts.min()),
        "prediction_count_max": int(counts.max()),
    }
    output["teacher_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.parent / f".{args.output.name}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, **output)
    os.replace(tmp, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
