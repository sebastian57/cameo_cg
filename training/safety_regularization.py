"""Safety-frame regularization helpers for residual-prior fine-tuning.

The losses in this module are deliberately opt-in and data-driven.  Normal
force matching still uses the dataset's residual force targets.  Safety frames
carry zero force-loss weights and only contribute through the additional
Chemtrain targets defined here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from utils.logging import training_logger


SAFETY_CONTACT_TARGET = "SafetyContact"
SAFETY_FORCE_CAP_TARGET = "SafetyForceCap"

SAFETY_FIELD_KEYS = (
    "safety_loss_mask",
    "safety_pair_i",
    "safety_pair_j",
    "safety_pair_mask",
    "sample_kind",
    "source_step",
    SAFETY_CONTACT_TARGET,
    SAFETY_FORCE_CAP_TARGET,
)


def safety_config(config) -> Dict[str, Any]:
    """Parse ``training.safety_regularization`` with conservative defaults."""
    cfg = config.get("training", "safety_regularization", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    contact_cfg = cfg.get("contact_attraction", {}) or {}
    cap_cfg = cfg.get("force_cap", {}) or {}
    dataset_paths = cfg.get("dataset_paths", []) or []
    if isinstance(dataset_paths, (str, os.PathLike)):
        dataset_paths = [dataset_paths]

    max_safety_fraction = float(cfg.get("max_safety_fraction", 0.10))
    if max_safety_fraction < 0.0 or max_safety_fraction >= 1.0:
        raise ValueError(
            "training.safety_regularization.max_safety_fraction must satisfy "
            f"0 <= value < 1, got {max_safety_fraction}."
        )

    max_pairs = int(contact_cfg.get("max_pairs", 64))
    if max_pairs <= 0:
        raise ValueError(
            "training.safety_regularization.contact_attraction.max_pairs must be > 0."
        )

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "dataset_paths": [str(p) for p in dataset_paths],
        "max_safety_fraction": max_safety_fraction,
        "normal_force_loss_on_safety": bool(
            cfg.get("normal_force_loss_on_safety", False)
        ),
        "seed_offset": int(cfg.get("seed_offset", 3119)),
        "contact_attraction": {
            "enabled": bool(contact_cfg.get("enabled", True)),
            "lambda": float(contact_cfg.get("lambda", 0.05)),
            "r_cut": float(contact_cfg.get("r_cut", 3.2)),
            "pull_tolerance": float(contact_cfg.get("pull_tolerance", 2.0)),
            "max_pairs": max_pairs,
        },
        "force_cap": {
            "enabled": bool(cap_cfg.get("enabled", True)),
            "lambda": float(cap_cfg.get("lambda", 0.01)),
            "mode": str(cap_cfg.get("mode", "prior_relative")).strip().lower(),
            "alpha": float(cap_cfg.get("alpha", 1.0)),
            "offset": float(cap_cfg.get("offset", 10.0)),
            "min_cap": float(cap_cfg.get("min_cap", 20.0)),
        },
    }


def safety_enabled(config) -> bool:
    return bool(safety_config(config)["enabled"])


def _resolve_path(path: str, config_path: Path, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = (config_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    return (repo_root / p).resolve()


def _zero_safety_pairs(n_frames: int, max_pairs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((n_frames, max_pairs), dtype=np.int32),
        np.zeros((n_frames, max_pairs), dtype=np.int32),
        np.zeros((n_frames, max_pairs), dtype=np.float32),
    )


def attach_default_safety_fields(split: Dict[str, np.ndarray], config) -> Dict[str, np.ndarray]:
    """Ensure a split has all safety target fields, filled with zeros."""
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return split

    out = dict(split)
    n_frames = int(out["R"].shape[0])
    n_atoms = int(out["R"].shape[1])
    max_pairs = int(cfg["contact_attraction"]["max_pairs"])

    out.setdefault("safety_loss_mask", np.zeros((n_frames, n_atoms), dtype=np.float32))
    pi, pj, pm = _zero_safety_pairs(n_frames, max_pairs)
    out.setdefault("safety_pair_i", pi)
    out.setdefault("safety_pair_j", pj)
    out.setdefault("safety_pair_mask", pm)
    out.setdefault(SAFETY_CONTACT_TARGET, np.zeros((n_frames, max_pairs), dtype=np.float32))
    out.setdefault(SAFETY_FORCE_CAP_TARGET, np.zeros((n_frames, n_atoms), dtype=np.float32))
    out.setdefault("sample_kind", np.zeros((n_frames,), dtype=np.int32))
    out.setdefault("source_step", np.full((n_frames,), -1, dtype=np.int32))
    return out


def _normalize_safety_dataset(raw: Dict[str, Any], config) -> Dict[str, np.ndarray]:
    cfg = safety_config(config)
    R = np.asarray(raw["R"], dtype=np.float32)
    n_frames, n_atoms = int(R.shape[0]), int(R.shape[1])
    max_pairs = int(cfg["contact_attraction"]["max_pairs"])

    mask = np.asarray(
        raw["mask"] if "mask" in raw else np.ones((n_frames, n_atoms), dtype=np.float32),
        dtype=np.float32,
    )
    species = np.asarray(
        raw["species"] if "species" in raw else np.zeros((n_frames, n_atoms), dtype=np.int32),
        dtype=np.int32,
    )

    if "F" in raw:
        F = np.asarray(raw["F"], dtype=np.float32)
    else:
        F = np.zeros_like(R, dtype=np.float32)

    force_loss_mask = np.asarray(
        raw["force_loss_mask"]
        if cfg["normal_force_loss_on_safety"] and "force_loss_mask" in raw
        else np.zeros((n_frames, n_atoms), dtype=np.float32),
        dtype=np.float32,
    )
    if cfg["normal_force_loss_on_safety"] and "force_loss_mask" not in raw:
        force_loss_mask = mask.astype(np.float32)

    safety_loss_mask = np.asarray(
        raw["safety_loss_mask"] if "safety_loss_mask" in raw else mask,
        dtype=np.float32,
    )
    if safety_loss_mask.shape != mask.shape:
        raise ValueError(
            f"safety_loss_mask shape {safety_loss_mask.shape} does not match mask {mask.shape}."
        )

    pi, pj, pm = _zero_safety_pairs(n_frames, max_pairs)
    if "safety_pair_i" in raw:
        src = np.asarray(raw["safety_pair_i"], dtype=np.int32)
        pi[:, : min(max_pairs, src.shape[1])] = src[:, :max_pairs]
    if "safety_pair_j" in raw:
        src = np.asarray(raw["safety_pair_j"], dtype=np.int32)
        pj[:, : min(max_pairs, src.shape[1])] = src[:, :max_pairs]
    if "safety_pair_mask" in raw:
        src = np.asarray(raw["safety_pair_mask"], dtype=np.float32)
        pm[:, : min(max_pairs, src.shape[1])] = src[:, :max_pairs]

    return {
        "R": R,
        "F": F,
        "mask": mask,
        "species": species,
        "force_loss_mask": force_loss_mask,
        "safety_loss_mask": safety_loss_mask,
        "safety_pair_i": pi,
        "safety_pair_j": pj,
        "safety_pair_mask": pm,
        SAFETY_CONTACT_TARGET: np.zeros((n_frames, max_pairs), dtype=np.float32),
        SAFETY_FORCE_CAP_TARGET: np.zeros((n_frames, n_atoms), dtype=np.float32),
        "sample_kind": np.asarray(
            raw["sample_kind"] if "sample_kind" in raw else np.ones((n_frames,), dtype=np.int32),
            dtype=np.int32,
        ),
        "source_step": np.asarray(
            raw["source_step"] if "source_step" in raw else np.full((n_frames,), -1, dtype=np.int32),
            dtype=np.int32,
        ),
    }


def load_safety_datasets(config, repo_root: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load and concatenate configured safety NPZ files."""
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return None
    paths = cfg["dataset_paths"]
    if not paths:
        training_logger.warning("[Safety] Enabled but no dataset_paths configured.")
        return None

    datasets = []
    for raw_path in paths:
        path = _resolve_path(raw_path, Path(config.config_path), repo_root)
        if not path.exists():
            raise FileNotFoundError(f"Safety dataset not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            datasets.append(_normalize_safety_dataset(dict(data), config))
        training_logger.info("[Safety] Loaded safety dataset: %s", path)

    keys = datasets[0].keys()
    merged = {key: np.concatenate([d[key] for d in datasets], axis=0) for key in keys}
    training_logger.info(
        "[Safety] Loaded %d safety frames from %d file(s).",
        int(merged["R"].shape[0]),
        len(datasets),
    )
    return merged


def _select_safety_subset(
    safety: Dict[str, np.ndarray],
    n_normal: int,
    max_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    n_safety = int(safety["R"].shape[0])
    if n_safety == 0 or max_fraction <= 0.0:
        return {k: v[:0] for k, v in safety.items()}
    max_count = int(np.floor((max_fraction * n_normal) / max(1.0 - max_fraction, 1e-6)))
    max_count = max(0, min(max_count, n_safety))
    if max_count == n_safety:
        return safety
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(np.arange(n_safety), size=max_count, replace=False))
    return {key: value[idx] for key, value in safety.items()}


def mix_safety_into_train_split(
    train_split: Dict[str, np.ndarray],
    safety: Optional[Dict[str, np.ndarray]],
    config,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Append a capped random subset of safety frames to a standard train split."""
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return train_split
    if safety is None or int(safety["R"].shape[0]) == 0:
        return attach_default_safety_fields(train_split, config)

    normal = attach_default_safety_fields(train_split, config)
    chosen = _select_safety_subset(
        safety,
        n_normal=int(normal["R"].shape[0]),
        max_fraction=float(cfg["max_safety_fraction"]),
        seed=int(seed) + int(cfg["seed_offset"]),
    )
    if int(chosen["R"].shape[0]) == 0:
        return normal

    if normal["R"].shape[1:] != chosen["R"].shape[1:]:
        raise ValueError(
            "Safety frames must match standard training frame shape. "
            f"normal R shape={normal['R'].shape}, safety R shape={chosen['R'].shape}."
        )

    merge_keys = (
        "R",
        "F",
        "mask",
        "species",
        "force_loss_mask",
        "safety_loss_mask",
        "safety_pair_i",
        "safety_pair_j",
        "safety_pair_mask",
        SAFETY_CONTACT_TARGET,
        SAFETY_FORCE_CAP_TARGET,
        "sample_kind",
        "source_step",
    )
    merged: Dict[str, np.ndarray] = {
        key: np.concatenate([normal[key], chosen[key]], axis=0)
        for key in merge_keys
    }

    n_total = int(merged["R"].shape[0])
    rng = np.random.RandomState(int(seed) + int(cfg["seed_offset"]) + 1)
    order = rng.permutation(n_total)
    for key, value in list(merged.items()):
        if isinstance(value, np.ndarray) and value.shape[0] == n_total:
            merged[key] = value[order]

    n_safety = int(chosen["R"].shape[0])
    n_valid = np.asarray(np.sum(merged["mask"] > 0, axis=1), dtype=np.int32)
    force_weight_base = np.asarray(merged["force_loss_mask"], dtype=np.float32)
    safe_n_valid = np.maximum(np.sum(force_weight_base > 0, axis=1, keepdims=True), 1.0)
    merged["force_loss_weights"] = np.where(
        force_weight_base > 0,
        force_weight_base / safe_n_valid,
        0.0,
    ).astype(np.float32)
    merged["n_valid"] = n_valid
    merged["n_segments"] = np.ones((n_total,), dtype=np.int32)
    merged["meta_batch_item_id"] = np.arange(n_total, dtype=np.int32)
    merged["meta_capacity"] = np.full((n_total,), int(merged["R"].shape[1]), dtype=np.int32)
    merged["meta_fill_ratio"] = n_valid.astype(np.float32) / max(int(merged["R"].shape[1]), 1)
    merged["meta_n_force_components"] = np.asarray(n_valid * 3, dtype=np.int32)
    merged["meta_source_structure_ids"] = np.arange(n_total, dtype=np.int32)[:, None]
    merged["meta_source_structure_n_valid"] = n_valid[:, None]
    merged["meta_structure_size_min"] = n_valid
    merged["meta_structure_size_mean"] = n_valid.astype(np.float32)
    merged["meta_structure_size_max"] = n_valid
    merged["meta_structure_size_std"] = np.zeros((n_total,), dtype=np.float32)
    training_logger.info(
        "[Safety] Mixed %d safety frames with %d normal frames (fraction=%.3f).",
        n_safety,
        int(normal["R"].shape[0]),
        n_safety / max(n_total, 1),
    )
    return merged


def safety_error(predictions, targets, weights=None):
    """Mean squared safety target error, normalized over active weights."""
    sq = jnp.square(predictions - targets)
    if weights is None:
        return jnp.mean(sq)
    weights = jnp.asarray(weights, dtype=sq.dtype)
    while weights.ndim < sq.ndim:
        weights = weights[..., None]
    weights = jnp.broadcast_to(weights, sq.shape)
    return jnp.sum(sq * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def _ml_force_fn(model, energy_params, state, neighbor, mask, species, segment_id):
    species_safe = jnp.where(mask > 0, species, 0).astype(jnp.int32)

    def energy_of_R(R_eval):
        return model.ml_model.compute_energy(
            energy_params["ml"],
            R_eval,
            mask,
            species_safe,
            neighbor,
            segment_id=segment_id,
        )

    return -jax.grad(energy_of_R)(state.position)


def _prior_force_fn(model, state, mask, species, segment_id):
    prior = getattr(model, "_residual_dsm_prior", None)
    if prior is None:
        prior = getattr(model, "prior", None)
    if prior is None:
        return jnp.zeros_like(state.position)

    species_safe = jnp.where(mask > 0, species, 0).astype(jnp.int32)

    def energy_of_R(R_eval):
        R_detached = jax.lax.stop_gradient(R_eval)
        R_masked = jnp.where(mask[:, None] > 0, R_eval, R_detached)
        return prior.compute_total_energy(
            R_masked,
            mask,
            species=species_safe,
            segment_id=segment_id,
        )

    return -jax.grad(energy_of_R)(state.position)


def make_contact_attraction_quantity(model, pull_tolerance: float, r_cut: float) -> Callable:
    """Return per-pair attractive ML-force excess for close contacts."""
    pull_tol = float(pull_tolerance)
    cutoff = float(r_cut)

    def quantity(
        state,
        neighbor=None,
        energy_params=None,
        mask=None,
        species=None,
        segment_id=None,
        safety_pair_i=None,
        safety_pair_j=None,
        safety_pair_mask=None,
        **kwargs,
    ):
        if mask is None or species is None:
            raise ValueError("Safety contact quantity requires mask and species.")
        if safety_pair_i is None or safety_pair_j is None or safety_pair_mask is None:
            raise ValueError("Safety contact quantity requires safety pair fields.")

        F_ml = _ml_force_fn(model, energy_params, state, neighbor, mask, species, segment_id)
        pair_i = jnp.asarray(safety_pair_i, dtype=jnp.int32)
        pair_j = jnp.asarray(safety_pair_j, dtype=jnp.int32)
        pair_mask = jnp.asarray(safety_pair_mask, dtype=state.position.dtype)

        Ri = state.position[pair_i]
        Rj = state.position[pair_j]
        rij = Ri - Rj
        dist = jnp.linalg.norm(rij, axis=-1)
        rhat = rij / jnp.maximum(dist[:, None], 1e-6)
        rel_force = F_ml[pair_i] - F_ml[pair_j]
        radial = jnp.sum(rel_force * rhat, axis=-1)

        close = (dist < cutoff).astype(state.position.dtype)
        active = pair_mask * close
        return jnp.maximum(0.0, -radial - pull_tol) * active

    return quantity


def make_force_cap_quantity(model, cfg: Dict[str, Any]) -> Callable:
    """Return per-atom ML force excess above a fixed or prior-relative cap."""
    mode = str(cfg.get("mode", "prior_relative")).strip().lower()
    alpha = float(cfg.get("alpha", 1.0))
    offset = float(cfg.get("offset", 10.0))
    min_cap = float(cfg.get("min_cap", 20.0))

    if mode not in ("prior_relative", "fixed"):
        raise ValueError(
            "training.safety_regularization.force_cap.mode must be "
            "'prior_relative' or 'fixed'."
        )

    def quantity(
        state,
        neighbor=None,
        energy_params=None,
        mask=None,
        species=None,
        segment_id=None,
        safety_loss_mask=None,
        **kwargs,
    ):
        if mask is None or species is None:
            raise ValueError("Safety force-cap quantity requires mask and species.")
        if safety_loss_mask is None:
            raise ValueError("Safety force-cap quantity requires safety_loss_mask.")

        F_ml = _ml_force_fn(model, energy_params, state, neighbor, mask, species, segment_id)
        ml_norm = jnp.linalg.norm(F_ml, axis=-1)
        if mode == "prior_relative":
            F_prior = _prior_force_fn(model, state, mask, species, segment_id)
            prior_norm = jnp.linalg.norm(F_prior, axis=-1)
            cap = jnp.maximum(min_cap, alpha * prior_norm + offset)
        else:
            cap = jnp.full_like(ml_norm, min_cap)
        active = jnp.asarray(safety_loss_mask, dtype=state.position.dtype)
        return jnp.maximum(0.0, ml_norm - cap) * active

    return quantity


def safety_gammas(config) -> Dict[str, float]:
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return {}
    out: Dict[str, float] = {}
    if cfg["contact_attraction"]["enabled"]:
        out[SAFETY_CONTACT_TARGET] = float(cfg["contact_attraction"]["lambda"])
    if cfg["force_cap"]["enabled"]:
        out[SAFETY_FORCE_CAP_TARGET] = float(cfg["force_cap"]["lambda"])
    return out


def safety_error_fns(config) -> Dict[str, Callable]:
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return {}
    return {key: safety_error for key in safety_gammas(config)}


def safety_weights_keys(config) -> Dict[str, str]:
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return {}
    keys: Dict[str, str] = {}
    if cfg["contact_attraction"]["enabled"]:
        keys[SAFETY_CONTACT_TARGET] = "safety_pair_mask"
    if cfg["force_cap"]["enabled"]:
        keys[SAFETY_FORCE_CAP_TARGET] = "safety_loss_mask"
    return keys


def make_safety_quantities(model, config) -> Dict[str, Callable]:
    cfg = safety_config(config)
    if not cfg["enabled"]:
        return {}
    quantities: Dict[str, Callable] = {}
    if cfg["contact_attraction"]["enabled"]:
        quantities[SAFETY_CONTACT_TARGET] = make_contact_attraction_quantity(
            model,
            pull_tolerance=float(cfg["contact_attraction"]["pull_tolerance"]),
            r_cut=float(cfg["contact_attraction"]["r_cut"]),
        )
    if cfg["force_cap"]["enabled"]:
        quantities[SAFETY_FORCE_CAP_TARGET] = make_force_cap_quantity(
            model,
            cfg["force_cap"],
        )
    return quantities
