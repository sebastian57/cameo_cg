"""Noised-residual training for prior-residual force matching.

Generates noised copies of clean frames and trains the ML residual force
to approach zero on off-manifold structures. This is a stability
regularization method — it should not be described as physically correct
force labelling for noised structures.

Requires ``training.prior_residual.enabled = true``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from models.prior_energy import PriorEnergy
from models.topology import TopologyBuilder
from utils.logging import training_logger


def noised_residual_enabled(config) -> bool:
    cfg = _noised_residual_config(config)
    return bool(cfg.get("enabled", False))


def _noised_residual_config(config) -> Dict[str, Any]:
    cfg = config.get("training", "noised_residual_training", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


def _parse_noise_levels(
    cfg: Dict[str, Any],
) -> List[Dict[str, float]]:
    levels_raw = cfg.get("noise_levels", [])
    if not levels_raw:
        raise ValueError(
            "training.noised_residual_training.noise_levels must be a non-empty list."
        )
    parsed = []
    for item in levels_raw:
        if isinstance(item, (int, float)):
            sigma = float(item)
            name = f"sigma_{sigma}"
            attenuation = 0.0
            weight = 2.0
        elif isinstance(item, dict):
            sigma = float(item["sigma"])
            name = str(item.get("name", f"sigma_{sigma}"))
            attenuation = float(item.get("attenuation", 0.0))
            weight = float(item.get("weight", 2.0))
        else:
            raise ValueError(
                f"noise_levels entries must be numbers or dicts with sigma/name/attenuation/weight, "
                f"got {item!r}."
            )
        if sigma <= 0:
            raise ValueError(f"noise_levels sigma must be > 0, got {sigma}.")
        if not (0.0 <= attenuation <= 1.0):
            raise ValueError(
                f"noise_levels attenuation must be in [0, 1], got {attenuation}."
            )
        if weight < 0:
            raise ValueError(f"noise_levels weight must be >= 0, got {weight}.")
        parsed.append({"sigma": sigma, "name": name, "attenuation": attenuation, "weight": weight})
    return parsed


def noised_residual_config_parsed(config) -> Dict[str, Any]:
    cfg = _noised_residual_config(config)
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    if not config.prior_residual_enabled():
        raise ValueError(
            "training.noised_residual_training requires prior_residual.enabled = true."
        )
    noise_levels = _parse_noise_levels(cfg)
    duplicate_every_raw = cfg.get("duplicate_every", None)
    if duplicate_every_raw is None:
        duplicate_every = None
    else:
        duplicate_every = int(duplicate_every_raw)
        if duplicate_every < 2:
            raise ValueError(
                "training.noised_residual_training.duplicate_every must be >= 2 or null (default). "
                f"Got {duplicate_every}."
            )
    duplicate_offset = int(cfg.get("duplicate_offset", 0))
    if duplicate_offset < 0:
        raise ValueError(
            f"training.noised_residual_training.duplicate_offset must be >= 0, got {duplicate_offset}."
        )
    refresh_interval_epochs = int(cfg.get("refresh_interval_epochs", 1))
    if refresh_interval_epochs < 0:
        raise ValueError(
            "training.noised_residual_training.refresh_interval_epochs must be >= 0 "
            f"(0 disables epoch-wise regeneration), got {refresh_interval_epochs}."
        )
    min_pair_distance = float(cfg.get("min_pair_distance", 0.0) or 0.0)
    if min_pair_distance < 0.0:
        raise ValueError(
            "training.noised_residual_training.min_pair_distance must be >= 0, "
            f"got {min_pair_distance}."
        )
    max_rescale_attempts = int(cfg.get("max_rescale_attempts", 8))
    if max_rescale_attempts < 0:
        raise ValueError(
            "training.noised_residual_training.max_rescale_attempts must be >= 0, "
            f"got {max_rescale_attempts}."
        )
    rescale_factor = float(cfg.get("rescale_factor", 0.5))
    if not (0.0 < rescale_factor < 1.0):
        raise ValueError(
            "training.noised_residual_training.rescale_factor must be in (0, 1), "
            f"got {rescale_factor}."
        )
    return {
        "enabled": True,
        "seed_offset": int(cfg.get("seed_offset", 9999)),
        "noise_levels": noise_levels,
        "compute_batch_size": int(cfg.get("compute_batch_size", 100)),
        "duplicate_every": duplicate_every,
        "duplicate_offset": duplicate_offset,
        "refresh_interval_epochs": refresh_interval_epochs,
        "min_pair_distance": min_pair_distance,
        "max_rescale_attempts": max_rescale_attempts,
        "rescale_factor": rescale_factor,
    }


def _build_prior_energy(
    config,
    R: np.ndarray,
    id_to_aa: Optional[Dict[int, str]] = None,
):
    """Build a PriorEnergy instance for prior force computation."""
    n_atoms = int(R.shape[1])
    topology = TopologyBuilder(N_max=n_atoms, min_repulsive_sep=config.get_min_repulsive_sep())
    displacement = lambda Ra, Rb: Ra - Rb
    return PriorEnergy(config, topology, displacement, id_to_aa=id_to_aa)


def _center_noise_per_structure(
    R: np.ndarray, mask: np.ndarray, eps: np.ndarray
) -> np.ndarray:
    """Center noise so it does not shift the centroid of valid atoms."""
    mask3 = mask[..., None]
    n_valid = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0).astype(np.float32)
    eps_centered = eps * mask3
    eps_mean = np.sum(eps_centered, axis=1, keepdims=True) / n_valid[:, :, None]
    return (eps_centered - eps_mean) * mask3


def _min_pair_distances(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the closest valid bead-pair distance for each frame."""
    mins = np.full((int(R.shape[0]),), np.inf, dtype=np.float32)
    for frame_idx, (coords_all, mask_i) in enumerate(zip(np.asarray(R), np.asarray(mask))):
        valid = np.asarray(mask_i) > 0
        coords = np.asarray(coords_all[valid], dtype=np.float32)
        if coords.shape[0] <= 1:
            continue
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1, dtype=np.float32)
        np.fill_diagonal(dist_sq, np.inf)
        mins[frame_idx] = np.float32(np.sqrt(np.min(dist_sq, initial=np.inf)))
    return mins


def _apply_min_distance_guard(
    R_clean: np.ndarray,
    eps: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    min_pair_distance: float,
    max_rescale_attempts: int,
    rescale_factor: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Shrink per-frame noise until generated structures avoid close contacts."""
    R_clean = np.asarray(R_clean, dtype=np.float32)
    eps = np.asarray(eps, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    R_noisy = (R_clean + float(sigma) * eps).astype(np.float32)
    if min_pair_distance <= 0.0 or R_noisy.shape[0] == 0:
        return R_noisy, {"guarded": 0, "remaining": 0, "min_before": np.inf, "min_after": np.inf}

    min_before = _min_pair_distances(R_noisy, mask)
    needs_guard = min_before < float(min_pair_distance)
    guarded = int(np.sum(needs_guard))
    if guarded == 0:
        return R_noisy, {
            "guarded": 0,
            "remaining": 0,
            "min_before": float(np.min(min_before)),
            "min_after": float(np.min(min_before)),
        }

    scales = np.ones((R_noisy.shape[0], 1, 1), dtype=np.float32)
    min_after = min_before.copy()
    for _ in range(int(max_rescale_attempts)):
        active = min_after < float(min_pair_distance)
        if not np.any(active):
            break
        scales[active] *= float(rescale_factor)
        R_noisy[active] = (R_clean[active] + float(sigma) * eps[active] * scales[active]).astype(np.float32)
        min_after[active] = _min_pair_distances(R_noisy[active], mask[active])

    remaining = int(np.sum(min_after < float(min_pair_distance)))
    return R_noisy, {
        "guarded": guarded,
        "remaining": remaining,
        "min_before": float(np.min(min_before)),
        "min_after": float(np.min(min_after)),
        "mean_scale_guarded": float(np.mean(scales[needs_guard])) if guarded else 1.0,
    }


def _compute_prior_forces(
    R: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    prior_energy,
    compute_batch_size: int = 100,
) -> np.ndarray:
    """Compute prior forces for all frames via chunked JAX vmap+grad."""
    def single_force(R_i, mask_i, species_i):
        def energy_of_R(R_var):
            R_detached = jax.lax.stop_gradient(R_var)
            R_masked = jnp.where(mask_i[:, None] > 0, R_var, R_detached)
            return prior_energy.compute_total_energy(R_masked, mask_i, species=species_i)
        return -jax.grad(energy_of_R)(R_i)

    batched_fn = jax.jit(jax.vmap(single_force, in_axes=(0, 0, 0)))
    n_frames = int(R.shape[0])
    f_prior = np.zeros_like(R, dtype=np.float32)
    for start in range(0, n_frames, int(compute_batch_size)):
        end = min(start + int(compute_batch_size), n_frames)
        R_chunk = jnp.asarray(R[start:end], dtype=jnp.float32)
        mask_chunk = jnp.asarray(mask[start:end], dtype=np.float32)
        species_chunk = jnp.asarray(species[start:end], dtype=np.int32)
        f_chunk = batched_fn(R_chunk, mask_chunk, species_chunk)
        f_prior[start:end] = np.asarray(f_chunk, dtype=np.float32)
    return f_prior


def attach_noised_residual_fields(
    split: Dict[str, np.ndarray],
    config,
    id_to_aa: Optional[Dict[int, str]],
    seed: int,
    split_seed: int,
    fitted_params: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Generate noised copies of clean frames with residual force targets.

    For each noise level (sigma):
      R_noisy   = R_clean + sigma * eps
      F_ref     = a * F_res_clean  (a = attenuation)
      F_stored  = F_prior(R_noisy) + F_ref
                 = F_prior(R_noisy) + a * (F_clean - F_prior(R_clean))

    Since the trainer subtracts F_prior online during loss, the ML target becomes:
      F_ML,target = F_stored - F_prior(R_noisy)
                  = a * (F_clean - F_prior(R_clean))

    At a=0: F_ML,target = 0  (zero-residual training on high-noise frames)
    At a=1: F_ML,target = F_res_clean  (clean residual, tiny perturbation)

    Only every ``duplicate_every``-th frame (plus ``duplicate_offset``) gets duplicated.
    ``duplicate_every = null`` (default) means all frames are duplicated.
    """
    cfg = noised_residual_config_parsed(config)
    if not cfg["enabled"]:
        return split

    R_clean = np.asarray(split["R"], dtype=np.float32)
    F_res_clean = np.asarray(split["F"], dtype=np.float32)
    mask = np.asarray(split["mask"], dtype=np.float32)
    species = np.asarray(split["species"], dtype=np.int32)
    n_clean = int(R_clean.shape[0])
    n_atoms = int(R_clean.shape[1])

    duplicate_every = cfg.get("duplicate_every")
    duplicate_offset = cfg.get("duplicate_offset", 0)
    if duplicate_every is None:
        dup_mask = np.ones(n_clean, dtype=bool)
    else:
        indices = np.arange(n_clean)
        dup_mask = ((indices - duplicate_offset) % duplicate_every) == 0
    n_dup = int(np.sum(dup_mask))

    prior_energy = _build_prior_energy(config, R_clean, id_to_aa=id_to_aa)
    if fitted_params is not None:
        prior_energy.params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in fitted_params.items()}

    compute_batch = int(cfg["compute_batch_size"])

    rng = np.random.RandomState(int(seed) + int(cfg["seed_offset"]))
    noise_levels = cfg["noise_levels"]

    all_blocks: List[Dict[str, np.ndarray]] = []
    for level_idx, level in enumerate(noise_levels):
        sigma = float(level["sigma"])
        attenuation = float(level["attenuation"])
        weight = float(level["weight"])

        R_sel = R_clean[dup_mask]
        F_sel = F_res_clean[dup_mask]
        mask_sel = mask[dup_mask]
        species_sel = species[dup_mask]
        box_sel = np.asarray(split["box"][dup_mask], dtype=np.float32) if "box" in split else None
        n_sel = int(R_sel.shape[0])

        rng_l = np.random.RandomState(int(rng.randint(2**31)) + level_idx)
        eps_raw = rng_l.normal(size=R_sel.shape).astype(np.float32)
        eps = _center_noise_per_structure(R_sel, mask_sel, eps_raw)

        R_noisy, guard_stats = _apply_min_distance_guard(
            R_sel,
            eps,
            mask_sel,
            sigma,
            float(cfg["min_pair_distance"]),
            int(cfg["max_rescale_attempts"]),
            float(cfg["rescale_factor"]),
        )
        if int(guard_stats.get("guarded", 0)) > 0:
            training_logger.info(
                "[NoisedResidual][Guard] level=%s sigma=%.4g min_pair_distance=%.3f "
                "guarded=%d remaining=%d min_before=%.4g min_after=%.4g mean_scale_guarded=%.4g",
                level["name"],
                sigma,
                float(cfg["min_pair_distance"]),
                int(guard_stats["guarded"]),
                int(guard_stats["remaining"]),
                float(guard_stats["min_before"]),
                float(guard_stats["min_after"]),
                float(guard_stats.get("mean_scale_guarded", 1.0)),
            )
        F_noised_ref = (attenuation * F_sel).astype(np.float32)

        f_prior_noisy = _compute_prior_forces(
            R_noisy, mask_sel, species_sel, prior_energy, compute_batch_size=compute_batch
        )
        F_stored = (f_prior_noisy + F_noised_ref).astype(np.float32)

        n_repeat = max(1, int(round(weight)))
        for _ in range(n_repeat):
            block = {
                "R": R_noisy,
                "F": F_stored,
                "mask": mask_sel.copy(),
                "species": species_sel.copy(),
                **({"box": box_sel.copy()} if box_sel is not None else {}),
                "force_loss_mask": np.asarray(
                    split.get("force_loss_mask", mask)[dup_mask], dtype=np.float32
                ),
                "is_noised_frame": np.ones((n_sel,), dtype=np.int32),
                "noise_level_id": np.full((n_sel,), level_idx, dtype=np.int32),
            }
            if "structure_ids" in split:
                block["structure_ids"] = np.asarray(split["structure_ids"], dtype=np.int32)[dup_mask]
            all_blocks.append(block)

    if not all_blocks:
        return split

    clean_block = {
        "R": R_clean,
        "F": np.asarray(F_res_clean, dtype=np.float32),
        "mask": mask.copy(),
        "species": species.copy(),
        **({"box": np.asarray(split["box"], dtype=np.float32)} if "box" in split else {}),
        "force_loss_mask": np.asarray(
            split.get("force_loss_mask", mask), dtype=np.float32
        ),
        "is_noised_frame": np.zeros((n_clean,), dtype=np.int32),
        "noise_level_id": np.full((n_clean,), -1, dtype=np.int32),
    }
    if "structure_ids" in split:
        clean_block["structure_ids"] = np.asarray(split["structure_ids"], dtype=np.int32)
    all_blocks.insert(0, clean_block)

    merged: Dict[str, np.ndarray] = {}
    merge_keys = ["R", "F", "mask", "species", "force_loss_mask", "is_noised_frame", "noise_level_id"]
    if "box" in clean_block:
        merge_keys.append("box")
    for key in merge_keys:
        merged[key] = np.concatenate([b[key] for b in all_blocks], axis=0)
    if "structure_ids" in clean_block:
        merged["structure_ids"] = np.concatenate([b["structure_ids"] for b in all_blocks], axis=0)

    n_total = int(merged["R"].shape[0])
    rng_final = np.random.RandomState(int(split_seed) + int(cfg["seed_offset"]) + 7)
    order = rng_final.permutation(n_total)
    for key in merged:
        merged[key] = merged[key][order]

    force_loss_base = np.asarray(merged["force_loss_mask"], dtype=np.float32)
    safe_n_valid = np.maximum(
        np.sum(force_loss_base > 0, axis=1, keepdims=True), 1.0
    )
    if "force_loss_weights" in split:
        base_w = np.asarray(split["force_loss_weights"], dtype=np.float32)
        weight_blocks = []
        for b in all_blocks:
            if b is all_blocks[0]:
                weight_blocks.append(np.broadcast_to(base_w, (b["R"].shape[0],) + base_w.shape[1:]))
            else:
                weight_blocks.append(np.broadcast_to(base_w[dup_mask], (b["R"].shape[0],) + base_w.shape[1:]))
        base_w_expanded = np.concatenate(weight_blocks, axis=0)
        merged["force_loss_weights"] = (
            base_w_expanded[order] * force_loss_base[order] / safe_n_valid[order]
        ).astype(np.float32)

    merged["n_valid"] = np.asarray(np.sum(merged["mask"] > 0, axis=1), dtype=np.int32)
    merged["n_segments"] = np.ones((n_total,), dtype=np.int32)
    merged["meta_batch_item_id"] = np.arange(n_total, dtype=np.int32)
    merged["meta_capacity"] = np.full((n_total,), n_atoms, dtype=np.int32)
    merged["meta_fill_ratio"] = merged["n_valid"].astype(np.float32) / max(n_atoms, 1)
    merged["meta_n_force_components"] = np.asarray(merged["n_valid"] * 3, dtype=np.int32)
    merged["meta_source_structure_ids"] = np.arange(n_total, dtype=np.int32)[:, None]
    merged["meta_source_structure_n_valid"] = merged["n_valid"][:, None]
    merged["meta_structure_size_min"] = merged["n_valid"]
    merged["meta_structure_size_mean"] = merged["n_valid"].astype(np.float32)
    merged["meta_structure_size_max"] = merged["n_valid"]
    merged["meta_structure_size_std"] = np.zeros((n_total,), dtype=np.float32)

    total_noised = n_total - n_clean
    dup_every = cfg.get("duplicate_every")
    training_logger.info(
        "[NoisedResidual] Attached noised-residual fields: "
        "clean=%d duplicated=%d (every=%s) noised=%d total=%d, levels=%s.",
        n_clean, n_dup, dup_every, total_noised, n_total,
        [(l["name"], l["sigma"], l["attenuation"], l["weight"]) for l in noise_levels],
    )
    return merged


def noised_residual_tiled_split_extension(
    tiled: Dict[str, np.ndarray],
    config,
    id_to_aa: Optional[Dict[int, str]],
    epoch_seed: int,
    fitted_params: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Regenerate noised-residual fields on tiled data rebuilt from expanded source."""
    cfg = noised_residual_config_parsed(config)
    if not cfg["enabled"]:
        return tiled

    is_noised = tiled.get("is_noised_frame", None)
    if is_noised is None:
        return tiled

    clean_mask = is_noised == 0
    if not np.any(clean_mask):
        return tiled

    clean_tiled = {k: v[clean_mask] for k, v in tiled.items() if v.shape[0] == is_noised.shape[0]}
    expanded = attach_noised_residual_fields(
        clean_tiled, config, id_to_aa=id_to_aa,
        seed=epoch_seed, split_seed=epoch_seed,
        fitted_params=fitted_params,
    )

    out = {}
    for key, value in tiled.items():
        out[key] = value
    for key, value in expanded.items():
        out[key] = value
    return out