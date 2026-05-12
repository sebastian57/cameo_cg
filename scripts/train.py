"""Training entry point for CAMEO coarse-grained force-field models."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
from utils.jax_setup import apply_numpy_dataloader_patch
from utils.distributed import initialize_jax_distributed, sync_all_ranks

_DISTRIBUTED = initialize_jax_distributed()
_IS_DISTRIBUTED = _DISTRIBUTED.is_distributed
_RANK = _DISTRIBUTED.rank
_WORLD_SIZE = _DISTRIBUTED.world_size

import copy
import pickle
import time
import json
import jax
import numpy as np
import jax.numpy as jnp
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from chemtrain.data.data_loaders import DataLoaders

from config.manager import ConfigManager
from data.loader import DatasetLoader, BucketedDatasetLoader, build_tiled_dataset
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from training.prior_residual import apply_prior_force_residual_targets, pretrain_prior_for_residual
from training.trainer import Trainer
from training.dsm import add_dsm_noise_fields, dsm_enabled
from training.diagnostics import find_training_log, write_dataset_summary
from training.path_utils import repo_root_from_file, resolve_from_config_or_repo
from export.exporter import ModelExporter
from analysis_tests.visualizer import LossPlotter
from utils.logging import data_logger, model_logger, training_logger, export_logger
import logging


def _sync_all_ranks(tag: str) -> None:
    sync_all_ranks(_DISTRIBUTED, tag)


def _make_export_model(
    config: ConfigManager,
    model: CombinedModel,
    R0,
    box,
    species0,
    id_to_aa,
    n_max: int,
) -> CombinedModel:
    """Build the model variant that should be used for export/evaluation."""
    export_with_priors = config.export_combined_ml_priors_enabled()
    has_prior_config = config.get("model", "priors", default=None) is not None

    if not export_with_priors:
        export_logger.info("Export mode: ML-only (training.export_combined_ml_priors=false)")
        return model

    if model.use_priors:
        export_logger.info("Export mode: ML+priors (model already has priors enabled)")
        return model

    if not has_prior_config:
        export_logger.warning(
            "Export mode requested ML+priors, but model.priors is missing; falling back to ML-only export."
        )
        return model

    export_config = ConfigManager(config.config_path)
    export_config._config = copy.deepcopy(config._config)
    export_config.set("model", "use_priors", True)
    export_config.set("model", "train_priors", False)

    export_logger.info(
        "Export mode: ML+priors (training.export_combined_ml_priors=true; priors re-enabled for export)"
    )
    return CombinedModel(
        config=export_config,
        R0=R0,
        box=box,
        species=species0,
        N_max=n_max,
        id_to_aa=id_to_aa,
        prior_only=False,
    )


def _apply_grad_accum_overrides(config: ConfigManager) -> None:
    """
    Allow per-config gradient accumulation overrides.

    This intentionally overrides values pre-set in the launcher script so
    experiment YAMLs can define their own accumulation behavior.
    """
    grad_acc_steps = config.get("training", "grad_accum_steps", default=None)
    grad_acc_mode = config.get("training", "grad_accum_mode", default=None)

    if grad_acc_steps is not None:
        os.environ["CHEMTRAIN_GRAD_ACCUM_STEPS"] = str(int(grad_acc_steps))
        training_logger.info(
            "[Config] Override CHEMTRAIN_GRAD_ACCUM_STEPS=%s from YAML",
            os.environ["CHEMTRAIN_GRAD_ACCUM_STEPS"],
        )
    if grad_acc_mode is not None and str(grad_acc_mode).strip() != "":
        os.environ["CHEMTRAIN_GRAD_ACCUM_MODE"] = str(grad_acc_mode)
        training_logger.info(
            "[Config] Override CHEMTRAIN_GRAD_ACCUM_MODE=%s from YAML",
            os.environ["CHEMTRAIN_GRAD_ACCUM_MODE"],
        )


def find_latest_checkpoint(checkpoint_dir: Path):
    """
    Find the most recent checkpoint file in a directory.

    Looks for chemtrain checkpoint files (epoch*.pkl or stage_*.pkl)
    and returns the most recently modified one.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    # Look for chemtrain checkpoints (epoch*.pkl) and stage checkpoints (stage_*.pkl)
    checkpoints = list(checkpoint_dir.glob("epoch*.pkl")) + list(checkpoint_dir.glob("stage_*.pkl"))

    # Filter out metadata files
    checkpoints = [p for p in checkpoints if not p.name.endswith(".meta.pkl")]

    if not checkpoints:
        return None

    # Return most recently modified checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def _shuffle_dataset_for_split(dataset: dict, seed: int) -> dict:
    """
    Shuffle frame-ordered dataset arrays before train/validation split.

    Only arrays whose leading dimension matches n_frames are shuffled.
    Non-frame metadata entries (if any) are left untouched.
    """
    if "R" not in dataset:
        return dataset

    n_frames = int(dataset["R"].shape[0])
    if n_frames <= 1:
        return dataset

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n_frames)

    shuffled = {}
    for key, value in dataset.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == n_frames:
            shuffled[key] = np.asarray(value)[permutation]
        else:
            shuffled[key] = value

    data_logger.info(
        "[Split] Shuffled %d frames before train/val split (seed=%d).",
        n_frames,
        seed,
    )
    return shuffled


def _validate_tiled_mode_constraints(config: ConfigManager) -> None:
    """Validate currently supported constraints for tiled training mode."""
    if config.get_batch_mode() != "tiled":
        return
    if config.use_priors():
        raise ValueError(
            "data.batch_mode='tiled' currently supports only pure-ML training. "
            "Set model.use_priors=false for Phase 1 tiled training."
        )
    if not config.tile_train_only_enabled():
        raise ValueError(
            "data.tile_train_only=false is not supported yet. "
            "Tiled runs currently assume training tiles are built only from the training split."
        )


def _validate_prior_residual_mode_constraints(config: ConfigManager) -> None:
    """Validate configuration constraints for residual-prior target mode."""
    if not config.prior_residual_enabled():
        return

    if config.use_priors():
        raise ValueError(
            "training.prior_residual.enabled=true requires model.use_priors=false. "
            "Residual mode trains ML against F_ref - F_prior and does not add prior "
            "energy at runtime."
        )
    if config.train_priors_enabled():
        raise ValueError(
            "training.prior_residual.enabled=true requires model.train_priors=false. "
            "Residual mode assumes frozen prior forces."
        )

    gammas = config.get_gammas()
    gamma_u = float(gammas.get("U", 0.0))
    if abs(gamma_u) > 0.0:
        raise ValueError(
            "training.prior_residual.enabled=true currently requires "
            "training.gammas.U == 0.0."
        )

    if config.get("model", "priors", default=None) is None:
        raise ValueError(
            "training.prior_residual.enabled=true requires model.priors "
            "configuration to be present."
        )


def _pretrain_for_residual_if_needed(
    config: ConfigManager,
    dataset: dict,
    id_to_aa: Optional[Dict[int, str]],
) -> Optional[dict]:
    """
    Run LBFGS prior pretraining against raw F_ref if both pretrain_prior and
    prior_residual are enabled.  Must be called BEFORE _apply_prior_residual_if_enabled
    so that fitting sees the original reference forces, not already-subtracted residuals.

    Returns fitted_params dict (numpy arrays) or None if pretraining was skipped.
    """
    fitted_params = pretrain_prior_for_residual(config, dataset, id_to_aa)
    if fitted_params is not None:
        training_logger.info(
            "[PriorResidual] Prior pretraining complete. "
            "Residuals will be computed with fitted parameters."
        )
    return fitted_params


def _apply_prior_residual_if_enabled(
    *,
    config: ConfigManager,
    dataset: dict,
    dataset_path: Path,
    dataset_tag: str,
    id_to_aa: Optional[Dict[int, str]],
    seed: int,
    max_frames: Optional[int],
    cutoff: float,
    buffer_mult: float,
    park_mult: float,
    fitted_params: Optional[dict] = None,
) -> dict:
    """Apply prior-force residual targets on untiled preprocessed dataset."""
    if not config.prior_residual_enabled():
        return dataset

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    stats = apply_prior_force_residual_targets(
        config=config,
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_tag=dataset_tag,
        id_to_aa=id_to_aa,
        project_root=project_root,
        seed=seed,
        max_frames=max_frames,
        cutoff=cutoff,
        buffer_multiplier=buffer_mult,
        park_multiplier=park_mult,
        fitted_params=fitted_params,
    )
    data_logger.info(
        "[PriorResidual] Applied residual targets for '%s' "
        "(cache_hit=%s, cache_path=%s, mean_prior_norm=%.4f, mean_residual_norm=%.4f).",
        dataset_tag,
        bool(stats.get("cache_hit", False)),
        stats.get("cache_path"),
        float(stats.get("mean_prior_norm", 0.0)),
        float(stats.get("mean_residual_norm", 0.0)),
    )
    return dataset


def _compute_force_loss_weights(split: dict) -> np.ndarray:
    """Build per-particle weights that average force MSE uniformly by structure."""
    mask = np.asarray(split["mask"], dtype=np.float32)
    weights = np.zeros_like(mask, dtype=np.float32)

    if "segment_id" in split:
        segment_id = np.asarray(split["segment_id"], dtype=np.int32)
        for batch_idx in range(mask.shape[0]):
            valid = mask[batch_idx] > 0
            if not np.any(valid):
                continue
            valid_segments = segment_id[batch_idx][valid]
            if valid_segments.size == 0:
                continue
            counts = np.bincount(valid_segments, minlength=int(valid_segments.max()) + 1).astype(np.float32)
            weights[batch_idx, valid] = 1.0 / np.maximum(counts[valid_segments], 1.0)
        return weights

    n_valid = np.sum(mask > 0, axis=1, dtype=np.float32)
    safe_n_valid = np.maximum(n_valid[:, None], 1.0)
    weights = np.where(mask > 0, mask / safe_n_valid, 0.0)
    return np.asarray(weights, dtype=np.float32)


def _estimate_avg_num_neighbors(
    *,
    R: np.ndarray,
    mask: np.ndarray,
    cutoff: float,
    segment_id: Optional[np.ndarray] = None,
    max_samples: int = 32,
    seed: int = 0,
) -> float:
    """
    Estimate the average number of neighbors from a representative sample.

    This avoids seeding Allegro normalization from a single initialization
    frame or tile.
    """
    n_items = int(R.shape[0])
    if n_items <= 0:
        return 0.0

    n_pick = min(n_items, max(1, int(max_samples)))
    if n_pick == n_items:
        sample_indices = np.arange(n_items, dtype=np.int32)
    else:
        rng = np.random.RandomState(seed)
        sample_indices = np.sort(rng.choice(n_items, size=n_pick, replace=False).astype(np.int32))

    cutoff_sq = float(cutoff) ** 2
    total_neighbors = 0.0
    total_valid_nodes = 0

    for idx in sample_indices:
        valid = np.asarray(mask[idx] > 0, dtype=bool)
        if not np.any(valid):
            continue

        coords = np.asarray(R[idx][valid], dtype=np.float32)
        if coords.shape[0] <= 1:
            total_valid_nodes += int(coords.shape[0])
            continue

        diffs = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diffs * diffs, axis=-1, dtype=np.float32)
        within_cutoff = (dist_sq < cutoff_sq) & (dist_sq > 0.0)

        if segment_id is not None:
            seg = np.asarray(segment_id[idx][valid], dtype=np.int32)
            within_cutoff &= seg[:, None] == seg[None, :]

        total_neighbors += float(np.sum(within_cutoff, dtype=np.float64))
        total_valid_nodes += int(coords.shape[0])

    if total_valid_nodes <= 0:
        return 0.0
    return float(total_neighbors / float(total_valid_nodes))


def _get_runtime_allegro_config_section(config: ConfigManager) -> dict:
    """Return the config section that the active Allegro backend will read."""
    model_cfg = config._config.setdefault("model", {})
    ml_model_type = str(config.get_ml_model_type()).strip().lower()

    if ml_model_type in ("allegro_cueq", "allegro_cueq_fast"):
        for key in ("allegro_cueq", "allegro_cuEq"):
            section = model_cfg.get(key)
            if isinstance(section, dict):
                return section

    return model_cfg.setdefault("allegro", {})


def _configure_runtime_avg_num_neighbors(
    config: ConfigManager,
    train_split: dict,
    seed: int,
) -> None:
    """Estimate avg_num_neighbors from sampled training structures/tiles and pin it in config."""
    ml_model_type = str(config.get_ml_model_type()).strip().lower()
    if not ml_model_type.startswith("allegro"):
        return

    estimate = _estimate_avg_num_neighbors(
        R=np.asarray(train_split["R"], dtype=np.float32),
        mask=np.asarray(train_split["mask"], dtype=np.float32),
        cutoff=float(config.get_cutoff()),
        segment_id=np.asarray(train_split["segment_id"], dtype=np.int32)
        if "segment_id" in train_split
        else None,
        max_samples=32,
        seed=seed,
    )

    allegro_cfg = _get_runtime_allegro_config_section(config)
    previous = allegro_cfg.get("avg_num_neighbors")
    allegro_cfg["avg_num_neighbors"] = float(estimate)
    allegro_cfg["avg_num_neighbors_source"] = "dataset_sample"
    batch_mode = config.get_batch_mode()
    training_logger.info(
        "[RuntimeConfig] avg_num_neighbors set from sampled training %s: %.3f "
        "(previous config value: %s, samples<=32).",
        "tiles" if batch_mode == "tiled" else "structures",
        float(estimate),
        previous,
    )


def _attach_batch_metadata(split: dict, sample_ids: np.ndarray) -> dict:
    """Attach uniform profiling metadata to a training split."""
    annotated = dict(split)
    sample_ids = np.asarray(sample_ids, dtype=np.int32)
    n_items = int(sample_ids.shape[0])
    n_valid = np.asarray(np.sum(annotated["mask"] > 0, axis=1), dtype=np.int32)
    capacity = int(annotated["R"].shape[1])

    annotated.setdefault("force_loss_mask", np.asarray(annotated["mask"], dtype=np.float32))
    annotated.setdefault("force_loss_weights", _compute_force_loss_weights(annotated))
    annotated.setdefault("n_valid", n_valid)
    annotated.setdefault("n_segments", np.ones((n_items,), dtype=np.int32))
    annotated.setdefault("meta_batch_item_id", sample_ids)
    annotated.setdefault("meta_capacity", np.full((n_items,), capacity, dtype=np.int32))
    annotated.setdefault("meta_fill_ratio", n_valid.astype(np.float32) / max(capacity, 1))
    annotated.setdefault("meta_n_force_components", np.asarray(n_valid * 3, dtype=np.int32))
    annotated.setdefault("meta_source_structure_ids", sample_ids[:, None])
    annotated.setdefault("meta_source_structure_n_valid", n_valid[:, None])
    annotated.setdefault("meta_structure_size_min", n_valid)
    annotated.setdefault("meta_structure_size_mean", n_valid.astype(np.float32))
    annotated.setdefault("meta_structure_size_max", n_valid)
    annotated.setdefault("meta_structure_size_std", np.zeros((n_items,), dtype=np.float32))
    return annotated


def _build_loader_kwargs(split: dict) -> dict:
    """Build NumpyDataLoader kwargs while preserving auxiliary arrays."""
    loader_kwargs = {"copy": False}
    for key, value in split.items():
        if key in (
            "R",
            "F",
            "mask",
            "species",
            "segment_id",
            "force_loss_mask",
            "force_loss_weights",
            "DSM",
            "dsm_eps",
            "dsm_sigma",
            "dsm_loss_mask",
        ):
            loader_kwargs[key] = value
        elif key.startswith("meta_") or key in ("n_valid", "n_segments"):
            loader_kwargs[key] = value
    return loader_kwargs


def _max_pairwise_distance_percentile(R: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    """Compute a percentile of per-frame maximum valid pair distances."""
    values = []
    for coords, frame_mask in zip(np.asarray(R), np.asarray(mask)):
        valid = np.flatnonzero(frame_mask > 0)
        if valid.shape[0] <= 1:
            continue
        xyz = np.asarray(coords[valid], dtype=np.float64)
        disp = xyz[:, None, :] - xyz[None, :, :]
        dist = np.sqrt(np.sum(disp * disp, axis=-1))
        values.append(float(np.max(dist)))
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _configure_auto_leash_d_safe(config: ConfigManager, train_split: dict) -> None:
    """Fill model.priors.leash.d_safe from training frames when requested."""
    leash_cfg = config.get("model", "priors", "leash", default={}) or {}
    if not isinstance(leash_cfg, dict):
        return
    d_safe = leash_cfg.get("d_safe", None)
    if d_safe is not None:
        try:
            if float(d_safe) > 0.0:
                return
        except (TypeError, ValueError):
            pass
    weights = config.get_prior_weights()
    enabled = bool(leash_cfg.get("enabled", False)) or float(weights.get("leash", 0.0)) != 0.0
    if not enabled:
        return

    factor = float(leash_cfg.get("auto_factor", 1.5))
    percentile = float(leash_cfg.get("auto_percentile", 99.9))
    base = _max_pairwise_distance_percentile(train_split["R"], train_split["mask"], percentile)
    auto_d_safe = float(factor * base)
    config.set("model", "priors", "leash", "d_safe", auto_d_safe)
    data_logger.info(
        "[SafetyPrior] Set model.priors.leash.d_safe=%.6g from %.3f x p%.3f(max_pair_distance=%.6g).",
        auto_d_safe,
        factor,
        percentile,
        base,
    )


def _build_validation_split(dataset: dict, start: int, stop: int) -> dict:
    """Build an untiled validation split with auxiliary loss-mask fields."""
    mask_slice = np.asarray(dataset["mask"][start:stop], dtype=np.float32)
    species_raw = dataset.get("species")
    val_split = {
        "R": np.asarray(dataset["R"][start:stop], dtype=np.float32),
        "F": np.asarray(dataset["F"][start:stop], dtype=np.float32),
        "mask": mask_slice,
        "species": (
            np.asarray(species_raw[start:stop], dtype=np.int32)
            if species_raw is not None
            else np.zeros_like(mask_slice, dtype=np.int32)
        ),
    }
    sample_ids = np.arange(start, stop, dtype=np.int32)
    return _attach_batch_metadata(val_split, sample_ids)


def _build_tiled_validation_split(
    dataset: dict,
    start: int,
    stop: int,
    config: ConfigManager,
    seed: int,
) -> dict:
    """Build a held-out tiled validation split with real validation structures."""
    val_R = np.asarray(dataset["R"][start:stop], dtype=np.float32)
    val_F = np.asarray(dataset["F"][start:stop], dtype=np.float32)
    val_mask = np.asarray(dataset["mask"][start:stop], dtype=np.float32)
    _sp = dataset.get("species")
    val_species = (
        np.asarray(_sp[start:stop], dtype=np.int32)
        if _sp is not None
        else np.zeros_like(val_mask, dtype=np.int32)
    )

    if val_R.shape[0] == 0:
        raise ValueError("Validation split is empty; cannot build tiled validation dataset.")

    structure_ids = np.arange(start, stop, dtype=np.int32)
    tiled = build_tiled_dataset(
        R=val_R,
        F=val_F,
        mask=val_mask,
        species=val_species,
        structure_ids=structure_ids,
        target_beads=config.get_tile_target_beads(),
        bucket_beads=config.get_tile_bucket_beads(),
        target_edges=config.get_tile_target_edges(),
        bucket_edges=config.get_tile_bucket_edges(),
        edge_estimate_scale=config.get_tile_edge_estimate_scale(),
        edge_estimate_mode=config.get_tile_edge_estimate_mode(),
        edge_estimate_cutoff=config.get_tile_edge_estimate_cutoff(),
        shuffle_structures=False,
        sort_by_size=config.tile_sort_by_size_enabled(),
        sort_by_estimated_edges=config.tile_sort_by_estimated_edges_enabled(),
        drop_incomplete=False,
        isolate_large_structures=config.tile_isolate_large_structures_enabled(),
        large_structure_threshold=config.get_tile_large_structure_threshold(),
        large_structure_edge_threshold=config.get_tile_large_structure_edge_threshold(),
        spatial_separation=config.tile_spatial_separation_enabled(),
        structure_gap=config.get_tile_structure_gap(),
        seed=seed,
    )
    tiled = _attach_batch_metadata(tiled, np.arange(tiled["R"].shape[0], dtype=np.int32))
    data_logger.info(
        "[Tiling][Validation] Built %d validation tiles from %d held-out structures "
        "(target_beads=%d, bucket_beads=%s, target_edges=%s, bucket_edges=%s, "
        "edge_mode=%s, edge_cutoff=%s, spatial_separation=%s, structure_gap=%.2f, "
        "sort_by_size=%s, sort_by_estimated_edges=%s).",
        int(tiled["R"].shape[0]),
        int(val_R.shape[0]),
        int(config.get_tile_target_beads()),
        config.get_tile_bucket_beads(),
        config.get_tile_target_edges(),
        config.get_tile_bucket_edges(),
        config.get_tile_edge_estimate_mode(),
        config.get_tile_edge_estimate_cutoff(),
        bool(config.tile_spatial_separation_enabled()),
        float(config.get_tile_structure_gap()),
        bool(config.tile_sort_by_size_enabled()),
        bool(config.tile_sort_by_estimated_edges_enabled()),
    )
    return tiled


def _build_tiled_train_source(dataset: dict, n_train: int) -> dict:
    """Capture the untiled training structures used to rebuild random tiles."""
    mask_slice = np.asarray(dataset["mask"][:n_train], dtype=np.float32)
    _sp = dataset.get("species")
    return {
        "R": np.asarray(dataset["R"][:n_train], dtype=np.float32),
        "F": np.asarray(dataset["F"][:n_train], dtype=np.float32),
        "mask": mask_slice,
        "species": (
            np.asarray(_sp[:n_train], dtype=np.int32)
            if _sp is not None
            else np.zeros_like(mask_slice, dtype=np.int32)
        ),
        "structure_ids": np.arange(n_train, dtype=np.int32),
    }


def _log_train_split_profile(train_split: dict, config: ConfigManager) -> None:
    """Log dataset-level profiling summaries for baseline or tiled training."""
    n_items = int(train_split["R"].shape[0])
    n_valid = np.asarray(train_split["n_valid"], dtype=np.int32)
    n_segments = np.asarray(train_split["n_segments"], dtype=np.int32)
    fill_ratio = np.asarray(train_split["meta_fill_ratio"], dtype=np.float32)
    n_force_components = np.asarray(train_split["meta_n_force_components"], dtype=np.int32)
    unique_structures = np.asarray(train_split["meta_source_structure_ids"], dtype=np.int32)
    unique_structures = unique_structures[unique_structures >= 0]

    if config.get_batch_mode() == "tiled":
        data_logger.info(
            "[TilingProfile] items=%d unique_structures=%d mean_structures_per_tile=%.2f "
            "mean_valid_beads=%.1f fill_ratio(mean/min/max)=%.3f/%.3f/%.3f",
            n_items,
            int(np.unique(unique_structures).size),
            float(np.mean(n_segments)),
            float(np.mean(n_valid)),
            float(np.mean(fill_ratio)),
            float(np.min(fill_ratio)),
            float(np.max(fill_ratio)),
        )
        if config.tile_rebuild_each_epoch_enabled():
            data_logger.info(
                "[TilingProfile] tile compositions are rebuilt every epoch with shuffle=%s and sort_by_size=%s. "
                "Loader shuffling still randomizes tile order within each epoch.",
                bool(config.tile_shuffle_structures_enabled()),
                bool(config.tile_sort_by_size_enabled()),
            )
        else:
            data_logger.info(
                "[TilingProfile] tile compositions are built once before training; loader shuffling changes tile order only. "
                "shuffle=%s sort_by_size=%s.",
                bool(config.tile_shuffle_structures_enabled()),
                bool(config.tile_sort_by_size_enabled()),
            )
    else:
        data_logger.info(
            "[BatchProfile] structures=%d mean_valid_beads=%.1f force_components(total)=%d "
            "fill_ratio(mean/min/max)=%.3f/%.3f/%.3f",
            n_items,
            float(np.mean(n_valid)),
            int(np.sum(n_force_components)),
            float(np.mean(fill_ratio)),
            float(np.min(fill_ratio)),
            float(np.max(fill_ratio)),
        )


def _shard_tiles_by_rank(tiled: dict, rank: int, world_size: int) -> dict:
    """
    Shard tiled dataset across ranks for true data parallelism.
    
    Each rank gets a contiguous subset of tiles to process.
    
    Args:
        tiled: Tiled dataset dict with R, F, mask, species, and metadata arrays
        rank: Current rank (0 to world_size-1)
        world_size: Total number of ranks
    
    Returns:
        Sharded tiled dataset dict containing only this rank's tiles
    """
    n_tiles = int(tiled["R"].shape[0])
    tiles_per_rank = n_tiles // world_size
    start_idx = rank * tiles_per_rank
    end_idx = start_idx + tiles_per_rank if rank < world_size - 1 else n_tiles
    
    # Log the sharding
    data_logger.info(
        "[Tiling][Rank %d] Sharding tiles: rank %d/%d gets tiles %d-%d (out of %d total)",
        rank, rank, world_size, start_idx, end_idx - 1, n_tiles
    )
    
    # Shard all arrays in the tiled dict
    sharded = {}
    for key, value in tiled.items():
        if isinstance(value, np.ndarray) and value.shape[0] == n_tiles:
            # This array has tiles as the first dimension - shard it
            sharded[key] = value[start_idx:end_idx]
        else:
            # Keep non-tiled data as-is
            sharded[key] = value
    
    return sharded


def _build_train_split(
    dataset: dict,
    n_train: int,
    config: ConfigManager,
    seed: int,
) -> dict:
    """Build the training split and optionally tile it."""
    mask_slice = np.asarray(dataset["mask"][:n_train], dtype=np.float32)
    species_raw = dataset.get("species")
    train_split = {
        "R": np.asarray(dataset["R"][:n_train], dtype=np.float32),
        "F": np.asarray(dataset["F"][:n_train], dtype=np.float32),
        "mask": mask_slice,
        "species": (
            np.asarray(species_raw[:n_train], dtype=np.int32)
            if species_raw is not None
            else np.zeros_like(mask_slice, dtype=np.int32)
        ),
    }
    structure_ids = np.arange(n_train, dtype=np.int32)
    if config.get_batch_mode() != "tiled":
        train_split = _attach_batch_metadata(train_split, structure_ids)
        _log_train_split_profile(train_split, config)
        return train_split

    t_tile_build_start = time.perf_counter()
    tiled = build_tiled_dataset(
        R=train_split["R"],
        F=train_split["F"],
        mask=train_split["mask"],
        species=train_split["species"],
        structure_ids=structure_ids,
        target_beads=config.get_tile_target_beads(),
        bucket_beads=config.get_tile_bucket_beads(),
        target_edges=config.get_tile_target_edges(),
        bucket_edges=config.get_tile_bucket_edges(),
        edge_estimate_scale=config.get_tile_edge_estimate_scale(),
        edge_estimate_mode=config.get_tile_edge_estimate_mode(),
        edge_estimate_cutoff=config.get_tile_edge_estimate_cutoff(),
        shuffle_structures=config.tile_shuffle_structures_enabled(),
        sort_by_size=config.tile_sort_by_size_enabled(),
        sort_by_estimated_edges=config.tile_sort_by_estimated_edges_enabled(),
        drop_incomplete=config.tile_drop_incomplete_enabled(),
        isolate_large_structures=config.tile_isolate_large_structures_enabled(),
        large_structure_threshold=config.get_tile_large_structure_threshold(),
        large_structure_edge_threshold=config.get_tile_large_structure_edge_threshold(),
        spatial_separation=config.tile_spatial_separation_enabled(),
        structure_gap=config.get_tile_structure_gap(),
        seed=seed,
    )
    t_tile_build_end = time.perf_counter()
    tiled = _attach_batch_metadata(tiled, np.arange(tiled["R"].shape[0], dtype=np.int32))
    t_tile_meta_end = time.perf_counter()

    # Shard tiles across ranks for data parallelism
    if _WORLD_SIZE > 1:
        tiled = _shard_tiles_by_rank(tiled, _RANK, _WORLD_SIZE)

    data_logger.info(
        "[Tiling] Built %d tiles from %d structures "
        "(target_beads=%d, bucket_beads=%s, target_edges=%s, bucket_edges=%s, "
        "edge_mode=%s, edge_cutoff=%s, "
        "shuffled=%s, sort_by_size=%s, sort_by_estimated_edges=%s, "
        "isolate_large=%s, large_threshold=%s, large_edge_threshold=%s, "
        "spatial_separation=%s, structure_gap=%.2f, drop_incomplete=%s).",
        int(tiled["R"].shape[0]),
        int(train_split["R"].shape[0]),
        int(config.get_tile_target_beads()),
        config.get_tile_bucket_beads(),
        config.get_tile_target_edges(),
        config.get_tile_bucket_edges(),
        config.get_tile_edge_estimate_mode(),
        config.get_tile_edge_estimate_cutoff(),
        bool(config.tile_shuffle_structures_enabled()),
        bool(config.tile_sort_by_size_enabled()),
        bool(config.tile_sort_by_estimated_edges_enabled()),
        bool(config.tile_isolate_large_structures_enabled()),
        config.get_tile_large_structure_threshold(),
        config.get_tile_large_structure_edge_threshold(),
        bool(config.tile_spatial_separation_enabled()),
        float(config.get_tile_structure_gap()),
        bool(config.tile_drop_incomplete_enabled()),
    )
    data_logger.info(
        "[Tiling] Tile shape: R=%s, segment_id=%s, mean_valid=%.1f, "
        "mean_segments=%.2f, mean_est_edges=%.1f, max_est_edges=%.1f",
        tuple(tiled["R"].shape),
        tuple(tiled["segment_id"].shape),
        float(np.mean(tiled["n_valid"])),
        float(np.mean(tiled["n_segments"])),
        float(np.mean(tiled.get("meta_estimated_edges", np.zeros((1,), dtype=np.float32)))),
        float(np.max(tiled.get("meta_estimated_edges", np.zeros((1,), dtype=np.float32)))),
    )
    data_logger.info(
        "[TilingTiming] build_ms=%.3f metadata_ms=%.3f total_ms=%.3f",
        (t_tile_build_end - t_tile_build_start) * 1e3,
        (t_tile_meta_end - t_tile_build_end) * 1e3,
        (t_tile_meta_end - t_tile_build_start) * 1e3,
    )
    _log_train_split_profile(tiled, config)
    return tiled


def main(config_file: str, job_id: str = None, resume_checkpoint: str = None):
    """
    Main training pipeline.

    Args:
        config_file: Path to YAML configuration file
        job_id: SLURM job ID (optional, for logging)
        resume_checkpoint: Path to checkpoint file to resume from (optional)
    """
    # Use global distributed state (initialized at module load)
    is_distributed = _IS_DISTRIBUTED
    rank = _RANK
    world_size = _WORLD_SIZE

    apply_numpy_dataloader_patch()

    config = ConfigManager(config_file)
    _validate_tiled_mode_constraints(config)
    _validate_prior_residual_mode_constraints(config)
    _apply_grad_accum_overrides(config)
    env_neighbor_fmt = os.environ.get("CHEMTRAIN_NEIGHBOR_LIST_FORMAT")
    effective_neighbor_fmt = config.get_neighbor_list_format()
    if env_neighbor_fmt is not None and str(env_neighbor_fmt).strip() != "":
        training_logger.info(
            "[Config] Effective neighbor_list_format=%s (from CHEMTRAIN_NEIGHBOR_LIST_FORMAT=%s)",
            effective_neighbor_fmt,
            env_neighbor_fmt,
        )
    else:
        training_logger.info(
            "[Config] Effective neighbor_list_format=%s (from YAML)",
            effective_neighbor_fmt,
        )

    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Log distributed info
    logging.info("=" * 60)
    logging.info("JAX DISTRIBUTED STATUS")
    logging.info("=" * 60)
    logging.info(f"Distributed: {is_distributed}")
    logging.info(f"Rank: {rank}/{world_size}")
    logging.info(f"Local devices: {jax.local_device_count()}")
    logging.info(f"Global devices: {jax.device_count()}")
    logging.info("=" * 60)

    # ===== Output directories =====
    export_dir = Path(config.get_export_path())
    export_dir.mkdir(parents=True, exist_ok=True)

    model_context = config.get_model_context()
    model_id = config.get_model_id()
    model_name = f"{model_context}_{model_id}"

    # Save config copy
    config_path = export_dir / f"{model_name}_config.yaml"
    config.save(config_path)
    training_logger.info(f"[Config] Saved to: {config_path}")

    logging.info("\n" + "=" * 60)
    logging.info("LOADING DATA")
    logging.info("=" * 60)

    data_path = config.get_data_path()
    data_path_obj = resolve_from_config_or_repo(
        data_path,
        config.config_path,
        repo_root_from_file(__file__),
    )
    if Path(data_path) != data_path_obj:
        data_logger.info("Resolved data.path: %s -> %s", data_path, data_path_obj)

    # Guard against accidentally using bucketed data with the single-dataset path.
    if data_path_obj.is_dir():
        bucket_files = sorted(data_path_obj.glob("bucket_N*.npz"))
        if bucket_files:
            raise ValueError(
                f"data.path points to bucketed directory '{data_path_obj}'. "
                "Use: python scripts/train.py <config.yaml> --multi-protein-dir "
                f"{data_path_obj} [job_id]"
            )

    max_frames = config.get_max_frames()
    seed = config.get_seed()
    loader = DatasetLoader(str(data_path_obj), max_frames=max_frames, seed=seed)

    N_max = loader.N_max
    species0 = loader.species[0]
    n_species_global = int(np.max(loader.species)) + 1

    data_logger.info(f"N_max: {N_max}")
    data_logger.info(f"n_species_global: {n_species_global}")
    data_logger.info(f"Species: {species0}")
    data_logger.info(f"Total frames: {len(loader)}")

    cutoff = config.get_cutoff()
    buffer_mult = config.get_buffer_multiplier()
    park_mult = config.get_park_multiplier()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=buffer_mult,
        park_multiplier=park_mult
    )

    extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)

    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, R_shift)

    # LBFGS prior pretraining must happen against raw F_ref, before residuals are subtracted.
    fitted_prior_params = _pretrain_for_residual_if_needed(config, dataset, loader.id_to_aa)

    dataset = _apply_prior_residual_if_enabled(
        config=config,
        dataset=dataset,
        dataset_path=data_path_obj,
        dataset_tag=data_path_obj.stem,
        id_to_aa=loader.id_to_aa,
        seed=seed,
        max_frames=max_frames,
        cutoff=cutoff,
        buffer_mult=buffer_mult,
        park_mult=park_mult,
        fitted_params=fitted_prior_params,
    )
    write_dataset_summary(
        dataset=dataset,
        id_to_aa=loader.id_to_aa,
        dataset_path=data_path_obj,
        export_dir=export_dir,
    )

    box = extent
    data_logger.info(f"[Preprocessing] Computed box: {jax.device_get(box)}")
    data_logger.info(f"[Preprocessing] R_shift: {jax.device_get(R_shift)}")

    # Spline priors are an add-on mode; LBFGS pretraining is only valid for
    # parametric prior parameters. Disable pretraining if both are enabled.
    if config.use_spline_priors_enabled() and config.pretrain_prior_enabled():
        config.set_pretrain_prior_enabled(False)
        training_logger.warning(
            "Both spline priors and training.pretrain_prior=true were configured. "
            "Disabling prior pretraining because LBFGS is only supported for "
            "parametric priors."
        )

    # ===== Setup data loaders =====
    logging.info("\n" + "=" * 60)
    logging.info("PREPARING TRAINING")
    logging.info("=" * 60)

    split_seed = int(config.get_seed())
    dataset = _shuffle_dataset_for_split(dataset, seed=split_seed)

    val_fraction = config.get_val_fraction()
    N_train = int(np.round(len(dataset["R"]) * (1 - val_fraction)))
    N_val = len(dataset["R"]) - N_train

    # Check if validation set is too small for batching
    batch_per_device = config.get_batch_per_device()
    n_devices = jax.local_device_count()
    min_val_samples = batch_per_device * n_devices

    if N_val < min_val_samples:
        data_logger.warning(f"[Split] Validation set ({N_val} samples) is too small for batch size "
                          f"({batch_per_device} per device * {n_devices} devices = {min_val_samples})")
        data_logger.warning(f"[Split] Using training data for validation")
        N_train = len(dataset["R"])
        N_val = 0

    data_logger.info(f"[Split] Training: {N_train}, Validation: {N_val if N_val > 0 else 'using train'}")

    tiled_train_source = None
    if config.get_batch_mode() == "tiled":
        tiled_train_source = _build_tiled_train_source(dataset, N_train)

    train_split = _build_train_split(
        dataset=dataset,
        n_train=N_train,
        config=config,
        seed=split_seed,
    )
    _configure_auto_leash_d_safe(
        config, tiled_train_source if tiled_train_source is not None else train_split
    )
    if dsm_enabled(config):
        train_split = add_dsm_noise_fields(train_split, config, seed=split_seed)
    _configure_runtime_avg_num_neighbors(
        config,
        tiled_train_source if tiled_train_source is not None else train_split,
        split_seed,
    )
    config.save(config_path)
    training_logger.info(f"[Config] Updated runtime config saved to: {config_path}")

    # ===== Initialize model =====
    # Important for tiled mode: initialize neighbor list state with the same
    # particle axis as training batches to avoid shape mismatches at neighbor_update().
    logging.info("\n" + "=" * 60)
    logging.info("INITIALIZING MODEL")
    logging.info("=" * 60)
    if config.get_batch_mode() == "tiled":
        R0 = train_split["R"][0]
        species0 = train_split["species"][0]
        init_mask0 = train_split["mask"][0]
        N_max = int(train_split["R"].shape[1])
        data_logger.info(
            "[Tiling] Model initialization uses tiled particle axis N_max=%d.",
            N_max,
        )
    else:
        R0 = dataset["R"][0]
        species0 = loader.species[0]
        init_mask0 = dataset["mask"][0]
        N_max = int(loader.N_max)

    model = CombinedModel(
        config=config,
        R0=R0,
        box=box,
        species=species0,
        N_max=N_max,
        init_mask=init_mask0,
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=config.prior_only_enabled()
    )

    model_logger.info(f"Initialized: {model}")

    train_loader_kwargs = _build_loader_kwargs(train_split)
    train_loader = NumpyDataLoader(**train_loader_kwargs)

    # Use training data for validation only when the held-out split is disabled
    # or too small for batching. Tiled mode now builds a separate packed
    # validation set so model selection sees held-out structures.
    if config.get_batch_mode() == "tiled":
        if N_val == 0 or val_fraction == 0.0:
            data_logger.warning(
                "[Tiling] No held-out validation structures available; using training loader for validation."
            )
            val_loader = train_loader
        else:
            val_split = _build_tiled_validation_split(
                dataset=dataset,
                start=N_train,
                stop=N_train + N_val,
                config=config,
                seed=split_seed,
            )
            if dsm_enabled(config):
                val_split = add_dsm_noise_fields(val_split, config, seed=split_seed + 100000)
            val_loader = NumpyDataLoader(**_build_loader_kwargs(val_split))
    elif N_val == 0 or val_fraction == 0.0:
        val_loader = train_loader
    else:
        val_split = _build_validation_split(dataset, N_train, N_train + N_val)
        if dsm_enabled(config):
            val_split = add_dsm_noise_fields(val_split, config, seed=split_seed + 100000)
        val_loader = NumpyDataLoader(**_build_loader_kwargs(val_split))

    loaders = DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None
    )

    # ===== Training =====
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING")
    logging.info("=" * 60)

    # Prepare training data dict for prior pre-training (avoid _chains access)
    train_data = {
        "R": jnp.asarray(train_split["R"]),
        "F": jnp.asarray(train_split["F"]),
        "mask": jnp.asarray(train_split["mask"]),
        "species": jnp.asarray(train_split["species"]),
        "force_loss_mask": jnp.asarray(train_split["force_loss_mask"]),
        "force_loss_weights": jnp.asarray(train_split["force_loss_weights"]),
    }
    if "segment_id" in train_split:
        train_data["segment_id"] = jnp.asarray(train_split["segment_id"])
    for key, value in train_split.items():
        if key.startswith("meta_") or key in ("n_valid", "n_segments"):
            train_data[key] = jnp.asarray(value)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_data=train_data,
        tiled_train_source=tiled_train_source,
    )

    # Handle resume from checkpoint
    resolved_checkpoint = None
    if resume_checkpoint:
        if resume_checkpoint == "auto":
            # Auto-find latest checkpoint
            checkpoint_dir = Path(config.get_checkpoint_path())
            resolved_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if resolved_checkpoint:
                training_logger.info(f"\nAuto-resume: found checkpoint {resolved_checkpoint}")
            else:
                training_logger.info("\nAuto-resume: no checkpoints found, starting fresh")
        else:
            # Use specified checkpoint
            resolved_checkpoint = resume_checkpoint
            training_logger.info(f"\nResuming from checkpoint: {resolved_checkpoint}")

    # Run full pipeline (prior pretrain + stage1 + stage2)
    results = trainer.train_full_pipeline(resume_from=resolved_checkpoint)

    # Save checkpoint after training (rank 0 only to avoid write races)
    if rank == 0:
        checkpoint_path = export_dir / f"{model_name}_checkpoint.pkl"
        trainer.save_checkpoint(checkpoint_path, metadata={"job_id": job_id, "results": results})

    training_logger.info("\nComplete!")
    for stage, result in results.items():
        training_logger.info(f"  {stage}: {result}")

    best_params = trainer.get_best_params()

    # Keep ranks in sync before rank-0-only post-processing to avoid distributed
    # shutdown barrier mismatches.
    _sync_all_ranks("post_train_ready_for_export")

    # In multi-process jobs, only rank 0 should write artifacts and perform
    # post-training export to avoid concurrent file writes to the same path.
    if rank != 0:
        training_logger.info(
            "Rank %d: skipping diagnostics/export/plot (rank 0 only); waiting for rank 0 post-processing.",
            rank,
        )
        _sync_all_ranks("rank0_postprocessing_done")
        return

    # Diagnostics are intentionally skipped to keep post-training fast:
    # export artifacts + loss plots only.

    # ===== Export =====
    logging.info("\n" + "=" * 60)
    logging.info("EXPORTING MODEL")
    logging.info("=" * 60)

    mlir_path = export_dir / f"{model_name}.mlir"
    export_model = _make_export_model(
        config=config,
        model=model,
        R0=R0,
        box=box,
        species0=species0,
        id_to_aa=loader.id_to_aa,
        n_max=model.N_max,
    )
    exporter = ModelExporter.from_combined_model(
        model=export_model,
        params=best_params,
        box=box,
        species=species0
    )
    exporter.export_to_file(mlir_path)
    export_logger.info(f"MLIR: {mlir_path}")

    # Save parameters as pickle
    params_path = export_dir / f"{model_name}_params.pkl"
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    export_logger.info(f"Parameters: {params_path}")

    # ===== Plotting =====
    logging.info("\n" + "=" * 60)
    logging.info("GENERATING PLOTS")
    logging.info("=" * 60)

    log_file = find_training_log(
        job_id=job_id,
        output_dir=Path(config.get_output_dir()),
        slurm_dir=Path(config.get_slurm_dir()),
        export_dir=export_dir,
    )

    if log_file is not None:
        plotter = LossPlotter(str(log_file), config=config)
        plotter.parse_log()

        plot_path = export_dir / f"loss_curve_{job_id}_{model_id}.png"
        plotter.plot(plot_path)
        logging.info(f"[Plot] Loss curve: {plot_path}")

        data_path = export_dir / f"loss_data_{job_id}_{model_id}.txt"
        plotter.save_loss_data(data_path)
        logging.info(f"[Plot] Loss data: {data_path}")
    else:
        logging.warning(
            f"[Plot] Log file not found for job_id={job_id} "
            f"(checked paths.output_dir/paths.slurm_dir/export_dir parent, outputs/, and cwd)"
        )

    logging.info("\n" + "=" * 60)
    logging.info("TRAINING PIPELINE COMPLETE")
    logging.info("=" * 60)
    _sync_all_ranks("rank0_postprocessing_done")


def main_multi_protein(config_file: str, bucket_dir: str, job_id: str = None):
    """
    Multi-protein training on bucketed datasets.

    Trains sequentially over all bucket_N*.npz files found in bucket_dir,
    transferring parameters from each bucket to the next (warm-start).
    This ensures all proteins of different lengths contribute to the model
    while each bucket uses its own (smaller) N_max, reducing wasted compute.

    Args:
        config_file:  Path to YAML config (shared across all buckets)
        bucket_dir:   Directory containing bucket_N*.npz files from
                      run_pipeline.py --bucket_boundaries or --n_buckets
        job_id:       SLURM job ID (optional, for logging/file naming)
    """
    is_distributed = _IS_DISTRIBUTED
    rank = _RANK

    apply_numpy_dataloader_patch()

    config = ConfigManager(config_file)
    _validate_tiled_mode_constraints(config)
    _validate_prior_residual_mode_constraints(config)
    _apply_grad_accum_overrides(config)
    env_neighbor_fmt = os.environ.get("CHEMTRAIN_NEIGHBOR_LIST_FORMAT")
    effective_neighbor_fmt = config.get_neighbor_list_format()
    if env_neighbor_fmt is not None and str(env_neighbor_fmt).strip() != "":
        training_logger.info(
            "[Config] Effective neighbor_list_format=%s (from CHEMTRAIN_NEIGHBOR_LIST_FORMAT=%s)",
            effective_neighbor_fmt,
            env_neighbor_fmt,
        )
    else:
        training_logger.info(
            "[Config] Effective neighbor_list_format=%s (from YAML)",
            effective_neighbor_fmt,
        )

    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

    export_dir = Path(config.get_export_path())
    export_dir.mkdir(parents=True, exist_ok=True)

    model_context = config.get_model_context()
    model_id = config.get_model_id()
    model_name = f"{model_context}_{model_id}"

    config_path = export_dir / f"{model_name}_config.yaml"
    config.save(config_path)
    training_logger.info(f"[Config] Saved to: {config_path}")

    bucketed = BucketedDatasetLoader(bucket_dir, max_frames=config.get_max_frames(),
                                     seed=config.get_seed())
    training_logger.info(bucketed.summary())
    global_n_species = max(int(np.max(dl.species)) + 1 for _, dl in bucketed.buckets)
    training_logger.info(f"Global species cardinality across buckets: {global_n_species}")

    cutoff = config.get_cutoff()
    buffer_mult = config.get_buffer_multiplier()
    park_mult = config.get_park_multiplier()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff, buffer_multiplier=buffer_mult, park_multiplier=park_mult
    )

    prev_params = None
    all_results = {}

    for bucket_idx, (n_max, loader) in enumerate(bucketed.buckets):
        training_logger.info(
            f"\n{'=' * 60}\n"
            f"Bucket {bucket_idx + 1}/{bucketed.n_buckets}: "
            f"{loader.npz_path.name}  (N_max={n_max}, frames={len(loader)})\n"
            f"{'=' * 60}"
        )

        extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
        dataset = loader.get_all()
        dataset["R"] = preprocessor.center_and_park(
            dataset["R"], dataset["mask"], extent, R_shift
        )

        if config.pretrain_prior_enabled() and config.prior_residual_enabled():
            training_logger.warning(
                "[PriorResidual] pretrain_prior=true is not supported in multi-protein "
                "bucketed mode (different topologies per bucket). "
                "Residuals will use initial config prior parameters."
            )

        dataset = _apply_prior_residual_if_enabled(
            config=config,
            dataset=dataset,
            dataset_path=loader.npz_path,
            dataset_tag=loader.npz_path.stem,
            id_to_aa=loader.id_to_aa,
            seed=config.get_seed(),
            max_frames=config.get_max_frames(),
            cutoff=cutoff,
            buffer_mult=buffer_mult,
            park_mult=park_mult,
        )
        box = extent

        split_seed = int(config.get_seed()) + int(bucket_idx)
        dataset = _shuffle_dataset_for_split(dataset, seed=split_seed)

        val_fraction = config.get_val_fraction()
        N_train = int(np.round(len(dataset["R"]) * (1 - val_fraction)))
        N_val = len(dataset["R"]) - N_train
        batch_per_device = config.get_batch_per_device()
        n_devices = jax.local_device_count()
        min_val_samples = batch_per_device * n_devices
        if N_val < min_val_samples:
            N_train = len(dataset["R"])
            N_val = 0

        tiled_train_source = None
        if config.get_batch_mode() == "tiled":
            tiled_train_source = _build_tiled_train_source(dataset, N_train)

        train_split = _build_train_split(
            dataset=dataset,
            n_train=N_train,
            config=config,
            seed=split_seed,
        )
        _configure_auto_leash_d_safe(
            config, tiled_train_source if tiled_train_source is not None else train_split
        )
        if dsm_enabled(config):
            train_split = add_dsm_noise_fields(train_split, config, seed=split_seed)
        _configure_runtime_avg_num_neighbors(
            config,
            tiled_train_source if tiled_train_source is not None else train_split,
            split_seed,
        )
        config.save(config_path)
        training_logger.info(
            "[Config][Bucket %d] Updated runtime config saved to: %s",
            bucket_idx,
            config_path,
        )

        if config.get_batch_mode() == "tiled":
            R0 = train_split["R"][0]
            species0 = train_split["species"][0]
            init_mask0 = train_split["mask"][0]
            model_n_max = int(train_split["R"].shape[1])
            data_logger.info(
                "[Tiling][Bucket %d] Model initialization uses tiled particle axis N_max=%d.",
                bucket_idx,
                model_n_max,
            )
        else:
            R0 = dataset["R"][0]
            species0 = loader.species[0]
            init_mask0 = dataset["mask"][0]
            model_n_max = int(n_max)

        model = CombinedModel(
            config=config, R0=R0, box=box, species=species0, N_max=model_n_max,
            init_mask=init_mask0,
            n_species_override=global_n_species,
            id_to_aa=loader.id_to_aa,
            prior_only=config.prior_only_enabled()
        )
        model_logger.info(f"Bucket {bucket_idx}: {model}")

        train_loader_kwargs = _build_loader_kwargs(train_split)
        train_loader = NumpyDataLoader(**train_loader_kwargs)
        if config.get_batch_mode() == "tiled":
            if N_val == 0 or val_fraction == 0.0:
                val_loader = train_loader
                data_logger.warning(
                    "[Tiling][Bucket %d] No held-out validation structures available; using training loader for validation.",
                    bucket_idx,
                )
            else:
                val_split = _build_tiled_validation_split(
                    dataset=dataset,
                    start=N_train,
                    stop=N_train + N_val,
                    config=config,
                    seed=split_seed,
                )
                if dsm_enabled(config):
                    val_split = add_dsm_noise_fields(val_split, config, seed=split_seed + 100000)
                val_loader = NumpyDataLoader(**_build_loader_kwargs(val_split))
        else:
            if N_val > 0:
                val_split = _build_validation_split(dataset, N_train, N_train + N_val)
                if dsm_enabled(config):
                    val_split = add_dsm_noise_fields(val_split, config, seed=split_seed + 100000)
                val_loader = NumpyDataLoader(**_build_loader_kwargs(val_split))
            else:
                val_loader = train_loader

        train_data = {
            "R": jnp.asarray(train_split["R"]),
            "F": jnp.asarray(train_split["F"]),
            "mask": jnp.asarray(train_split["mask"]),
            "species": jnp.asarray(train_split["species"]),
            "force_loss_mask": jnp.asarray(train_split["force_loss_mask"]),
            "force_loss_weights": jnp.asarray(train_split["force_loss_weights"]),
        }
        if "segment_id" in train_split:
            train_data["segment_id"] = jnp.asarray(train_split["segment_id"])
        for key, value in train_split.items():
            if key.startswith("meta_") or key in ("n_valid", "n_segments"):
                train_data[key] = jnp.asarray(value)

        trainer = Trainer(
            model=model, config=config,
            train_loader=train_loader, val_loader=val_loader,
            train_data=train_data,
            tiled_train_source=tiled_train_source,
        )

        if prev_params is not None:
            trainer.params = prev_params
            trainer.best_params = prev_params
            training_logger.info(
                f"  Bucket {bucket_idx}: warm-start from previous bucket params"
            )

        results = trainer.train_full_pipeline()
        prev_params = trainer.get_best_params()
        all_results[loader.npz_path.stem] = results

        training_logger.info(f"Bucket {bucket_idx} done: {results}")

    final_n_max, final_loader = bucketed.buckets[-1]
    final_species0 = species0

    training_logger.info(f"\nExporting model (N_max={final_n_max})…")

    _sync_all_ranks("multi_protein_post_train_ready_for_export")

    # Only rank 0 writes exported artifacts/checkpoints.
    if rank == 0:
        checkpoint_path = export_dir / f"{model_name}_checkpoint.pkl"
        trainer.save_checkpoint(checkpoint_path, metadata={"job_id": job_id, "results": all_results})

        mlir_path = export_dir / f"{model_name}.mlir"
        export_model = _make_export_model(
            config=config,
            model=model,
            R0=R0,
            box=box,
            species0=final_species0,
            id_to_aa=loader.id_to_aa,
            n_max=final_n_max,
        )
        exporter = ModelExporter.from_combined_model(
            model=export_model, params=prev_params,
            box=box, species=final_species0
        )
        exporter.export_to_file(mlir_path)
        export_logger.info(f"MLIR: {mlir_path}")

        params_path = export_dir / f"{model_name}_params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(prev_params, f)
        export_logger.info(f"Parameters: {params_path}")
    else:
        training_logger.info(
            "Rank %d: skipping final export/checkpoint write (rank 0 only); waiting for rank 0 export.",
            rank,
        )

    _sync_all_ranks("multi_protein_rank0_export_done")

    training_logger.info("\nMulti-protein training complete.")
    for bucket_name, result in all_results.items():
        training_logger.info(f"  {bucket_name}: {result}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error(
            "Usage:\n"
            "  python train.py <config.yaml> [job_id] [--resume checkpoint.pkl]\n"
            "  python train.py <config.yaml> --multi-protein-dir <bucket_dir> [job_id]"
        )
        sys.exit(1)

    config_file = sys.argv[1]
    job_id = None
    resume_checkpoint = None
    multi_protein_dir = None

    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--resume":
            if i + 1 < len(sys.argv):
                resume_checkpoint = sys.argv[i + 1]
                i += 2
            else:
                logging.error("--resume requires a checkpoint path")
                sys.exit(1)
        elif arg == "--multi-protein-dir":
            if i + 1 < len(sys.argv):
                multi_protein_dir = sys.argv[i + 1]
                i += 2
            else:
                logging.error("--multi-protein-dir requires a directory path")
                sys.exit(1)
        else:
            if job_id is None:
                job_id = arg
            i += 1

    if multi_protein_dir is not None:
        main_multi_protein(config_file, multi_protein_dir, job_id)
    else:
        main(config_file, job_id, resume_checkpoint)
