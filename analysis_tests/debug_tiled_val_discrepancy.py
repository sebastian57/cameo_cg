#!/usr/bin/env python3
"""
Debug: tiled val_loss vs individual val MSE discrepancy.

The training run 1pro_4zohB01_320_aggforce_fm_residual_500ep shows:
  - Training val_loss (tiled evaluation): 116 -> 83 -> 143 (diverges)
  - Post-training individual analysis on same val frames: RMSE=6.6, MSE~43.6

These should agree but differ by ~3.3x at epoch 500. This script isolates the cause.

Stages:
  1. Reproduce both metrics from the final checkpoint.
  2. Per-structure force comparison: tiled vs individual forces.
  3. Neighbor list edge counts and overflow check per val tile.
  4. Checkpoint sweep to reconstruct both curves.
  5. Force decomposition: ML vs prior contribution on raw val frames.
     Bypasses the prior-residual pipeline entirely — loads raw F_ref from
     NPZ, computes F_ml and F_prior separately via CombinedModel VJP, and
     reports: RMS per component, RMSE(combined vs F_ref), RMSE(ML vs residual),
     Pearson correlations. Directly tests whether F_ml is substantial.

Usage:
  python debug_tiled_val_discrepancy.py \\
    <config_yaml> \\
    --checkpoint <epoch500_pkl> \\
    [--checkpoint-dir <ckpt_dir>  # for stage 4 sweep] \\
    --output-dir <out_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.jax_setup import apply_jax_compat_shims as _apply_jax_compat_shims
_apply_jax_compat_shims()

import jax
import jax.numpy as jnp

from config.manager import ConfigManager
from data.loader import DatasetLoader, build_tiled_dataset
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from training.trainer import valid_component_mse


# ---------------------------------------------------------------------------
# Helpers: dataset loading (mirrors train.py pipeline)
# ---------------------------------------------------------------------------

def _shuffle_dataset(dataset: dict, seed: int) -> dict:
    """Shuffle frame-ordered dataset arrays (mirrors _shuffle_dataset_for_split in train.py)."""
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
    return shuffled


def _load_dataset_with_residuals(config: ConfigManager):
    """Load NPZ, preprocess coords, apply prior residual, shuffle for split.

    Returns dataset in SHUFFLED order (matching training pipeline).
    First n_train rows = train, remaining = val.
    """
    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(), seed=config.get_seed())
    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset: Dict[str, Any] = {
        "R": preprocessor.center_and_park(loader.R, loader.mask, extent, r_shift),
        "F": np.asarray(loader.F, dtype=np.float32),
        "mask": np.asarray(loader.mask, dtype=np.float32),
        "species": np.asarray(loader.species, dtype=np.int32),
    }
    box = np.asarray(extent, dtype=np.float32)

    # Apply prior residual BEFORE shuffle (same as training)
    if config.prior_residual_enabled():
        # Temporarily disable force_recompute to use cache for speed
        orig_recompute = config.get("training", "prior_residual", "force_recompute", default=False)
        if orig_recompute:
            print("[debug_tiled_val] prior_residual.force_recompute=true in config; "
                  "using cache instead (set to false temporarily).")
            config.set("training", "prior_residual", "force_recompute", False)
        from training.prior_residual import apply_prior_force_residual_targets
        stats = apply_prior_force_residual_targets(
            config=config,
            dataset=dataset,
            dataset_path=data_path,
            dataset_tag=data_path.stem,
            id_to_aa=loader.id_to_aa,
            project_root=PROJECT_ROOT,
            seed=int(config.get_seed()),
            max_frames=config.get_max_frames(),
            cutoff=float(config.get_cutoff()),
            buffer_multiplier=float(config.get_buffer_multiplier()),
            park_multiplier=float(config.get_park_multiplier()),
        )
        if orig_recompute:
            config.set("training", "prior_residual", "force_recompute", True)
        print(f"[debug_tiled_val] prior residual: cache_hit={stats.get('cache_hit')}, "
              f"mean_prior_norm={stats.get('mean_prior_norm', float('nan')):.4f}, "
              f"mean_residual_norm={stats.get('mean_residual_norm', float('nan')):.4f}")

    # Shuffle AFTER prior residual (same order as training)
    split_seed = int(config.get_seed())
    dataset = _shuffle_dataset(dataset, seed=split_seed)
    print(f"[debug_tiled_val] dataset shuffled with seed={split_seed}, n_frames={dataset['R'].shape[0]}")

    return dataset, loader, box


# ---------------------------------------------------------------------------
# Helpers: build val tiles (mirrors _build_tiled_validation_split in train.py)
# ---------------------------------------------------------------------------

def _build_val_tiles(dataset: dict, val_start: int, val_stop: int, config: ConfigManager) -> dict:
    """Rebuild the fixed val tiles exactly as done in training."""
    val_R = dataset["R"][val_start:val_stop]
    val_F = dataset["F"][val_start:val_stop]
    val_mask = dataset["mask"][val_start:val_stop]
    val_species = dataset["species"][val_start:val_stop]
    structure_ids = np.arange(val_start, val_stop, dtype=np.int32)

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
        seed=int(config.get_seed()),
    )
    return tiled


# ---------------------------------------------------------------------------
# Helpers: build train tiles (to get R0 for model initialization, like training)
# ---------------------------------------------------------------------------

def _build_train_tiles(dataset: dict, n_train: int, config: ConfigManager) -> dict:
    """Build training tiles to get R0 for model init (same as training code)."""
    train_R = dataset["R"][:n_train]
    train_F = dataset["F"][:n_train]
    train_mask = dataset["mask"][:n_train]
    train_species = dataset["species"][:n_train]
    structure_ids = np.arange(n_train, dtype=np.int32)

    tiled = build_tiled_dataset(
        R=train_R,
        F=train_F,
        mask=train_mask,
        species=train_species,
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
        seed=int(config.get_seed()),
    )
    return tiled


# ---------------------------------------------------------------------------
# Helpers: load params from checkpoint
# ---------------------------------------------------------------------------

def _load_params(checkpoint_path: Path) -> dict:
    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)
    params = payload
    if isinstance(payload, dict):
        if "params" in payload and isinstance(payload["params"], dict):
            params = payload["params"]
        elif "trainer_state" in payload and isinstance(payload["trainer_state"], dict):
            if "params" in payload["trainer_state"]:
                params = payload["trainer_state"]["params"]
    if isinstance(params, dict) and "ml" not in params and "allegro" in params:
        params = dict(params)
        params["ml"] = params["allegro"]
    return params


# ---------------------------------------------------------------------------
# Stage 1: Reproduce both metrics at one checkpoint
# ---------------------------------------------------------------------------

def stage1_reproduce_metrics(
    config: ConfigManager,
    dataset: dict,
    loader: DatasetLoader,
    box: np.ndarray,
    val_start: int,
    val_stop: int,
    train_tiles: dict,
    val_tiles: dict,
    params: dict,
    n_species_global: int,
) -> Dict[str, Any]:
    """Compute tiled val_mse and individual val_mse from the same params."""
    print("\n" + "=" * 60)
    print("STAGE 1: Reproduce both metrics")
    print("=" * 60)

    # ---- Individual model ----
    val_frames = dataset["R"][val_start:val_stop]
    val_F_ref = dataset["F"][val_start:val_stop]
    val_mask = dataset["mask"][val_start:val_stop]
    val_species = dataset["species"][val_start:val_stop]
    n_val = val_frames.shape[0]

    config_indiv = ConfigManager(str(config.config_path))
    config_indiv.set("model", "use_priors", False)
    config_indiv.set("model", "train_priors", False)

    R0_indiv = val_frames[0]
    model_indiv = CombinedModel(
        config=config_indiv,
        R0=jnp.asarray(R0_indiv, dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(val_species[0], dtype=jnp.int32),
        N_max=int(loader.N_max),
        init_mask=jnp.asarray(val_mask[0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    def _forces_single_indiv(R_i, mask_i, species_i):
        def energy_fn(R_):
            return model_indiv.compute_energy(params, R_, mask_i, species_i, None)
        return -jax.grad(energy_fn)(R_i)

    batched_forces_indiv = jax.jit(jax.vmap(_forces_single_indiv))

    R_val_jax = jnp.asarray(val_frames, dtype=jnp.float32)
    mask_val_jax = jnp.asarray(val_mask, dtype=jnp.float32)
    species_val_jax = jnp.asarray(val_species, dtype=jnp.int32)

    print(f"  Computing individual forces for {n_val} val frames...")
    F_val_pred_indiv = np.asarray(batched_forces_indiv(R_val_jax, mask_val_jax, species_val_jax))
    indiv_sq = (F_val_pred_indiv - val_F_ref) ** 2
    indiv_mask3 = (val_mask > 0)[..., None]
    indiv_mse = float(np.sum(indiv_sq * indiv_mask3) / max(np.sum(indiv_mask3) * 3, 1.0))
    indiv_rmse = float(np.sqrt(indiv_mse))
    print(f"  Individual val MSE = {indiv_mse:.6f}  RMSE = {indiv_rmse:.6f}")

    # ---- Tiled model ----
    config_tiled = ConfigManager(str(config.config_path))
    config_tiled.set("model", "use_priors", False)
    config_tiled.set("model", "train_priors", False)

    R0_tiled = train_tiles["R"][0]
    mask0_tiled = train_tiles["mask"][0]
    species0_tiled = train_tiles["species"][0]
    N_max_tiled = int(train_tiles["R"].shape[1])

    model_tiled = CombinedModel(
        config=config_tiled,
        R0=jnp.asarray(R0_tiled, dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(species0_tiled, dtype=jnp.int32),
        N_max=N_max_tiled,
        init_mask=jnp.asarray(mask0_tiled, dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    n_tiles = val_tiles["R"].shape[0]
    tile_losses = []
    tile_edge_counts = []
    tile_nbrs_capacities = []

    def _forces_single_tiled(R_i, mask_i, species_i, seg_i):
        def energy_fn(R_):
            return model_tiled.compute_energy(params, R_, mask_i, species_i, None, segment_id=seg_i)
        return -jax.grad(energy_fn)(R_i)

    force_fn_tiled = jax.jit(_forces_single_tiled)

    print(f"  Computing tiled forces for {n_tiles} val tiles...")
    for tile_idx in range(n_tiles):
        R_tile = jnp.asarray(val_tiles["R"][tile_idx], dtype=jnp.float32)
        F_tile_ref = jnp.asarray(val_tiles["F"][tile_idx], dtype=jnp.float32)
        mask_tile = jnp.asarray(val_tiles["mask"][tile_idx], dtype=jnp.float32)
        species_tile = jnp.asarray(val_tiles["species"][tile_idx], dtype=jnp.int32)
        seg_tile = jnp.asarray(val_tiles["segment_id"][tile_idx], dtype=jnp.int32)

        F_tile_pred = force_fn_tiled(R_tile, mask_tile, species_tile, seg_tile)
        tile_loss = float(valid_component_mse(
            F_tile_pred, F_tile_ref, mask_tile
        ))
        tile_losses.append(tile_loss)

        # Count actual edges in neighbor list for overflow detection
        nbrs = model_tiled.ml_model.nneigh_fn.update(
            R_tile,
            model_tiled.ml_model.nbrs_init,
            mask=(mask_tile > 0).astype(jnp.bool_),
        )
        # Count valid (non-self-padding) entries in sparse neighbor list
        if hasattr(nbrs, "idx"):
            idx_arr = np.asarray(nbrs.idx)
            N = idx_arr.shape[0]
            valid_edges = int(np.sum(idx_arr < N))
            capacity = idx_arr.shape[-1] if idx_arr.ndim > 1 else idx_arr.shape[0]
            tile_edge_counts.append(valid_edges)
            tile_nbrs_capacities.append(int(np.prod(idx_arr.shape)))
        else:
            tile_edge_counts.append(-1)
            tile_nbrs_capacities.append(-1)

        if (tile_idx + 1) % 5 == 0 or tile_idx == n_tiles - 1:
            print(f"    tile {tile_idx+1}/{n_tiles}: loss={tile_loss:.4f}, "
                  f"edges={tile_edge_counts[-1]}, capacity={tile_nbrs_capacities[-1]}")

    tiled_val_mse = float(np.mean(tile_losses))
    tiled_val_rmse = float(np.sqrt(max(tiled_val_mse, 0.0)))
    print(f"\n  Tiled val MSE = {tiled_val_mse:.6f}  RMSE = {tiled_val_rmse:.6f}")
    print(f"  Discrepancy ratio (tiled/individual) = {tiled_val_mse / max(indiv_mse, 1e-12):.4f}")

    # Did any tile overflow?
    overflow_tiles = [i for i, (e, c) in enumerate(zip(tile_edge_counts, tile_nbrs_capacities))
                      if c > 0 and e >= c - N_max_tiled]
    if overflow_tiles:
        print(f"  !! Potential overflow in tiles: {overflow_tiles}")
    else:
        print(f"  No obvious edge overflow detected (max_edges={max(tile_edge_counts)}, "
              f"capacity={tile_nbrs_capacities[0] if tile_nbrs_capacities else '?'})")

    return {
        "indiv_val_mse": indiv_mse,
        "indiv_val_rmse": indiv_rmse,
        "tiled_val_mse": tiled_val_mse,
        "tiled_val_rmse": tiled_val_rmse,
        "ratio_tiled_over_individual": tiled_val_mse / max(indiv_mse, 1e-12),
        "n_val_frames": n_val,
        "n_val_tiles": n_tiles,
        "per_tile_mse": tile_losses,
        "per_tile_edge_counts": tile_edge_counts,
        "per_tile_nbrs_capacities": tile_nbrs_capacities,
    }


# ---------------------------------------------------------------------------
# Stage 2: Per-structure force comparison (tiled vs individual)
# ---------------------------------------------------------------------------

def stage2_force_comparison(
    config: ConfigManager,
    dataset: dict,
    loader: DatasetLoader,
    box: np.ndarray,
    val_start: int,
    val_stop: int,
    train_tiles: dict,
    val_tiles: dict,
    params: dict,
    n_species_global: int,
    n_structures_to_compare: int = 10,
) -> List[Dict[str, Any]]:
    """
    For each of the first n_structures_to_compare val structures:
    1. Find which val tile contains that structure.
    2. Compute forces on the full tile (tiled model, N_max=1024).
    3. Extract forces for this structure's atoms from the tile.
    4. Compute forces individually (individual model, N_max=n_atoms).
    5. Compare forces, report max abs diff and MSE ratio.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Per-structure force comparison (tiled vs individual)")
    print("=" * 60)

    val_frames = dataset["R"][val_start:val_stop]
    val_F_ref = dataset["F"][val_start:val_stop]
    val_mask = dataset["mask"][val_start:val_stop]
    val_species = dataset["species"][val_start:val_stop]

    # ---- Individual model ----
    config_indiv = ConfigManager(str(config.config_path))
    config_indiv.set("model", "use_priors", False)
    config_indiv.set("model", "train_priors", False)

    model_indiv = CombinedModel(
        config=config_indiv,
        R0=jnp.asarray(val_frames[0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(val_species[0], dtype=jnp.int32),
        N_max=int(loader.N_max),
        init_mask=jnp.asarray(val_mask[0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    def _forces_indiv(R_i, mask_i, species_i):
        def energy_fn(R_):
            return model_indiv.compute_energy(params, R_, mask_i, species_i, None)
        return -jax.grad(energy_fn)(R_i)

    force_fn_indiv = jax.jit(_forces_indiv)

    # ---- Tiled model ----
    config_tiled = ConfigManager(str(config.config_path))
    config_tiled.set("model", "use_priors", False)
    config_tiled.set("model", "train_priors", False)

    N_max_tiled = int(train_tiles["R"].shape[1])
    model_tiled = CombinedModel(
        config=config_tiled,
        R0=jnp.asarray(train_tiles["R"][0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(train_tiles["species"][0], dtype=jnp.int32),
        N_max=N_max_tiled,
        init_mask=jnp.asarray(train_tiles["mask"][0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    def _forces_tile(R_i, mask_i, species_i, seg_i):
        def energy_fn(R_):
            return model_tiled.compute_energy(params, R_, mask_i, species_i, None, segment_id=seg_i)
        return -jax.grad(energy_fn)(R_i)

    force_fn_tiled = jax.jit(_forces_tile)

    # Build a mapping: shuffled_structure_idx -> (tile_idx, segment_id_in_tile)
    meta_source_ids = np.asarray(val_tiles["meta_source_structure_ids"], dtype=np.int32)
    # shape: (n_tiles, max_segments)
    struct_to_tile: Dict[int, Tuple[int, int]] = {}
    for tile_idx in range(val_tiles["R"].shape[0]):
        for seg_pos in range(meta_source_ids.shape[1]):
            struct_id = int(meta_source_ids[tile_idx, seg_pos])
            if struct_id >= 0:
                struct_to_tile[struct_id] = (tile_idx, seg_pos)

    # Pick structures to compare: first n_structures_to_compare val structures
    n_compare = min(n_structures_to_compare, val_stop - val_start)
    results = []

    print(f"  Comparing forces for {n_compare} val structures...")
    for local_i in range(n_compare):
        global_struct_idx = val_start + local_i
        if global_struct_idx not in struct_to_tile:
            print(f"  Structure {local_i} (global={global_struct_idx}): not found in tiles, skipping")
            continue

        tile_idx, seg_pos = struct_to_tile[global_struct_idx]
        R_struct = val_frames[local_i]
        mask_struct = val_mask[local_i]
        species_struct = val_species[local_i]
        F_ref_struct = val_F_ref[local_i]
        valid_atoms = np.flatnonzero(mask_struct > 0)
        n_valid = len(valid_atoms)

        # Individual forces
        F_indiv = np.asarray(force_fn_indiv(
            jnp.asarray(R_struct, dtype=jnp.float32),
            jnp.asarray(mask_struct, dtype=jnp.float32),
            jnp.asarray(species_struct, dtype=jnp.int32),
        ))

        # Tiled forces: compute forces on full tile, extract for this structure's atoms
        R_tile = jnp.asarray(val_tiles["R"][tile_idx], dtype=jnp.float32)
        mask_tile = jnp.asarray(val_tiles["mask"][tile_idx], dtype=jnp.float32)
        species_tile = jnp.asarray(val_tiles["species"][tile_idx], dtype=jnp.int32)
        seg_tile = jnp.asarray(val_tiles["segment_id"][tile_idx], dtype=jnp.int32)

        F_tile_full = np.asarray(force_fn_tiled(R_tile, mask_tile, species_tile, seg_tile))

        # Extract atoms belonging to this segment (seg_pos within tile)
        seg_arr = np.asarray(val_tiles["segment_id"][tile_idx], dtype=np.int32)
        tile_atom_positions = np.flatnonzero(seg_arr == seg_pos)

        if len(tile_atom_positions) != n_valid:
            print(f"  WARNING: structure {local_i}: expected {n_valid} atoms in tile segment, "
                  f"got {len(tile_atom_positions)}")

        n_match = min(len(tile_atom_positions), n_valid)
        F_tiled_struct = F_tile_full[tile_atom_positions[:n_match]]  # forces in tile
        F_indiv_valid = F_indiv[valid_atoms[:n_match]]                # individual forces

        # Compare forces (both on VALID atoms)
        max_abs_diff = float(np.max(np.abs(F_tiled_struct - F_indiv_valid)))
        mean_abs_diff = float(np.mean(np.abs(F_tiled_struct - F_indiv_valid)))
        rel_max_diff = max_abs_diff / max(float(np.max(np.abs(F_indiv_valid))), 1e-12)

        # Per-structure MSE from each model
        mse_indiv_vs_ref = float(np.mean((F_indiv_valid - F_ref_struct[valid_atoms[:n_match]]) ** 2))
        mse_tiled_vs_ref = float(np.mean((F_tiled_struct - F_ref_struct[valid_atoms[:n_match]]) ** 2))

        # Also compare the COORDINATES fed to each model
        tile_coords = np.asarray(val_tiles["R"][tile_idx])[tile_atom_positions[:n_match]]
        indiv_coords = R_struct[valid_atoms[:n_match]]
        max_coord_diff = float(np.max(np.abs(tile_coords - indiv_coords)))
        # (tile coords are spatially shifted, so this will be nonzero!)
        coord_diff_relative = np.abs(tile_coords - indiv_coords)
        # Y and Z should match; only X should differ (due to spatial separation)
        max_yz_diff = float(np.max(np.abs(tile_coords[:, 1:] - indiv_coords[:, 1:])))

        result = {
            "local_i": local_i,
            "global_struct_idx": global_struct_idx,
            "tile_idx": tile_idx,
            "seg_pos_in_tile": seg_pos,
            "n_valid": n_valid,
            "n_segs_in_tile": int(val_tiles["n_segments"][tile_idx]),
            "max_force_abs_diff_tiled_vs_indiv": max_abs_diff,
            "mean_force_abs_diff_tiled_vs_indiv": mean_abs_diff,
            "rel_max_force_diff": rel_max_diff,
            "mse_indiv_vs_ref": mse_indiv_vs_ref,
            "mse_tiled_vs_ref": mse_tiled_vs_ref,
            "ratio_mse_tiled_over_indiv": mse_tiled_vs_ref / max(mse_indiv_vs_ref, 1e-12),
            "max_coord_x_diff_tiled_vs_indiv": float(np.max(np.abs(tile_coords[:, 0] - indiv_coords[:, 0]))),
            "max_coord_yz_diff_tiled_vs_indiv": max_yz_diff,
        }
        results.append(result)

        print(f"  struct {local_i:3d} (tile={tile_idx}, seg={seg_pos}, "
              f"n_segs={result['n_segs_in_tile']}): "
              f"force_max_abs_diff={max_abs_diff:.4e}  "
              f"mse_ratio(tiled/indiv)={result['ratio_mse_tiled_over_indiv']:.3f}  "
              f"coord_yz_diff={max_yz_diff:.2e}")

    forces_match = all(r["max_force_abs_diff_tiled_vs_indiv"] < 1e-3 for r in results)
    print(f"\n  Force equivalence (tol=1e-3): {'PASS' if forces_match else 'FAIL'}")
    if not forces_match:
        worst = max(results, key=lambda r: r["max_force_abs_diff_tiled_vs_indiv"])
        print(f"  Worst: struct={worst['local_i']}, tile={worst['tile_idx']}, "
              f"seg={worst['seg_pos_in_tile']}, max_diff={worst['max_force_abs_diff_tiled_vs_indiv']:.4e}")

    return results


# ---------------------------------------------------------------------------
# Stage 3: Neighbor list diagnostics
# ---------------------------------------------------------------------------

def stage3_nbrs_diagnostics(
    config: ConfigManager,
    val_tiles: dict,
    train_tiles: dict,
    loader: DatasetLoader,
    box: np.ndarray,
    n_species_global: int,
) -> Dict[str, Any]:
    """Examine neighbor list edge counts for all val tiles and training tile 0."""
    print("\n" + "=" * 60)
    print("STAGE 3: Neighbor list diagnostics")
    print("=" * 60)

    config_tiled = ConfigManager(str(config.config_path))
    config_tiled.set("model", "use_priors", False)
    config_tiled.set("model", "train_priors", False)

    N_max_tiled = int(train_tiles["R"].shape[1])
    model_tiled = CombinedModel(
        config=config_tiled,
        R0=jnp.asarray(train_tiles["R"][0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(train_tiles["species"][0], dtype=jnp.int32),
        N_max=N_max_tiled,
        init_mask=jnp.asarray(train_tiles["mask"][0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    # Check init tile 0 (training)
    nbrs_init = model_tiled.ml_model.nbrs_init
    idx_init = np.asarray(nbrs_init.idx)
    edges_init = int(np.sum(idx_init < N_max_tiled))
    print(f"  Training tile 0: N_max={N_max_tiled}, "
          f"nbrs_init.idx.shape={idx_init.shape}, "
          f"n_edges_in_init={edges_init}")

    # For sparse format: idx_init shape is (N_atoms, max_neighbors_per_atom)
    # Total capacity = N_atoms * max_neighbors_per_atom
    if idx_init.ndim == 2:
        n_atoms_cap, max_nbrs_per_atom = idx_init.shape
        total_capacity = n_atoms_cap * max_nbrs_per_atom
        print(f"  Neighbor list format: DENSE sparse  "
              f"(N_atoms={n_atoms_cap}, max_nbrs={max_nbrs_per_atom}, capacity={total_capacity})")
    else:
        total_capacity = idx_init.shape[0]
        print(f"  Neighbor list format: FLAT sparse  (capacity={total_capacity})")

    overflow_flag_init = getattr(nbrs_init, "did_buffer_overflow", None)
    print(f"  did_buffer_overflow (init): {overflow_flag_init}")

    # Check each val tile
    n_tiles = val_tiles["R"].shape[0]
    tile_diagnostics = []

    for tile_idx in range(n_tiles):
        R_tile = jnp.asarray(val_tiles["R"][tile_idx], dtype=jnp.float32)
        mask_tile = jnp.asarray(val_tiles["mask"][tile_idx], dtype=jnp.float32)
        valid_mask = (mask_tile > 0).astype(jnp.bool_)

        nbrs_tile = model_tiled.ml_model.nneigh_fn.update(
            R_tile,
            nbrs_init,
            mask=valid_mask,
        )
        idx_tile = np.asarray(nbrs_tile.idx)
        edges_tile = int(np.sum(idx_tile < N_max_tiled))
        did_overflow = getattr(nbrs_tile, "did_buffer_overflow", None)
        n_valid = int(np.sum(val_tiles["mask"][tile_idx] > 0))

        # Count edges per atom (for dense format)
        if idx_tile.ndim == 2:
            edges_per_atom = np.sum(idx_tile < N_max_tiled, axis=1)
            max_edges_per_atom = int(np.max(edges_per_atom))
            max_neighbors_per_atom_capacity = idx_tile.shape[1]
            nearly_full = bool(max_edges_per_atom >= max_neighbors_per_atom_capacity - 1)
        else:
            max_edges_per_atom = -1
            max_neighbors_per_atom_capacity = -1
            nearly_full = False

        tile_diag = {
            "tile_idx": tile_idx,
            "n_valid": n_valid,
            "n_segments": int(val_tiles["n_segments"][tile_idx]),
            "total_edges": edges_tile,
            "did_buffer_overflow": bool(did_overflow) if did_overflow is not None else None,
            "max_edges_per_atom": max_edges_per_atom,
            "max_nbrs_capacity_per_atom": max_neighbors_per_atom_capacity,
            "any_atom_nearly_full": nearly_full,
        }
        tile_diagnostics.append(tile_diag)

    # Summary
    overflowed = [d for d in tile_diagnostics if d["did_buffer_overflow"]]
    nearly_full = [d for d in tile_diagnostics if d["any_atom_nearly_full"]]
    max_edges = max(d["total_edges"] for d in tile_diagnostics)
    print(f"\n  Val tile summary: n_tiles={n_tiles}")
    print(f"  Tiles with did_buffer_overflow=True: {len(overflowed)}")
    print(f"  Tiles with any atom nearly at capacity: {len(nearly_full)}")
    print(f"  Init tile edges: {edges_init}  Max val tile edges: {max_edges}")
    print(f"  Total capacity: {total_capacity}")
    if max_edges > edges_init:
        print(f"  !! Val tiles have MORE edges ({max_edges}) than init tile ({edges_init})")
        print(f"     This could cause capacity issues if any per-atom count exceeds limit")
    else:
        print(f"  Val tiles have <= edges compared to init tile. No bead overflow.")

    # Print first few tiles
    print("\n  Per-tile details (first 5):")
    for d in tile_diagnostics[:5]:
        print(f"    tile={d['tile_idx']}: n_valid={d['n_valid']}, "
              f"n_segs={d['n_segments']}, "
              f"edges={d['total_edges']}, "
              f"overflow={d['did_buffer_overflow']}, "
              f"max_nbrs_per_atom={d['max_edges_per_atom']}/{d['max_nbrs_capacity_per_atom']}")

    return {
        "init_tile_edges": edges_init,
        "total_capacity": total_capacity,
        "max_val_tile_edges": max_edges,
        "n_overflow_tiles": len(overflowed),
        "n_nearly_full_tiles": len(nearly_full),
        "tile_diagnostics": tile_diagnostics,
    }


# ---------------------------------------------------------------------------
# Stage 4: Checkpoint sweep
# ---------------------------------------------------------------------------

def stage4_checkpoint_sweep(
    config: ConfigManager,
    dataset: dict,
    loader: DatasetLoader,
    box: np.ndarray,
    val_start: int,
    val_stop: int,
    train_tiles: dict,
    val_tiles: dict,
    n_species_global: int,
    checkpoint_dir: Path,
) -> List[Dict[str, Any]]:
    """For each checkpoint, compute both tiled_val_mse and individual_val_mse."""
    print("\n" + "=" * 60)
    print("STAGE 4: Checkpoint sweep")
    print("=" * 60)

    ckpt_files = sorted(checkpoint_dir.glob("epoch*.pkl"))
    # Also include final stage checkpoint
    final_ckpts = sorted(checkpoint_dir.glob("stage_*.pkl"))
    all_ckpts = ckpt_files + final_ckpts

    if not all_ckpts:
        print(f"  No checkpoints found in {checkpoint_dir}")
        return []

    val_frames = dataset["R"][val_start:val_stop]
    val_F_ref = dataset["F"][val_start:val_stop]
    val_mask = dataset["mask"][val_start:val_stop]
    val_species = dataset["species"][val_start:val_stop]
    n_val = val_frames.shape[0]
    n_tiles = val_tiles["R"].shape[0]

    # Pre-build models (fixed architecture, just swap params)
    config_indiv = ConfigManager(str(config.config_path))
    config_indiv.set("model", "use_priors", False)
    config_indiv.set("model", "train_priors", False)

    model_indiv = CombinedModel(
        config=config_indiv,
        R0=jnp.asarray(val_frames[0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(val_species[0], dtype=jnp.int32),
        N_max=int(loader.N_max),
        init_mask=jnp.asarray(val_mask[0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    def _forces_indiv(params_, R_i, mask_i, species_i):
        def energy_fn(R_):
            return model_indiv.compute_energy(params_, R_, mask_i, species_i, None)
        return -jax.grad(energy_fn)(R_i)

    batched_forces_indiv = jax.jit(jax.vmap(lambda R_i, m_i, s_i: _forces_indiv(None, R_i, m_i, s_i)))
    # Need params-aware version
    def _batched_forces_indiv_with_params(params_, R_b, m_b, s_b):
        def single(R_i, m_i, s_i):
            def energy_fn(R_):
                return model_indiv.compute_energy(params_, R_, m_i, s_i, None)
            return -jax.grad(energy_fn)(R_i)
        return jax.vmap(single)(R_b, m_b, s_b)

    batched_forces_indiv_fn = jax.jit(_batched_forces_indiv_with_params)

    config_tiled = ConfigManager(str(config.config_path))
    config_tiled.set("model", "use_priors", False)
    config_tiled.set("model", "train_priors", False)

    N_max_tiled = int(train_tiles["R"].shape[1])
    model_tiled = CombinedModel(
        config=config_tiled,
        R0=jnp.asarray(train_tiles["R"][0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(train_tiles["species"][0], dtype=jnp.int32),
        N_max=N_max_tiled,
        init_mask=jnp.asarray(train_tiles["mask"][0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    def _forces_tile_with_params(params_, R_i, mask_i, species_i, seg_i):
        def energy_fn(R_):
            return model_tiled.compute_energy(params_, R_, mask_i, species_i, None, segment_id=seg_i)
        return -jax.grad(energy_fn)(R_i)

    force_fn_tiled_fn = jax.jit(_forces_tile_with_params)

    R_val_jax = jnp.asarray(val_frames, dtype=jnp.float32)
    mask_val_jax = jnp.asarray(val_mask, dtype=jnp.float32)
    species_val_jax = jnp.asarray(val_species, dtype=jnp.int32)

    results = []
    print(f"  Sweeping {len(all_ckpts)} checkpoints...")

    for ckpt_path in all_ckpts:
        # Parse epoch from filename
        stem = ckpt_path.stem
        if stem.startswith("epoch"):
            try:
                epoch = int(stem[5:])
            except ValueError:
                epoch = -1
        elif "epoch" in stem:
            try:
                epoch = int(stem.split("epoch")[1].split(".")[0])
            except (ValueError, IndexError):
                epoch = -1
        else:
            epoch = -1

        if ckpt_path.name.endswith(".meta.pkl"):
            continue  # Skip meta files

        try:
            params = _load_params(ckpt_path)
        except Exception as e:
            print(f"  Skipping {ckpt_path.name}: {e}")
            continue

        # Individual val MSE
        F_pred_indiv = np.asarray(batched_forces_indiv_fn(
            params, R_val_jax, mask_val_jax, species_val_jax
        ))
        sq_indiv = (F_pred_indiv - val_F_ref) ** 2
        mask3 = (val_mask > 0)[..., None]
        indiv_mse = float(np.sum(sq_indiv * mask3) / max(float(np.sum(mask3)), 1.0))

        # Tiled val MSE
        tile_losses = []
        for tile_idx in range(n_tiles):
            R_tile = jnp.asarray(val_tiles["R"][tile_idx], dtype=jnp.float32)
            F_tile_ref = jnp.asarray(val_tiles["F"][tile_idx], dtype=jnp.float32)
            mask_tile = jnp.asarray(val_tiles["mask"][tile_idx], dtype=jnp.float32)
            species_tile = jnp.asarray(val_tiles["species"][tile_idx], dtype=jnp.int32)
            seg_tile = jnp.asarray(val_tiles["segment_id"][tile_idx], dtype=jnp.int32)

            F_tile_pred = force_fn_tiled_fn(params, R_tile, mask_tile, species_tile, seg_tile)
            tile_loss = float(valid_component_mse(F_tile_pred, F_tile_ref, mask_tile))
            tile_losses.append(tile_loss)

        tiled_mse = float(np.mean(tile_losses))

        row = {
            "epoch": epoch,
            "checkpoint": ckpt_path.name,
            "indiv_val_mse": indiv_mse,
            "indiv_val_rmse": float(np.sqrt(max(indiv_mse, 0.0))),
            "tiled_val_mse": tiled_mse,
            "tiled_val_rmse": float(np.sqrt(max(tiled_mse, 0.0))),
            "ratio_tiled_over_indiv": tiled_mse / max(indiv_mse, 1e-12),
        }
        results.append(row)
        print(f"  epoch={epoch:4d}: indiv_mse={indiv_mse:.2f}  tiled_mse={tiled_mse:.2f}  "
              f"ratio={row['ratio_tiled_over_indiv']:.3f}")

    results.sort(key=lambda r: r["epoch"])
    return results


# ---------------------------------------------------------------------------
# Stage 5: Force decomposition — ML vs prior
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stage5_force_decomposition(
    config: ConfigManager,
    dataset: dict,
    loader: DatasetLoader,
    box: np.ndarray,
    val_start: int,
    val_stop: int,
    params: dict,
    n_species_global: int,
    n_frames: int = 20,
) -> Dict[str, Any]:
    """
    Decompose val-frame forces into ML and prior contributions.

    Bypasses the prior-residual pipeline: loads raw F_ref from the NPZ
    directly, computes F_ml and F_prior separately via compute_force_components,
    and answers: is F_ml substantial? Does F_ml + F_prior reproduce F_ref?
    Does F_ml correlate well with F_residual = F_ref - F_prior?

    Expected values from metrics.json (post-analysis):
      rmse(F_ml + F_prior, F_ref)   ~6.6
      pearson(F_ml + F_prior, F_ref) ~0.73
    """
    print("\n" + "=" * 60)
    print("STAGE 5: Force decomposition (ML vs prior)")
    print("=" * 60)

    # Load raw forces directly from NPZ (no prior-residual applied)
    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    raw_loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(),
                               seed=config.get_seed())
    # Shuffle raw frames with same seed so val indices align with dataset["R"]
    n_raw = int(raw_loader.R.shape[0])
    rng = np.random.RandomState(int(config.get_seed()))
    perm = rng.permutation(n_raw)
    F_raw_shuffled = np.asarray(raw_loader.F, dtype=np.float32)[perm]
    F_ref_val_raw = F_raw_shuffled[val_start:val_stop]  # raw forces for val frames

    val_frames = dataset["R"][val_start:val_stop]
    val_mask = dataset["mask"][val_start:val_stop]
    val_species = dataset["species"][val_start:val_stop]
    n_eval = min(n_frames, int(val_frames.shape[0]))

    # Build CombinedModel with priors enabled (to get both F_ml and F_prior)
    config_priors = ConfigManager(str(config.config_path))
    config_priors.set("model", "use_priors", True)
    config_priors.set("model", "train_priors", False)

    model_combined = CombinedModel(
        config=config_priors,
        R0=jnp.asarray(val_frames[0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(val_species[0], dtype=jnp.int32),
        N_max=int(loader.N_max),
        init_mask=jnp.asarray(val_mask[0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    # JIT the decomposition function (compiled once, reused per frame)
    def _decomp(R_i, mask_i, species_i):
        return model_combined.compute_force_components(params, R_i, mask_i, species_i)

    decomp_fn = jax.jit(_decomp)

    rows = []
    F_ml_all, F_prior_all, F_total_all, F_ref_all = [], [], [], []

    print(f"  Computing force components for {n_eval} val frames...")
    for i in range(n_eval):
        R_i = jnp.asarray(val_frames[i], dtype=jnp.float32)
        mask_i = jnp.asarray(val_mask[i], dtype=jnp.float32)
        species_i = jnp.asarray(val_species[i], dtype=jnp.int32)

        comps = decomp_fn(R_i, mask_i, species_i)
        F_ml_i = np.asarray(comps["F_ml"])
        F_total_i = np.asarray(comps["F_total"])
        F_prior_i = F_total_i - F_ml_i
        F_ref_i = F_ref_val_raw[i]

        valid = val_mask[i] > 0
        F_ml_v = F_ml_i[valid]
        F_prior_v = F_prior_i[valid]
        F_total_v = F_total_i[valid]
        F_ref_v = F_ref_i[valid]
        F_residual_v = F_ref_v - F_prior_v  # what ML was trained to predict

        rms_ml = float(np.sqrt(np.mean(F_ml_v ** 2)))
        rms_prior = float(np.sqrt(np.mean(F_prior_v ** 2)))
        rms_total = float(np.sqrt(np.mean(F_total_v ** 2)))
        rms_ref = float(np.sqrt(np.mean(F_ref_v ** 2)))
        rms_residual = float(np.sqrt(np.mean(F_residual_v ** 2)))

        row = {
            "frame_i": i,
            "rms_F_ml": rms_ml,
            "rms_F_prior": rms_prior,
            "rms_F_total": rms_total,
            "rms_F_ref": rms_ref,
            "rms_F_residual": rms_residual,
            "rmse_total_vs_ref": float(np.sqrt(np.mean((F_total_v - F_ref_v) ** 2))),
            "rmse_ml_vs_residual": float(np.sqrt(np.mean((F_ml_v - F_residual_v) ** 2))),
            "rmse_prior_vs_ref": float(np.sqrt(np.mean((F_prior_v - F_ref_v) ** 2))),
            "pearson_total_vs_ref": _pearson(F_total_v, F_ref_v),
            "pearson_ml_vs_residual": _pearson(F_ml_v, F_residual_v),
            "pearson_prior_vs_ref": _pearson(F_prior_v, F_ref_v),
            "ml_fraction_of_residual_rms": rms_ml / max(rms_residual, 1e-12),
        }
        rows.append(row)
        F_ml_all.append(F_ml_v)
        F_prior_all.append(F_prior_v)
        F_total_all.append(F_total_v)
        F_ref_all.append(F_ref_v)

        if (i + 1) % 5 == 0 or i == n_eval - 1:
            print(f"    frame {i+1}/{n_eval}: "
                  f"rms_ml={rms_ml:.3f}  rms_prior={rms_prior:.3f}  "
                  f"rms_ref={rms_ref:.3f}  "
                  f"pearson_total={row['pearson_total_vs_ref']:.3f}  "
                  f"pearson_ml_res={row['pearson_ml_vs_residual']:.3f}")

    # Global aggregated statistics across all frames
    F_ml_flat = np.concatenate([f.ravel() for f in F_ml_all])
    F_prior_flat = np.concatenate([f.ravel() for f in F_prior_all])
    F_total_flat = np.concatenate([f.ravel() for f in F_total_all])
    F_ref_flat = np.concatenate([f.ravel() for f in F_ref_all])
    F_residual_flat = F_ref_flat - F_prior_flat

    g_rms_ml = float(np.sqrt(np.mean(F_ml_flat ** 2)))
    g_rms_prior = float(np.sqrt(np.mean(F_prior_flat ** 2)))
    g_rms_total = float(np.sqrt(np.mean(F_total_flat ** 2)))
    g_rms_ref = float(np.sqrt(np.mean(F_ref_flat ** 2)))
    g_rms_residual = float(np.sqrt(np.mean(F_residual_flat ** 2)))

    g_rmse_total_ref = float(np.sqrt(np.mean((F_total_flat - F_ref_flat) ** 2)))
    g_rmse_ml_res = float(np.sqrt(np.mean((F_ml_flat - F_residual_flat) ** 2)))
    g_rmse_prior_ref = float(np.sqrt(np.mean((F_prior_flat - F_ref_flat) ** 2)))

    g_pearson_total_ref = _pearson(F_total_flat, F_ref_flat)
    g_pearson_ml_res = _pearson(F_ml_flat, F_residual_flat)
    g_pearson_prior_ref = _pearson(F_prior_flat, F_ref_flat)

    print(f"\n  === Global force statistics ({n_eval} frames) ===")
    print(f"  RMS F_ref     :  {g_rms_ref:.4f}  (raw reference forces)")
    print(f"  RMS F_prior   :  {g_rms_prior:.4f}  ({100*g_rms_prior/g_rms_ref:.1f}% of F_ref)")
    print(f"  RMS F_residual:  {g_rms_residual:.4f}  (F_ref - F_prior; ML training target)")
    print(f"  RMS F_ml      :  {g_rms_ml:.4f}  ({100*g_rms_ml/g_rms_residual:.1f}% of residual, {100*g_rms_ml/g_rms_ref:.1f}% of F_ref)")
    print(f"  RMS F_total   :  {g_rms_total:.4f}  (F_ml + F_prior)")
    print()
    print(f"  Pearson(F_total, F_ref)   :  {g_pearson_total_ref:.4f}  [expect ~0.73 from post-analysis]")
    print(f"  Pearson(F_prior, F_ref)   :  {g_pearson_prior_ref:.4f}  [prior alone]")
    print(f"  Pearson(F_ml, F_residual) :  {g_pearson_ml_res:.4f}  [ML model on its target]")
    print()
    print(f"  RMSE(F_total vs F_ref)   :  {g_rmse_total_ref:.4f}  [expect ~6.6 from post-analysis]")
    print(f"  RMSE(F_prior vs F_ref)   :  {g_rmse_prior_ref:.4f}  [prior alone vs ref]")
    print(f"  RMSE(F_ml vs F_residual) :  {g_rmse_ml_res:.4f}  [ML error on training target]")
    print()

    # Diagnosis
    ml_fraction = g_rms_ml / max(g_rms_residual, 1e-12)
    if ml_fraction > 0.3:
        print(f"  ML model IS contributing: RMS_ml = {100*ml_fraction:.0f}% of RMS_residual")
    else:
        print(f"  !! ML model is QUIET: RMS_ml = {100*ml_fraction:.0f}% of RMS_residual")

    if abs(g_pearson_total_ref - 0.73) < 0.10:
        print(f"  Pearson(total, ref) = {g_pearson_total_ref:.3f} matches post-analysis (0.73) ✓")
    else:
        print(f"  Pearson(total, ref) = {g_pearson_total_ref:.3f} DOES NOT match post-analysis (0.73)!")
        print(f"  -> checkpoint or prior params differ from what post-analysis used")

    if g_rmse_ml_res < g_rms_residual * 0.9:
        print(f"  ML model reduces residual error: RMSE_ml/RMS_residual = "
              f"{g_rmse_ml_res/g_rms_residual:.3f} < 1  (model generalizes to val)")
    else:
        print(f"  ML model does NOT reduce residual error on val: RMSE_ml = {g_rmse_ml_res:.3f} "
              f"vs baseline RMS_residual = {g_rms_residual:.3f}  (overfitting?)")

    return {
        "n_frames": n_eval,
        "global_rms_F_ref": g_rms_ref,
        "global_rms_F_prior": g_rms_prior,
        "global_rms_F_residual": g_rms_residual,
        "global_rms_F_ml": g_rms_ml,
        "global_rms_F_total": g_rms_total,
        "global_rmse_total_vs_ref": g_rmse_total_ref,
        "global_rmse_ml_vs_residual": g_rmse_ml_res,
        "global_rmse_prior_vs_ref": g_rmse_prior_ref,
        "global_pearson_total_vs_ref": g_pearson_total_ref,
        "global_pearson_ml_vs_residual": g_pearson_ml_res,
        "global_pearson_prior_vs_ref": g_pearson_prior_ref,
        "ml_fraction_of_residual_rms": ml_fraction,
        "per_frame": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug tiled val_loss vs individual val MSE discrepancy")
    parser.add_argument("config", type=str, help="Path to training config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single checkpoint (for stages 1-3). "
                             "Defaults to stage_sgd_nesterov_epoch500.pkl if omitted.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (enables stage 4 sweep)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write output JSON/CSV files")
    parser.add_argument("--stages", type=str, default="1,2,3",
                        help="Comma-separated stages to run: 1,2,3,4,5 (default: 1,2,3)")
    parser.add_argument("--n-compare", type=int, default=10,
                        help="Number of structures to compare in stage 2 (default: 10)")
    parser.add_argument("--n-decomp", type=int, default=20,
                        help="Number of val frames for stage 5 force decomposition (default: 20)")
    args = parser.parse_args()

    stages = set(int(s.strip()) for s in args.stages.split(","))
    config_path = Path(args.config).resolve()
    config = ConfigManager(str(config_path))

    # Setup output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config_path.parent.parent / "analysis" / "debug_tiled_val"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Resolve checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        ckpt_dir_guess = Path(config.get("paths", "checkpoint_dir", default="checkpoints"))
        if not ckpt_dir_guess.is_absolute():
            ckpt_dir_guess = Path(config.get("paths", "output_dir", default=".")) / "checkpoints"
        checkpoint_path = ckpt_dir_guess / "stage_sgd_nesterov_epoch500.pkl"
        if not checkpoint_path.exists():
            # Try any last checkpoint
            candidates = sorted(ckpt_dir_guess.glob("*.pkl"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir_guess}")
            checkpoint_path = candidates[-1]
    print(f"Using checkpoint: {checkpoint_path}")

    # Load dataset
    print("\nLoading dataset and applying prior residual...")
    dataset, loader, box = _load_dataset_with_residuals(config)
    n_frames = dataset["R"].shape[0]
    n_species_global = int(np.max(dataset["species"])) + 1

    # Split train/val — dataset is already shuffled, so slice contiguously like training does
    val_fraction = float(config.get_val_fraction())
    n_train = int(np.round(n_frames * (1.0 - val_fraction)))
    n_val = n_frames - n_train
    val_start = n_train
    val_stop = n_frames
    print(f"  n_frames={n_frames}, n_train={n_train}, n_val={n_val}")
    print(f"  val frames: shuffled_dataset[{val_start}:{val_stop}]")

    # Build tiles
    print("\nBuilding tiles...")
    val_tiles = _build_val_tiles(dataset, val_start, val_stop, config)
    print(f"  Val tiles: {val_tiles['R'].shape[0]} tiles, N_max={val_tiles['R'].shape[1]}")

    train_tiles = _build_train_tiles(dataset, n_train, config)
    print(f"  Train tiles: {train_tiles['R'].shape[0]} tiles, N_max={train_tiles['R'].shape[1]}")

    # Load checkpoint params
    params = _load_params(checkpoint_path)
    print(f"  Checkpoint params loaded from {checkpoint_path.name}")

    all_outputs: Dict[str, Any] = {
        "config_path": str(config_path),
        "checkpoint": str(checkpoint_path),
        "n_frames": n_frames,
        "n_train": n_train,
        "n_val": val_stop - val_start,
        "n_val_tiles": int(val_tiles["R"].shape[0]),
        "N_max_tiled": int(val_tiles["R"].shape[1]),
        "N_max_individual": int(loader.N_max),
    }

    # Run stages
    if 1 in stages:
        s1 = stage1_reproduce_metrics(
            config, dataset, loader, box,
            val_start, val_stop,
            train_tiles, val_tiles, params, n_species_global,
        )
        all_outputs["stage1"] = {k: v for k, v in s1.items()
                                  if not isinstance(v, list) or len(v) < 100}
        # Save per-tile CSV
        per_tile_path = output_dir / "stage1_per_tile.csv"
        with open(per_tile_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tile_idx", "tile_mse", "tile_rmse",
                                                    "edge_count", "nbrs_capacity"])
            writer.writeheader()
            for i, (loss, edges, cap) in enumerate(zip(
                s1["per_tile_mse"], s1["per_tile_edge_counts"], s1["per_tile_nbrs_capacities"]
            )):
                writer.writerow({
                    "tile_idx": i,
                    "tile_mse": loss,
                    "tile_rmse": float(np.sqrt(max(loss, 0.0))),
                    "edge_count": edges,
                    "nbrs_capacity": cap,
                })
        print(f"\n  Per-tile CSV saved: {per_tile_path}")

    if 2 in stages:
        s2 = stage2_force_comparison(
            config, dataset, loader, box,
            val_start, val_stop,
            train_tiles, val_tiles, params, n_species_global,
            n_structures_to_compare=args.n_compare,
        )
        all_outputs["stage2"] = s2
        # Save CSV
        stage2_path = output_dir / "stage2_force_comparison.csv"
        if s2:
            with open(stage2_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(s2[0].keys()))
                writer.writeheader()
                writer.writerows(s2)
            print(f"\n  Force comparison CSV saved: {stage2_path}")

    if 3 in stages:
        s3 = stage3_nbrs_diagnostics(
            config, val_tiles, train_tiles, loader, box, n_species_global,
        )
        all_outputs["stage3"] = {k: v for k, v in s3.items() if k != "tile_diagnostics"}
        # Save tile diagnostics CSV
        nbrs_path = output_dir / "stage3_nbrs_diagnostics.csv"
        if s3["tile_diagnostics"]:
            with open(nbrs_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(s3["tile_diagnostics"][0].keys()))
                writer.writeheader()
                writer.writerows(s3["tile_diagnostics"])
            print(f"\n  Neighbor list diagnostics CSV saved: {nbrs_path}")

    if 4 in stages:
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else checkpoint_path.parent
        s4 = stage4_checkpoint_sweep(
            config, dataset, loader, box,
            val_start, val_stop,
            train_tiles, val_tiles, n_species_global,
            ckpt_dir,
        )
        all_outputs["stage4"] = s4
        sweep_path = output_dir / "stage4_epoch_sweep.csv"
        if s4:
            with open(sweep_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(s4[0].keys()))
                writer.writeheader()
                writer.writerows(s4)
            print(f"\n  Epoch sweep CSV saved: {sweep_path}")

    if 5 in stages:
        s5 = stage5_force_decomposition(
            config, dataset, loader, box,
            val_start, val_stop,
            params, n_species_global,
            n_frames=args.n_decomp,
        )
        per_frame = s5.pop("per_frame")
        all_outputs["stage5"] = s5
        decomp_path = output_dir / "stage5_force_decomposition.csv"
        if per_frame:
            with open(decomp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
                writer.writeheader()
                writer.writerows(per_frame)
            print(f"\n  Force decomposition CSV saved: {decomp_path}")

    # Save summary JSON
    summary_path = output_dir / "debug_summary.json"
    def _json_safe(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        return obj

    with open(summary_path, "w") as f:
        json.dump(_json_safe(all_outputs), f, indent=2)
    print(f"\nSummary JSON saved: {summary_path}")

    # Print key conclusions
    print("\n" + "=" * 60)
    print("KEY CONCLUSIONS")
    print("=" * 60)
    if "stage1" in all_outputs:
        s1 = all_outputs["stage1"]
        print(f"  Individual val MSE = {s1['indiv_val_mse']:.4f} (RMSE={s1['indiv_val_rmse']:.4f})")
        print(f"  Tiled val MSE      = {s1['tiled_val_mse']:.4f} (RMSE={s1['tiled_val_rmse']:.4f})")
        print(f"  Ratio (tiled/indiv) = {s1['ratio_tiled_over_individual']:.4f}")
        if abs(s1["ratio_tiled_over_individual"] - 1.0) < 0.05:
            print("  -> Metrics agree: discrepancy was in training logging, not force computation.")
        elif s1["tiled_val_mse"] > s1["indiv_val_mse"] * 1.5:
            print("  -> Tiled MSE is significantly higher: FORCE COMPUTATION or OVERFITTING issue.")
        else:
            print("  -> Moderate discrepancy: possible aggregation or normalization issue.")

    if "stage2" in all_outputs:
        s2 = all_outputs["stage2"]
        if s2:
            max_force_diff = max(r["max_force_abs_diff_tiled_vs_indiv"] for r in s2)
            mean_force_diff = np.mean([r["mean_force_abs_diff_tiled_vs_indiv"] for r in s2])
            mean_mse_ratio = np.mean([r["ratio_mse_tiled_over_indiv"] for r in s2])
            print(f"\n  Force comparison (tiled vs individual):")
            print(f"  Max abs force diff: {max_force_diff:.4e}")
            print(f"  Mean abs force diff: {mean_force_diff:.4e}")
            print(f"  Mean MSE ratio (tiled/indiv): {mean_mse_ratio:.4f}")
            if max_force_diff < 1e-3:
                print("  -> Forces are equivalent: discrepancy is in LOSS AGGREGATION or TARGET MISMATCH.")
            else:
                print("  -> Forces DIFFER: force computation is different in tiled vs individual mode!")
                print("     Check neighbor list overflow (stage 3) and coordinate handling.")

    if "stage5" in all_outputs:
        s5 = all_outputs["stage5"]
        print(f"\n  Force decomposition ({s5['n_frames']} frames):")
        print(f"  RMS F_ml / RMS F_residual = {s5['ml_fraction_of_residual_rms']:.3f}  "
              f"({'substantial' if s5['ml_fraction_of_residual_rms'] > 0.3 else 'near-zero — model is quiet'})")
        print(f"  Pearson(F_total, F_ref)   = {s5['global_pearson_total_vs_ref']:.4f}  [expect ~0.73]")
        print(f"  RMSE(F_total vs F_ref)    = {s5['global_rmse_total_vs_ref']:.4f}  [expect ~6.6]")
        print(f"  RMSE(F_ml vs F_residual)  = {s5['global_rmse_ml_vs_residual']:.4f}  [training target]")
        print(f"  Pearson(F_ml, F_residual) = {s5['global_pearson_ml_vs_residual']:.4f}")


if __name__ == "__main__":
    main()
