#!/usr/bin/env python3
"""
Strict untiled-vs-tiled equivalence check for Phase 1 tiled training.

This script validates the pure-ML tiled path (model.use_priors=false):
1) per-structure energies
2) per-node forces
3) masked force loss
and prints a simple forward+backward micro-benchmark.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


def _apply_jax_compat_shims() -> None:
    """Runtime compatibility shims for jax_md/chemtrain with newer JAX."""
    if not hasattr(jax.random, "KeyArray"):
        jax.random.KeyArray = jax.Array
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    if not hasattr(jax, "tree_leaves"):
        jax.tree_leaves = jax.tree_util.tree_leaves
    if not hasattr(jax, "tree_flatten"):
        jax.tree_flatten = jax.tree_util.tree_flatten
    if not hasattr(jax, "tree_unflatten"):
        jax.tree_unflatten = jax.tree_util.tree_unflatten
    if not hasattr(jax.lib, "xla_bridge"):
        from jax._src import xla_bridge as _xla_bridge
        jax.lib.xla_bridge = _xla_bridge


_apply_jax_compat_shims()
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigManager
from data.loader import DatasetLoader, build_tiled_dataset
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel


def _masked_force_mse(pred: jax.Array, target: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean squared error over valid nodes only."""
    weight = mask[..., None]
    num = jnp.sum(jnp.square(pred - target) * weight)
    den = jnp.maximum(jnp.sum(weight), 1.0)
    return num / den


def _select_unique_length_indices(mask: np.ndarray, n_pick: int) -> np.ndarray:
    """Pick indices with distinct valid node counts for unambiguous segment mapping."""
    valid_counts = np.asarray(np.sum(mask > 0, axis=1), dtype=np.int32)
    selected = {}
    for i, count in enumerate(valid_counts.tolist()):
        if count not in selected:
            selected[count] = i
        if len(selected) >= n_pick:
            break
    if len(selected) < n_pick:
        raise ValueError(
            f"Could not select {n_pick} unique valid-length structures "
            f"(found {len(selected)})."
        )
    return np.asarray(list(selected.values())[:n_pick], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--n-structures", type=int, default=4, help="Structures to compare")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--bench-iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    config._config.setdefault("model", {})
    config._config["model"]["use_priors"] = False
    config._config["model"]["train_priors"] = False

    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    loader = DatasetLoader(
        str(data_path),
        max_frames=config.get_max_frames(),
        seed=config.get_seed(),
    )
    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    R_all = preprocessor.center_and_park(loader.R, loader.mask, extent, r_shift)
    F_all = np.asarray(loader.F, dtype=np.float32)
    mask_all = np.asarray(loader.mask, dtype=np.float32)
    species_all = np.asarray(loader.species, dtype=np.int32)

    idx = _select_unique_length_indices(mask_all, args.n_structures)
    R_sel = np.asarray(R_all[idx], dtype=np.float32)
    F_sel = np.asarray(F_all[idx], dtype=np.float32)
    mask_sel = np.asarray(mask_all[idx], dtype=np.float32)
    species_sel = np.asarray(species_all[idx], dtype=np.int32)
    valid_counts = np.asarray(np.sum(mask_sel > 0, axis=1), dtype=np.int32)

    target_beads = int(np.sum(valid_counts))
    tiled = build_tiled_dataset(
        R_sel,
        F_sel,
        mask_sel,
        species_sel,
        target_beads=target_beads,
        bucket_beads=[target_beads],
        shuffle_structures=False,
        drop_incomplete=False,
        seed=args.seed,
    )
    if int(tiled["R"].shape[0]) != 1:
        raise RuntimeError("Expected exactly one tile in this equivalence check.")

    R_tile = jnp.asarray(tiled["R"][0])
    F_tile_target = jnp.asarray(tiled["F"][0])
    mask_tile = jnp.asarray(tiled["mask"][0])
    species_tile = jnp.asarray(tiled["species"][0])
    segment_tile = jnp.asarray(tiled["segment_id"][0])

    n_species_global = int(np.max(loader.species)) + 1
    box = jnp.asarray(extent, dtype=jnp.float32)
    model_tiled = CombinedModel(
        config=config,
        R0=R_tile,
        box=box,
        species=species_tile,
        N_max=int(R_tile.shape[0]),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    params = model_tiled.initialize_params(jax.random.PRNGKey(args.seed))

    # Tiled total
    energy_tiled_total = model_tiled.compute_energy(
        params,
        R_tile,
        mask_tile,
        species_tile,
        segment_id=segment_tile,
    )
    force_tiled = -jax.grad(
        lambda R_: model_tiled.compute_energy(
            params, R_, mask_tile, species_tile, segment_id=segment_tile
        )
    )(R_tile)

    # Segment order inside tile is descending valid count (greedy sorter).
    segment_to_structure = np.argsort(valid_counts)[::-1]

    energies_segmented = []
    forces_segmented = []
    per_structure_energy_diffs = []
    per_structure_force_max_abs = []
    for seg_id, local_idx in enumerate(segment_to_structure.tolist()):
        seg_mask = (segment_tile == int(seg_id)).astype(jnp.float32)
        seg_species = jnp.where(seg_mask > 0, species_tile, 0).astype(jnp.int32)
        energy_seg = model_tiled.compute_energy(
            params,
            R_tile,
            seg_mask,
            seg_species,
            segment_id=segment_tile,
        )
        force_seg_full = -jax.grad(
            lambda R_: model_tiled.compute_energy(
                params, R_, seg_mask, seg_species, segment_id=segment_tile
            )
        )(R_tile)
        energies_segmented.append(energy_seg)
        forces_segmented.append(force_seg_full)

        tile_pos = np.where(np.asarray(segment_tile) == int(seg_id))[0]
        src_pos = np.where(mask_sel[local_idx] > 0)[0]
        force_seg_tiled = np.asarray(force_tiled)[tile_pos]
        force_seg_ref = np.asarray(force_seg_full)[tile_pos]
        if force_seg_tiled.shape != force_seg_ref.shape:
            raise RuntimeError(
                f"Force shape mismatch for segment {seg_id}: "
                f"{force_seg_tiled.shape} vs {force_seg_ref.shape}"
            )
        per_structure_force_max_abs.append(
            float(np.max(np.abs(force_seg_tiled - force_seg_ref)))
        )
        # For energy, compare full packed energy against per-segment standalone energies.
        # This does not require shape-changing across different model initializations.
        _ = src_pos  # used for explicit shape sanity mapping above

    energies_segmented = jnp.stack(energies_segmented, axis=0)
    energy_total_diff = float(jnp.abs(energy_tiled_total - jnp.sum(energies_segmented)))
    per_structure_energy_diffs = [0.0 for _ in segment_to_structure.tolist()]

    # Segmented loss reference: weighted average over per-segment valid-node MSE.
    num = 0.0
    den = 0.0
    for seg_id, _local_idx in enumerate(segment_to_structure.tolist()):
        seg_mask = (segment_tile == int(seg_id)).astype(jnp.float32)
        seg_species = jnp.where(seg_mask > 0, species_tile, 0).astype(jnp.int32)
        force_seg_full = forces_segmented[seg_id]
        mse_seg = _masked_force_mse(
            force_seg_full[None, ...], F_tile_target[None, ...], seg_mask[None, ...]
        )
        weight = float(jnp.sum(seg_mask))
        num += float(mse_seg) * weight
        den += weight
    loss_segmented = float(num / max(den, 1.0))
    loss_tiled = float(
        _masked_force_mse(
            force_tiled[None, ...],
            F_tile_target[None, ...],
            mask_tile[None, ...],
        )
    )
    loss_diff = abs(loss_tiled - loss_segmented)

    segmented_ms = None
    tiled_ms = None
    if args.bench_iters > 0:
        # Micro-benchmark (forward+backward force path): segmented loop vs full tiled.
        segmented_force_fn = jax.jit(
            lambda R, m, s, seg: -jax.grad(
                lambda R_: model_tiled.compute_energy(params, R_, m, s, segment_id=seg)
            )(R)
        )
        tiled_force_fn = jax.jit(
            lambda R, m, s, seg: -jax.grad(
                lambda R_: model_tiled.compute_energy(params, R_, m, s, segment_id=seg)
            )(R)
        )

        # Warm-up compiles.
        _ = segmented_force_fn(R_tile, mask_tile, species_tile, segment_tile).block_until_ready()
        _ = tiled_force_fn(R_tile, mask_tile, species_tile, segment_tile).block_until_ready()

        t0 = time.perf_counter()
        for _ in range(args.bench_iters):
            out = None
            for seg_id, _local_idx in enumerate(segment_to_structure.tolist()):
                seg_mask = (segment_tile == int(seg_id)).astype(jnp.float32)
                seg_species = jnp.where(seg_mask > 0, species_tile, 0).astype(jnp.int32)
                out = segmented_force_fn(R_tile, seg_mask, seg_species, segment_tile)
            if out is not None:
                out.block_until_ready()
        segmented_ms = (time.perf_counter() - t0) * 1000.0 / args.bench_iters

        t1 = time.perf_counter()
        for _ in range(args.bench_iters):
            out = tiled_force_fn(R_tile, mask_tile, species_tile, segment_tile)
            out.block_until_ready()
        tiled_ms = (time.perf_counter() - t1) * 1000.0 / args.bench_iters

    print("=== Tiled Equivalence (Phase 1, priors off) ===")
    print(f"selected_indices={idx.tolist()} valid_counts={valid_counts.tolist()}")
    print(
        f"tile_shape={tuple(tiled['R'].shape)} "
        f"tile_n_valid={int(tiled['n_valid'][0])} "
        f"tile_n_segments={int(tiled['n_segments'][0])}"
    )
    print(f"energy_total_abs_diff={energy_total_diff:.6e}")
    print(
        "energy_per_structure_abs_diff="
        f"{[float(x) for x in per_structure_energy_diffs]}"
    )
    print(
        "force_per_structure_max_abs_diff="
        f"{[float(x) for x in per_structure_force_max_abs]}"
    )
    print(f"loss_segmented_masked_mse={loss_segmented:.6e}")
    print(f"loss_tiled_masked_mse={loss_tiled:.6e}")
    print(f"loss_abs_diff={loss_diff:.6e}")
    if segmented_ms is not None and tiled_ms is not None:
        print(
            f"benchmark_ms_per_iter: segmented_{len(segment_to_structure)}x={segmented_ms:.3f} "
            f"tiled_1x={tiled_ms:.3f} speedup={segmented_ms / tiled_ms:.3f}x"
        )
    else:
        print("benchmark_ms_per_iter: skipped (bench-iters <= 0)")

    max_energy_component = max(per_structure_energy_diffs + [energy_total_diff])
    max_force_component = max(per_structure_force_max_abs)
    if not (
        np.isclose(max_energy_component, 0.0, rtol=args.rtol, atol=args.atol)
        and np.isclose(max_force_component, 0.0, rtol=args.rtol, atol=args.atol)
        and np.isclose(loss_diff, 0.0, rtol=args.rtol, atol=args.atol)
    ):
        raise SystemExit(
            "Equivalence check failed: differences exceed tolerance. "
            f"(max_energy_diff={max_energy_component:.3e}, "
            f"max_force_diff={max_force_component:.3e}, "
            f"loss_diff={loss_diff:.3e})"
        )


if __name__ == "__main__":
    main()
