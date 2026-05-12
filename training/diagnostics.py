"""Optional diagnostics for training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from utils.logging import data_logger, training_logger


def write_dataset_summary(
    *,
    dataset: dict,
    id_to_aa: dict[int, str] | None,
    dataset_path: Path,
    export_dir: Path,
) -> None:
    if "mask" not in dataset or "species" not in dataset:
        return

    mask = np.asarray(dataset["mask"], dtype=np.float32) > 0
    species = np.asarray(dataset["species"], dtype=np.int32)
    n_frames = int(mask.shape[0])
    n_valid = np.asarray(np.sum(mask, axis=1), dtype=np.int32)

    source_name = dataset.get("source_name")
    if source_name is None or np.asarray(source_name).shape[0] != n_frames:
        source_name = np.full((n_frames,), dataset_path.stem, dtype=object)
    source_name = np.asarray(source_name).astype(str)

    valid_species = species[mask]
    if valid_species.size == 0:
        data_logger.warning("[DataDebug] No valid beads found; skipping dataset summary.")
        return

    species_ids, species_counts = np.unique(valid_species, return_counts=True)
    species_summary = [
        {
            "species_id": int(species_id),
            "label": str(id_to_aa.get(int(species_id), f"id{int(species_id)}"))
            if id_to_aa
            else f"id{int(species_id)}",
            "count": int(count),
        }
        for species_id, count in zip(species_ids.tolist(), species_counts.tolist())
    ]

    proteins = []
    for protein in np.unique(source_name).tolist():
        protein_mask = source_name == protein
        valid_per_frame = n_valid[protein_mask]
        protein_species = species[protein_mask][mask[protein_mask]]
        ps_ids, ps_counts = np.unique(protein_species, return_counts=True)
        proteins.append(
            {
                "protein": str(protein),
                "frames": int(np.sum(protein_mask)),
                "beads_per_frame_min": int(np.min(valid_per_frame)),
                "beads_per_frame_mean": float(np.mean(valid_per_frame)),
                "beads_per_frame_max": int(np.max(valid_per_frame)),
                "total_valid_beads": int(np.sum(valid_per_frame, dtype=np.int64)),
                "species_counts": {
                    str(int(species_id)): int(count)
                    for species_id, count in zip(ps_ids.tolist(), ps_counts.tolist())
                },
            }
        )

    out_path = export_dir / "dataset_debug_summary.json"
    out_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "n_frames": n_frames,
                "global_valid_beads": int(valid_species.size),
                "global_species_counts": species_summary,
                "proteins": proteins,
            },
            indent=2,
        )
    )
    data_logger.info("[DataDebug] Wrote dataset summary: %s", out_path)


def find_training_log(
    *,
    job_id: str,
    output_dir: Path,
    slurm_dir: Path,
    export_dir: Path,
) -> Path | None:
    log_name = f"train_{job_id}.log"
    candidates = []
    for base in (output_dir, slurm_dir, export_dir.parent):
        candidates.extend(
            [
                base / log_name,
                base / f"slurm-{job_id}.out",
                base / f"slurm-{job_id}.err",
            ]
        )
    candidates.extend(
        [
            Path("outputs") / log_name,
            Path(log_name),
            Path("outputs") / f"slurm-{job_id}.out",
            Path(f"slurm-{job_id}.out"),
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return None


def _coerce_neighbor_meta(value: Any) -> str:
    if value is None:
        return "None"
    try:
        arr = np.asarray(jax.device_get(value))
        return str(arr.item()) if arr.shape == () else str(arr)
    except Exception:
        return str(value)


def log_neighbor_debug_once(trainer: Any) -> bool:
    ml_model = getattr(trainer.model, "ml_model", None)
    if ml_model is None:
        return True
    if not hasattr(ml_model, "summarize_neighborlist"):
        training_logger.warning("[NeighborDebug] ML model has no summarize_neighborlist(); skipping.")
        return True
    if not hasattr(ml_model, "nneigh_fn") or not hasattr(ml_model, "nbrs_init"):
        training_logger.warning("[NeighborDebug] ML model has no neighbor function/init list; skipping.")
        return True

    R_src = getattr(trainer.train_loader, "R", None)
    mask_src = getattr(trainer.train_loader, "mask", None)
    species_src = getattr(trainer.train_loader, "species", None)
    if R_src is None or mask_src is None:
        training_logger.warning("[NeighborDebug] train_loader is missing R/mask arrays; skipping.")
        return True

    from chemtrain import util as chemtrain_util
    from chemtrain.ensemble import evaluation as chemtrain_eval
    from jax_md import partition
    from jax_md_mod import custom_partition

    compute_dtype = getattr(ml_model, "compute_dtype", jnp.float32)
    R0 = jnp.asarray(R_src[0], dtype=compute_dtype)
    mask0 = jnp.asarray(mask_src[0]) > 0
    species0 = (
        jnp.asarray(species_src[0], dtype=jnp.int32)
        if species_src is not None
        else jnp.zeros((R0.shape[0],), dtype=jnp.int32)
    )

    state0 = chemtrain_eval.SimpleState(R0)
    nbrs_updated = chemtrain_util.neighbor_update(ml_model.nbrs_init, state0)
    nbrs_masked = custom_partition.mask_neighbor_list(nbrs_updated, mask0)
    stats_masked = ml_model.summarize_neighborlist(nbrs_masked, mask0)

    ref_position = getattr(nbrs_masked, "reference_position", None)
    target_dtype = getattr(ref_position, "dtype", compute_dtype)
    nbrs_post = ml_model.nneigh_fn.update(jnp.asarray(R0, dtype=target_dtype), nbrs_masked)
    stats_post = ml_model.summarize_neighborlist(nbrs_post, mask0)

    if stats_masked["format"] == "dense":
        training_logger.info(
            "[NeighborDebug][runtime][dense] N_max=%d M_slots(masked)=%d "
            "max_neighbors=%d mean_neighbors=%.2f util_max=%.3f "
            "M_slots(post_update)=%d util_max(post)=%.3f shape_changed=%s "
            "error(masked)=%s did_buffer_overflow(masked)=%s overflow(masked)=%s "
            "error(post)=%s did_buffer_overflow(post)=%s overflow(post)=%s",
            stats_masked["n_atoms"],
            stats_masked["capacity"],
            stats_masked["max_neighbors"],
            stats_masked["mean_neighbors"],
            stats_masked["utilization"],
            stats_post["capacity"],
            stats_post["utilization"],
            stats_masked["idx_shape"] != stats_post["idx_shape"],
            stats_masked["error"],
            stats_masked["did_buffer_overflow"],
            stats_masked["overflow"],
            stats_post["error"],
            stats_post["did_buffer_overflow"],
            stats_post["overflow"],
        )
    else:
        training_logger.info(
            "[NeighborDebug][runtime][sparse] N_max=%d E_capacity(masked)=%d "
            "E_valid=%d util=%.3f E_capacity(post_update)=%d E_valid(post)=%d "
            "util(post)=%.3f shape_changed=%s error(masked)=%s "
            "did_buffer_overflow(masked)=%s overflow(masked)=%s error(post)=%s "
            "did_buffer_overflow(post)=%s overflow(post)=%s",
            stats_masked["n_atoms"],
            stats_masked["capacity"],
            stats_masked["e_valid"],
            stats_masked["utilization"],
            stats_post["capacity"],
            stats_post["e_valid"],
            stats_post["utilization"],
            stats_masked["idx_shape"] != stats_post["idx_shape"],
            stats_masked["error"],
            stats_masked["did_buffer_overflow"],
            stats_masked["overflow"],
            stats_post["error"],
            stats_post["did_buffer_overflow"],
            stats_post["overflow"],
        )

    if getattr(nbrs_post, "format", None) == partition.Dense:
        from jax_md_mod.model import sparse_graph

        cutoff = jnp.asarray(getattr(ml_model, "cutoff"), dtype=jnp.float32)
        species_valid = jnp.where(mask0, species0, 0).astype(jnp.int32)
        max_edges = getattr(ml_model, "max_edges", None)
        dense_shape = tuple(int(x) for x in np.asarray(jax.device_get(nbrs_post.idx)).shape)
        graph, capped = sparse_graph.sparse_graph_from_neighborlist(
            ml_model.displacement,
            jnp.asarray(R0, dtype=jnp.float32),
            nbrs_post,
            cutoff,
            species=species_valid,
            max_edges=max_edges,
            species_mask=mask0,
        )
        training_logger.info(
            "[NeighborDebug][dense_to_sparse] dense_shape=%s max_edges=%s "
            "sparse_idx_i_shape=%s sparse_idx_j_shape=%s E_capacity=%d n_edges=%d capped=%s",
            dense_shape,
            str(max_edges),
            tuple(int(x) for x in np.asarray(jax.device_get(graph.idx_i)).shape),
            tuple(int(x) for x in np.asarray(jax.device_get(graph.idx_j)).shape),
            int(np.asarray(jax.device_get(graph.idx_i)).shape[0]),
            int(np.asarray(jax.device_get(graph.n_edges)).item()),
            _coerce_neighbor_meta(capped),
        )

    return True
