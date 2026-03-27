#!/usr/bin/env python3
"""
Temporary avg_num_neighbors inspection utility.

This script loads a dataset, removes any per-structure padding before analysis,
auto-detects the unique bead counts present, computes avg_num_neighbors over a
random subset of structures, and reports per-structure neighbor statistics for
all bead-count groups.

Examples:
    python data_prep/tmp_avg_num_neighbors.py \
        --npz data/my_dataset.npz \
        --cutoff 10.0 \
        --sample-size 256

    python data_prep/tmp_avg_num_neighbors.py \
        --npz data/my_dataset.npz \
        --cutoff 10.0 \
        --sample-size all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_npz


def _parse_sample_size(raw: str) -> Optional[int]:
    value = str(raw).strip().lower()
    if value in {"all", "full", "none"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"sample size must be positive or 'all', got {raw!r}")
    return parsed


def _resolve_indices(indices: np.ndarray, sample_size: Optional[int], seed: int) -> np.ndarray:
    if sample_size is None or sample_size >= indices.size:
        return np.asarray(indices, dtype=np.int32)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(indices, size=int(sample_size), replace=False)
    return np.sort(chosen.astype(np.int32))


def _strip_padding(
    R: np.ndarray,
    mask: Optional[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray, bool, int]:
    """Return unpadded per-structure coordinates and valid bead counts."""
    if mask is None:
        mask = np.ones(R.shape[:2], dtype=np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)

    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape (n_frames, n_atoms), got {mask.shape}")

    valid_counts = np.asarray(np.sum(mask > 0, axis=1), dtype=np.int32)
    capacity = int(R.shape[1])
    padded_detected = bool(np.any(valid_counts < capacity))

    unpadded_structures: list[np.ndarray] = []
    for frame_idx in range(R.shape[0]):
        valid = np.asarray(mask[frame_idx] > 0, dtype=bool)
        unpadded_structures.append(np.asarray(R[frame_idx][valid], dtype=np.float32))

    return unpadded_structures, valid_counts, padded_detected, capacity


def _compute_structure_avg_num_neighbors(coords: np.ndarray, cutoff: float) -> tuple[float, int, int]:
    n_nodes = int(coords.shape[0])
    if n_nodes <= 1:
        return 0.0, 0, n_nodes

    diffs = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diffs * diffs, axis=-1, dtype=np.float32)
    within_cutoff = (dist_sq < float(cutoff) ** 2) & (dist_sq > 0.0)
    total_neighbors = int(np.sum(within_cutoff, dtype=np.int64))
    avg_neighbors = float(total_neighbors / float(n_nodes))
    return avg_neighbors, total_neighbors, n_nodes


def _compute_sample_stats(structures: list[np.ndarray], indices: np.ndarray, cutoff: float) -> dict:
    per_structure = []
    total_neighbors = 0
    total_nodes = 0

    for idx in indices:
        coords = np.asarray(structures[int(idx)], dtype=np.float32)
        avg_neighbors, structure_neighbors, n_nodes = _compute_structure_avg_num_neighbors(coords, cutoff)
        per_structure.append((int(idx), int(n_nodes), float(avg_neighbors), int(structure_neighbors)))
        total_neighbors += int(structure_neighbors)
        total_nodes += int(n_nodes)

    structure_avgs = np.asarray([item[2] for item in per_structure], dtype=np.float64)
    weighted_avg = 0.0 if total_nodes == 0 else float(total_neighbors / float(total_nodes))
    return {
        "indices": np.asarray(indices, dtype=np.int32),
        "per_structure": per_structure,
        "weighted_avg": weighted_avg,
        "mean_structure_avg": float(np.mean(structure_avgs)) if structure_avgs.size else 0.0,
        "std_structure_avg": float(np.std(structure_avgs)) if structure_avgs.size else 0.0,
        "min_structure_avg": float(np.min(structure_avgs)) if structure_avgs.size else 0.0,
        "max_structure_avg": float(np.max(structure_avgs)) if structure_avgs.size else 0.0,
        "n_structures": int(len(per_structure)),
        "n_nodes": int(total_nodes),
        "total_neighbors": int(total_neighbors),
    }


def _format_counts(valid_counts: np.ndarray, max_rows: int) -> str:
    unique_counts, frequencies = np.unique(valid_counts, return_counts=True)
    rows = []
    for bead_count, frequency in zip(unique_counts, frequencies):
        rows.append(f"  beads={int(bead_count):>4}  structures={int(frequency):>6}")
    if len(rows) <= max_rows:
        return "\n".join(rows)
    head = rows[:max_rows]
    hidden = len(rows) - max_rows
    head.append(f"  ... ({hidden} more bead-count groups omitted)")
    return "\n".join(head)


def _format_group_summary_rows(
    valid_counts: np.ndarray,
    unique_bead_counts: list[int],
    structures: list[np.ndarray],
    valid_indices: np.ndarray,
    cutoff: float,
) -> str:
    rows = [
        "  beads  structures  weighted_avg  mean_avg   std_avg   min_avg   max_avg"
    ]
    for bead_count in unique_bead_counts:
        group_indices = valid_indices[valid_counts[valid_indices] == bead_count]
        group_stats = _compute_sample_stats(structures, group_indices, cutoff)
        rows.append(
            f"  {bead_count:>5}  {group_indices.size:>10}  "
            f"{group_stats['weighted_avg']:>12.4f}  "
            f"{group_stats['mean_structure_avg']:>8.4f}  "
            f"{group_stats['std_structure_avg']:>8.4f}  "
            f"{group_stats['min_structure_avg']:>8.4f}  "
            f"{group_stats['max_structure_avg']:>8.4f}"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect avg_num_neighbors across dataset-wide and per-bead-count samples."
    )
    parser.add_argument("--npz", required=True, help="Path to dataset .npz file or compatible directory")
    parser.add_argument("--cutoff", type=float, required=True, help="Neighbor cutoff in Angstrom")
    parser.add_argument(
        "--sample-size",
        default="all",
        help="Random subset size for the overall estimate, or 'all'",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument(
        "--max-count-rows",
        type=int,
        default=20,
        help="Maximum number of bead-count histogram rows to print",
    )
    args = parser.parse_args()

    sample_size = _parse_sample_size(args.sample_size)
    if args.cutoff <= 0:
        raise ValueError("--cutoff must be > 0")

    dataset = load_npz(args.npz)
    R = np.asarray(dataset["R"], dtype=np.float32)
    mask = dataset.get("mask", None)
    structures, valid_counts, padded_detected, capacity = _strip_padding(R, mask)
    valid_indices = np.flatnonzero(valid_counts > 0).astype(np.int32)

    if valid_indices.size == 0:
        raise ValueError("dataset contains no valid structures")

    unique_bead_counts = sorted({int(valid_counts[idx]) for idx in valid_indices})
    sampled_indices = _resolve_indices(valid_indices, sample_size, args.seed)
    overall = _compute_sample_stats(structures, sampled_indices, args.cutoff)

    print("=" * 80)
    print("AVG_NUM_NEIGHBORS DATASET SCAN")
    print("=" * 80)
    print(f"dataset:               {args.npz}")
    print(f"frames:                {R.shape[0]}")
    print(f"original capacity:     {capacity}")
    print(f"valid structures:      {valid_indices.size}")
    print(f"padding detected:      {padded_detected}")
    print(f"cutoff:                {args.cutoff:.3f} A")
    print(f"overall sample size:   {sampled_indices.size}")
    print(f"seed:                  {args.seed}")
    print(f"unique bead counts:    {unique_bead_counts}")

    print("\nOverall sampled estimate")
    print("-" * 80)
    print(f"weighted avg_num_neighbors:   {overall['weighted_avg']:.4f}")
    print(f"mean(structure averages):     {overall['mean_structure_avg']:.4f}")
    print(f"std(structure averages):      {overall['std_structure_avg']:.4f}")
    print(f"min/max structure average:    {overall['min_structure_avg']:.4f} / {overall['max_structure_avg']:.4f}")
    print(f"total valid beads sampled:    {overall['n_nodes']}")

    print("\nBead-count distribution")
    print("-" * 80)
    print(_format_counts(valid_counts[valid_indices], max_rows=args.max_count_rows))

    print("\nPer unique bead-count summary")
    print("-" * 80)
    print(
        _format_group_summary_rows(
            valid_counts=valid_counts,
            unique_bead_counts=unique_bead_counts,
            structures=structures,
            valid_indices=valid_indices,
            cutoff=args.cutoff,
        )
    )

    print("\nInterpretation")
    print("-" * 80)
    print(
        "If the per-size averages are close, a single fixed avg_num_neighbors is probably "
        "a reasonable global scale. If different bead-count groups separate clearly, one "
        "fixed value becomes a compromise and can bias force amplitude calibration across subsets."
    )


if __name__ == "__main__":
    main()
