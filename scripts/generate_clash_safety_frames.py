#!/usr/bin/env python3
"""Generate mild close-contact safety frames from normal training structures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessor import CoordinatePreprocessor


DEFAULT_DATASET = Path(
    "data_prep/datasets/dataset_1604_25pro_320_aggforce_allframes_padded/combined_dataset.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--variants-per-frame", type=int, default=1)
    p.add_argument("--seed", type=int, default=4129)
    p.add_argument("--target-min", type=float, default=2.4)
    p.add_argument("--target-max", type=float, default=3.1)
    p.add_argument("--candidate-min-distance", type=float, default=4.0)
    p.add_argument("--min-seq-sep", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=64)
    p.add_argument("--pair-r-cut", type=float, default=3.2)
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--buffer-multiplier", type=float, default=2.0)
    p.add_argument("--park-multiplier", type=float, default=0.95)
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Do not center/park input coordinates before generating clashes.",
    )
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return p.parse_args()


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def valid_pairs(
    coords: np.ndarray,
    valid_idx: np.ndarray,
    min_seq_sep: int,
    min_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    seq_sep = np.abs(valid_idx[:, None] - valid_idx[None, :])
    allowed = (
        np.triu(np.ones_like(dist, dtype=bool), k=1)
        & (seq_sep > int(min_seq_sep))
        & (dist > float(min_distance))
    )
    i, j = np.where(allowed)
    return i, j, dist


def close_pairs(
    R: np.ndarray,
    mask: np.ndarray,
    max_pairs: int,
    r_cut: float,
    min_seq_sep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.flatnonzero(mask > 0)
    pair_i = np.zeros((max_pairs,), dtype=np.int32)
    pair_j = np.zeros((max_pairs,), dtype=np.int32)
    pair_mask = np.zeros((max_pairs,), dtype=np.float32)
    if valid.size < 2:
        return pair_i, pair_j, pair_mask
    coords = R[valid]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    seq_sep = np.abs(valid[:, None] - valid[None, :])
    allowed = (
        np.triu(np.ones_like(dist, dtype=bool), k=1)
        & (seq_sep > int(min_seq_sep))
        & (dist < float(r_cut))
    )
    local_i, local_j = np.where(allowed)
    if local_i.size == 0:
        allowed = np.triu(np.ones_like(dist, dtype=bool), k=1) & (seq_sep > int(min_seq_sep))
        local_i, local_j = np.where(allowed)
    if local_i.size == 0:
        return pair_i, pair_j, pair_mask
    order = np.argsort(dist[local_i, local_j])
    n = min(max_pairs, order.size)
    pair_i[:n] = valid[local_i[order[:n]]].astype(np.int32)
    pair_j[:n] = valid[local_j[order[:n]]].astype(np.int32)
    pair_mask[:n] = 1.0
    return pair_i, pair_j, pair_mask


def compress_pair(
    R: np.ndarray,
    i: int,
    j: int,
    target_distance: float,
) -> np.ndarray:
    out = np.array(R, copy=True)
    vec = out[i] - out[j]
    dist = float(np.linalg.norm(vec))
    if dist <= 1e-6:
        return out
    direction = vec / dist
    delta = dist - float(target_distance)
    if delta <= 0.0:
        return out
    # Move both beads symmetrically, preserving the pair midpoint and COM.
    out[i] = out[i] - 0.5 * delta * direction
    out[j] = out[j] + 0.5 * delta * direction
    return out


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input, args.repo_root)
    output_path = resolve(args.output, args.repo_root)
    rng = np.random.default_rng(args.seed)

    with np.load(input_path, allow_pickle=True) as data:
        R_all = np.asarray(data["R"], dtype=np.float32)
        mask_all = np.asarray(
            data["mask"] if "mask" in data else np.ones(R_all.shape[:2], dtype=np.float32),
            dtype=np.float32,
        )
        species_all = np.asarray(
            data["species"] if "species" in data else np.zeros(R_all.shape[:2], dtype=np.int32),
            dtype=np.int32,
        )

    if not args.no_preprocess:
        preprocessor = CoordinatePreprocessor(
            cutoff=float(args.cutoff),
            buffer_multiplier=float(args.buffer_multiplier),
            park_multiplier=float(args.park_multiplier),
        )
        extent, shift = preprocessor.compute_box_extent(R_all, mask_all)
        R_all = np.asarray(
            preprocessor.center_and_park(R_all, mask_all, extent, shift),
            dtype=np.float32,
        )

    frame_count = min(int(args.n_frames), int(R_all.shape[0]))
    source_frames = rng.choice(np.arange(R_all.shape[0]), size=frame_count, replace=False)

    out_R: List[np.ndarray] = []
    out_mask: List[np.ndarray] = []
    out_species: List[np.ndarray] = []
    out_pair_i: List[np.ndarray] = []
    out_pair_j: List[np.ndarray] = []
    out_pair_mask: List[np.ndarray] = []
    out_source: List[int] = []

    for frame_idx in source_frames:
        R0 = R_all[frame_idx]
        mask = mask_all[frame_idx]
        species = species_all[frame_idx]
        valid = np.flatnonzero(mask > 0)
        if valid.size < 2:
            continue
        coords = R0[valid]
        cand_i, cand_j, dist = valid_pairs(
            coords,
            valid,
            min_seq_sep=args.min_seq_sep,
            min_distance=args.candidate_min_distance,
        )
        if cand_i.size == 0:
            continue
        for _ in range(int(args.variants_per_frame)):
            pick = int(rng.integers(0, cand_i.size))
            i = int(valid[cand_i[pick]])
            j = int(valid[cand_j[pick]])
            target = float(rng.uniform(args.target_min, args.target_max))
            R_bad = compress_pair(R0, i, j, target)
            pi, pj, pm = close_pairs(
                R_bad,
                mask,
                max_pairs=args.max_pairs,
                r_cut=args.pair_r_cut,
                min_seq_sep=args.min_seq_sep,
            )
            out_R.append(R_bad.astype(np.float32))
            out_mask.append(mask.astype(np.float32))
            out_species.append(species.astype(np.int32))
            out_pair_i.append(pi)
            out_pair_j.append(pj)
            out_pair_mask.append(pm)
            out_source.append(int(frame_idx))

    if not out_R:
        raise ValueError("No clash frames generated; relax candidate thresholds.")

    R = np.stack(out_R, axis=0).astype(np.float32)
    mask = np.stack(out_mask, axis=0).astype(np.float32)
    species = np.stack(out_species, axis=0).astype(np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        R=R,
        F=np.zeros_like(R, dtype=np.float32),
        mask=mask,
        species=species,
        force_loss_mask=np.zeros(mask.shape, dtype=np.float32),
        safety_loss_mask=mask,
        safety_pair_i=np.stack(out_pair_i, axis=0),
        safety_pair_j=np.stack(out_pair_j, axis=0),
        safety_pair_mask=np.stack(out_pair_mask, axis=0),
        sample_kind=np.full((R.shape[0],), 2, dtype=np.int32),
        source_step=np.asarray(out_source, dtype=np.int32),
    )
    print(f"Wrote {R.shape[0]} generated clash frames to {output_path}")


if __name__ == "__main__":
    main()
