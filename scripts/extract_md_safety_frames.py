#!/usr/bin/env python3
"""Extract off-manifold safety frames from an MD trajectory NPZ.

The output is a small NPZ dataset for ``training.safety_regularization``.  It
does not invent reference forces: ``F`` is zero and ``force_loss_mask`` is zero,
so normal force matching ignores these frames.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


DEFAULT_TRAJ = Path(
    "local_work/md_runs/20260519_25pro_aggforce/base_bond_angle_dihedral_ev_wca/"
    "traj_4zoh_25pro_aggforce_base_bond_angle_dihedral_ev_wca_epoch00420_nvt_10000steps.npz"
)
DEFAULT_METRICS = Path(
    "local_work/md_runs/20260519_25pro_aggforce/base_bond_angle_dihedral_ev_wca/"
    "force_component_analysis_stride5/frame_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=str, default="", help="Comma-separated saved frame indices")
    p.add_argument("--d-min-lt", type=float, default=3.2)
    p.add_argument("--wca-gt", type=float, default=None)
    p.add_argument("--ml-force-rms-gt", type=float, default=None)
    p.add_argument("--pre-window", type=int, default=2)
    p.add_argument("--post-window", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=128)
    p.add_argument("--max-pairs", type=int, default=64)
    p.add_argument("--pair-r-cut", type=float, default=3.2)
    p.add_argument("--min-seq-sep", type=int, default=1)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    return p.parse_args()


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_frames(raw: str) -> List[int]:
    out = []
    for piece in raw.replace("[", "").replace("]", "").split(","):
        text = piece.strip()
        if text:
            out.append(int(text))
    return list(dict.fromkeys(out))


def load_metrics(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            frame = int(float(row["frame_index"]))
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows[frame] = parsed
    return rows


def min_pair_distance(R: np.ndarray, mask: np.ndarray, min_seq_sep: int) -> float:
    valid = np.flatnonzero(mask > 0)
    if valid.size < 2:
        return float("inf")
    coords = R[valid]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    seq_sep = np.abs(valid[:, None] - valid[None, :])
    allowed = seq_sep > int(min_seq_sep)
    dist = np.where(allowed, dist, np.inf)
    return float(np.min(dist))


def select_frames(args: argparse.Namespace, R: np.ndarray, mask: np.ndarray, steps: np.ndarray) -> List[int]:
    metrics = load_metrics(args.metrics)
    selected = set(parse_frames(args.frames))

    for frame_idx in range(R.shape[0]):
        row = metrics.get(frame_idx)
        hit = False
        if row is not None:
            if args.d_min_lt is not None and row.get("min_pair_distance", np.inf) < args.d_min_lt:
                hit = True
            if args.wca_gt is not None and row.get("E_wca", -np.inf) > args.wca_gt:
                hit = True
            if args.ml_force_rms_gt is not None and row.get("F_ml_rms", -np.inf) > args.ml_force_rms_gt:
                hit = True
        elif args.d_min_lt is not None:
            hit = min_pair_distance(R[frame_idx], mask, args.min_seq_sep) < args.d_min_lt

        if hit:
            for idx in range(frame_idx - args.pre_window, frame_idx + args.post_window + 1):
                if 0 <= idx < R.shape[0]:
                    selected.add(idx)

    final = sorted(selected)
    if args.max_frames > 0 and len(final) > args.max_frames:
        # Preserve temporal coverage with an even subsample.
        picks = np.linspace(0, len(final) - 1, num=args.max_frames).round().astype(int)
        final = [final[i] for i in sorted(set(picks))]
    if not final:
        raise ValueError("No frames selected; relax thresholds or pass --frames.")
    for idx in final:
        if idx < 0 or idx >= R.shape[0]:
            raise ValueError(f"Frame index {idx} out of range for trajectory with {R.shape[0]} frames")
    return final


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
    upper = np.triu(np.ones_like(dist, dtype=bool), k=1)
    allowed = upper & (seq_sep > int(min_seq_sep)) & (dist < float(r_cut))
    local_i, local_j = np.where(allowed)
    if local_i.size == 0:
        allowed = upper & (seq_sep > int(min_seq_sep))
        local_i, local_j = np.where(allowed)
    if local_i.size == 0:
        return pair_i, pair_j, pair_mask

    order = np.argsort(dist[local_i, local_j])
    n = min(max_pairs, order.size)
    chosen_i = valid[local_i[order[:n]]]
    chosen_j = valid[local_j[order[:n]]]
    pair_i[:n] = chosen_i.astype(np.int32)
    pair_j[:n] = chosen_j.astype(np.int32)
    pair_mask[:n] = 1.0
    return pair_i, pair_j, pair_mask


def main() -> None:
    args = parse_args()
    traj_path = resolve(args.traj, args.repo_root)
    args.metrics = resolve(args.metrics, args.repo_root)
    out_path = resolve(args.output, args.repo_root)

    with np.load(traj_path, allow_pickle=True) as data:
        R_all = np.asarray(data["R"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.float32)
        species = np.asarray(data["species"], dtype=np.int32)
        steps = np.asarray(data["step"], dtype=np.int32) if "step" in data else np.arange(R_all.shape[0])

    if mask.ndim == 1:
        mask_all = np.broadcast_to(mask[None, :], R_all.shape[:2]).astype(np.float32)
    else:
        mask_all = mask.astype(np.float32)
    if species.ndim == 1:
        species_all = np.broadcast_to(species[None, :], R_all.shape[:2]).astype(np.int32)
    else:
        species_all = species.astype(np.int32)

    frames = select_frames(args, R_all, mask_all[0], steps)
    R = R_all[frames]
    mask_out = mask_all[frames] if mask_all.shape[0] == R_all.shape[0] else mask_all[: len(frames)]
    species_out = (
        species_all[frames] if species_all.shape[0] == R_all.shape[0] else species_all[: len(frames)]
    )

    pair_i = []
    pair_j = []
    pair_mask = []
    for coords, m in zip(R, mask_out):
        pi, pj, pm = close_pairs(
            coords,
            m,
            max_pairs=args.max_pairs,
            r_cut=args.pair_r_cut,
            min_seq_sep=args.min_seq_sep,
        )
        pair_i.append(pi)
        pair_j.append(pj)
        pair_mask.append(pm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        R=R.astype(np.float32),
        F=np.zeros_like(R, dtype=np.float32),
        mask=mask_out.astype(np.float32),
        species=species_out.astype(np.int32),
        force_loss_mask=np.zeros(mask_out.shape, dtype=np.float32),
        safety_loss_mask=mask_out.astype(np.float32),
        safety_pair_i=np.stack(pair_i, axis=0),
        safety_pair_j=np.stack(pair_j, axis=0),
        safety_pair_mask=np.stack(pair_mask, axis=0),
        sample_kind=np.ones((len(frames),), dtype=np.int32),
        source_step=steps[frames].astype(np.int32),
        selected_frame=np.asarray(frames, dtype=np.int32),
    )
    print(f"Wrote {len(frames)} safety frames to {out_path}")


if __name__ == "__main__":
    main()
