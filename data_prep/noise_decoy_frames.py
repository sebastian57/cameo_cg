#!/usr/bin/env python3
"""Add Gaussian-noised decoy frames to an NPZ force-matching dataset.

The selected frames are duplicated, noised, and appended to the dataset.  By
default decoys are assigned zero force labels for delta/residual learning: they
teach the learned ML correction to vanish in distorted regions while fixed
priors provide the restoring force.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def _frame_count(dataset: Dict[str, Any]) -> int:
    if "R" not in dataset:
        raise KeyError("Dataset must contain key 'R'.")
    return int(np.asarray(dataset["R"]).shape[0])


def _selected_indices(
    n_frames: int,
    every_n: int,
    source_name: Optional[np.ndarray] = None,
) -> np.ndarray:
    if every_n <= 0:
        raise ValueError(f"every_n must be > 0, got {every_n}.")
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.int32)

    if source_name is None:
        return np.arange(0, n_frames, every_n, dtype=np.int32)

    src = np.asarray(source_name)
    if src.shape[0] != n_frames:
        return np.arange(0, n_frames, every_n, dtype=np.int32)

    selected = []
    for value in np.unique(src):
        group_idx = np.flatnonzero(src == value)
        selected.extend(group_idx[::every_n].tolist())
    return np.asarray(sorted(selected), dtype=np.int32)


def add_noised_decoy_frames(
    dataset: Dict[str, Any],
    every_n: int,
    sigma: float,
    seed: int = 0,
    source_key: str = "source_name",
    zero_force_decoys: bool = True,
) -> Dict[str, Any]:
    """Return a copy of ``dataset`` with appended noised decoy frames."""
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}.")

    n_frames = _frame_count(dataset)
    source_name = dataset.get(source_key)
    selected = _selected_indices(n_frames, int(every_n), source_name=source_name)
    if selected.size == 0:
        out = dict(dataset)
        out["noise_decoy_source_index"] = np.full((n_frames,), -1, dtype=np.int32)
        out["noise_decoy_mask"] = np.zeros((n_frames,), dtype=np.float32)
        return out

    out: Dict[str, Any] = {}
    R = np.asarray(dataset["R"], dtype=np.float32)
    mask = np.asarray(
        dataset.get("mask", np.ones(R.shape[:2], dtype=np.float32)),
        dtype=np.float32,
    )

    rng = np.random.default_rng(int(seed))
    R_decoy = np.array(R[selected], copy=True)
    mask_decoy = np.asarray(mask[selected], dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=R_decoy.shape).astype(np.float32)
    noise *= mask_decoy[..., None]
    n_valid = np.maximum(np.sum(mask_decoy, axis=1, keepdims=True), 1.0).astype(np.float32)
    noise_mean = np.sum(noise, axis=1, keepdims=True) / n_valid[:, :, None]
    noise = (noise - noise_mean) * mask_decoy[..., None]
    R_decoy = R_decoy + noise

    for key, value in dataset.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n_frames,):
            if key == "R":
                decoy = R_decoy
            elif key == "F" and zero_force_decoys:
                decoy = np.zeros_like(arr[selected], dtype=arr.dtype)
            else:
                decoy = arr[selected]
            out[key] = np.concatenate([arr, decoy], axis=0)
        else:
            out[key] = value

    out["noise_decoy_source_index"] = np.concatenate(
        [
            np.full((n_frames,), -1, dtype=np.int32),
            selected.astype(np.int32),
        ],
        axis=0,
    )
    out["noise_decoy_mask"] = np.concatenate(
        [
            np.zeros((n_frames,), dtype=np.float32),
            np.ones((selected.size,), dtype=np.float32),
        ],
        axis=0,
    )
    out["noise_decoy_zero_force"] = np.concatenate(
        [
            np.zeros((n_frames,), dtype=np.float32),
            np.full((selected.size,), 1.0 if zero_force_decoys else 0.0, dtype=np.float32),
        ],
        axis=0,
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--every-n", type=int, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-key", type=str, default="source_name")
    p.add_argument(
        "--copy-force-labels",
        action="store_true",
        help="Copy source force labels onto decoys instead of assigning zero delta labels.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        dataset = {key: data[key] for key in data.files}

    augmented = add_noised_decoy_frames(
        dataset,
        every_n=int(args.every_n),
        sigma=float(args.sigma),
        seed=int(args.seed),
        source_key=str(args.source_key),
        zero_force_decoys=not bool(args.copy_force_labels),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **augmented)
    n_original = int(np.asarray(dataset["R"]).shape[0])
    n_total = int(np.asarray(augmented["R"]).shape[0])
    print(
        f"Wrote {args.output}: original={n_original} decoys={n_total - n_original} total={n_total}"
    )


if __name__ == "__main__":
    main()
