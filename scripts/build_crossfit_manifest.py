#!/usr/bin/env python3
"""Build deterministic contiguous/grouped cross-fit index manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(n_frames: int, groups: np.ndarray, n_folds: int, guard: int):
    fold_id = np.full((n_frames,), -1, dtype=np.int16)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        for fold, block in enumerate(np.array_split(indices, n_folds)):
            fold_id[block] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("Not every frame was assigned to a fold.")

    result = {"fold_id": fold_id, "n_frames": np.asarray(n_frames, dtype=np.int64)}
    for fold in range(n_folds):
        holdout = fold_id == fold
        excluded = holdout.copy()
        if guard > 0:
            for group in np.unique(groups):
                group_indices = np.flatnonzero(groups == group)
                local_holdout = holdout[group_indices]
                if not np.any(local_holdout):
                    continue
                local_excluded = np.zeros(local_holdout.shape, dtype=bool)
                for center in np.flatnonzero(local_holdout):
                    lo = max(0, int(center) - guard)
                    hi = min(local_holdout.size, int(center) + guard + 1)
                    local_excluded[lo:hi] = True
                excluded[group_indices[local_excluded]] = True
        result[f"train_indices_{fold}"] = np.flatnonzero(~excluded).astype(np.int64)
        result[f"holdout_indices_{fold}"] = np.flatnonzero(holdout).astype(np.int64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--guard-frames", type=int, default=0)
    parser.add_argument("--group-key", default=None)
    args = parser.parse_args()
    if args.n_folds < 2:
        raise ValueError("--n-folds must be >= 2")
    if args.guard_frames < 0:
        raise ValueError("--guard-frames must be >= 0")

    with np.load(args.dataset, allow_pickle=True) as source:
        n_frames = int(source["R"].shape[0])
        if args.group_key is None:
            groups = np.zeros((n_frames,), dtype=np.int64)
        elif args.group_key not in source:
            raise KeyError(f"Dataset does not contain group key {args.group_key!r}")
        else:
            raw_groups = np.asarray(source[args.group_key])
            _, groups = np.unique(raw_groups, return_inverse=True)

    arrays = build_manifest(n_frames, groups, args.n_folds, args.guard_frames)
    metadata = {
        "version": 1,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "n_frames": n_frames,
        "n_folds": args.n_folds,
        "guard_frames": args.guard_frames,
        "group_key": args.group_key,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.parent / f".{args.output.name}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, args.output)
    print(json.dumps(metadata, indent=2))
    for fold in range(args.n_folds):
        print(
            f"fold {fold}: train={len(arrays[f'train_indices_{fold}'])} "
            f"holdout={len(arrays[f'holdout_indices_{fold}'])}"
        )


if __name__ == "__main__":
    main()
