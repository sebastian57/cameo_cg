"""Leakage-resistant index manifests for teacher cross-fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.logging import data_logger


def apply_crossfit_split(
    config,
    dataset: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[int]]:
    """Reorder a dataset as manifest train indices followed by held-out indices."""
    cfg = config.get_crossfit_config()
    if not cfg["enabled"]:
        return dataset, None
    path = Path(cfg["manifest_path"])
    if not path.is_absolute():
        path = (Path(config.config_path).parent / path).resolve()
    fold = int(cfg["held_out_fold"])
    with np.load(path, allow_pickle=False) as manifest:
        train_key = f"train_indices_{fold}"
        holdout_key = f"holdout_indices_{fold}"
        if train_key not in manifest or holdout_key not in manifest:
            raise ValueError(
                f"Cross-fit manifest {path} does not contain fold {fold}."
            )
        train_idx = np.asarray(manifest[train_key], dtype=np.int64)
        holdout_idx = np.asarray(manifest[holdout_key], dtype=np.int64)
        manifest_n = int(np.asarray(manifest["n_frames"]).item())
    n_frames = int(np.asarray(dataset["R"]).shape[0])
    if manifest_n != n_frames:
        raise ValueError(
            f"Cross-fit manifest has {manifest_n} frames but dataset has {n_frames}."
        )
    if np.intersect1d(train_idx, holdout_idx).size:
        raise ValueError("Cross-fit train and holdout indices overlap.")
    order = np.concatenate((train_idx, holdout_idx))
    if np.unique(order).size != order.size:
        raise ValueError("Cross-fit manifest contains duplicate selected indices.")

    reordered: Dict[str, Any] = {}
    for key, value in dataset.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == n_frames:
            reordered[key] = np.asarray(value)[order]
        else:
            reordered[key] = value
    data_logger.info(
        "[CrossFit] fold=%d manifest=%s training=%d holdout=%d excluded_by_guard=%d",
        fold,
        path,
        train_idx.size,
        holdout_idx.size,
        n_frames - order.size,
    )
    return reordered, int(train_idx.size)

