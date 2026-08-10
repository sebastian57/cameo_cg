from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.build_crossfit_manifest import build_manifest
from training.crossfit import apply_crossfit_split


def test_grouped_manifest_has_complete_holdout_and_guarded_training() -> None:
    groups = np.asarray([0] * 9 + [1] * 7)
    arrays = build_manifest(len(groups), groups, n_folds=3, guard=1)
    holdout_count = np.zeros((len(groups),), dtype=np.int32)
    for fold in range(3):
        train = arrays[f"train_indices_{fold}"]
        holdout = arrays[f"holdout_indices_{fold}"]
        holdout_count[holdout] += 1
        assert np.intersect1d(train, holdout).size == 0
        for idx in holdout:
            same_group_neighbors = [
                j for j in (idx - 1, idx + 1)
                if 0 <= j < len(groups) and groups[j] == groups[idx]
            ]
            assert not np.intersect1d(train, same_group_neighbors).size
    np.testing.assert_array_equal(holdout_count, 1)


def test_apply_crossfit_split_reorders_train_then_holdout(tmp_path) -> None:
    groups = np.zeros((12,), dtype=np.int32)
    arrays = build_manifest(12, groups, n_folds=3, guard=0)
    path = tmp_path / "manifest.npz"
    np.savez(path, **arrays)
    config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        get_crossfit_config=lambda: {
            "enabled": True,
            "manifest_path": str(path),
            "held_out_fold": 1,
        },
    )
    dataset = {
        "R": np.arange(12, dtype=np.float32)[:, None, None],
        "F": np.zeros((12, 1, 1), dtype=np.float32),
        "metadata": np.asarray(7),
    }
    reordered, n_train = apply_crossfit_split(config, dataset)
    train = arrays["train_indices_1"]
    holdout = arrays["holdout_indices_1"]
    assert n_train == len(train)
    np.testing.assert_array_equal(reordered["R"][:, 0, 0], np.concatenate((train, holdout)))
    assert int(reordered["metadata"]) == 7

