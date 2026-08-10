import numpy as np
import pytest

from config.manager import ConfigManager
from data.loader import build_tiled_dataset
from scripts.train import _cell_list_box_for_tiled_split


def _toy_structures(n_structures=27):
    base = np.asarray(
        [[-1.0, -0.4, 0.0], [0.2, 0.6, -0.3], [1.1, -0.2, 0.5]],
        dtype=np.float32,
    )
    R = np.stack([base + np.asarray([i, -2 * i, i / 3]) for i in range(n_structures)])
    F = np.zeros_like(R)
    mask = np.ones(R.shape[:2], dtype=np.float32)
    species = np.ones(R.shape[:2], dtype=np.int32)
    return R, F, mask, species


def _tile(layout):
    R, F, mask, species = _toy_structures()
    return build_tiled_dataset(
        R,
        F,
        mask,
        species,
        target_beads=81,
        spatial_separation=True,
        spatial_layout=layout,
        structure_gap=10.0,
    )


def test_grid_3d_preserves_structures_and_separates_segments():
    original, _, _, _ = _toy_structures()
    tiled = _tile("grid_3d")
    coords = tiled["R"][0, tiled["mask"][0] > 0]
    segments = tiled["segment_id"][0, tiled["mask"][0] > 0]

    assert np.min(coords) >= 0.0
    reference_distances = np.linalg.norm(
        original[0, :, None, :] - original[0, None, :, :], axis=-1
    )
    for segment in np.unique(segments):
        segment_coords = coords[segments == segment]
        distances = np.linalg.norm(
            segment_coords[:, None, :] - segment_coords[None, :, :], axis=-1
        )
        np.testing.assert_allclose(distances, reference_distances, atol=2e-6)

    for left in np.unique(segments):
        for right in np.unique(segments):
            if right <= left:
                continue
            distances = np.linalg.norm(
                coords[segments == left, None, :] - coords[None, segments == right, :],
                axis=-1,
            )
            assert np.min(distances) >= 10.0 - 2e-6


def test_grid_3d_is_compact_relative_to_legacy_line_layout():
    grid = _tile("grid_3d")
    line = _tile("line_x")
    grid_coords = grid["R"][0, grid["mask"][0] > 0]
    line_coords = line["R"][0, line["mask"][0] > 0]
    grid_extent = np.ptp(grid_coords, axis=0)
    line_extent = np.ptp(line_coords, axis=0)

    assert np.all(grid_extent > 0.0)
    assert grid_extent[0] < line_extent[0] / 3.0


def test_unknown_spatial_layout_is_rejected():
    R, F, mask, species = _toy_structures(2)
    with pytest.raises(ValueError, match="Unsupported spatial_layout"):
        build_tiled_dataset(
            R,
            F,
            mask,
            species,
            target_beads=6,
            spatial_separation=True,
            spatial_layout="diagonal",
        )


def test_config_spatial_layout_aliases_and_validation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  tile_spatial_layout: 3d\nmodel: {}\ntraining: {}\noptimizer: {}\n"
    )
    config = ConfigManager(config_path)
    assert config.get_tile_spatial_layout() == "grid_3d"
    config_path.write_text(
        "data:\n  tile_spatial_layout: diagonal\nmodel: {}\ntraining: {}\noptimizer: {}\n"
    )
    invalid = ConfigManager(config_path)
    with pytest.raises(ValueError, match="tile_spatial_layout"):
        invalid.get_tile_spatial_layout()


class _CellListConfig:
    def get_batch_mode(self):
        return "tiled"

    def neighbor_disable_cell_list_enabled(self):
        return False

    def use_pbc_enabled(self):
        return False

    def get_cutoff(self):
        return 5.5

    def get_dr_threshold(self):
        return 1.0

    def get_tile_spatial_layout(self):
        return "grid_3d"

    def tile_spatial_separation_enabled(self):
        return True

    def tile_rebuild_each_epoch_enabled(self):
        return False


def test_cell_list_box_covers_positive_packed_coordinates():
    split = {
        "R": np.asarray([[[1.0, 2.0, 3.0], [8.0, 9.0, 10.0]]], dtype=np.float32),
        "mask": np.ones((1, 2), dtype=np.float32),
    }
    box = np.asarray(_cell_list_box_for_tiled_split(_CellListConfig(), split, None))
    np.testing.assert_allclose(box, [14.5, 15.5, 16.5])


def test_cell_list_box_rejects_negative_legacy_packing():
    split = {
        "R": np.asarray([[[1.0, -0.1, 3.0], [8.0, 9.0, 10.0]]], dtype=np.float32),
        "mask": np.ones((1, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="grid_3d packing"):
        _cell_list_box_for_tiled_split(_CellListConfig(), split, None)


def test_cell_list_box_requires_spatial_separation():
    class NoSeparation(_CellListConfig):
        def tile_spatial_separation_enabled(self):
            return False

    split = {
        "R": np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        "mask": np.ones((1, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="tile_spatial_separation=true"):
        _cell_list_box_for_tiled_split(NoSeparation(), split, None)


def test_cell_list_box_rejects_epoch_rebuild_with_fixed_box():
    class RebuildEachEpoch(_CellListConfig):
        def tile_rebuild_each_epoch_enabled(self):
            return True

    split = {
        "R": np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        "mask": np.ones((1, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="tile_rebuild_each_epoch=false"):
        _cell_list_box_for_tiled_split(RebuildEachEpoch(), split, None)
