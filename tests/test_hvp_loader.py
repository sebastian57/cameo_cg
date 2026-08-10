import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.loader import DatasetLoader, build_tiled_dataset, load_npz
from scripts.train import _build_tiled_split_from_source, _build_tiled_train_source


class HVPLoaderTests(unittest.TestCase):
    def test_hvp_arrays_survive_load_slice_get_all_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hvp_data.npz"
            R = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
            F = -R
            mask = np.ones((4, 3), dtype=np.float32)
            species = np.tile(np.array([[1, 2, 3]], dtype=np.int32), (4, 1))
            hvp_probe = np.arange(4 * 2 * 3 * 3, dtype=np.float32).reshape(4, 2, 3, 3)
            HVP = hvp_probe + 1000.0
            hvp_loss_mask = np.ones((4, 2, 3), dtype=np.float32)
            np.savez(
                path,
                R=R,
                F=F,
                mask=mask,
                species=species,
                hvp_probe=hvp_probe,
                HVP=HVP,
                hvp_loss_mask=hvp_loss_mask,
            )

            raw = load_npz(path)
            loader = DatasetLoader(path, max_frames=3, seed=7)
            all_data = loader.get_all()
            train_loader, val_loader = loader.split_train_val(val_fraction=1 / 3)

        self.assertIn("hvp_probe", raw)
        self.assertIn("HVP", raw)
        self.assertEqual(loader.hvp_probe.shape, (3, 2, 3, 3))
        self.assertEqual(loader.HVP.shape, (3, 2, 3, 3))
        self.assertEqual(loader.hvp_loss_mask.shape, (3, 2, 3))
        np.testing.assert_allclose(all_data["hvp_probe"], loader.hvp_probe)
        np.testing.assert_allclose(all_data["HVP"], loader.HVP)
        self.assertEqual(train_loader.hvp_probe.shape[0], 2)
        self.assertEqual(val_loader.hvp_probe.shape[0], 1)
        np.testing.assert_allclose(train_loader.HVP, loader.HVP[:2])
        np.testing.assert_allclose(val_loader.HVP, loader.HVP[2:])


class HVPTiledPackingTests(unittest.TestCase):
    def test_tiled_dataset_packs_hvp_fields_by_valid_bead_order(self):
        R = np.zeros((2, 3, 3), dtype=np.float32)
        F = np.zeros_like(R)
        mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        species = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        hvp_probe = np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
        HVP = hvp_probe + 100.0
        hvp_loss_mask = np.array(
            [
                [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
                [[0.0, 3.0, 0.0], [0.0, 4.0, 0.0]],
            ],
            dtype=np.float32,
        )

        tiled = build_tiled_dataset(
            R=R,
            F=F,
            mask=mask,
            species=species,
            structure_ids=np.array([0, 1], dtype=np.int32),
            target_beads=4,
            sort_by_size=False,
            extra_per_atom_fields={
                "hvp_probe": hvp_probe,
                "HVP": HVP,
                "hvp_loss_mask": hvp_loss_mask,
            },
        )

        expected_probe = np.zeros((2, 4, 3), dtype=np.float32)
        expected_probe[:, 0:2, :] = np.take(hvp_probe[0], [0, 2], axis=1)
        expected_probe[:, 2:3, :] = np.take(hvp_probe[1], [1], axis=1)
        expected_hvp = np.zeros((2, 4, 3), dtype=np.float32)
        expected_hvp[:, 0:2, :] = np.take(HVP[0], [0, 2], axis=1)
        expected_hvp[:, 2:3, :] = np.take(HVP[1], [1], axis=1)
        expected_mask = np.zeros((2, 4), dtype=np.float32)
        expected_mask[:, 0:2] = np.take(hvp_loss_mask[0], [0, 2], axis=1)
        expected_mask[:, 2:3] = np.take(hvp_loss_mask[1], [1], axis=1)

        self.assertEqual(tiled["hvp_probe"].shape, (1, 2, 4, 3))
        self.assertEqual(tiled["HVP"].shape, (1, 2, 4, 3))
        self.assertEqual(tiled["hvp_loss_mask"].shape, (1, 2, 4))
        np.testing.assert_allclose(tiled["hvp_probe"][0], expected_probe)
        np.testing.assert_allclose(tiled["HVP"][0], expected_hvp)
        np.testing.assert_allclose(tiled["hvp_loss_mask"][0], expected_mask)



class _TiledConfigStub:
    def get_tile_target_beads(self):
        return 4

    def get_tile_bucket_beads(self):
        return None

    def get_tile_target_edges(self):
        return None

    def get_tile_bucket_edges(self):
        return None

    def get_tile_edge_estimate_scale(self):
        return 1.0

    def get_tile_edge_estimate_mode(self):
        return "valid_scaled"

    def get_tile_edge_estimate_cutoff(self):
        return None

    def tile_shuffle_structures_enabled(self):
        return False

    def tile_sort_by_size_enabled(self):
        return False

    def tile_sort_by_estimated_edges_enabled(self):
        return False

    def tile_drop_incomplete_enabled(self):
        return False

    def tile_isolate_large_structures_enabled(self):
        return False

    def get_tile_large_structure_threshold(self):
        return None

    def get_tile_large_structure_edge_threshold(self):
        return None

    def tile_spatial_separation_enabled(self):
        return False

    def get_tile_structure_gap(self):
        return 25.0

    def get_tile_spatial_layout(self):
        return "line_x"

    def get_batch_mode(self):
        return "tiled"

    def tile_rebuild_each_epoch_enabled(self):
        return False

    def get_static_neighbors_config(self):
        return {
            "enabled": False,
            "backend": "kdtree",
            "block_size": 1024,
            "capacity_multiplier": 1.0,
            "r_list": 6.0,
        }


class HVPTiledTrainScriptTests(unittest.TestCase):
    def test_tiled_train_source_and_split_preserve_hvp_fields(self):
        R = np.zeros((2, 3, 3), dtype=np.float32)
        F = np.zeros_like(R)
        mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        species = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        hvp_probe = np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
        HVP = hvp_probe + 100.0
        dataset = {
            "R": R,
            "F": F,
            "mask": mask,
            "species": species,
            "hvp_probe": hvp_probe,
            "HVP": HVP,
        }

        train_source = _build_tiled_train_source(dataset, n_train=2)
        tiled = _build_tiled_split_from_source(train_source, _TiledConfigStub(), seed=0)

        self.assertIn("hvp_probe", train_source)
        self.assertIn("HVP", train_source)
        self.assertIn("hvp_loss_mask", train_source)
        self.assertEqual(tiled["hvp_probe"].shape, (1, 2, 4, 3))
        self.assertEqual(tiled["HVP"].shape, (1, 2, 4, 3))
        self.assertEqual(tiled["hvp_loss_mask"].shape, (1, 2, 4))
        np.testing.assert_allclose(
            tiled["hvp_loss_mask"][0, :, 0:2],
            np.broadcast_to(mask[0, [0, 2]][None, :], (2, 2)),
        )
        np.testing.assert_allclose(
            tiled["hvp_loss_mask"][0, :, 2:3],
            np.broadcast_to(mask[1, [1]][None, :], (2, 1)),
        )

if __name__ == "__main__":
    unittest.main()
