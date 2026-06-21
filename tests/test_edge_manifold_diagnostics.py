import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.edge_manifold_diagnostics import (
    compute_latent_mahalanobis_scores,
    compute_radial_alpha,
    fit_latent_type_pair_stats,
    generate_ood_batches,
    load_radial_stats_from_artifact,
    split_indices_with_optional_protein_holdout,
    summarize_outlier_scores,
)


class EdgeManifoldDiagnosticsTests(unittest.TestCase):
    def test_ood_batches_preserve_masks_and_expected_geometries(self):
        R = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            ],
            dtype=np.float32,
        )
        mask = np.ones((1, 4), dtype=np.float32)
        species = np.array([[0, 1, 1, 2]], dtype=np.int32)
        radial_min = np.full((3, 3), 0.8, dtype=np.float32)

        batches = generate_ood_batches(
            R,
            mask,
            species,
            radial_min=radial_min,
            seed=3,
            noise_stds=(0.05,),
            clash_factors=(0.5,),
            stretch_scales=(1.5,),
            clash_edges_per_frame=2,
        )

        self.assertIn("clean", batches)
        self.assertIn("noise_0.05", batches)
        self.assertIn("clash_0.5xmin", batches)
        self.assertIn("stretch_1.5x", batches)
        self.assertIn("angular_shell_shuffle", batches)
        np.testing.assert_allclose(batches["clean"].R, R)
        np.testing.assert_allclose(batches["noise_0.05"].mask, mask)

        clean_d = np.linalg.norm(R[0, 1:] - R[0, :1], axis=1)
        stretch_d = np.linalg.norm(batches["stretch_1.5x"].R[0, 1:] - batches["stretch_1.5x"].R[0, :1], axis=1)
        self.assertGreater(float(stretch_d.mean()), float(clean_d.mean()))

        clash = batches["clash_0.5xmin"]
        self.assertGreaterEqual(int(clash.touched_mask.sum()), 3)
        touched = np.flatnonzero(clash.touched_mask[0])
        self.assertGreaterEqual(touched.size, 3)
        clash_d = np.linalg.norm(clash.R[0, touched[1]] - clash.R[0, touched[0]])
        self.assertLessEqual(float(clash_d), 0.4 + 1e-5)

        angular_d = np.sort(np.linalg.norm(batches["angular_shell_shuffle"].R[0, 1:] - batches["angular_shell_shuffle"].R[0, :1], axis=1))
        np.testing.assert_allclose(angular_d, np.sort(clean_d), atol=1e-5)

    def test_radial_alpha_matches_onset_offset_and_unseen_pair_floor(self):
        distances = np.array([1.2, 1.0, 0.9, 1.8, 2.0, 2.2, 1.5], dtype=np.float32)
        sender_types = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int32)
        receiver_types = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        min_distance = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        max_distance = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        count = np.array([[0, 5], [0, 0]], dtype=np.int32)

        alpha = compute_radial_alpha(
            distances,
            sender_types,
            receiver_types,
            min_distance,
            max_distance,
            count,
            onset_percent=0.10,
            offset_percent=0.10,
            floor=0.0,
        )

        self.assertAlmostEqual(float(alpha[0]), 1.0, places=6)
        self.assertLess(float(alpha[1]), 1.0)
        self.assertGreater(float(alpha[1]), 0.0)
        self.assertAlmostEqual(float(alpha[2]), 0.0, places=6)
        self.assertAlmostEqual(float(alpha[3]), 1.0, places=6)
        self.assertLess(float(alpha[4]), 1.0)
        self.assertGreater(float(alpha[4]), 0.0)
        self.assertAlmostEqual(float(alpha[5]), 0.0, places=6)
        self.assertAlmostEqual(float(alpha[6]), 0.0, places=6)

    def test_latent_type_pair_mahalanobis_scores_shifted_features_high(self):
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [-0.1, 0.1],
                [4.0, 4.0],
                [4.1, 3.9],
                [3.9, 4.1],
            ],
            dtype=np.float32,
        )
        sender_types = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        receiver_types = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
        valid = np.ones((6,), dtype=bool)

        stats = fit_latent_type_pair_stats(features, sender_types, receiver_types, valid, n_species=2)
        train_scores = compute_latent_mahalanobis_scores(features, sender_types, receiver_types, valid, stats)
        shifted_scores = compute_latent_mahalanobis_scores(features + 2.0, sender_types, receiver_types, valid, stats)
        unseen_score = compute_latent_mahalanobis_scores(
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([True]),
            stats,
        )

        self.assertLess(float(np.nanmean(train_scores)), 2.0)
        self.assertGreater(float(np.nanmean(shifted_scores)), float(np.nanmean(train_scores)) + 5.0)
        self.assertTrue(np.isinf(unseen_score[0]))

    def test_split_indices_uses_true_protein_holdout_when_available(self):
        protein_id = np.array([0, 0, 1, 1, 2, 2, 2, 3], dtype=np.int32)

        splits = split_indices_with_optional_protein_holdout(
            len(protein_id),
            seed=0,
            val_fraction=0.25,
            protein_id=protein_id,
            holdout_protein_id=2,
        )

        self.assertTrue(np.all(protein_id[splits["holdout"]] == 2))
        self.assertFalse(np.any(protein_id[splits["train"]] == 2))
        self.assertFalse(np.any(protein_id[splits["val"]] == 2))
        self.assertGreater(splits["train"].size, 0)
        self.assertGreater(splits["val"].size, 0)
        self.assertEqual(splits["holdout_source"], "protein_id=2")

    def test_load_radial_stats_from_artifact(self):
        out = Path("/tmp/cameo_edge_diag_radial_artifact_test.npz")
        min_distance = np.array([[0.0, 1.0], [1.2, 0.0]], dtype=np.float32)
        max_distance = np.array([[0.0, 2.0], [2.2, 0.0]], dtype=np.float32)
        count = np.array([[0, 5], [7, 0]], dtype=np.int32)
        np.savez(out, min_distance=min_distance, max_distance=max_distance, count=count)

        loaded_min, loaded_max, loaded_count = load_radial_stats_from_artifact(out, n_species=3)

        self.assertEqual(loaded_min.shape, (3, 3))
        np.testing.assert_allclose(loaded_min[:2, :2], min_distance)
        np.testing.assert_allclose(loaded_max[:2, :2], max_distance)
        np.testing.assert_array_equal(loaded_count[:2, :2], count)
        self.assertEqual(int(loaded_count[2, 2]), 0)

    def test_summarize_outlier_scores_uses_touched_subset_and_quantiles(self):
        alpha = np.array([1.0, 0.9, 0.2, 0.0], dtype=np.float32)
        score = np.array([0.0, 1.0, 5.0, 10.0], dtype=np.float32)
        valid = np.array([True, True, True, True])
        touched = np.array([False, True, True, False])

        summary = summarize_outlier_scores("latent", alpha=alpha, score=score, valid=valid, touched=touched)

        self.assertAlmostEqual(summary["latent_alpha_p01"], 0.006, places=3)
        self.assertAlmostEqual(summary["latent_score_p99"], 9.85, places=2)
        self.assertAlmostEqual(summary["latent_touched_frac_alpha_lt_0.95"], 1.0, places=6)
        self.assertAlmostEqual(summary["latent_touched_score_p99"], 4.96, places=2)


if __name__ == "__main__":
    unittest.main()
