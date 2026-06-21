import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
import numpy as np

from training.edge_distance_gate import (
    EdgeDistanceGateBank,
    build_edge_distance_gate_stats,
    compute_edge_distance_gate,
    save_edge_distance_gate_stats,
    select_clean_training_frames,
)


class EdgeDistanceGateTest(unittest.TestCase):
    def test_training_frame_selection_matches_shuffle_and_split(self):
        R = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
        mask = np.ones((5, 2), dtype=np.float32)
        species = np.zeros((5, 2), dtype=np.int32)

        selected_R, selected_mask, selected_species = select_clean_training_frames(
            R=R,
            mask=mask,
            species=species,
            seed=7,
            val_fraction=0.4,
        )

        rng = np.random.RandomState(7)
        order = np.arange(5)
        rng.shuffle(order)
        expected = R[order[:3]]

        self.assertTrue(np.array_equal(selected_R, expected))
        self.assertEqual(selected_mask.shape, (3, 2))
        self.assertEqual(selected_species.shape, (3, 2))

    def test_stats_are_directed_type_pair_bounds_with_cutoff(self):
        R = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        mask = np.ones((2, 3), dtype=np.float32)
        species = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)

        stats = build_edge_distance_gate_stats(
            R=R,
            mask=mask,
            species=species,
            cutoff=2.0,
            n_species=3,
        )

        np.testing.assert_allclose(stats.min_distance[0, 1], 1.0)
        np.testing.assert_allclose(stats.max_distance[0, 1], 1.5)
        np.testing.assert_allclose(stats.min_distance[1, 0], 1.0)
        np.testing.assert_allclose(stats.max_distance[1, 0], 1.5)
        self.assertEqual(int(stats.count[0, 1]), 2)
        self.assertEqual(int(stats.count[1, 0]), 2)
        self.assertEqual(int(stats.count[0, 2]), 0)
        self.assertEqual(int(stats.count[2, 0]), 0)

    def test_gate_is_one_inside_smooth_to_floor_outside_and_unseen_is_floor(self):
        bank = EdgeDistanceGateBank(
            min_distance=jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
            max_distance=jnp.array([[0.0, 2.0], [2.0, 0.0]], dtype=jnp.float32),
            count=jnp.array([[0, 3], [3, 0]], dtype=jnp.int32),
            falloff_percent=0.05,
            floor=0.0,
            stop_gradient=True,
        )
        distances = jnp.array([1.5, 2.05, 2.1, 1.0], dtype=jnp.float32)
        senders = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        receivers = jnp.array([1, 1, 1, 0], dtype=jnp.int32)
        species = jnp.array([0, 1], dtype=jnp.int32)
        valid_edges = jnp.array([True, True, True, True])

        alpha = compute_edge_distance_gate(
            distances=distances,
            senders=senders,
            receivers=receivers,
            species=species,
            valid_edges=valid_edges,
            bank=bank,
        )

        self.assertAlmostEqual(float(alpha[0]), 1.0, places=6)
        self.assertGreater(float(alpha[1]), 0.0)
        self.assertLess(float(alpha[1]), 1.0)
        self.assertAlmostEqual(float(alpha[2]), 0.0, places=6)
        self.assertAlmostEqual(float(alpha[3]), 0.0, places=6)

    def test_onset_offset_gate_starts_inside_observed_range(self):
        bank = EdgeDistanceGateBank(
            min_distance=jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
            max_distance=jnp.array([[0.0, 2.0], [2.0, 0.0]], dtype=jnp.float32),
            count=jnp.array([[0, 3], [3, 0]], dtype=jnp.int32),
            falloff_percent=0.05,
            onset_percent=0.10,
            offset_percent=0.10,
            floor=0.0,
            stop_gradient=True,
        )
        distances = jnp.array([1.2, 1.1, 0.9, 1.8, 2.0, 2.2], dtype=jnp.float32)
        senders = jnp.zeros((6,), dtype=jnp.int32)
        receivers = jnp.ones((6,), dtype=jnp.int32)
        species = jnp.array([0, 1], dtype=jnp.int32)
        valid_edges = jnp.ones((6,), dtype=jnp.bool_)

        alpha = compute_edge_distance_gate(
            distances=distances,
            senders=senders,
            receivers=receivers,
            species=species,
            valid_edges=valid_edges,
            bank=bank,
        )

        self.assertAlmostEqual(float(alpha[0]), 1.0, places=6)
        self.assertAlmostEqual(float(alpha[1]), 1.0, places=6)
        self.assertAlmostEqual(float(alpha[2]), 0.0, places=6)
        self.assertAlmostEqual(float(alpha[3]), 1.0, places=6)
        self.assertLess(float(alpha[4]), 1.0)
        self.assertGreater(float(alpha[4]), 0.0)
        self.assertAlmostEqual(float(alpha[5]), 0.0, places=6)



    def test_stop_gradient_removes_gate_derivative(self):
        bank = EdgeDistanceGateBank(
            min_distance=jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
            max_distance=jnp.array([[0.0, 2.0], [2.0, 0.0]], dtype=jnp.float32),
            count=jnp.array([[0, 3], [3, 0]], dtype=jnp.int32),
            falloff_percent=0.05,
            floor=0.0,
            stop_gradient=True,
        )

        def gated_distance_energy(distance):
            alpha = compute_edge_distance_gate(
                distances=jnp.array([distance], dtype=jnp.float32),
                senders=jnp.array([0], dtype=jnp.int32),
                receivers=jnp.array([1], dtype=jnp.int32),
                species=jnp.array([0, 1], dtype=jnp.int32),
                valid_edges=jnp.array([True]),
                bank=bank,
            )
            return alpha[0] * distance

        grad = jax.grad(gated_distance_energy)(jnp.asarray(2.05, dtype=jnp.float32))
        alpha = gated_distance_energy(jnp.asarray(2.05, dtype=jnp.float32)) / 2.05

        self.assertAlmostEqual(float(grad), float(alpha), places=6)


    def test_stats_round_trip_to_gate_bank(self):
        stats = build_edge_distance_gate_stats(
            R=np.array([[[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]], dtype=np.float32),
            mask=np.ones((1, 2), dtype=np.float32),
            species=np.array([[0, 1]], dtype=np.int32),
            cutoff=2.0,
            n_species=2,
            falloff_percent_default=0.07,
        )
        out = Path("/tmp/cameo_edge_distance_gate_test.npz")

        save_edge_distance_gate_stats(stats, out)
        bank = EdgeDistanceGateBank.from_file(out, floor=0.1, stop_gradient=False)

        self.assertAlmostEqual(float(bank.min_distance[0, 1]), 1.25, places=6)
        self.assertAlmostEqual(float(bank.max_distance[1, 0]), 1.25, places=6)
        self.assertAlmostEqual(float(bank.falloff_percent), 0.07, places=6)
        self.assertAlmostEqual(float(bank.floor), 0.1, places=6)
        self.assertFalse(bank.stop_gradient)



if __name__ == "__main__":
    unittest.main()
