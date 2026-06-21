import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
import numpy as np

from training.support_gate import (
    build_pairwise_distance_descriptors,
    build_support_gate_bank,
    rbf_segment_supports,
    rbf_structure_support,
)


class SupportGateTest(unittest.TestCase):
    def test_pairwise_descriptor_is_translation_invariant(self):
        R = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ])
        mask = jnp.array([[1.0, 1.0, 1.0]])
        shifted = R + jnp.array([3.0, -2.0, 1.0])

        d0 = build_pairwise_distance_descriptors(R, mask)
        d1 = build_pairwise_distance_descriptors(shifted, mask)

        self.assertTrue(bool(jnp.allclose(d0, d1)))
        self.assertEqual(d0.shape, (1, 3))

    def test_rbf_support_is_high_at_training_structure_and_low_far_away(self):
        R = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
        ])
        mask = jnp.ones((2, 3))
        bank = build_support_gate_bank(
            R=np.asarray(R),
            mask=np.asarray(mask),
            max_centers=2,
            sigma_multiplier=1.0,
            seed=0,
        )

        near = rbf_structure_support(R[0], mask[0], bank)
        far = rbf_structure_support(R[0] * 4.0, mask[0], bank)

        self.assertGreater(float(near), 0.99)
        self.assertLess(float(far), float(near))

    def test_segment_supports_match_individual_structures_in_a_tile(self):
        structures = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
        ])
        structure_mask = jnp.ones((2, 3))
        bank = build_support_gate_bank(
            R=np.asarray(structures),
            mask=np.asarray(structure_mask),
            max_centers=2,
            seed=0,
        )
        tile_R = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [10.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        tile_mask = jnp.array([1, 1, 1, 1, 1, 1, 0], dtype=jnp.float32)
        segment_id = jnp.array([0, 0, 0, 1, 1, 1, -1], dtype=jnp.int32)

        alphas = rbf_segment_supports(tile_R, tile_mask, segment_id, bank, num_segments=3)
        expected0 = rbf_structure_support(tile_R[:3], tile_mask[:3], bank)
        expected1 = rbf_structure_support(tile_R[3:6], tile_mask[3:6], bank)

        self.assertTrue(bool(jnp.allclose(alphas[0], expected0)))
        self.assertTrue(bool(jnp.allclose(alphas[1], expected1)))
        self.assertEqual(float(alphas[2]), 0.0)

    def test_support_gate_is_differentiable_with_coordinates(self):
        R = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
        ])
        mask = jnp.ones((2, 3))
        bank = build_support_gate_bank(R=np.asarray(R), mask=np.asarray(mask), max_centers=2, seed=0)

        def gated_energy(coords):
            return rbf_structure_support(coords, mask[0], bank) * jnp.sum(coords * coords)

        grad = jax.grad(gated_energy)(R[0])

        self.assertEqual(grad.shape, R[0].shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))


if __name__ == "__main__":
    unittest.main()
