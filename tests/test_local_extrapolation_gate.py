import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import numpy as np

from models.local_extrapolation_gate import (
    LocalExtrapolationGate,
    build_jax_geometric_bio_descriptor,
    build_jax_geometric_descriptor,
    smoothstep_gate,
)


class LocalExtrapolationGateTests(unittest.TestCase):
    def test_descriptor_returns_one_row_per_atom_and_fixed_width(self):
        R = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float32)
        mask = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)

        desc = build_jax_geometric_descriptor(R, mask, cutoff=3.0)

        self.assertEqual(desc.shape, (3, 20))
        self.assertTrue(np.all(np.isfinite(np.asarray(desc))))


    def test_runtime_bio_descriptor_detects_sequence_stretch(self):
        clean_R = jnp.asarray(
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [0.0, 6.0, 0.0]],
            dtype=jnp.float32,
        )
        stretched_R = clean_R.at[1].set(jnp.asarray([7.0, 0.0, 0.0], dtype=jnp.float32))
        mask = jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        clean = np.asarray(build_jax_geometric_bio_descriptor(clean_R, mask, cutoff=10.0))
        stretched = np.asarray(build_jax_geometric_bio_descriptor(stretched_R, mask, cutoff=10.0))

        self.assertEqual(clean.shape, stretched.shape)
        self.assertGreater(stretched[0, -2], clean[0, -2])
        self.assertGreater(stretched[1, -2], clean[1, -2])

    def test_smoothstep_gate_is_bounded_and_decreasing(self):
        scores = jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
        gates = np.asarray(smoothstep_gate(scores, onset=1.0, offset=3.0))

        self.assertAlmostEqual(float(gates[0]), 1.0, places=6)
        self.assertAlmostEqual(float(gates[-1]), 0.0, places=6)
        self.assertTrue(np.all(gates >= 0.0))
        self.assertTrue(np.all(gates <= 1.0))
        self.assertTrue(np.all(np.diff(gates) <= 1e-6))

    def test_constant_zero_teacher_artifact_gates_clean_descriptor_as_one(self):
        R = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        mask = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
        desc = build_jax_geometric_descriptor(R, mask, cutoff=3.0)
        artifact = {
            "descriptor": "jax_geometric",
            "cutoff": 3.0,
            "scale_mean": np.asarray(desc[0]),
            "scale_std": np.ones((20,), dtype=np.float32),
            "center_z": np.zeros((20,), dtype=np.float32),
            "onset": 10.0,
            "offset": 11.0,
            "mode": "center_distance",
        }
        gate = LocalExtrapolationGate(artifact)

        gates = np.asarray(gate.compute_gates(R, mask))

        np.testing.assert_allclose(gates, np.ones((2,), dtype=np.float32), atol=1e-6)

    def test_mahalanobis_artifact_scores_shifted_descriptor_high(self):
        R = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        mask = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
        desc = np.asarray(build_jax_geometric_descriptor(R, mask, cutoff=3.0))
        mean = desc[0] + 1.0
        inv_cov = np.eye(desc.shape[1], dtype=np.float32)
        artifact = {
            "descriptor": "jax_geometric",
            "cutoff": 3.0,
            "scale_mean": np.zeros((desc.shape[1],), dtype=np.float32),
            "scale_std": np.ones((desc.shape[1],), dtype=np.float32),
            "mahalanobis_mean": mean.astype(np.float32),
            "mahalanobis_inv_cov": inv_cov,
            "onset": 0.5,
            "offset": 2.0,
            "mode": "mahalanobis",
        }
        gate = LocalExtrapolationGate(artifact)

        scores = np.asarray(gate.compute_scores(R, mask))

        self.assertGreater(float(scores[0]), 0.5)
        np.testing.assert_allclose(scores, np.asarray(scores[0]) * np.ones_like(scores), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
