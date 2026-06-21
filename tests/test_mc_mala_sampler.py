import unittest

import jax
import jax.numpy as jnp
import numpy as np

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from mc.config import MALAConfig
from mc.samplers import BlackJaxMALASampler, run_mala_chain


class MALAChainTests(unittest.TestCase):
    def test_harmonic_chain_samples_boltzmann_variance(self):
        beta = 2.0
        k = 4.0

        def logdensity_fn(x):
            return -0.5 * beta * k * jnp.sum(x * x)

        result = run_mala_chain(
            jnp.array([2.0], dtype=jnp.float32),
            logdensity_fn,
            rng_key=jax.random.PRNGKey(0),
            steps=3500,
            burn_in=500,
            sample_stride=3,
            step_size=0.35,
        )

        samples = np.asarray(result.samples[:, 0])
        self.assertEqual(samples.shape[0], 1000)
        self.assertGreater(float(result.acceptance_rate), 0.2)
        self.assertLess(float(result.acceptance_rate), 1.0)
        self.assertAlmostEqual(float(np.mean(samples)), 0.0, delta=0.12)
        self.assertAlmostEqual(float(np.var(samples)), 1.0 / (beta * k), delta=0.04)


class BlackJaxMALASamplerTests(unittest.TestCase):
    def test_sampler_matches_relative_entropy_sampler_interface(self):
        cfg = MALAConfig(
            n_chains=2,
            steps_per_iteration=12,
            burn_in_steps=4,
            sample_stride=2,
            step_size=0.02,
            beta=1.0,
        )

        def energy_fn(R, mask=None, species=None):
            del species
            return 0.5 * jnp.sum(jnp.where(mask[:, None] > 0, R, 0.0) ** 2)

        sampler = BlackJaxMALASampler(energy_fn, cfg)
        R0 = jnp.array(
            [
                [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
            ],
            dtype=jnp.float32,
        )
        mask = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
        species = jnp.zeros((2, 2), dtype=jnp.int32)

        result = sampler.run(R0, mask, species, rng_key=jax.random.PRNGKey(1))

        self.assertEqual(result["R"].shape, (8, 2, 3))
        self.assertEqual(result["mask"].shape, (8, 2))
        self.assertEqual(result["species"].shape, (8, 2))
        self.assertIn("acceptance_rate_mean", result["diagnostics"])
        self.assertIn("logdensity_mean", result["diagnostics"])
        self.assertFalse(result["diagnostics"]["has_nan_or_inf"])
        padded_samples = np.asarray(result["R"][:4, 1, :])
        expected_padded = np.repeat(np.asarray(R0[0, 1, :])[None, :], padded_samples.shape[0], axis=0)
        np.testing.assert_allclose(padded_samples, expected_padded, atol=1e-6)

    def test_sampler_rejects_wrong_chain_count(self):
        cfg = MALAConfig(n_chains=2)
        sampler = BlackJaxMALASampler(lambda R, **kwargs: jnp.sum(R * R), cfg)

        with self.assertRaisesRegex(ValueError, "n_chains"):
            sampler.run(
                jnp.zeros((1, 2, 3), dtype=jnp.float32),
                jnp.ones((1, 2), dtype=jnp.float32),
                jnp.zeros((1, 2), dtype=jnp.int32),
                rng_key=jax.random.PRNGKey(2),
            )


if __name__ == "__main__":
    unittest.main()
