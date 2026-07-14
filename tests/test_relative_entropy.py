import pickle
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from config.manager import ConfigManager
from training.relative_entropy import (
    InProcessLangevinSampler,
    RelativeEntropyTrainer,
    RelativeEntropyConfig,
    apply_ml_updates,
    extract_params_from_checkpoint_payload,
    compute_sample_diagnostics,
    is_unstable,
    relative_entropy_config,
    relative_entropy_gradient,
    write_relative_entropy_history_artifacts,
)


def _write_minimal_config(path, relative_entropy=None):
    re_block = relative_entropy or {}
    block_lines = "\n".join(f"    {key}: {value}" for key, value in re_block.items())
    path.write_text(
        """
seed: 123
data:
  path: data/example.npz
model:
  ml_model: allegro
optimizer:
  adam:
    lr: 0.001
training:
  relative_entropy:
"""
        + block_lines
        + "\n",
        encoding="utf-8",
    )


class RelativeEntropyConfigTests(unittest.TestCase):
    def test_relative_entropy_config_defaults_to_data_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_minimal_config(config_path, {"enabled": "true"})
            config = ConfigManager(config_path)

            cfg = relative_entropy_config(config)

        self.assertIsInstance(cfg, RelativeEntropyConfig)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.reference_data_path, "data/example.npz")
        self.assertEqual(cfg.optimizer, "adam")
        self.assertEqual(cfg.iterations, 100)
        self.assertEqual(cfg.retained_samples_per_replica, 15)
        self.assertEqual(cfg.total_model_samples, 120)
        self.assertEqual(cfg.optimizer_gradient_scale, 1.0)
        self.assertEqual(cfg.gradient_batch_size, 0)
        self.assertEqual(cfg.diagnostics_interval, 1)

    def test_relative_entropy_config_parses_mc_mala_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
seed: 123
data:
  path: data/example.npz
model:
  ml_model: allegro
optimizer:
  adam:
    lr: 0.001
training:
  relative_entropy:
    enabled: true
    sampler: mc_mala
    n_replicas: 6
    steps_per_iteration: 40
    burn_in_steps: 10
    sample_stride: 5
    kT: 2.0
    mc:
      mala:
        step_size: 0.025
        n_chains: 4
        steps_per_iteration: 30
        burn_in_steps: 6
        sample_stride: 3
""",
                encoding="utf-8",
            )
            config = ConfigManager(config_path)

            cfg = relative_entropy_config(config)

        self.assertEqual(cfg.sampler, "mc_mala")
        self.assertIsNotNone(cfg.mc_mala)
        self.assertEqual(cfg.mc_mala.n_chains, 4)
        self.assertEqual(cfg.mc_mala.steps_per_iteration, 30)
        self.assertEqual(cfg.mc_mala.burn_in_steps, 6)
        self.assertEqual(cfg.mc_mala.sample_stride, 3)
        self.assertAlmostEqual(cfg.mc_mala.step_size, 0.025)
        self.assertAlmostEqual(cfg.mc_mala.beta, 0.5)
        self.assertEqual(cfg.mc_mala.total_samples, 32)
        self.assertEqual(cfg.model_start_count, 4)

    def test_relative_entropy_config_rejects_zero_retained_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_minimal_config(
                config_path,
                {
                    "enabled": "true",
                    "steps_per_iteration": 20,
                    "burn_in_steps": 20,
                    "sample_stride": 5,
                },
            )
            config = ConfigManager(config_path)

            with self.assertRaisesRegex(ValueError, "retain at least one model sample"):
                relative_entropy_config(config)

    def test_retained_sample_count_matches_strict_post_burn_in_stride(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            steps_per_iteration=100,
            burn_in_steps=25,
            sample_stride=10,
            n_replicas=4,
        )

        self.assertEqual(cfg.retained_samples_per_replica, 7)
        self.assertEqual(cfg.total_model_samples, 28)


class RelativeEntropyCheckpointTests(unittest.TestCase):
    def test_extract_params_from_checkpoint_payload_accepts_existing_formats(self):
        params = {"ml": {"w": jnp.array([1.0])}, "prior": {"k": jnp.array(2.0)}}

        np.testing.assert_allclose(
            extract_params_from_checkpoint_payload(params)["ml"]["w"],
            jnp.array([1.0]),
        )
        np.testing.assert_allclose(
            extract_params_from_checkpoint_payload({"best_params": params})["ml"]["w"],
            jnp.array([1.0]),
        )
        np.testing.assert_allclose(
            extract_params_from_checkpoint_payload({"trainer_state": {"params": params}})["ml"]["w"],
            jnp.array([1.0]),
        )

    def test_extract_params_from_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = {"ml": {"w": np.array([3.0], dtype=np.float32)}}
            path = Path(tmp) / "checkpoint.pkl"
            with path.open("wb") as handle:
                pickle.dump({"params": params}, handle)

            loaded = extract_params_from_checkpoint_payload(path)

        np.testing.assert_allclose(loaded["ml"]["w"], np.array([3.0], dtype=np.float32))

    def test_apply_ml_updates_preserves_prior_tree(self):
        params = {
            "ml": {"w": jnp.array([1.0, -2.0])},
            "prior": {"k": jnp.array([4.0])},
        }
        updates = {"w": jnp.array([0.5, 0.25])}

        updated = apply_ml_updates(params, updates)

        np.testing.assert_allclose(updated["ml"]["w"], jnp.array([1.5, -1.75]))
        self.assertIs(updated["prior"], params["prior"])
        np.testing.assert_allclose(updated["prior"]["k"], params["prior"]["k"])


class RelativeEntropyGradientTests(unittest.TestCase):
    def test_relative_entropy_gradient_sign_matches_expectation_difference(self):
        params = {"w": jnp.array(2.0)}
        R_ref = jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32)
        R_model = jnp.array([[[0.5, 0.0, 0.0]]], dtype=jnp.float32)
        mask_ref = jnp.ones((1, 1), dtype=jnp.float32)
        mask_model = jnp.ones((1, 1), dtype=jnp.float32)
        species = jnp.zeros((1, 1), dtype=jnp.int32)

        def energy_fn(ml_params, R, mask, species):
            del mask, species
            return ml_params["w"] * jnp.sum(R * R)

        grad, metrics = relative_entropy_gradient(
            params,
            R_ref,
            mask_ref,
            species,
            R_model,
            mask_model,
            species,
            energy_fn,
            beta=0.5,
        )

        self.assertAlmostEqual(float(grad["w"]), 1.875, places=6)
        self.assertAlmostEqual(float(metrics["ref_energy_mean"]), 8.0, places=6)
        self.assertAlmostEqual(float(metrics["model_energy_mean"]), 0.5, places=6)

    def test_chunked_gradient_matches_full_batch(self):
        params = {"w": jnp.array(2.0)}
        R_ref = jnp.array(
            [
                [[2.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        R_model = jnp.array(
            [
                [[0.5, 0.0, 0.0]],
                [[0.25, 0.0, 0.0]],
                [[0.125, 0.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        mask = jnp.ones((3, 1), dtype=jnp.float32)
        species = jnp.zeros((3, 1), dtype=jnp.int32)

        def energy_fn(ml_params, R, mask, species):
            del mask, species
            return ml_params["w"] * jnp.sum(R * R)

        full_grad, full_metrics = relative_entropy_gradient(
            params, R_ref, mask, species, R_model, mask, species, energy_fn, beta=0.5
        )
        chunked_grad, chunked_metrics = relative_entropy_gradient(
            params,
            R_ref,
            mask,
            species,
            R_model,
            mask,
            species,
            energy_fn,
            beta=0.5,
            gradient_batch_size=2,
        )

        np.testing.assert_allclose(chunked_grad["w"], full_grad["w"], rtol=1e-6)
        np.testing.assert_allclose(
            chunked_metrics["re_energy_gap"], full_metrics["re_energy_gap"], rtol=1e-6
        )

    def test_sample_diagnostics_detect_instability(self):
        R = jnp.array([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=jnp.float32)
        F = jnp.array([[[jnp.nan, 0.0, 0.0], [2.0e4, 0.0, 0.0]]], dtype=jnp.float32)
        mask = jnp.ones((1, 2), dtype=jnp.float32)

        diagnostics = compute_sample_diagnostics(R, F, mask)

        self.assertTrue(diagnostics["has_nan_or_inf"])
        self.assertLess(diagnostics["min_pair_distance"], 1.5)
        self.assertTrue(is_unstable(diagnostics, max_force=1.0e4, min_pair_distance=1.5))


class RelativeEntropySamplerTests(unittest.TestCase):
    def test_sampler_returns_retained_samples_and_finite_diagnostics(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_data_path="data/example.npz",
            n_replicas=2,
            steps_per_iteration=6,
            burn_in_steps=2,
            sample_stride=2,
            dt=0.001,
            kT=0.1,
            gamma=0.1,
            mass=12.0,
        )

        def energy_fn(R, mask=None, species=None):
            del species
            return 0.5 * jnp.sum(jnp.where(mask[:, None] > 0, R, 0.0) ** 2)

        def shift_fn(R, dR):
            return R + dR

        sampler = InProcessLangevinSampler(energy_fn, shift_fn, cfg)
        R0 = jnp.array(
            [
                [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
            ],
            dtype=jnp.float32,
        )
        mask = jnp.ones((2, 2), dtype=jnp.float32)
        species = jnp.zeros((2, 2), dtype=jnp.int32)

        result = sampler.run(R0, mask, species, rng_key=jnp.array([0, 1], dtype=jnp.uint32))

        self.assertEqual(result["R"].shape, (4, 2, 3))
        self.assertEqual(result["mask"].shape, (4, 2))
        self.assertEqual(result["species"].shape, (4, 2))
        self.assertFalse(result["diagnostics"]["has_nan_or_inf"])
        diagnostics = sampler.diagnostics_for_samples(result["R"], result["mask"], result["species"])
        self.assertTrue(np.isfinite(diagnostics["max_force"]))

    def test_sampler_rejects_wrong_replica_count(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_data_path="data/example.npz",
            n_replicas=2,
        )
        sampler = InProcessLangevinSampler(lambda R, **kwargs: jnp.sum(R * R), lambda R, dR: R + dR, cfg)

        with self.assertRaisesRegex(ValueError, "n_replicas"):
            sampler.run(
                jnp.zeros((1, 2, 3), dtype=jnp.float32),
                jnp.ones((1, 2), dtype=jnp.float32),
                jnp.zeros((1, 2), dtype=jnp.int32),
                rng_key=jnp.array([0, 2], dtype=jnp.uint32),
            )


class _FakeSampler:
    def __init__(self):
        self.calls = 0

    def run(self, R0, mask, species, rng_key):
        del R0, rng_key
        self.calls += 1
        model_R = jnp.array([[[0.5, 0.0, 0.0]]], dtype=jnp.float32)
        model_mask = jnp.ones((1, 1), dtype=jnp.float32)
        model_species = jnp.zeros((1, 1), dtype=jnp.int32)
        return {
            "R": model_R,
            "mask": model_mask,
            "species": model_species,
            "diagnostics": {
                "has_nan_or_inf": False,
                "max_force": 1.0,
                "min_pair_distance": float("inf"),
                "n_samples": 1,
            },
        }


class RelativeEntropyTrainerTests(unittest.TestCase):
    def test_train_step_updates_ml_only_and_records_metrics(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_data_path="data/example.npz",
            reference_batch_size=1,
            n_replicas=1,
            iterations=1,
            steps_per_iteration=2,
            burn_in_steps=0,
            sample_stride=1,
            kT=2.0,
            reject_on_instability=True,
        )
        params = {
            "ml": {"w": jnp.array(2.0)},
            "prior": {"k": jnp.array(7.0)},
        }
        reference = {
            "R": jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }

        def energy_fn(ml_params, R, mask, species):
            del mask, species
            return ml_params["w"] * jnp.sum(R * R)

        trainer = RelativeEntropyTrainer(
            params=params,
            reference_data=reference,
            sampler=_FakeSampler(),
            energy_fn=energy_fn,
            optimizer=optax.sgd(learning_rate=0.1),
            config=cfg,
            seed=0,
        )

        metrics = trainer.train_step(0)

        self.assertLess(float(trainer.params["ml"]["w"]), 2.0)
        self.assertIs(trainer.params["prior"], params["prior"])
        self.assertFalse(metrics["rejected"])
        self.assertEqual(len(trainer.history), 1)

    def test_history_artifacts_write_log_csv_and_plot_status(self):
        history = [
            {
                "iteration": 0,
                "rejected": False,
                "objective": 1.5,
                "re_energy_gap": -1.5,
                "grad_norm": 2.0,
                "update_norm": 0.1,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_relative_entropy_history_artifacts(history, tmp)
            csv_path = Path(artifacts["csv"])
            log_path = Path(artifacts["log"])

            self.assertTrue(csv_path.exists())
            self.assertIn("objective", csv_path.read_text(encoding="utf-8"))
            self.assertTrue(log_path.exists())
            self.assertIn("objective=1.5", log_path.read_text(encoding="utf-8"))
            self.assertIn("plot", artifacts)

    def test_train_step_rejects_unstable_sampler_output(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_data_path="data/example.npz",
            reference_batch_size=1,
            n_replicas=1,
            max_force=10.0,
            reject_on_instability=True,
        )
        params = {"ml": {"w": jnp.array(2.0)}}
        reference = {
            "R": jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }

        class BadSampler(_FakeSampler):
            def run(self, R0, mask, species, rng_key):
                out = super().run(R0, mask, species, rng_key)
                out["diagnostics"] = {
                    "has_nan_or_inf": False,
                    "max_force": 99.0,
                    "min_pair_distance": float("inf"),
                    "n_samples": 1,
                }
                return out

        trainer = RelativeEntropyTrainer(
            params=params,
            reference_data=reference,
            sampler=BadSampler(),
            energy_fn=lambda ml_params, R, mask, species: ml_params["w"] * jnp.sum(R * R),
            optimizer=optax.sgd(learning_rate=0.1),
            config=cfg,
            seed=0,
        )

        metrics = trainer.train_step(0)

        self.assertTrue(metrics["rejected"])
        np.testing.assert_allclose(trainer.params["ml"]["w"], jnp.array(2.0))


if __name__ == "__main__":
    unittest.main()
