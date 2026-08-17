import pickle
import tempfile
import unittest
from pathlib import Path

import jax
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
    RelativeEntropyRolloutStage,
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

    def test_relative_entropy_config_parses_persistent_configured_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_minimal_config(
                config_path,
                {
                    "enabled": "true",
                    "n_replicas": 2,
                    "persistent_chains": "true",
                    "start_frame_mode": "configured_phi_targets",
                    "initial_state_data_path": "starts.npz",
                    "initial_state_phi_targets_deg": "[120, -120]",
                    "initial_state_phi_indices": "[0, 1, 2, 3]",
                },
            )
            cfg = relative_entropy_config(ConfigManager(config_path))

        self.assertTrue(cfg.persistent_chains)
        self.assertEqual(cfg.start_frame_mode, "configured_phi_targets")
        self.assertEqual(cfg.initial_state_phi_targets_deg, (120.0, -120.0))
        self.assertEqual(cfg.initial_state_phi_indices, (0, 1, 2, 3))

    def test_relative_entropy_config_parses_2d_starts_and_rollout_schedule(self):
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
    iterations: 20
    n_replicas: 2
    persistent_chains: true
    start_frame_mode: configured_cv_targets
    initial_state_data_path: starts.npz
    initial_state_cv_indices: [[0, 1, 2, 3], [1, 2, 3, 4]]
    initial_state_cv_shift_deg: [180, 180]
    initial_state_cv_targets_deg: [[120, -110], [125, 120]]
    rollout_schedule:
      - start_iteration: 0
        steps_per_iteration: 100
        burn_in_steps: 20
        sample_stride: 10
      - start_iteration: 10
        steps_per_iteration: 250
        burn_in_steps: 50
        sample_stride: 20
""",
                encoding="utf-8",
            )
            cfg = relative_entropy_config(ConfigManager(config_path))

        self.assertEqual(cfg.start_frame_mode, "configured_cv_targets")
        self.assertEqual(cfg.initial_state_cv_targets_deg[1], (125.0, 120.0))
        self.assertEqual(len(cfg.rollout_schedule), 2)
        self.assertEqual(cfg.rollout_for_iteration(0).steps_per_iteration, 100)
        self.assertEqual(cfg.rollout_for_iteration(9).steps_per_iteration, 100)
        self.assertEqual(cfg.rollout_for_iteration(10).steps_per_iteration, 250)
        self.assertEqual(cfg.rollout_for_iteration(19).retained_samples_per_replica, 10)

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
        self.assertEqual(result["final_R"].shape, (2, 2, 3))
        self.assertFalse(result["diagnostics"]["has_nan_or_inf"])
        diagnostics = sampler.diagnostics_for_samples(result["R"], result["mask"], result["species"])
        self.assertTrue(np.isfinite(diagnostics["max_force"]))

    def test_nested_scan_rollout_is_bit_identical_to_the_flat_retention_buffer(self):
        """REGRESSION GUARD for the 2026-08-11 nested-scan rewrite.

        The old rollout carried the retained buffer through every MD step
        (dynamic_update_index_in_dim + a full-buffer jnp.where select per step). The new one
        uses an outer scan over retained samples and an inner scan over `sample_stride`.
        Both perform exactly the same number of step_fn calls in the same order, so they
        consume the RNG stream identically and MUST agree bit-for-bit. If this test ever
        fails, the rewrite changed the trajectory, not just its bookkeeping.
        """
        from jax_md import simulate

        cfg = RelativeEntropyConfig(
            enabled=True, reference_data_path="data/example.npz", n_replicas=3,
            steps_per_iteration=12, burn_in_steps=4, sample_stride=2,
            dt=0.002, kT=0.15, gamma=0.3, mass=12.0,
        )

        def energy_fn(R, mask=None, species=None):
            del species
            return 0.5 * jnp.sum(jnp.where(mask[:, None] > 0, R, 0.0) ** 2)

        def shift_fn(R, dR):
            return R + dR

        sampler = InProcessLangevinSampler(energy_fn, shift_fn, cfg)

        def legacy_run_single(R0, mask, species, key):
            """Verbatim pre-2026-08-11 implementation, kept only as the reference."""
            mask = jnp.asarray(mask, dtype=jnp.float32)
            species = jnp.asarray(species, dtype=jnp.int32)
            R0 = jnp.asarray(R0, dtype=jnp.float32)

            def energy_for_md(R):
                R_masked = jnp.where(mask[:, None] > 0, R, jax.lax.stop_gradient(R0))
                return energy_fn(R_masked, mask=mask, species=species)

            init_fn, step_fn = simulate.nvt_langevin(
                energy_for_md, shift_fn, float(cfg.dt), float(cfg.kT), float(cfg.gamma)
            )
            state = init_fn(key, R0, mass=jnp.full((species.shape[0], 1), 12.0, jnp.float32))
            n_retained = int(cfg.retained_samples_per_replica)
            retained0 = jnp.zeros((n_retained,) + R0.shape, dtype=R0.dtype)

            def scan_step(carry, step_idx):
                state, retained, retained_count = carry
                state = step_fn(state)
                position = jnp.where(mask[:, None] > 0, state.position, R0)
                state = state.set(position=position)
                after_burn = step_idx > int(cfg.burn_in_steps)
                on_stride = (step_idx - int(cfg.burn_in_steps)) % int(cfg.sample_stride) == 0
                should_retain = jnp.logical_and(after_burn, on_stride)
                safe_index = jnp.minimum(retained_count, n_retained - 1)
                updated = jax.lax.dynamic_update_index_in_dim(
                    retained, jax.lax.stop_gradient(position[None, ...]), safe_index, axis=0
                )
                retained = jnp.where(should_retain, updated, retained)
                return (state, retained, retained_count + should_retain.astype(jnp.int32)), None

            steps = jnp.arange(1, int(cfg.steps_per_iteration) + 1)
            (final_state, retained, _), _ = jax.lax.scan(
                scan_step, (state, retained0, jnp.asarray(0, dtype=jnp.int32)), steps
            )
            return retained, jnp.where(mask[:, None] > 0, final_state.position, R0)

        R0 = jnp.array(
            [[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
             [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
             [[0.2, 0.1, 0.3], [0.9, 0.2, 0.1]]], dtype=jnp.float32
        )
        mask = jnp.ones((3, 2), dtype=jnp.float32)
        species = jnp.zeros((3, 2), dtype=jnp.int32)
        keys = jax.random.split(jnp.array([0, 1], dtype=jnp.uint32), 3)

        new_retained, new_final = jax.vmap(sampler._run_single)(R0, mask, species, keys)
        old_retained, old_final = jax.vmap(legacy_run_single)(R0, mask, species, keys)

        self.assertEqual(new_retained.shape, old_retained.shape)
        np.testing.assert_array_equal(np.asarray(new_retained), np.asarray(old_retained))
        np.testing.assert_array_equal(np.asarray(new_final), np.asarray(old_final))

    def test_rollout_recompiles_when_the_schedule_changes_stage(self):
        """jax.jit keys its cache on the function object, NOT on closed-over Python values.

        Caching the compiled rollout without invalidating it on configure_rollout() would
        silently keep running the previous stage's step count for the rest of training.
        """
        cfg = RelativeEntropyConfig(
            enabled=True, reference_data_path="data/example.npz", n_replicas=2,
            steps_per_iteration=6, burn_in_steps=2, sample_stride=2,
            dt=0.001, kT=0.1, gamma=0.1, mass=12.0,
        )

        def energy_fn(R, mask=None, species=None):
            del species
            return 0.5 * jnp.sum(jnp.where(mask[:, None] > 0, R, 0.0) ** 2)

        sampler = InProcessLangevinSampler(energy_fn, lambda R, dR: R + dR, cfg)
        R0 = jnp.zeros((2, 2, 3), dtype=jnp.float32) + 0.1
        mask = jnp.ones((2, 2), dtype=jnp.float32)
        species = jnp.zeros((2, 2), dtype=jnp.int32)
        key = jnp.array([0, 1], dtype=jnp.uint32)

        first = sampler.run(R0, mask, species, key)
        self.assertEqual(first["R"].shape[0], 2 * 2)  # (6-2)//2 = 2 retained per replica
        fn_before = sampler._compiled_rollout

        sampler.configure_rollout(
            RelativeEntropyRolloutStage(start_iteration=1, steps_per_iteration=12,
                                        burn_in_steps=2, sample_stride=2)
        )
        self.assertIsNone(sampler._compiled_rollout, "cache must be dropped on stage change")
        second = sampler.run(R0, mask, species, key)
        self.assertEqual(second["R"].shape[0], 2 * 5)  # (12-2)//2 = 5 retained per replica
        self.assertIsNot(sampler._compiled_rollout, fn_before)

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


class RelativeEntropyResumeTests(unittest.TestCase):
    """REM had no resume, so a run that hit the 10 h QOS wall lost everything after its
    last checkpoint. These cover the restart path added 2026-08-11."""

    @staticmethod
    def _make_trainer(tmpdir, iterations=4, seed=0):
        cfg = RelativeEntropyConfig(
            enabled=True, reference_data_path="data/example.npz", reference_batch_size=1,
            n_replicas=1, iterations=iterations, steps_per_iteration=2, burn_in_steps=0,
            sample_stride=1, kT=2.0, checkpoint_freq=1,
        )
        params = {"ml": {"w": jnp.array(2.0)}, "prior": {"k": jnp.array(7.0)}}
        reference = {
            "R": jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }

        def energy_fn(ml_params, R, mask, species):
            del mask, species
            return ml_params["w"] * jnp.sum(R * R)

        return RelativeEntropyTrainer(
            params=params, reference_data=reference, sampler=_FakeSampler(),
            energy_fn=energy_fn, optimizer=optax.sgd(learning_rate=0.1), config=cfg,
            seed=seed, checkpoint_dir=Path(tmpdir) / "checkpoints",
        )

    def test_resume_continues_from_checkpoint_without_redoing_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._make_trainer(tmpdir, iterations=2)
            first.train()
            self.assertEqual(len(first.history), 2)
            w_after_two = float(first.params["ml"]["w"])

            ckpt = RelativeEntropyTrainer.latest_checkpoint(Path(tmpdir) / "checkpoints")
            self.assertIsNotNone(ckpt)

            resumed = self._make_trainer(tmpdir, iterations=4)
            start = resumed.resume_from(ckpt)

            self.assertEqual(start, 2)
            self.assertEqual(len(resumed.history), 2)
            self.assertAlmostEqual(float(resumed.params["ml"]["w"]), w_after_two, places=6)

            resumed.train()
            # 4 total, not 6: the first two are not repeated.
            self.assertEqual(len(resumed.history), 4)
            self.assertEqual([row["iteration"] for row in resumed.history], [0, 1, 2, 3])

    def test_resume_restores_optimizer_state_and_rng_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._make_trainer(tmpdir, iterations=2)
            first.train()
            ckpt = RelativeEntropyTrainer.latest_checkpoint(Path(tmpdir) / "checkpoints")

            resumed = self._make_trainer(tmpdir, iterations=4)
            fresh_key = np.asarray(resumed.rng_key).copy()
            resumed.resume_from(ckpt)

            # RNG must come from the checkpoint, or resumed rollouts replay old randomness.
            self.assertFalse(np.array_equal(np.asarray(resumed.rng_key), fresh_key))
            np.testing.assert_array_equal(
                np.asarray(resumed.rng_key), np.asarray(first.rng_key)
            )
            # Optimizer moments must survive: restarting them at zero silently changes the
            # effective learning-rate schedule mid-run.
            self.assertEqual(
                jax.tree_util.tree_structure(resumed.opt_state),
                jax.tree_util.tree_structure(first.opt_state),
            )

    def test_latest_checkpoint_picks_the_highest_iteration_and_tolerates_no_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "checkpoints"
            self.assertIsNone(RelativeEntropyTrainer.latest_checkpoint(root))
            root.mkdir(parents=True)
            self.assertIsNone(RelativeEntropyTrainer.latest_checkpoint(root))
            for i in (5, 40, 100):
                (root / f"relative_entropy_iter{i:06d}.pkl").write_bytes(b"x")
            self.assertEqual(
                RelativeEntropyTrainer.latest_checkpoint(root).name,
                "relative_entropy_iter000100.pkl",
            )

    def test_resume_appends_to_the_live_csv_without_a_second_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._make_trainer(tmpdir, iterations=2)
            first.train()
            live = Path(tmpdir) / "relative_entropy_history_live.csv"
            self.assertTrue(live.exists())

            resumed = self._make_trainer(tmpdir, iterations=4)
            resumed.resume_from(RelativeEntropyTrainer.latest_checkpoint(Path(tmpdir) / "checkpoints"))
            resumed.train()

            rows = live.read_text().strip().splitlines()
            headers = [r for r in rows if r.startswith("iteration,")]
            self.assertEqual(len(headers), 1, f"expected exactly one header, got {len(headers)}")
            self.assertEqual(len(rows), 5)  # 1 header + 4 iterations


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

    def test_train_step_reports_selected_gradient_and_parameter_norms(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_batch_size=1,
            n_replicas=1,
            iterations=1,
            steps_per_iteration=2,
            burn_in_steps=0,
            sample_stride=1,
            kT=2.0,
            trainable_param_substring="head",
        )
        params = {"ml": {"backbone": jnp.array(2.0), "head": jnp.array(3.0)}}
        reference = {
            "R": jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }

        def energy_fn(ml_params, R, mask, species):
            del mask, species
            return (ml_params["backbone"] + ml_params["head"]) * jnp.sum(R * R)

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

        self.assertEqual(float(trainer.params["ml"]["backbone"]), 2.0)
        self.assertNotEqual(float(trainer.params["ml"]["head"]), 3.0)
        self.assertGreater(metrics["trainable_grad_norm"], 0.0)
        self.assertAlmostEqual(
            metrics["trainable_param_norm"],
            abs(float(trainer.params["ml"]["head"])),
        )

    def test_persistent_chains_continue_from_previous_final_coordinates(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_batch_size=1,
            n_replicas=1,
            iterations=2,
            steps_per_iteration=2,
            burn_in_steps=0,
            sample_stride=1,
            kT=2.0,
            persistent_chains=True,
        )
        reference = {
            "R": jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }
        initial = {
            "R": jnp.array([[[5.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }

        class AdvancingSampler:
            def __init__(self):
                self.starts = []
            def run(self, R0, mask, species, rng_key):
                del rng_key
                self.starts.append(np.asarray(R0))
                return {
                    "R": R0,
                    "mask": mask,
                    "species": species,
                    "final_R": R0 + 1.0,
                    "diagnostics": {"has_nan_or_inf": False, "n_samples": 1},
                }

        sampler = AdvancingSampler()
        trainer = RelativeEntropyTrainer(
            params={"ml": {"w": jnp.array(0.1)}},
            reference_data=reference,
            sampler=sampler,
            energy_fn=lambda ml_params, R, mask, species: ml_params["w"] * jnp.sum(R * R),
            optimizer=optax.sgd(learning_rate=1.0e-3),
            config=cfg,
            seed=0,
            initial_states=initial,
        )
        first = trainer.train_step(0)
        second = trainer.train_step(1)

        np.testing.assert_allclose(sampler.starts[0], initial["R"])
        np.testing.assert_allclose(sampler.starts[1], initial["R"] + 1.0)
        self.assertEqual(first["chain_start_source"], "configured_initial")
        self.assertEqual(second["chain_start_source"], "persistent")
        self.assertTrue(second["persistent_chain_advanced"])

    def test_progressive_rollout_reconfigures_sampler_without_resetting_chain(self):
        cfg = RelativeEntropyConfig(
            enabled=True,
            reference_batch_size=1,
            n_replicas=1,
            iterations=2,
            steps_per_iteration=2,
            burn_in_steps=0,
            sample_stride=1,
            kT=2.0,
            persistent_chains=True,
            rollout_schedule=(
                RelativeEntropyRolloutStage(0, 2, 0, 1),
                RelativeEntropyRolloutStage(1, 6, 2, 2),
            ),
        )
        reference = {
            "R": jnp.array([[[2.0, 0.0, 0.0]]], dtype=jnp.float32),
            "mask": jnp.ones((1, 1), dtype=jnp.float32),
            "species": jnp.zeros((1, 1), dtype=jnp.int32),
        }

        class ScheduledSampler:
            def __init__(self):
                self.stages = []
                self.starts = []

            def configure_rollout(self, stage):
                self.stages.append(stage)

            def run(self, R0, mask, species, rng_key):
                del rng_key
                self.starts.append(np.asarray(R0))
                return {
                    "R": R0,
                    "mask": mask,
                    "species": species,
                    "final_R": R0 + 1.0,
                    "diagnostics": {"has_nan_or_inf": False, "n_samples": 1},
                }

        sampler = ScheduledSampler()
        trainer = RelativeEntropyTrainer(
            params={"ml": {"w": jnp.array(0.1)}},
            reference_data=reference,
            sampler=sampler,
            energy_fn=lambda ml_params, R, mask, species: ml_params["w"] * jnp.sum(R * R),
            optimizer=optax.sgd(learning_rate=1.0e-3),
            config=cfg,
            seed=0,
        )
        first = trainer.train_step(0)
        second = trainer.train_step(1)

        self.assertEqual([stage.steps_per_iteration for stage in sampler.stages], [2, 6])
        np.testing.assert_allclose(sampler.starts[1], sampler.starts[0] + 1.0)
        self.assertEqual(first["rollout_steps"], 2)
        self.assertEqual(second["rollout_steps"], 6)
        self.assertEqual(second["rollout_burn_in_steps"], 2)

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

class _BasinRecorder:
    def __init__(self):
        self.calls = []
        self.finalized = 0

    def should_record(self, step, *, final_step):
        return step == 0 or step == final_step or step % 1 == 0

    def record(self, params, **metadata):
        self.calls.append(
            {
                **metadata,
                "w": float(np.asarray(params["ml"]["w"])),
            }
        )

    def finalize(self):
        self.finalized += 1


class RelativeEntropyBasinMonitorTests(unittest.TestCase):
    def test_monitor_stride_records_off_stride_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RelativeEntropyResumeTests._make_trainer(tmpdir, iterations=5)
            recorder = _BasinRecorder()
            recorder.should_record = lambda step, *, final_step: (
                step == 0 or step == final_step or step % 3 == 0
            )
            trainer.basin_energy_monitor = recorder
            trainer.train()
            self.assertEqual([row["step"] for row in recorder.calls], [0, 3, 5])


    def test_monitor_records_initial_and_accepted_updated_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RelativeEntropyResumeTests._make_trainer(
                tmpdir, iterations=1
            )
            recorder = _BasinRecorder()
            trainer.basin_energy_monitor = recorder

            trainer.train()

            self.assertEqual([row["step"] for row in recorder.calls], [0, 1])
            self.assertFalse(recorder.calls[1]["rejected"])
            self.assertLess(recorder.calls[1]["w"], recorder.calls[0]["w"])
            self.assertEqual(recorder.finalized, 1)

    def test_monitor_marks_rejected_iteration_with_unchanged_params(self):
        class UnstableSampler(_FakeSampler):
            def run(self, R0, mask, species, rng_key):
                out = super().run(R0, mask, species, rng_key)
                out["diagnostics"]["has_nan_or_inf"] = True
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RelativeEntropyResumeTests._make_trainer(
                tmpdir, iterations=1
            )
            trainer.sampler = UnstableSampler()
            recorder = _BasinRecorder()
            trainer.basin_energy_monitor = recorder

            trainer.train()

            self.assertTrue(recorder.calls[1]["rejected"])
            self.assertEqual(recorder.calls[1]["w"], recorder.calls[0]["w"])


if __name__ == "__main__":
    unittest.main()
