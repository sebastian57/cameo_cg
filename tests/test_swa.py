import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import numpy as np

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from config.manager import ConfigManager
from training.swa import SWAState, save_swa_checkpoint
from training.trainer import Trainer


def _write_config(path: Path, swa_block: str) -> None:
    path.write_text(
        f"""
seed: 123
data:
  path: data/example.npz
model:
  ml_model: allegro
optimizer:
  adam:
    lr: 0.001
training:
  stages:
    - optimizer: adam
      epochs: 10
    - optimizer: sgd_nesterov
      epochs: 20
  swa:
{swa_block}
""",
        encoding="utf-8",
    )


class SWAStateTests(unittest.TestCase):
    def test_first_update_copies_nested_params_and_records_epoch(self):
        params = {
            "ml": {"w": jnp.array([1.0, 3.0])},
            "prior": {"k": jnp.array(2.0)},
        }
        state = SWAState(stage="adam", start_epoch=2, sample_freq_epochs=1)

        state.update(params, epoch=2)

        params["ml"]["w"] = jnp.array([100.0, 100.0])
        np.testing.assert_allclose(state.averaged_params["ml"]["w"], np.array([1.0, 3.0]))
        np.testing.assert_allclose(state.averaged_params["prior"]["k"], np.array(2.0))
        self.assertEqual(state.n_samples, 1)
        self.assertEqual(state.sample_epochs, [2])

    def test_subsequent_updates_compute_arithmetic_mean(self):
        state = SWAState(stage="adam", start_epoch=2, sample_freq_epochs=1)

        state.update({"ml": {"w": jnp.array([1.0, 3.0])}}, epoch=2)
        state.update({"ml": {"w": jnp.array([3.0, 7.0])}}, epoch=3)
        state.update({"ml": {"w": jnp.array([5.0, 11.0])}}, epoch=4)

        np.testing.assert_allclose(state.averaged_params["ml"]["w"], np.array([3.0, 7.0]))
        self.assertEqual(state.n_samples, 3)
        self.assertEqual(state.sample_epochs, [2, 3, 4])

    def test_mismatched_tree_raises_clear_error(self):
        state = SWAState(stage="adam", start_epoch=0, sample_freq_epochs=1)
        state.update({"ml": {"w": jnp.array([1.0])}}, epoch=0)

        with self.assertRaisesRegex(ValueError, "SWA parameter tree changed"):
            state.update({"ml": {"b": jnp.array([2.0])}}, epoch=1)

    def test_should_sample_respects_start_and_frequency(self):
        state = SWAState(stage="adam", start_epoch=3, sample_freq_epochs=2)

        self.assertFalse(state.should_sample(2))
        self.assertTrue(state.should_sample(3))
        self.assertFalse(state.should_sample(4))
        self.assertTrue(state.should_sample(5))


class SWAConfigTests(unittest.TestCase):
    def test_swa_config_defaults_to_disabled_last_nonzero_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            _write_config(path, "    enabled: false\n")
            config = ConfigManager(path)

            cfg = config.get_swa_config()

        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["stage"], "sgd_nesterov")
        self.assertIsNone(cfg["start_epoch"])
        self.assertAlmostEqual(cfg["start_fraction"], 0.75)
        self.assertEqual(cfg["sample_freq_epochs"], 1)
        self.assertTrue(cfg["save_checkpoint"])
        self.assertFalse(cfg["use_best_params"])

    def test_swa_config_accepts_explicit_stage_and_start_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            _write_config(
                path,
                """
    enabled: true
    stage: adam
    start_epoch: 4
    start_fraction: 0.5
    sample_freq_epochs: 2
    save_checkpoint: false
    use_best_params: true
""",
            )
            config = ConfigManager(path)

            cfg = config.get_swa_config()

        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["stage"], "adam")
        self.assertEqual(cfg["start_epoch"], 4)
        self.assertAlmostEqual(cfg["start_fraction"], 0.5)
        self.assertEqual(cfg["sample_freq_epochs"], 2)
        self.assertFalse(cfg["save_checkpoint"])
        self.assertTrue(cfg["use_best_params"])

    def test_swa_config_rejects_invalid_values(self):
        invalid_blocks = [
            "    enabled: true\n    start_fraction: 1.5\n",
            "    enabled: true\n    start_epoch: -1\n",
            "    enabled: true\n    sample_freq_epochs: 0\n",
        ]
        for block in invalid_blocks:
            with self.subTest(block=block), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"
                _write_config(path, block)
                config = ConfigManager(path)

                with self.assertRaises(ValueError):
                    config.get_swa_config()


class _FakeChemtrainTrainer:
    def __init__(self):
        self._epoch = 0
        self.params = {"ml": {"w": jnp.array([0.0])}}
        self.best_inference_params = {"ml": {"w": jnp.array([-1.0])}}
        self.train_losses = [1.0]
        self.val_losses = [2.0]
        self.tasks = {}

    def add_task(self, trigger, fn):
        self.tasks.setdefault(trigger, []).append(fn)

    def run_post_epoch(self):
        for fn in self.tasks.get("post_epoch", []):
            fn(self)


class SWATrainerHookTests(unittest.TestCase):
    def test_post_epoch_hook_samples_stage_local_epochs_without_replacing_params(self):
        wrapper = object.__new__(Trainer)
        fake = _FakeChemtrainTrainer()
        state = SWAState(stage="adam", start_epoch=2, sample_freq_epochs=2)

        wrapper._install_swa_sampler(fake, state, stage_start_epoch=0)
        for epoch in range(5):
            fake._epoch = epoch
            fake.params = {"ml": {"w": jnp.array([float(epoch + 1)])}}
            fake.run_post_epoch()

        np.testing.assert_allclose(state.averaged_params["ml"]["w"], np.array([3.0]))
        self.assertEqual(state.sample_epochs, [2, 4])
        np.testing.assert_allclose(fake.params["ml"]["w"], np.array([5.0]))

    def test_save_swa_checkpoint_if_ready_writes_artifact_and_keeps_wrapper_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = object.__new__(Trainer)
            wrapper.checkpoint_path = Path(tmp)
            wrapper.params = {"ml": {"w": jnp.array([99.0])}}
            fake = _FakeChemtrainTrainer()
            state = SWAState(stage="adam", start_epoch=1, sample_freq_epochs=1)
            state.update({"ml": {"w": jnp.array([4.0])}}, epoch=1)
            final_losses = {}

            wrapper._save_swa_checkpoint_if_ready(
                state,
                {"save_checkpoint": True},
                "adam",
                10,
                final_losses,
                fake,
            )

            output = Path(tmp) / "swa_adam_epoch10.pkl"
            self.assertTrue(output.exists())
            self.assertEqual(final_losses["swa_checkpoint"], str(output))
            self.assertEqual(final_losses["swa_sample_epochs"], [1])
            np.testing.assert_allclose(wrapper.params["ml"]["w"], np.array([99.0]))


class SWACheckpointTests(unittest.TestCase):
    def test_save_swa_checkpoint_writes_downstream_friendly_payload_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = SWAState(stage="adam", start_epoch=2, sample_freq_epochs=1)
            state.update({"ml": {"w": jnp.array([1.0])}}, epoch=2)
            path = Path(tmp) / "swa_adam_epoch10.pkl"
            metadata = {"stage": "adam", "completed_epochs": 10}

            save_swa_checkpoint(path, state, metadata)

            with path.open("rb") as handle:
                payload = pickle.load(handle)
            with path.with_suffix(".meta.pkl").open("rb") as handle:
                meta_payload = pickle.load(handle)

        np.testing.assert_allclose(payload["params"]["ml"]["w"], np.array([1.0]))
        np.testing.assert_allclose(payload["best_params"]["ml"]["w"], np.array([1.0]))
        self.assertEqual(payload["metadata"]["sample_epochs"], [2])
        self.assertEqual(meta_payload["sample_count"], 1)
        self.assertEqual(meta_payload["stage"], "adam")
