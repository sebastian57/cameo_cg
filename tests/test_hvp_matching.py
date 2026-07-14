import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from config.manager import ConfigManager
from training.hvp_matching import hvp_config, hvp_enabled, hvp_error, make_hvp_quantity
from training.trainer import Trainer


def _write_config(path: Path, hvp_block: str = "") -> None:
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
  hvp:
{hvp_block}
""",
        encoding="utf-8",
    )


class HVPConfigTests(unittest.TestCase):
    def test_hvp_config_defaults_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_config(config_path, "    enabled: false\n")
            config = ConfigManager(config_path)

            cfg = hvp_config(config)

        self.assertFalse(hvp_enabled(config))
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["target_key"], "HVP")
        self.assertEqual(cfg["probe_key"], "hvp_probe")
        self.assertEqual(cfg["loss_mask_key"], "hvp_loss_mask")
        self.assertEqual(cfg["energy_template"], "auto")
        self.assertAlmostEqual(cfg["lambda"], 0.01)
        self.assertTrue(cfg["stop_gradient_target"])


class HVPLossTests(unittest.TestCase):
    def test_hvp_error_broadcasts_k_by_n_mask(self):
        predictions = jnp.array(
            [
                [[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]],
                [[2.0, 4.0, 6.0], [20.0, 20.0, 20.0]],
            ],
            dtype=jnp.float32,
        )
        targets = jnp.zeros_like(predictions)
        weights = jnp.array([[1.0, 0.0], [0.5, 0.0]], dtype=jnp.float32)

        loss = hvp_error(predictions, targets, weights=weights)

        expected = (
            np.sum(np.array([1.0, 4.0, 9.0]))
            + 0.5 * np.sum(np.array([4.0, 16.0, 36.0]))
        ) / ((1.0 + 0.5) * 3.0)
        self.assertAlmostEqual(float(loss), float(expected), places=6)


class HVPQuantityTests(unittest.TestCase):
    def test_hvp_quantity_matches_quadratic_hessian_vector_product(self):
        scale = 2.5

        def energy_fn_template(params):
            del params

            def energy_fn(R, **kwargs):
                del kwargs
                return 0.5 * scale * jnp.sum(R * R)

            return energy_fn

        quantity = make_hvp_quantity(energy_fn_template)
        state = type("State", (), {"position": jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])})()
        probes = jnp.array(
            [
                [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        mask = jnp.array([1.0, 0.0], dtype=jnp.float32)

        hvp = quantity(
            state,
            energy_params={},
            hvp_probe=probes,
            mask=mask,
            species=jnp.array([1, 2], dtype=jnp.int32),
        )

        expected = scale * probes * mask[None, :, None]
        np.testing.assert_allclose(np.asarray(hvp), np.asarray(expected), rtol=1e-6, atol=1e-6)


class HVPTrainerHookTests(unittest.TestCase):
    def test_trainer_force_matching_hooks_include_hvp_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_config(
                config_path,
                """
    enabled: true
    lambda: 0.25
    target_key: HVP
    probe_key: hvp_probe
    loss_mask_key: hvp_loss_mask
""",
            )
            config = ConfigManager(config_path)

        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer._hvp_cfg = hvp_config(config)
        trainer._dsm_cfg = {"enabled": False}
        trainer._safety_cfg = {"enabled": False}
        trainer._force_loss_normalization = "valid_components"
        trainer.model = type(
            "Model",
            (),
            {"hvp_energy_fn_template": staticmethod(lambda params: (lambda R, **kwargs: jnp.sum(R * R)))},
        )()

        self.assertIsNotNone(trainer._force_matching_error_fns()["HVP"])
        self.assertEqual(trainer._force_matching_weights_keys()["HVP"], "hvp_loss_mask")
        self.assertIn("HVP", trainer._force_matching_additional_targets())

    def test_trainer_hvp_can_force_ml_only_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_config(
                config_path,
                """
    enabled: true
    energy_template: ml_only
""",
            )
            config = ConfigManager(config_path)

        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer._hvp_cfg = hvp_config(config)
        trainer._dsm_cfg = {"enabled": False}
        trainer._safety_cfg = {"enabled": False}
        trainer._force_loss_normalization = "valid_components"
        trainer.model = type(
            "Model",
            (),
            {
                "energy_fn_template": staticmethod(lambda params: (lambda R, **kwargs: jnp.sum(R))),
                "hvp_energy_fn_template": staticmethod(lambda params: (lambda R, **kwargs: 2.0 * jnp.sum(R))),
            },
        )()

        quantity = trainer._force_matching_additional_targets()["HVP"]
        state = type("State", (), {"position": jnp.ones((1, 3), dtype=jnp.float32)})()
        out = quantity(
            state,
            energy_params={},
            hvp_probe=jnp.ones((1, 1, 3), dtype=jnp.float32),
            mask=jnp.ones((1,), dtype=jnp.float32),
            species=jnp.ones((1,), dtype=jnp.int32),
        )

        np.testing.assert_allclose(np.asarray(out), np.zeros((1, 1, 3), dtype=np.float32))



    def test_trainer_loader_kwargs_preserve_hvp_arrays(self):
        trainer = Trainer.__new__(Trainer)
        split = {
            "R": np.zeros((1, 2, 3), dtype=np.float32),
            "F": np.zeros((1, 2, 3), dtype=np.float32),
            "mask": np.ones((1, 2), dtype=np.float32),
            "species": np.ones((1, 2), dtype=np.int32),
            "hvp_probe": np.zeros((1, 3, 2, 3), dtype=np.float32),
            "HVP": np.zeros((1, 3, 2, 3), dtype=np.float32),
            "hvp_loss_mask": np.ones((1, 3, 2), dtype=np.float32),
        }

        kwargs = trainer._split_loader_kwargs(split)

        self.assertIn("hvp_probe", kwargs)
        self.assertIn("HVP", kwargs)
        self.assertIn("hvp_loss_mask", kwargs)

if __name__ == "__main__":
    unittest.main()
