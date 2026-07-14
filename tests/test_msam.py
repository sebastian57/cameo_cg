import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from config.manager import ConfigManager
from training.msam import (
    microbatch_sam_gradient,
    sam_perturbation,
    shmap_msam_update_fn,
    tree_l2_norm,
)
from training.trainer import Trainer


def _write_config(path: Path, msam_block: str) -> None:
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
  msam:
{msam_block}
""",
        encoding="utf-8",
    )


class MSAMConfigTests(unittest.TestCase):
    def test_msam_config_defaults_to_disabled_last_nonzero_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            _write_config(path, "    enabled: false\n")
            config = ConfigManager(path)

            cfg = config.get_msam_config()

        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["stage"], "sgd_nesterov")
        self.assertIsNone(cfg["start_epoch"])
        self.assertAlmostEqual(cfg["start_fraction"], 0.80)
        self.assertAlmostEqual(cfg["rho"], 0.01)
        self.assertAlmostEqual(cfg["epsilon"], 1.0e-12)

    def test_msam_config_accepts_explicit_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            _write_config(
                path,
                """
    enabled: true
    stage: adam
    start_epoch: 4
    start_fraction: 0.5
    rho: 0.02
    epsilon: 1.0e-8
""",
            )
            config = ConfigManager(path)

            cfg = config.get_msam_config()

        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["stage"], "adam")
        self.assertEqual(cfg["start_epoch"], 4)
        self.assertAlmostEqual(cfg["start_fraction"], 0.5)
        self.assertAlmostEqual(cfg["rho"], 0.02)
        self.assertAlmostEqual(cfg["epsilon"], 1.0e-8)

    def test_msam_config_rejects_invalid_values(self):
        invalid_blocks = [
            "    enabled: true\n    start_fraction: 1.5\n",
            "    enabled: true\n    start_epoch: -1\n",
            "    enabled: true\n    rho: 0.0\n",
            "    enabled: true\n    epsilon: 0.0\n",
            "    enabled: true\n    stage: missing\n",
        ]
        for block in invalid_blocks:
            with self.subTest(block=block), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"
                _write_config(path, block)
                config = ConfigManager(path)

                with self.assertRaises(ValueError):
                    config.get_msam_config()


class MSAMMathTests(unittest.TestCase):
    def test_sam_perturbation_has_requested_norm_and_zero_grad_is_finite(self):
        params = {"w": jnp.array([1.0, 2.0])}
        grad = {"w": jnp.array([3.0, 4.0])}

        perturb = sam_perturbation(params, grad, rho=0.2, epsilon=1.0e-12)

        self.assertAlmostEqual(float(tree_l2_norm(perturb)), 0.2, places=6)

        zero = sam_perturbation(params, {"w": jnp.zeros(2)}, rho=0.2, epsilon=1.0e-12)
        self.assertTrue(bool(jnp.all(jnp.isfinite(zero["w"]))))
        np.testing.assert_allclose(zero["w"], np.zeros(2), atol=1.0e-7)

    def test_microbatch_sam_gradient_matches_hand_computed_quadratic(self):
        def loss_fn(params, x):
            residual = params["w"] - x
            return 0.5 * residual * residual, {"quad": 0.5 * residual * residual}

        loss, per_target, grad = microbatch_sam_gradient(
            loss_fn,
            {"w": jnp.array(2.0)},
            jnp.array(0.0),
            rho=0.1,
            epsilon=1.0e-12,
        )

        self.assertAlmostEqual(float(loss), 2.0, places=6)
        self.assertAlmostEqual(float(per_target["quad"]), 2.0, places=6)
        self.assertAlmostEqual(float(grad["w"]), 2.1, places=6)

    def test_rho_zero_matches_base_gradient(self):
        def loss_fn(params, x):
            residual = params["w"] - x
            return 0.5 * residual * residual, {"quad": 0.5 * residual * residual}

        (_, _), base_grad = jax.value_and_grad(loss_fn, has_aux=True)(
            {"w": jnp.array(2.0)}, jnp.array(0.25)
        )
        _, _, sam_grad = microbatch_sam_gradient(
            loss_fn,
            {"w": jnp.array(2.0)},
            jnp.array(0.25),
            rho=0.0,
            epsilon=1.0e-12,
        )

        np.testing.assert_allclose(sam_grad["w"], base_grad["w"], atol=1.0e-7)

    def test_shmap_msam_update_matches_toy_two_microbatch_update(self):
        def model(params, batch):
            return jnp.ones_like(batch) * params["w"]

        def loss_fn(predictions, batch):
            residual = predictions - batch
            loss = 0.5 * jnp.mean(residual * residual)
            return loss, {"quad": loss}

        update_fn = shmap_msam_update_fn(
            model,
            loss_fn,
            optax.sgd(learning_rate=0.1),
            rho=0.1,
            epsilon=1.0e-12,
        )

        params = {"w": jnp.array(2.0)}
        opt_state = optax.sgd(learning_rate=0.1).init(params)
        batch = jnp.array([[0.0], [1.0]])

        new_params, _, loss, grad, per_target = update_fn(
            params,
            opt_state,
            batch,
            per_target=True,
            microbatch_count=2,
            accum_mode="stack_scan",
        )

        self.assertAlmostEqual(float(loss), 1.25, places=6)
        self.assertAlmostEqual(float(per_target["quad"]), 1.25, places=6)
        self.assertAlmostEqual(float(grad["w"]), 1.6, places=6)
        self.assertAlmostEqual(float(new_params["w"]), 1.84, places=6)


class _FakeChemtrainTrainer:
    def __init__(self):
        self._epoch = 0
        self._disable_shmap = False
        self.batched_model = object()
        self._loss_fn = object()
        self.base_update_fn = mock.Mock(return_value="base")
        self._update_fn = self.base_update_fn


class MSAMTrainerHookTests(unittest.TestCase):
    def test_trainer_hook_uses_base_before_start_and_msam_after_start(self):
        wrapper = object.__new__(Trainer)
        fake = _FakeChemtrainTrainer()
        optimizer = optax.sgd(learning_rate=0.1)
        msam_mock = mock.Mock(return_value="msam")

        with mock.patch("training.trainer.shmap_msam_update_fn", return_value=msam_mock):
            wrapper._install_msam_update(
                fake,
                optimizer,
                {"rho": 0.1, "epsilon": 1.0e-12, "start_epoch": 2},
                stage_start_epoch=0,
            )

        fake._epoch = 1
        self.assertEqual(fake._update_fn("params", "state", "batch"), "base")
        fake._epoch = 2
        self.assertEqual(fake._update_fn("params", "state", "batch"), "msam")
        self.assertEqual(fake._update_fn("params", "state", "batch"), "msam")
        self.assertEqual(fake.base_update_fn.call_count, 1)
        self.assertEqual(msam_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
