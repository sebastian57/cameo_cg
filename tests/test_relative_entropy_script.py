import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from scripts.train_relative_entropy import (
    _dihedral_degrees,
    _load_configured_initial_states,
)

PYTHON = "/e/project1/cameo/schmidt36/venv_cameocg_jupiter2026/bin/python"
MODULES = "source /e/project1/cameo/schmidt36/load_modules_2026.sh"


def _run_python(args):
    quoted = " ".join(args)
    return subprocess.run(
        f"{MODULES} && PYTHONPATH=. {PYTHON} {quoted}",
        cwd=Path(__file__).resolve().parents[1],
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )


class RelativeEntropyScriptTests(unittest.TestCase):
    def test_two_dimensional_periodic_initial_state_selection(self):
        base = np.asarray(
            [
                [-0.4, 0.8, 0.3],
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [1.5, 0.9, 0.7],
                [2.2, 1.1, -0.2],
            ],
            dtype=np.float32,
        )
        coordinates = np.stack(
            [base, base + np.asarray([0.0, 0.0, 0.1], dtype=np.float32), base.copy()]
        )
        coordinates[1, 4] += np.asarray([0.4, -0.2, 0.7], dtype=np.float32)
        coordinates[2, 0] += np.asarray([-0.3, 0.5, 0.2], dtype=np.float32)
        values = np.column_stack(
            [
                _dihedral_degrees(coordinates, (0, 1, 2, 3), 180.0),
                _dihedral_degrees(coordinates, (1, 2, 3, 4), 180.0),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "starts.npz"
            np.savez_compressed(
                path,
                R=coordinates,
                mask=np.ones((3, 5), dtype=np.float32),
                species=np.zeros((3, 5), dtype=np.int32),
            )
            config = SimpleNamespace(config_path=Path(tmp) / "config.yaml")
            re_cfg = SimpleNamespace(
                start_frame_mode="configured_cv_targets",
                initial_state_data_path=str(path),
                initial_state_phi_indices=(0, 1, 2, 3),
                initial_state_phi_shift_deg=180.0,
                initial_state_phi_targets_deg=(),
                initial_state_cv_indices=((0, 1, 2, 3), (1, 2, 3, 4)),
                initial_state_cv_shift_deg=(180.0, 180.0),
                initial_state_cv_targets_deg=(tuple(values[2]), tuple(values[1])),
            )
            reference = {
                "R": coordinates,
                "mask": np.ones((3, 5), dtype=np.float32),
                "species": np.zeros((3, 5), dtype=np.int32),
            }
            initial, metadata = _load_configured_initial_states(config, re_cfg, reference)

        self.assertEqual(metadata["selected_frame_indices"], [2, 1])
        np.testing.assert_allclose(initial["R"], coordinates[[2, 1]])
        np.testing.assert_allclose(metadata["periodic_euclidean_errors_deg"], 0.0, atol=1.0e-6)

    def test_help_works(self):
        result = _run_python(["scripts/train_relative_entropy.py", "--help"])

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("relative-entropy", result.stdout.lower())

    def test_disabled_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "disabled.yaml"
            config.write_text(
                """
seed: 1
data:
  path: missing.npz
model:
  ml_model: allegro
  use_priors: false
optimizer:
  adam:
    lr: 0.001
training:
  relative_entropy:
    enabled: false
""",
                encoding="utf-8",
            )
            result = _run_python(["scripts/train_relative_entropy.py", str(config)])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("training.relative_entropy.enabled=true", result.stderr)


if __name__ == "__main__":
    unittest.main()
