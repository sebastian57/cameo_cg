import subprocess
import tempfile
import unittest
from pathlib import Path

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
