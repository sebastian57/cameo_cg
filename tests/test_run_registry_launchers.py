import os
from pathlib import Path
import subprocess
import sys

import pytest

from runs.registry import Registry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHERS = [
    "scripts/run_training.sh",
    "scripts/submit_suite.sh",
    "scripts/submit_md.sh",
    "scripts/submit_md_parallel.sh",
    "scripts/submit_md_array.sh",
    "scripts/run_relative_entropy.sh",
    "scripts/run_analysis.sh",
    "scripts/run_profiling.sh",
    "scripts/submit_teacher_materialization.sh",
    "data_prep/run_pipeline_gpu.sh",
    "md_setup/submit_lammps_chemtrain.sh",
]


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_has_valid_bash_syntax(launcher: str):
    result = subprocess.run(
        ["bash", "-n", launcher],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_md_launcher_records_resolved_output_and_completion(tmp_path: Path):
    fake_root = tmp_path / "cameo_cg"
    (fake_root / "configs").mkdir(parents=True)
    (fake_root / "runs").mkdir()
    (fake_root / "runs" / "registry_hook.sh").symlink_to(
        PROJECT_ROOT / "runs" / "registry_hook.sh"
    )
    (fake_root / "runs" / "registry.py").symlink_to(
        PROJECT_ROOT / "runs" / "registry.py"
    )
    (tmp_path / "load_modules_2026.sh").write_text(":\n")
    (tmp_path / "set_lammps_paths_2026.sh").write_text(":\n")
    activate = tmp_path / "venv_cameocg_jupiter2026" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("python() { return 0; }\n")
    config = fake_root / "configs" / "md.yaml"
    config.write_text(
        "run:\n"
        "  description: launcher smoke\n"
        "  tags: [md]\n"
        "md:\n"
        "  output_dir: outputs/md-smoke\n"
    )

    env = os.environ.copy()
    env.update(
        {
            "CAMEO_CG_PROJECT_ROOT": str(fake_root),
            "CAMEO_RUN_REGISTRY_DB": str(tmp_path / "registry.sqlite3"),
            "CAMEO_RUN_REGISTRY_MD": str(tmp_path / "REGISTRY.md"),
            "CAMEO_RUN_REGISTRY_LOCK": str(tmp_path / "registry.lock"),
            "RUN_REGISTRY_PYTHON": sys.executable,
            "SLURM_JOB_ID": "8123",
            "SLURM_JOB_NAME": "cameo_md",
            "SLURM_JOB_USER": "alice",
            "SLURMD_NODENAME": "jwb001",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "submit_md.sh"), str(config)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    registry = Registry(
        tmp_path / "registry.sqlite3",
        tmp_path / "REGISTRY.md",
        tmp_path / "registry.lock",
    )
    record = registry.get("8123")
    assert record is not None
    assert record["state"] == "COMPLETED"
    assert record["run_type"] == "md"
    assert record["description"] == "launcher smoke"
    assert record["outputs"] == [str(fake_root / "outputs" / "md-smoke")]
