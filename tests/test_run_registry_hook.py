import os
from pathlib import Path
import subprocess
import sys

from runs.registry import Registry


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def registry_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CAMEO_RUN_REGISTRY_DB": str(tmp_path / "registry.sqlite3"),
            "CAMEO_RUN_REGISTRY_MD": str(tmp_path / "REGISTRY.md"),
            "CAMEO_RUN_REGISTRY_LOCK": str(tmp_path / "registry.lock"),
            "RUN_REGISTRY_PYTHON": sys.executable,
            "SLURM_JOB_ID": "777",
            "SLURM_JOB_NAME": "hook-test",
            "SLURM_JOB_USER": "alice",
        }
    )
    return env


def test_hook_records_start_and_finish(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "source runs/registry_hook.sh\n"
        f"run_registry_start md '' '{output}'\n"
        "run_registry_finish 0\n"
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=PROJECT_ROOT,
        env=registry_env(tmp_path),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    registry = Registry(
        tmp_path / "registry.sqlite3",
        tmp_path / "REGISTRY.md",
        tmp_path / "registry.lock",
    )
    record = registry.get("777")
    assert record["state"] == "COMPLETED"
    assert record["run_type"] == "md"
    assert record["outputs"] == [str(output)]


def test_registry_failure_does_not_change_workload_exit_code(tmp_path: Path):
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "source runs/registry_hook.sh\n"
        "RUN_REGISTRY_PYTHON=/does/not/exist\n"
        "run_registry_start training '' /tmp/output\n"
        "exit 7\n"
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=PROJECT_ROOT,
        env=registry_env(tmp_path),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert "WARNING: run registry start failed" in result.stderr


def test_installed_exit_trap_records_failure_and_preserves_exit_code(tmp_path: Path):
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n"
        "source runs/registry_hook.sh\n"
        "run_registry_start analysis '' /tmp/analysis\n"
        "run_registry_install_exit_trap\n"
        "exit 3\n"
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=PROJECT_ROOT,
        env=registry_env(tmp_path),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    registry = Registry(
        tmp_path / "registry.sqlite3",
        tmp_path / "REGISTRY.md",
        tmp_path / "registry.lock",
    )
    assert registry.get("777")["state"] == "FAILED"
    assert registry.get("777")["exit_code"] == 3
