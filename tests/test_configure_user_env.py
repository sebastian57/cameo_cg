from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_configure_user_env_migrates_legacy_block(tmp_path: Path):
    bashrc = tmp_path / "bashrc"
    bashrc.write_text(
        "export KEEP_ME=yes\n"
        "# >>> cameo_cg_pkgflow env >>>\n"
        "export CAMEO_CG_PROJECT_ROOT=/old/repo\n"
        "export CAMEO_STANDARD_VENV=/old/venv\n"
        "# <<< cameo_cg_pkgflow env <<<\n"
    )

    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "configure_user_env.sh"),
            "--bashrc",
            str(bashrc),
            "--project-root",
            "/new/repo",
            "--cueq-venv",
            "/new/cueq",
            "--standard-venv",
            "/new/standard",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    updated = bashrc.read_text()
    assert updated.count("# >>> cameo_cg env >>>") == 1
    assert "cameo_cg_pkgflow env" not in updated
    assert "export KEEP_ME=yes" in updated
    assert "export CAMEO_CG_PROJECT_ROOT=/new/repo" in updated
    assert "export CAMEO_CUEQ_VENV=/new/cueq" in updated
    assert "export CAMEO_STANDARD_VENV=/new/standard" in updated
