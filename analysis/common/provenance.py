"""Machine-readable provenance for analysis outputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .paths import resolve_output

_HASH_LIMIT_BYTES = 512 * 1024 * 1024


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _input_records(value: Any) -> Any:
    if isinstance(value, Path):
        return describe_path(value)
    if isinstance(value, Mapping):
        return {str(key): _input_records(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_input_records(item) for item in value]
    return _jsonable(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_path(path: str | os.PathLike[str], *, checksum: bool = True) -> dict[str, Any]:
    """Describe one resolved artifact without recursively walking directories."""
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "size_bytes": int(stat.st_size) if resolved.is_file() else None,
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if resolved.is_file() and checksum:
        if stat.st_size <= _HASH_LIMIT_BYTES:
            record["sha256"] = sha256_file(resolved)
        else:
            record["sha256"] = None
            record["sha256_skipped"] = f"file exceeds {_HASH_LIMIT_BYTES} bytes"
    return record


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_state(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status_lines": status.splitlines() if status else [],
    }


def _environment() -> dict[str, str | None]:
    names = (
        "CAMEO_CG_PROJECT_ROOT", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME", "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
    )
    values = {name: os.environ.get(name) for name in names}
    values.update({"python": sys.version.split()[0], "platform": platform.platform()})
    return values


def _output_records(output_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    return [
        describe_path(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]


def write_manifest(
    output_dir: str | os.PathLike[str],
    *,
    inputs: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    module: str | None = None,
    extra: Mapping[str, Any] | None = None,
    command: list[str] | None = None,
) -> Path:
    """Write a complete provenance manifest into an analysis output directory."""
    output = resolve_output(output_dir)
    manifest_path = output / "manifest.json"
    try:
        from .paths import repo_root
        root = repo_root()
        git_state = _git_state(root)
    except Exception as exc:  # pragma: no cover
        root = None
        git_state = {"error": str(exc)}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "command": command or sys.argv,
        "repository": str(root) if root else None,
        "git": git_state,
        "environment": _environment(),
        "inputs": _input_records(inputs or {}),
        "parameters": _jsonable(parameters or {}),
        "outputs": _output_records(output, manifest_path),
    }
    if extra:
        manifest["extra"] = _jsonable(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
