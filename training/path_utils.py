"""Path helpers for training entry points."""

from __future__ import annotations

from pathlib import Path


def repo_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parent.parent


def resolve_from_config_or_repo(path_value: str | Path, config_path: Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (config_path.parent / path, repo_root / path):
        if candidate.exists():
            return candidate
    return repo_root / path
