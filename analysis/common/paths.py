"""Strict, repository-aware path handling for analysis commands."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable


class AnalysisPathError(ValueError):
    """Raised when an analysis path is ambiguous or violates its contract."""


def repo_root(project_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the resolved cameo_cg repository root."""
    candidate = project_root or os.environ.get("CAMEO_CG_PROJECT_ROOT")
    root = Path(candidate).expanduser().resolve() if candidate else Path(__file__).resolve().parents[2]
    if not (root / "analysis").is_dir():
        raise AnalysisPathError(f"Not a cameo_cg checkout: {root}")
    return root


def _base(base: str | os.PathLike[str] | None) -> Path:
    return Path(base).expanduser().resolve() if base else repo_root()


def _candidate(path: str | os.PathLike[str], base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else base / value


def resolve_input(
    path: str | os.PathLike[str],
    *,
    base: str | os.PathLike[str] | None = None,
    label: str = "input",
) -> Path:
    """Resolve an existing input path or raise a useful error."""
    candidate = _candidate(path, _base(base))
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc


def resolve_output(
    path: str | os.PathLike[str],
    *,
    base: str | os.PathLike[str] | None = None,
    create: bool = True,
) -> Path:
    """Resolve an output directory and create it if requested."""
    candidate = _candidate(path, _base(base)).resolve(strict=False)
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def resolve_glob(
    pattern: str,
    *,
    base: str | os.PathLike[str] | None = None,
    label: str = "input pattern",
    files_only: bool = True,
) -> list[Path]:
    """Resolve a glob pattern against the project root and require a match."""
    root = _base(base)
    raw = Path(pattern).expanduser()
    full_pattern = str(raw if raw.is_absolute() else root / raw)
    matches = [Path(item).resolve() for item in sorted(glob.glob(full_pattern))]
    if files_only:
        matches = [item for item in matches if item.is_file()]
    if not matches:
        raise FileNotFoundError(f"{label} matched no paths: {full_pattern}")
    return matches


def resolved_paths(
    values: Iterable[str | os.PathLike[str]],
    *,
    base: str | os.PathLike[str] | None = None,
    label: str = "input",
) -> list[Path]:
    """Resolve a sequence of explicit input paths."""
    return [resolve_input(value, base=base, label=label) for value in values]
