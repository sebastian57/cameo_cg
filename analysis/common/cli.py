"""Shared CLI conventions for analysis entry points."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .paths import repo_root, resolve_output


def add_project_root_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="cameo_cg root for relative paths (default: environment or package root)",
    )


def add_output_argument(
    parser: ArgumentParser, *, required: bool = True, default: str | None = None
) -> None:
    parser.add_argument(
        "--outdir", "--output-dir", dest="outdir", type=Path,
        required=required, default=default,
        help="directory for figures, tables, reports, and manifest.json",
    )


def resolve_cli_output(args: Namespace, *, label: str = "analysis") -> Path:
    value = args.outdir
    if value is None:
        root = repo_root(getattr(args, "project_root", None))
        value = root / "local_work" / "analysis" / label
    return resolve_output(value, base=getattr(args, "project_root", None))


def configure_cpu_default() -> None:
    """Prevent accidental JAX GPU allocation for non-Slurm analysis commands."""
    if (
        "JAX_PLATFORMS" not in os.environ
        and "JAX_PLATFORM_NAME" not in os.environ
        and not os.environ.get("SLURM_JOB_ID")
    ):
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
