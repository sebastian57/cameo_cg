"""Canonical offline analysis package.

Reusable analyses live below this package; runtime, campaign, and collection code
remain in md and sampling. Use python -m analysis.<domain>.<module> for new commands.
"""

from .common import repo_root, resolve_glob, resolve_input, resolve_output, write_manifest

__all__ = ["repo_root", "resolve_glob", "resolve_input", "resolve_output", "write_manifest"]
