"""Compatibility import for the local BlackJAX checkout used on the cluster."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _ensure_local_blackjax_on_path() -> None:
    for candidate in (
        Path(__file__).resolve().parents[2] / "blackjax",
        Path("/e/project1/cameo/schmidt36/blackjax"),
    ):
        if (candidate / "blackjax" / "__init__.py").exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


def import_blackjax():
    """Import BlackJAX, tolerating editable checkouts without generated _version."""
    _ensure_local_blackjax_on_path()
    try:
        import blackjax  # type: ignore
        return blackjax
    except ModuleNotFoundError as exc:
        if exc.name != "blackjax._version":
            raise
        version_module = types.ModuleType("blackjax._version")
        version_module.__version__ = "local"
        sys.modules["blackjax._version"] = version_module
        import blackjax  # type: ignore
        return blackjax
