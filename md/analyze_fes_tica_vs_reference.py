"""Compatibility wrapper; use analysis.md.analyze_fes_tica_vs_reference instead."""
from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    runpy.run_module("analysis.md.analyze_fes_tica_vs_reference", run_name="__main__")
