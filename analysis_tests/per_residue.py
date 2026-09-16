"""Compatibility wrapper; use analysis.evaluation.per_residue instead."""
from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.evaluation import per_residue as _canonical
globals().update({name: getattr(_canonical, name) for name in dir(_canonical) if name not in {"__name__", "__package__", "__loader__"}})

if __name__ == "__main__":
    runpy.run_module("analysis.evaluation.per_residue", run_name="__main__")
