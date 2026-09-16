"""Compatibility wrapper; use analysis.md.analyze_traj instead."""
from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Forward the public API. The 2026-08-25 refactor moved the implementation to
# analysis/md/ and left this wrapper behind, but it forwarded no symbols -- which
# broke `md/__init__.py` (imports 8 names from here), and therefore
# `from md.runner import MDRunner`, and therefore ALL MD runs (job 1491932).
# analysis/md/analyze_traj.py imports nothing from `md`, so there is no cycle.
from analysis.md.analyze_traj import (  # noqa: E402,F401
    load_npz_coords,
    load_dump_coords,
    choose_pairs,
    load_pairs_csv,
    build_features,
    pair_hash,
    fit_tica,
    fit_pca,
    project_onto_model,
    compute_fes_2d,
    plot_fes,
    write_projection_csv,
    write_pairs_csv,
)

if __name__ == "__main__":
    runpy.run_module("analysis.md.analyze_traj", run_name="__main__")
