"""JAX-MD simulation runners and analysis tools for trained CG protein force fields."""

from .runner import MDRunner
from .units import to_akma, register_converter, describe_akma
from .dump import write_lammps_dump
from .analyze_traj import (
    load_npz_coords,
    load_dump_coords,
    choose_pairs,
    build_features,
    fit_tica,
    fit_pca,
    compute_fes_2d,
    plot_fes,
)

__all__ = [
    "MDRunner",
    "to_akma",
    "register_converter",
    "describe_akma",
    "write_lammps_dump",
    # Analysis helpers (analyze_traj.py)
    "load_npz_coords",
    "load_dump_coords",
    "choose_pairs",
    "build_features",
    "fit_tica",
    "fit_pca",
    "compute_fes_2d",
    "plot_fes",
]
