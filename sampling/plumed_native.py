"""Render native PLUMED input for biases that do not need the Python bias server.

WHY THIS EXISTS
    The socket route (GROMACS -> PLUMED -> CGBias.so -> Unix socket -> Python/JAX) costs a
    measured **1.86x in throughput**: on ala2 (2,642 atoms) the pure-PLUMED `harvest_v2`
    campaign runs at 937.6 ns/day against 505.0 ns/day for `allcorridor_metad`. It also
    forces one Python process (and, for the teacher, one JAX model copy) per replica, which
    is what makes packing several replicas onto a GPU expensive.

    Three of the four registered biases turn out not to need Python at all.

WHAT IS EXACTLY EXPRESSIBLE, AND WHY
    `TICAProjection.transform` is

        z = (pair_distances - mean) @ coefficients

    and PLUMED's COMBINE computes `sum_i c_i (x_i - a_i)^p_i`. With `POWERS=1` those are the
    SAME function, so the TICA CVs are reproduced exactly by `DISTANCE` + `COMBINE` and PLUMED
    differentiates them to atomic forces itself. No surrogate, no fit.

    From there:
      tica_metad     -> METAD, a PLUMED built-in (and a strict upgrade: it gets grids and a
                        real HILLS restart, which the Python term never had -- `save_hills()`
                        is not called anywhere in production)
      tica_regional  -> a STATIC scalar field of (tic1, tic2). Tabulate it once here and hand
                        PLUMED an EXTERNAL grid; the harmonic walls become UPPER/LOWER_WALLS.
      R_TP           -> same EXTERNAL mechanism, different precomputed field.

WHAT IS NOT PORTED, DELIBERATELY
    `local_inversion_umbrella` restrains the normalised signed volume chi, which is not a
    PLUMED CV. It can be rebuilt from three ANGLEs via Cayley-Menger + CUSTOM, but the sign is
    ambiguous, d(chi)/dR diverges as chi -> 0 -- precisely the last rung of the production
    chi_target ladder (-0.71 ... 0.0) -- and the ramp measures chi at step 0, which
    MOVINGRESTRAINT cannot express. It stays on the Python route. (PYCVINTERFACE is not built
    into the deployed kernel either, so there is no escape hatch.)

    `mlcg_teacher` needs the JAX model; it stays on the socket until the MLIR connector lands.

UNITS
    Every file starts with `UNITS LENGTH=A ENERGY=kcal/mol`, so the numbers written here are
    the numbers stored in the artifacts (Angstrom, kcal/mol) with no conversion anywhere.
    Without it each constant would need its own factor (coefficients x10, mean /10,
    energies x4.184) -- four more places to get a silent factor wrong.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .biases.tica_regional import SmoothTICABias
from .mapping import CGMapping

__all__ = [
    "tica_cv_block",
    "walls_block",
    "metad_block",
    "external_block",
    "write_external_grid",
    "PlumedNativeBias",
]

UNITS_HEADER = "UNITS LENGTH=A ENERGY=kcal/mol\n"


def _fmt(values: Sequence[float], places: int = 10) -> str:
    return ",".join(f"{float(v):.{places}g}" for v in values)


def _without_walls(bias: SmoothTICABias) -> SmoothTICABias:
    """Copy of `bias` with zero wall stiffness, for tabulating the KDE part alone.

    NOT `dataclasses.replace`: `attractor_weights` / `attractor_depth` / `attractor_norm` are
    injected by `SmoothTICABias.load` through `object.__setattr__` and are therefore NOT
    dataclass fields. `replace()` would silently drop them, and `tica_energy_gradient` would
    fall back to the log-ratio branch -- a completely different bias. `copy.copy` keeps the
    whole instance dict.
    """
    clone = copy.copy(bias)
    object.__setattr__(clone, "wall_k_kcal_mol", np.zeros_like(bias.wall_k_kcal_mol))
    return clone


def padded_bounds(bias: SmoothTICABias, pad: float = 0.15,
                  pad_sigma: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """TICA bounds widened by the thermal excursion the walls actually permit.

    Both PLUMED actions that take a grid -- `EXTERNAL` and gridded `METAD` -- abort mid-run
    if the CV leaves it, so the padding is sized from PHYSICS: a wall of stiffness k lets the
    CV wander `sigma = sqrt(kT/k)` past `bounds`. `pad` (a fraction of the range) is only a
    floor. For the ala2 artifact k = [1.29, 11.2] gives sigma = [0.68, 0.23] while a 15 % pad
    supplies 0.79 on tic1 -- barely one sigma -- so the fraction alone is not safe.

    Shared by the EXTERNAL grid and the METAD grid so the two cannot disagree about where
    the sampled region ends.
    """
    wall_k = np.asarray(bias.wall_k_kcal_mol, dtype=float)
    kbt = float(bias.kbt_kcal_mol)
    lo = np.asarray(bias.bounds[:, 0], dtype=float)
    hi = np.asarray(bias.bounds[:, 1], dtype=float)
    with np.errstate(divide="ignore"):
        sigma = np.where(wall_k > 0, np.sqrt(kbt / np.maximum(wall_k, 1e-30)), 0.0)
    margin = np.maximum(pad * (hi - lo), pad_sigma * sigma)
    return lo - margin, hi + margin


def tica_cv_block(bias: SmoothTICABias, mapping: CGMapping, prefix: str = "tic") -> str:
    """DISTANCE per bead pair + one COMBINE per TIC.

    Exact, not approximate: COMBINE's `sum_i c_i (x_i - a_i)^p_i` with POWERS=1 is literally
    `(d - mean) @ coefficients`. Bead indices are translated to the 1-based AA atom numbers
    PLUMED expects through the mapping, never hardcoded.
    """
    proj = bias.projection
    atoms = mapping.aa_atom_indices_1based
    lines = [f"# TICA CVs: {len(proj.pairs)} pair distances -> {proj.coefficients.shape[1]} TICs",
             "# z = (d - mean) @ coefficients   == COMBINE(COEFFICIENTS, PARAMETERS, POWERS=1)"]
    for k, (i, j) in enumerate(proj.pairs):
        lines.append(f"d{k}: DISTANCE ATOMS={atoms[int(i)]},{atoms[int(j)]}")
    args = ",".join(f"d{k}" for k in range(len(proj.pairs)))
    for t in range(proj.coefficients.shape[1]):
        lines.append(
            f"{prefix}{t + 1}: COMBINE ARG={args} "
            f"COEFFICIENTS={_fmt(proj.coefficients[:, t])} "
            f"PARAMETERS={_fmt(proj.mean)} "
            f"POWERS={_fmt(np.ones(len(proj.pairs)), places=1)} PERIODIC=NO"
        )
    return "\n".join(lines) + "\n"


def walls_block(bias: SmoothTICABias, prefix: str = "tic", label: str = "twall") -> str:
    """Harmonic walls at the TICA grid bounds.

    FACTOR OF TWO: the Python term adds `0.5 * wall_k * d**2`, while PLUMED's UPPER_WALLS is
    `KAPPA * ((x - AT)/EPS)**EXP` with NO 1/2. So KAPPA_plumed = 0.5 * wall_k. Getting this
    wrong doubles the wall stiffness silently.
    """
    lo, hi = bias.bounds[:, 0], bias.bounds[:, 1]
    kappa = 0.5 * np.asarray(bias.wall_k_kcal_mol, dtype=float)
    args = ",".join(f"{prefix}{t + 1}" for t in range(len(lo)))
    return (
        f"# walls: PLUMED has no 1/2 in KAPPA, Python does -> KAPPA = 0.5 * wall_k\n"
        f"{label}_lo: LOWER_WALLS ARG={args} AT={_fmt(lo)} KAPPA={_fmt(kappa)}\n"
        f"{label}_hi: UPPER_WALLS ARG={args} AT={_fmt(hi)} KAPPA={_fmt(kappa)}\n"
    )


def metad_block(prefix: str = "tic", *, height: float, sigma: Sequence[float], pace: int,
                bias_factor: float, temperature: float, equilibrate_steps: int,
                dt_ps: float, grid_min: Sequence[float] | None = None,
                grid_max: Sequence[float] | None = None, walkers_mpi: bool = False,
                label: str = "metad") -> str:
    """Well-tempered MetaD on the TICA CVs.

    Two differences from the Python term, both intentional:
      * PLUMED deposits on `step % PACE == 0`; the Python term used a catch-up schedule from
        `equilibrate_steps` because `step` advanced by RECOMPUTE_STRIDE. The equilibration is
        expressed here as UPDATE_FROM in PLUMED time units (ps), hence `dt_ps`.
      * Hills go to a PLUMED HILLS file, which supports RESTART. The Python NPZ path never
        wrote hills in production, so this only adds capability.

    `walkers_mpi=True` turns N replicas into N walkers on ONE shared bias -- required for
    wide-and-short discovery, see the comment at the emission site.
    """
    args = ",".join(f"{prefix}{t + 1}" for t in range(len(sigma)))
    line = (f"{label}: METAD ARG={args} HEIGHT={height:.10g} SIGMA={_fmt(sigma)} "
            f"PACE={int(pace)} BIASFACTOR={bias_factor:.10g} TEMP={temperature:.10g} "
            f"FILE=HILLS")
    if walkers_mpi:
        # MULTIPLE WALKERS. Without this, N replicas each build their OWN hill history from
        # zero: 64 replicas give 64x redundant filling and no replica gets far, which defeats
        # the whole "short but wide" discovery strategy. With it they share ONE growing bias.
        #
        # WALKERS_MPI shares hills over the MPI communicator, NOT through a shared directory
        # (WALKERS_DIR is the file-based alternative and is the only compatible WALKERS_*
        # option -- checked against the deployed 2.9.3 kernel's own manual). `-multidir`
        # already sets up the multi-replica communicator, so nothing else is needed.
        line += " WALKERS_MPI"
    if equilibrate_steps > 0:
        line += f" UPDATE_FROM={equilibrate_steps * dt_ps:.10g}"
    if grid_min is not None and grid_max is not None:
        line += (f" GRID_MIN={_fmt(grid_min)} GRID_MAX={_fmt(grid_max)}"
                 f" CALC_RCT")
    return (f"# well-tempered MetaD; UPDATE_FROM is ps, = equilibrate_steps * dt\n{line}\n")


def write_external_grid(bias: SmoothTICABias, path: Path, *, n_points: Sequence[int] = (401, 401),
                        pad: float = 0.15, pad_sigma: float = 4.0, label: str = "treg",
                        prefix: str = "tic") -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the static tica_regional field for PLUMED's EXTERNAL action.

    PLUMED EXTERNAL **errors out (hard, mid-run) if the CV leaves the grid**, so the padding
    beyond `bounds` is sized from PHYSICS, not as a fixed fraction of the range.

    A wall of stiffness k lets the CV wander past `bounds` with a thermal spread of
    `sigma = sqrt(kT/k)`; the grid must cover several of those. For the current ala2 artifact
    k = [0.647, 5.60] kcal/mol and kT = 0.592, giving sigma = [0.96, 0.33]. A naive 15%-of-
    range pad supplies only 0.79 along tic1 -- less than ONE sigma -- so a normal run would
    walk off the grid and abort. `pad_sigma` (default 4) fixes that; `pad` is kept as a floor.

    Grid SPACING is held fixed as the range grows, so n_points scales with the padded range
    rather than the resolution silently degrading.

    Values and analytic derivatives come from the SAME `tica_energy_gradient` the Python bias
    uses, so the two agree to grid-interpolation error and nothing else can drift.

    Measured convergence against the Python bias (20 reference frames, ala2 bb6 all-corridor
    attractor, via `plumed driver`; max force error as a fraction of max |F|):

        n_points   max|dE|    max|dF|    file
          161      1.3e-02     5.2%      1.6 MB
          401      5.2e-03     2.0%       10 MB      <- default
          801      2.6e-03     0.96%      40 MB

    401 is the default because a sampling bias does not need better than ~2% on the force,
    and the file is re-read by every replica at campaign start. Raise it if the field is
    being used quantitatively rather than to drive exploration.

    THE WALLS ARE EXCLUDED HERE ON PURPOSE. `tica_energy_gradient` adds `0.5*wall_k*d^2`
    internally, and `walls_block()` emits UPPER_WALLS/LOWER_WALLS as separate PLUMED actions.
    Tabulating the walls as well would apply them TWICE. Measured while building this: with
    the walls left in, the single test frame that sat outside `bounds` had a force error of
    0.207 kcal/mol/A against ~0.008 for the 19 frames inside -- a 25x outlier that pointed
    straight at the double count. Zeroing them here also makes the wall forces exact
    (analytic in PLUMED) instead of grid-interpolated.
    """
    lo_p, hi_p = padded_bounds(bias, pad=pad, pad_sigma=pad_sigma)
    lo = np.asarray(bias.bounds[:, 0], dtype=float)
    hi = np.asarray(bias.bounds[:, 1], dtype=float)
    span = hi - lo
    bias = _without_walls(bias)

    # keep the ORIGINAL spacing (from n_points over the fraction-padded range) as the range
    # grows, instead of stretching the same number of points over a wider window
    base_span = span * (1.0 + 2.0 * pad)
    spacing = base_span / (np.asarray(n_points, dtype=float) - 1.0)
    counts = np.maximum(np.ceil((hi_p - lo_p) / spacing).astype(int) + 1, 3)
    axes = [np.linspace(lo_p[a], hi_p[a], int(counts[a])) for a in range(2)]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"{prefix}1", f"{prefix}2"]
    with path.open("w") as fh:
        # The value column MUST be named "<action label>.bias" -- PLUMED looks the grid column
        # up by the label of the EXTERNAL action, not by a fixed name, and aborts otherwise.
        fh.write(f"#! FIELDS {names[0]} {names[1]} {label}.bias "
                 f"der_{names[0]} der_{names[1]}\n")
        for a in range(2):
            # nbins MUST come from the ACTUAL axis length, not the requested n_points: the
            # padding below grows the axes, and a stale nbins makes PLUMED reinterpret the
            # whole grid on the wrong stride (silently, as a plausible-looking wrong field).
            fh.write(f"#! SET min_{names[a]} {lo_p[a]:.10g}\n")
            fh.write(f"#! SET max_{names[a]} {hi_p[a]:.10g}\n")
            fh.write(f"#! SET nbins_{names[a]} {len(axes[a]) - 1}\n")
            fh.write(f"#! SET periodic_{names[a]} false\n")
        # PLUMED grids vary the FIRST field fastest, so tic1 is the inner loop.
        # Evaluated one ROW at a time rather than point-by-point: the scalar path is
        # O(n_points^2 * n_centres) Python-level calls (585 centres here) and a 1201^2 grid
        # did not finish in two minutes. Row-chunking keeps peak memory at
        # n_points x n_centres while giving numpy the whole row.
        # Zero-weight KDE centres make log(w) = -inf, which logsumexp handles correctly; the
        # numpy warning is noise, not a problem.
        with np.errstate(divide="ignore"):
            for y in axes[1]:
                energy, grad = _row_energy_gradient(bias, axes[0], float(y))
                for i, x in enumerate(axes[0]):
                    fh.write(f"{x:.10g} {y:.10g} {energy[i]:.10g} "
                             f"{grad[i, 0]:.10g} {grad[i, 1]:.10g}\n")
    return axes[0], axes[1]


def _row_energy_gradient(bias: SmoothTICABias, xs: np.ndarray, y: float):
    """Vectorised `tica_energy_gradient` over one row of constant tic2.

    Mirrors the scalar implementation exactly (both branches); the equivalence test pins it
    against `bias.tica_energy_gradient` point by point so the two cannot drift.
    """
    from scipy.special import logsumexp

    z = np.column_stack([np.asarray(xs, dtype=float), np.full(len(xs), float(y))])
    h = np.asarray(bias.bandwidth, dtype=float)

    def log_density(weights):
        # (n_x, n_centres, 2)
        delta = z[:, None, :] - bias.centers[None, :, :]
        expo = np.log(weights)[None, :] - 0.5 * np.sum((delta / h) ** 2, axis=2)
        ld = logsumexp(expo, axis=1)
        resp = np.exp(expo - ld[:, None])
        grad = np.sum(resp[:, :, None] * (-delta / h**2), axis=1)
        return ld, grad

    attractor = getattr(bias, "attractor_weights", None)
    if attractor is not None:
        ld, grad_ld = log_density(attractor)
        rho = np.exp(ld) / bias.attractor_norm
        energy = bias.attractor_depth * (1.0 - rho)
        gradient = -bias.attractor_depth * rho[:, None] * grad_ld
    else:
        ld_t, g_t = log_density(bias.target_weights)
        ld_r, g_r = log_density(bias.reference_weights)
        energy = -bias.kbt_kcal_mol * (ld_t - ld_r) + bias.energy_offset_kcal_mol
        gradient = -bias.kbt_kcal_mol * (g_t - g_r)

    for axis in range(2):
        lower, upper = bias.bounds[axis]
        val = z[:, axis]
        d = np.where(val < lower, val - lower, np.where(val > upper, val - upper, 0.0))
        energy = energy + 0.5 * bias.wall_k_kcal_mol[axis] * d**2
        gradient[:, axis] += bias.wall_k_kcal_mol[axis] * d
    return energy, gradient


def external_block(grid_path: Path, prefix: str = "tic", label: str = "treg") -> str:
    # `label:` already names the action -- adding LABEL= as well is a parse error.
    args = f"{prefix}1,{prefix}2"
    return (f"# static tica_regional field, tabulated from tica_energy_gradient\n"
            f"{label}: EXTERNAL ARG={args} FILE={grid_path}\n")


@dataclass(frozen=True)
class PlumedNativeBias:
    """One campaign's native PLUMED input, assembled from the same artifacts the Python
    biases load so there is a single source of truth for every constant."""

    mapping: CGMapping
    bias: SmoothTICABias

    def render(self, *, monitor_cvs: Sequence[str] = ("phi", "psi"),
               use_regional: bool = False, grid_path: Path | None = None,
               metad: dict | None = None, print_stride: int = 100,
               colvar: str = "colvar.dat") -> str:
        parts = [UNITS_HEADER,
                 f"WHOLEMOLECULES ENTITY0={self.mapping.plumed_atom_selection()}\n",
                 tica_cv_block(self.bias, self.mapping)]
        printed = ["tic1", "tic2"]
        if use_regional:
            if grid_path is None:
                raise ValueError("use_regional=True requires grid_path")
            parts.append(external_block(grid_path))
            parts.append(walls_block(self.bias))
            printed.append("treg.bias")
        if metad:
            parts.append(metad_block(**metad))
            printed.append("metad.bias")
        for name in monitor_cvs:
            cv = self.mapping.cvs[name]
            atoms = ",".join(str(a) for a in cv.atom_indices_1based(self.mapping))
            parts.append(f"{name}: TORSION ATOMS={atoms}\n")
            printed.append(name)
        parts.append(f"PRINT ARG={','.join(printed)} FILE={colvar} STRIDE={int(print_stride)}\n")
        return "".join(parts)


def write_grid_from_fn(fn, path: Path, lo, hi, *, n_points=(401, 401), label: str = "treg",
                       prefix: str = "tic") -> tuple[np.ndarray, np.ndarray]:
    """Tabulate ANY `z -> (V, grad V)` callable as a PLUMED EXTERNAL grid.

    Generic counterpart of `write_external_grid`, which is hard-wired to the KDE artifact.
    Used for the flow-based acquisition bias, whose `V` comes from `flow_bias.AcquisitionBias`.

    The same three header traps apply, each of which cost a debugging cycle when the KDE grid
    was first written:
      * the value column MUST be named `<label>.bias` -- PLUMED looks it up by the label of the
        EXTERNAL action, not by a fixed name, and aborts otherwise;
      * `nbins_` must come from the ACTUAL axis length, or PLUMED reads the whole grid on the
        wrong stride and produces a plausible-looking wrong field;
      * PLUMED varies the FIRST field fastest, so the first coordinate is the inner loop.

    WALLS ARE NOT INCLUDED, by the same argument as `write_external_grid`: they are emitted as
    separate UPPER_WALLS/LOWER_WALLS actions, and tabulating them too would apply them twice.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    axes = [np.linspace(lo[a], hi[a], int(n_points[a])) for a in range(2)]
    names = [f"{prefix}1", f"{prefix}2"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f"#! FIELDS {names[0]} {names[1]} {label}.bias "
                 f"der_{names[0]} der_{names[1]}\n")
        for a in range(2):
            fh.write(f"#! SET min_{names[a]} {axes[a][0]:.10g}\n")
            fh.write(f"#! SET max_{names[a]} {axes[a][-1]:.10g}\n")
            fh.write(f"#! SET nbins_{names[a]} {len(axes[a]) - 1}\n")
            fh.write(f"#! SET periodic_{names[a]} false\n")
        for y in axes[1]:
            row = np.stack([axes[0], np.full(len(axes[0]), float(y))], -1)
            V, dV = fn(row)
            for i, x in enumerate(axes[0]):
                fh.write(f"{x:.10g} {y:.10g} {V[i]:.10g} {dV[i, 0]:.10g} {dV[i, 1]:.10g}\n")
    return axes[0], axes[1]
