#!/usr/bin/env python3

import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
import argparse
from typing import Optional, Tuple

# JAX/jax_md compatibility patch (must be before any jax_md imports)
# jax_md uses jax.random.KeyArray which was removed in newer JAX versions
import jax
jax.random.KeyArray = jax.Array
_to_uncache = [mod for mod in sys.modules if mod.startswith('jax.random')]
for mod in _to_uncache:
    del sys.modules[mod]

from jax_md import space

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

# Logger — matches clean_code_base [Name] format from utils/logging.py
logger = logging.getLogger("PriorFit")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def order_from_resid(resid):
    r = np.asarray(resid)
    if r.ndim == 2:
        r = r[0]
    return np.argsort(r)

def build_bonds_angles_dihedrals(resid):

    order = order_from_resid(resid)
    N = len(order)

    bonds = np.stack([order[:-1], order[1:]], axis=1).astype(np.int32)
    angles = np.stack([order[:-2], order[1:-1], order[2:]], axis=1).astype(np.int32)

    if N >= 4:
        dihedrals = np.stack([order[:-3], order[1:-2], order[2:-1], order[3:]], axis=1).astype(np.int32)
    else:
        dihedrals = np.zeros((0, 4), dtype=np.int32)

    return (
        jnp.asarray(bonds, dtype=jnp.int32),
        jnp.asarray(angles, dtype=jnp.int32),
        jnp.asarray(dihedrals, dtype=jnp.int32),
    )


def bond_distances_single_frame(R, bonds, displacement):
    i, j = bonds[:, 0], bonds[:, 1]
    Ri, Rj = R[i], R[j]
    dR = jax.vmap(displacement)(Ri, Rj)
    return jnp.linalg.norm(dR, axis=-1)

def angles_single_frame(R, angles, displacement):
    ia, ib, ic = angles[:, 0], angles[:, 1], angles[:, 2]
    Ra, Rb, Rc = R[ia], R[ib], R[ic]

    v1 = jax.vmap(displacement)(Rb, Ra)  # a->b
    v2 = jax.vmap(displacement)(Rb, Rc)  # c->b

    dot = jnp.einsum("ij,ij->i", v1, v2)
    n1 = jnp.linalg.norm(v1, axis=-1)
    n2 = jnp.linalg.norm(v2, axis=-1)
    cos_th = dot / (n1 * n2 + 1e-12)
    cos_th = jnp.clip(cos_th, -1.0, 1.0)
    return jnp.arccos(cos_th)  # (0, pi)

def dihedral_angles_single_frame(R, dihedrals, displacement):

    if dihedrals.shape[0] == 0:
        return jnp.zeros((0,), dtype=R.dtype)

    i, j, k, l = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]
    Ri, Rj, Rk, Rl = R[i], R[j], R[k], R[l]

    b1 = jax.vmap(displacement)(Rj, Ri)  # i - j
    b2 = jax.vmap(displacement)(Rk, Rj)  # j - k
    b3 = jax.vmap(displacement)(Rl, Rk)  # k - l

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    b2_norm = jnp.linalg.norm(b2, axis=-1)
    b2_hat = b2 / (b2_norm[:, None] + 1e-12)

    m1 = jnp.cross(n1, b2_hat)

    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)

    return jnp.arctan2(y, x)


def fit_bond_harmonic(d_np, T=320.0, kB=0.0019872041):
    d_np = np.asarray(d_np, dtype=np.float64)
    r0 = float(np.mean(d_np))
    var = float(np.mean((d_np - r0) ** 2))
    kBT = kB * T
    kr = float(kBT / max(var, 1e-12))
    return r0, kr

def fit_fourier_angles_stable(
    a_np,
    n_terms=50,
    nbins=400,
    T=320.0,
    kB=0.0019872041,
    theta_min=1.0,
    theta_max=2.8,
    min_count=50,
    pseudocount=5.0,
    ridge=1e-2,
):
    beta = 1.0 / (kB * T)

    counts, edges = np.histogram(a_np, bins=nbins, range=(theta_min, theta_max), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_count
    if mask.sum() < 10:
        raise ValueError(f"Too few bins above min_count={min_count}. Lower min_count or increase data/nbins.")

    counts_smooth = counts.astype(np.float64) + pseudocount
    P_bin = counts_smooth / counts_smooth.sum()
    bin_width = edges[1] - edges[0]
    P = P_bin / bin_width

    sinth = np.clip(np.sin(centers), 1e-12, None)
    P_corr = np.clip(P / sinth, 1e-300, None)

    U = -(1.0 / beta) * np.log(P_corr)
    U = U - np.mean(U[mask])

    N = n_terms
    n = np.arange(1, N + 1)[None, :]
    th = centers[:, None]
    Xc = np.cos(n * th)
    Xs = np.sin(n * th)
    X = np.concatenate([Xc, Xs], axis=1)

    Xm = X[mask]
    Um = U[mask]

    w = np.clip(P[mask], 1e-12, None)
    Xm_w = Xm * w[:, None]
    Um_w = Um * w

    A = Xm_w.T @ Xm_w + ridge * np.eye(2 * N)
    y = Xm_w.T @ Um_w
    coeff = np.linalg.solve(A, y)

    a = coeff[:N]
    b = coeff[N:]
    return a, b, centers, U, mask, edges, counts


def U_fourier(theta, a, b):
    theta = np.asarray(theta)
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = np.arange(1, len(a) + 1)[None, :]
    th = theta[:, None]
    return (np.cos(n * th) @ a) + (np.sin(n * th) @ b)

def prior_pdf_from_U_theta(theta, U, T=320.0, kB=0.0019872041):
    beta = 1.0 / (kB * T)
    U_shift = U - np.min(U)
    w = np.sin(theta) * np.exp(-beta * U_shift)
    Z = np.trapz(w, theta)
    return w / Z


def fit_dihedral_fourier_2term(
    phi_np,
    nbins=360,
    T=320.0,
    kB=0.0019872041,
    min_count=20,
    pseudocount=0.1,
    ridge=1e-2,
):

    beta = 1.0 / (kB * T)

    counts, edges = np.histogram(phi_np, bins=nbins, range=(-np.pi, np.pi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_count
    if mask.sum() < 10:
        raise ValueError(f"Too few bins above min_count={min_count} for dihedrals.")

    counts_smooth = counts.astype(np.float64) + pseudocount
    P_bin = counts_smooth / counts_smooth.sum()
    bin_width = edges[1] - edges[0]
    P = P_bin / bin_width

    U = -(1.0 / beta) * np.log(np.clip(P, 1e-300, None))
    U = U - np.mean(U[mask])

    th = centers
    X = np.stack([np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], axis=1)  # (M,4)

    Xm = X[mask]
    Um = U[mask]

    w = np.clip(P[mask], 1e-12, None)
    Xm_w = Xm * w[:, None]
    Um_w = Um * w

    A = Xm_w.T @ Xm_w + ridge * np.eye(4)
    y = Xm_w.T @ Um_w
    coeff = np.linalg.solve(A, y)  # [A1,B1,A2,B2]

    A1, B1, A2, B2 = coeff
    k1 = float(np.sqrt(A1 * A1 + B1 * B1))
    g1 = float(np.arctan2(B1, A1))
    k2 = float(np.sqrt(A2 * A2 + B2 * B2))
    g2 = float(np.arctan2(B2, A2))

    return (k1, k2), (g1, g2), centers, U, mask, edges, counts, coeff


def prior_pdf_from_U_phi(phi, U, T=320.0, kB=0.0019872041):
    beta = 1.0 / (kB * T)
    U_shift = U - np.min(U)
    w = np.exp(-beta * U_shift)
    Z = np.trapz(w, phi)
    return w / Z


def _require_scipy_for_splines():
    """Import scipy spline/KDE dependencies only when spline mode is requested."""
    try:
        from scipy.interpolate import CubicSpline
        from scipy.stats import gaussian_kde
    except Exception as exc:
        raise ImportError(
            "Spline fitting requires scipy (scipy.interpolate + scipy.stats). "
            "Install scipy or run without --spline."
        ) from exc
    return CubicSpline, gaussian_kde


def _extract_spline_coeffs(cs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert scipy CubicSpline object into (knots, coeffs) for JAX runtime.

    scipy stores coeffs with shape (4, N-1) in descending powers:
      S_i(x) = c[0,i](dx)^3 + c[1,i](dx)^2 + c[2,i](dx) + c[3,i]
    Runtime evaluator expects shape (N-1, 4) in ascending order:
      [c0, c1, c2, c3] => c0 + c1*dx + c2*dx^2 + c3*dx^3
    """
    knots = np.asarray(cs.x, dtype=np.float64)
    c_desc = np.asarray(cs.c, dtype=np.float64)  # (4, N-1)
    coeffs = np.stack([c_desc[3], c_desc[2], c_desc[1], c_desc[0]], axis=1)
    return knots, coeffs


def _safe_probability_to_pmf(prob, T, kB, floor_rel=1e-8):
    """Convert probability density to PMF with floor and min-shift."""
    p = np.asarray(prob, dtype=np.float64)
    pmax = max(float(np.max(p)), 1e-300)
    p_floor = max(floor_rel * pmax, 1e-300)
    p_safe = np.clip(p, p_floor, None)
    U = -(kB * T) * np.log(p_safe)
    U -= float(np.min(U))
    return U


def _periodic_extend(values, period=2.0 * np.pi):
    """Duplicate angular samples by +/- one period for periodic KDE."""
    v = np.asarray(values, dtype=np.float64)
    return np.concatenate([v - period, v, v + period], axis=0)


def _kde_on_grid(samples, grid, bw_factor=1.0, weights=None):
    """Evaluate gaussian_kde on a fixed grid with optional bandwidth scaling."""
    _, gaussian_kde = _require_scipy_for_splines()
    kde = gaussian_kde(samples, weights=weights)
    if bw_factor != 1.0:
        kde.set_bandwidth(kde.factor * float(bw_factor))
    dens = kde(grid)
    return np.asarray(dens, dtype=np.float64)


def fit_bond_spline(all_bonds, T, kB, n_grid, bw_factor):
    """Fit bond PMF using KDE -> BI -> natural cubic spline."""
    CubicSpline, _ = _require_scipy_for_splines()

    x = np.asarray(all_bonds, dtype=np.float64)
    p1, p99 = np.percentile(x, [1.0, 99.0])
    span = max(p99 - p1, 1e-3)
    xmin = max(1e-6, p1 - 0.05 * span)
    xmax = p99 + 0.05 * span
    grid = np.linspace(xmin, xmax, int(n_grid), dtype=np.float64)

    dens = _kde_on_grid(x, grid, bw_factor=bw_factor)
    U = _safe_probability_to_pmf(dens, T=T, kB=kB)
    cs = CubicSpline(grid, U, bc_type="natural")
    knots, coeffs = _extract_spline_coeffs(cs)
    return knots, coeffs, grid, dens, U, cs


def fit_angle_spline(all_angles, T, kB, n_grid, bw_factor, theta_min=0.5, theta_max=None):
    """Fit global angle PMF with Jacobian-aware KDE weighting and natural spline."""
    CubicSpline, _ = _require_scipy_for_splines()
    if theta_max is None:
        theta_max = np.pi - 0.01

    x = np.asarray(all_angles, dtype=np.float64)
    x = x[(x > theta_min) & (x < theta_max)]
    grid = np.linspace(theta_min, theta_max, int(n_grid), dtype=np.float64)

    # Jacobian correction P_intrinsic(theta) ~ P_raw(theta) / sin(theta)
    sinx = np.clip(np.sin(x), 1e-8, None)
    weights = 1.0 / sinx
    dens = _kde_on_grid(x, grid, bw_factor=bw_factor, weights=weights)
    U = _safe_probability_to_pmf(dens, T=T, kB=kB)
    cs = CubicSpline(grid, U, bc_type="natural")
    knots, coeffs = _extract_spline_coeffs(cs)
    return knots, coeffs, grid, dens, U, cs


def fit_angle_spline_by_type(
    all_angles,
    all_central_species,
    n_types,
    T,
    kB,
    n_grid,
    bw_factor,
    min_samples=500,
    theta_min=0.5,
    theta_max=None,
):
    """Fit residue-specific angle splines with global fallback mask."""
    if theta_max is None:
        theta_max = np.pi - 0.01

    # Fit global spline first; this defines the shared grid for all types
    g_knots, g_coeffs, grid, _, _, _ = fit_angle_spline(
        all_angles, T=T, kB=kB, n_grid=n_grid, bw_factor=bw_factor,
        theta_min=theta_min, theta_max=theta_max
    )

    all_knots = np.tile(g_knots[None, :], (int(n_types), 1))
    all_coeffs = np.tile(g_coeffs[None, :, :], (int(n_types), 1, 1))
    type_mask = np.zeros((int(n_types),), dtype=np.int32)
    counts = np.zeros((int(n_types),), dtype=np.int64)

    CubicSpline, _ = _require_scipy_for_splines()
    angles = np.asarray(all_angles, dtype=np.float64)
    species = np.asarray(all_central_species, dtype=np.int32)

    for s in range(int(n_types)):
        sel = species == s
        x_s = angles[sel]
        x_s = x_s[(x_s > theta_min) & (x_s < theta_max)]
        counts[s] = int(x_s.size)
        if x_s.size < int(min_samples):
            continue

        sinx = np.clip(np.sin(x_s), 1e-8, None)
        w_s = 1.0 / sinx
        dens_s = _kde_on_grid(x_s, grid, bw_factor=bw_factor, weights=w_s)
        U_s = _safe_probability_to_pmf(dens_s, T=T, kB=kB)
        cs_s = CubicSpline(grid, U_s, bc_type="natural")
        k_s, c_s = _extract_spline_coeffs(cs_s)
        all_knots[s] = k_s
        all_coeffs[s] = c_s
        type_mask[s] = 1

    return g_knots, g_coeffs, all_knots, all_coeffs, type_mask, counts


def fit_dihedral_spline(all_dihedrals, T, kB, n_grid, bw_factor):
    """Fit periodic dihedral PMF via periodic KDE extension and periodic spline."""
    CubicSpline, _ = _require_scipy_for_splines()

    x = np.asarray(all_dihedrals, dtype=np.float64)
    x_ext = _periodic_extend(x, period=2.0 * np.pi)
    grid = np.linspace(-np.pi, np.pi, int(n_grid), endpoint=True, dtype=np.float64)

    dens = _kde_on_grid(x_ext, grid, bw_factor=bw_factor)
    # Required by periodic cubic BC: y[0] == y[-1]
    dens[-1] = dens[0]
    U = _safe_probability_to_pmf(dens, T=T, kB=kB)
    U[-1] = U[0]

    cs = CubicSpline(grid, U, bc_type="periodic")
    knots, coeffs = _extract_spline_coeffs(cs)
    return knots, coeffs, grid, dens, U, cs


def eval_piecewise_spline_numpy(x, knots, coeffs, periodic=False):
    """Evaluate exported spline coefficients in NumPy (for diagnostics)."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    c = np.asarray(coeffs, dtype=np.float64)
    if periodic:
        period = float(k[-1] - k[0])
        x = k[0] + np.mod(x - k[0], period)
    x = np.clip(x, k[0], np.nextafter(k[-1], k[0]))
    idx = np.searchsorted(k, x, side="right") - 1
    idx = np.clip(idx, 0, k.shape[0] - 2)
    dx = x - k[idx]
    ci = c[idx]
    return ci[:, 0] + ci[:, 1] * dx + ci[:, 2] * dx * dx + ci[:, 3] * dx * dx * dx


def eval_piecewise_spline_derivative_numpy(x, knots, coeffs, periodic=False):
    """Evaluate dU/dx from exported spline coefficients (NumPy diagnostics)."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    c = np.asarray(coeffs, dtype=np.float64)
    if periodic:
        period = float(k[-1] - k[0])
        x = k[0] + np.mod(x - k[0], period)
    x = np.clip(x, k[0], np.nextafter(k[-1], k[0]))
    idx = np.searchsorted(k, x, side="right") - 1
    idx = np.clip(idx, 0, k.shape[0] - 2)
    dx = x - k[idx]
    ci = c[idx]
    return ci[:, 1] + 2.0 * ci[:, 2] * dx + 3.0 * ci[:, 3] * dx * dx


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Fit CG priors from dataset distributions and output YAML + plots.")
    parser.add_argument("--data", action="append", default=[], help="Path to a dataset npz. Can be repeated.")
    parser.add_argument("--out_yaml", default="fitted_priors.yaml", help="Output YAML filename.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to save plots.")
    parser.add_argument("--T", type=float, default=320.0, help="Temperature (K).")
    parser.add_argument("--kB", type=float, default=0.0019872041, help="Boltzmann constant in kcal/mol/K.")
    parser.add_argument("--angle_terms", type=int, default=10, help="Number of Fourier terms for angle PMF fit.")
    parser.add_argument("--angle_bins", type=int, default=400, help="Histogram bins for angle PMF.")
    parser.add_argument("--dih_bins", type=int, default=360, help="Histogram bins for dihedral PMF.")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum count per bin to include in PMF fit.")
    parser.add_argument("--ridge", type=float, default=1e-2, help="Ridge regularization for PMF fits.")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Histogram pseudocount.")
    parser.add_argument("--theta_min", type=float, default=1e-4)
    parser.add_argument("--theta_max", type=float, default=np.pi - 1e-4)
    # Repulsion parameters: optionally set in output YAML
    parser.add_argument("--epsilon", type=float, default=1.0, help="Repulsion epsilon to write to YAML.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Repulsion sigma to write to YAML.")
    parser.add_argument("--rep_power", type=int, default=4, help="Repulsion power in (sigma/r)^power (for plotting only).")
    # Add-on spline fitting path (keeps legacy parametric fitting intact)
    parser.add_argument("--spline", action="store_true", default=False,
                        help="Enable spline fitting (KDE -> BI -> CubicSpline) and write NPZ output.")
    parser.add_argument("--spline_out", default="fitted_priors_spline.npz",
                        help="Output NPZ path for spline coefficients.")
    parser.add_argument("--residue_specific_angles", action="store_true", default=False,
                        help="Enable per-species angle splines with global fallback.")
    parser.add_argument("--angle_min_samples", type=int, default=500,
                        help="Minimum samples per species to fit type-specific angle spline.")
    parser.add_argument("--kde_bandwidth_factor", type=float, default=1.0,
                        help="Bandwidth multiplier for gaussian_kde (1.0 = default/Silverman).")
    parser.add_argument("--spline_grid_points", type=int, default=500,
                        help="Number of grid points for spline fitting.")
    parser.add_argument("--n_species", type=int, default=None,
                        help="Override number of species types for residue-specific angles.")
    parser.add_argument("--skip_plots", action="store_true", default=False,
                        help="Skip all matplotlib figure generation.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    dataset_paths = args.data if len(args.data) > 0 else [
        "datasets/4zohB01_temps_kcalmol_notnorm_1bead_aggforce.npz",
    ]

    if not args.skip_plots:
        ensure_dir(args.plots_dir)

    displacement, _ = space.free()

    all_bonds = []
    all_angles = []
    all_dihedrals = []
    all_angle_central_species = []

    all_rep_dists = []
    inferred_n_species = 0

    for path in dataset_paths:
        logger.info(f"Loading: {path}")
        data = np.load(path, allow_pickle=True)

        R = data["R"].astype(np.float32)       # (T,N,3)
        resid = data["resid"]                  # (T,N) or (N,)
        mask = data["mask"] if "mask" in data else np.ones(R.shape[:2], dtype=np.float32)
        species = data["species"] if "species" in data else None

        if mask.ndim == 1:
            mask = np.broadcast_to(mask[None, :], (R.shape[0], R.shape[1]))
        mask = np.asarray(mask, dtype=np.float32)

        if species is not None:
            species = np.asarray(species)
            if species.ndim == 1:
                species = np.broadcast_to(species[None, :], (R.shape[0], R.shape[1]))
            inferred_n_species = max(inferred_n_species, int(np.max(species)) + 1)

        Tframes, N, _ = R.shape
        bonds, angles, dihedrals = build_bonds_angles_dihedrals(resid)

        R_jax = jnp.asarray(R)

        bond_all = jax.vmap(bond_distances_single_frame, in_axes=(0, None, None))(R_jax, bonds, displacement)
        ang_all  = jax.vmap(angles_single_frame, in_axes=(0, None, None))(R_jax, angles, displacement)
        dih_all  = jax.vmap(dihedral_angles_single_frame, in_axes=(0, None, None))(R_jax, dihedrals, displacement)

        # Per-frame tuple validity masks to exclude padded atoms.
        bond_valid = (mask[:, bonds[:, 0]] * mask[:, bonds[:, 1]]) > 0
        angle_valid = (mask[:, angles[:, 0]] * mask[:, angles[:, 1]] * mask[:, angles[:, 2]]) > 0
        if dihedrals.shape[0] > 0:
            dih_valid = (
                mask[:, dihedrals[:, 0]] * mask[:, dihedrals[:, 1]] *
                mask[:, dihedrals[:, 2]] * mask[:, dihedrals[:, 3]]
            ) > 0
        else:
            dih_valid = np.zeros((R.shape[0], 0), dtype=bool)

        d_raw = np.asarray(bond_all)[bond_valid]
        a_raw = np.asarray(ang_all)[angle_valid]
        phi_raw = np.asarray(dih_all)[dih_valid]

        d_fin = np.isfinite(d_raw)
        a_fin = np.isfinite(a_raw)
        phi_fin = np.isfinite(phi_raw)

        d_np = d_raw[d_fin]
        a_np = a_raw[a_fin]
        phi_np = phi_raw[phi_fin]

        all_bonds.append(d_np)
        all_angles.append(a_np)
        all_dihedrals.append(phi_np)

        if species is not None:
            # Central atom species for each valid angle sample.
            central_sp = species[:, angles[:, 1]][angle_valid]
            central_sp = central_sp[a_fin]
            all_angle_central_species.append(np.asarray(central_sp, dtype=np.int32))

        order = order_from_resid(resid)
        if N > 6:
            ii = order[:-6]
            jj = order[6:]
            sample_frames = np.linspace(0, Tframes - 1, num=min(Tframes, 200), dtype=int)
            Rsub = R[sample_frames]
            Msub = mask[sample_frames]
            rep_valid = (Msub[:, ii] * Msub[:, jj]) > 0
            dr = Rsub[:, ii, :] - Rsub[:, jj, :]
            rep_d = np.linalg.norm(dr, axis=-1)[rep_valid]
            rep_d = rep_d[np.isfinite(rep_d)]
            all_rep_dists.append(rep_d)

    all_bonds = np.concatenate(all_bonds, axis=0)
    all_angles = np.concatenate(all_angles, axis=0)
    all_dihedrals = np.concatenate(all_dihedrals, axis=0)
    all_rep_dists = np.concatenate(all_rep_dists, axis=0) if len(all_rep_dists) else None
    all_angle_central_species = (
        np.concatenate(all_angle_central_species, axis=0)
        if len(all_angle_central_species) else None
    )

    n_species = int(args.n_species) if args.n_species is not None else int(max(inferred_n_species, 0))

    logger.info(f"Total samples: bonds={all_bonds.shape[0]}, angles={all_angles.shape[0]}, dihedrals={all_dihedrals.shape[0]}")

    # --- Raw histogram plots ---
    if not args.skip_plots:
        plt.figure()
        plt.hist(all_bonds, bins=100)
        plt.xlabel("Bond distance r (Å)")
        plt.ylabel("Count")
        plt.title("Bond distance histogram (all datasets)")
        savefig(Path(args.plots_dir) / "bond_distance_hist.png")

        plt.figure()
        plt.hist(all_angles, bins=100)
        plt.xlabel("Angle θ (rad)")
        plt.ylabel("Count")
        plt.title("Angle histogram (all datasets)")
        savefig(Path(args.plots_dir) / "angle_hist.png")

        plt.figure()
        plt.hist(all_dihedrals, bins=120, range=(-np.pi, np.pi))
        plt.xlabel("Dihedral φ (rad)")
        plt.ylabel("Count")
        plt.title("Dihedral histogram (all datasets)")
        savefig(Path(args.plots_dir) / "dihedral_hist.png")

        if all_rep_dists is not None and all_rep_dists.size > 0:
            plt.figure()
            plt.hist(all_rep_dists, bins=120)
            plt.xlabel("Rep pair distance r (Å) (i,i+6 sampled)")
            plt.ylabel("Count")
            plt.title("Repulsion-pair distance histogram (sampled)")
            savefig(Path(args.plots_dir) / "repulsion_pair_distance_hist.png")

    # --- Bond fit ---
    r0, kr = fit_bond_harmonic(all_bonds, T=args.T, kB=args.kB)
    logger.info(f"Bond fit: r0={r0:.5f} Å, kr={kr:.5f} kcal/mol/Å^2")

    if not args.skip_plots:
        r_grid = np.linspace(max(1e-3, np.min(all_bonds)), np.max(all_bonds), 2000)
        U_bond = 0.5 * kr * (r_grid - r0) ** 2
        U_bond -= U_bond.min()
        beta = 1.0 / (args.kB * args.T)
        pdf_bond = np.exp(-beta * U_bond)
        pdf_bond /= np.trapz(pdf_bond, r_grid)

        hist_r, edges_r = np.histogram(all_bonds, bins=120, density=True)
        cent_r = 0.5 * (edges_r[:-1] + edges_r[1:])

        plt.figure()
        plt.plot(cent_r, hist_r, label="Empirical P(r)", linewidth=2)
        plt.plot(r_grid, pdf_bond, label="Prior-implied (Gaussian sanity)", linewidth=2)
        plt.xlabel("r (Å)")
        plt.ylabel("density")
        plt.title("Bond prior sanity check")
        plt.legend()
        savefig(Path(args.plots_dir) / "bond_prior_implied_P.png")

    # --- Angle fit ---
    a_coeff, b_coeff, centers_th, U_bins_th, mask_th, edges_th, counts_th = fit_fourier_angles_stable(
        all_angles,
        n_terms=args.angle_terms,
        nbins=args.angle_bins,
        T=args.T,
        kB=args.kB,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        min_count=args.min_count,
        pseudocount=args.pseudocount,
        ridge=args.ridge,
    )

    logger.info(f"Angle Fourier fit: {args.angle_terms} terms")
    logger.debug(f"  a: {a_coeff}")
    logger.debug(f"  b: {b_coeff}")

    if not args.skip_plots:
        theta_grid = np.linspace(1.0, 2.8, 2000)
        U_fit_th = U_fourier(theta_grid, a_coeff, b_coeff)
        U_fit_th -= np.mean(U_fit_th)

        pdf_th_fit = prior_pdf_from_U_theta(theta_grid, U_fit_th, T=args.T, kB=args.kB)

        # Empirical angle density
        hist_th, edges = np.histogram(all_angles, bins=180, range=(args.theta_min, args.theta_max), density=True)
        cent_th = 0.5 * (edges[:-1] + edges[1:])

        # Plot PMF bins and fitted U(theta)
        plt.figure()
        plt.plot(theta_grid, U_fit_th, label="Fitted U(θ) Fourier")
        plt.scatter(centers_th[mask_th], U_bins_th[mask_th], s=10, alpha=0.6, label="PMF bins used")
        plt.xlabel("θ (rad)")
        plt.ylabel("U (kcal/mol) up to constant")
        plt.title("Angle PMF + Fourier fit")
        plt.legend()
        savefig(Path(args.plots_dir) / "angle_pmf_fit.png")

        # Plot implied angle density
        plt.figure()
        plt.plot(cent_th, hist_th, label="Empirical P(θ)", linewidth=2)
        plt.plot(theta_grid, pdf_th_fit, label="Prior-implied P(θ) (Fourier)", linewidth=2)
        plt.xlabel("θ (rad)")
        plt.ylabel("density")
        plt.title("Angle prior implied distribution")
        plt.legend()
        savefig(Path(args.plots_dir) / "angle_prior_implied_P.png")

    # --- Dihedral fit ---
    (k1, k2), (g1, g2), centers_phi, U_bins_phi, mask_phi, edges_phi, counts_phi, coeff_phi = fit_dihedral_fourier_2term(
        all_dihedrals,
        nbins=args.dih_bins,
        T=args.T,
        kB=args.kB,
        min_count=args.min_count,
        pseudocount=args.pseudocount,
        ridge=args.ridge,
    )

    logger.info(f"Dihedral fit (2-term): k=[{k1:.4f}, {k2:.4f}], gamma=[{g1:.4f}, {g2:.4f}]")
    logger.debug(f"  raw coeff [A1,B1,A2,B2]: {coeff_phi}")

    if not args.skip_plots:
        phi_grid = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
        A1, B1, A2, B2 = coeff_phi
        U_fit_phi = A1 * np.cos(phi_grid) + B1 * np.sin(phi_grid) + A2 * np.cos(2 * phi_grid) + B2 * np.sin(2 * phi_grid)
        U_fit_phi -= np.mean(U_fit_phi)

        pdf_phi_fit = prior_pdf_from_U_phi(phi_grid, U_fit_phi, T=args.T, kB=args.kB)

        # Empirical dihedral density
        hist_phi, edges = np.histogram(all_dihedrals, bins=180, range=(-np.pi, np.pi), density=True)
        cent_phi = 0.5 * (edges[:-1] + edges[1:])

        # Plot dihedral PMF and fit
        plt.figure()
        plt.plot(phi_grid, U_fit_phi, label="Fitted U(φ) (2-term)")
        plt.scatter(centers_phi[mask_phi], U_bins_phi[mask_phi], s=10, alpha=0.6, label="PMF bins used")
        plt.xlabel("φ (rad)")
        plt.ylabel("U (kcal/mol) up to constant")
        plt.title("Dihedral PMF + fit")
        plt.legend()
        savefig(Path(args.plots_dir) / "dihedral_pmf_fit.png")

        # Plot implied dihedral density
        plt.figure()
        plt.plot(cent_phi, hist_phi, label="Empirical P(φ)", linewidth=2)
        plt.plot(phi_grid, pdf_phi_fit, label="Prior-implied P(φ)", linewidth=2)
        plt.xlabel("φ (rad)")
        plt.ylabel("density")
        plt.title("Dihedral prior implied distribution")
        plt.legend()
        savefig(Path(args.plots_dir) / "dihedral_prior_implied_P.png")

    # --- Optional add-on: spline fits (keeps legacy parametric outputs) ---
    if args.spline:
        logger.info("Spline fitting enabled: KDE -> BI -> CubicSpline")

        theta_min_spline = max(0.5, float(args.theta_min))
        theta_max_spline = min(float(args.theta_max), float(np.pi - 0.01))

        # Global splines
        bond_knots, bond_coeffs, grid_bond, dens_bond, U_bond_kde, _ = fit_bond_spline(
            all_bonds, T=args.T, kB=args.kB, n_grid=args.spline_grid_points, bw_factor=args.kde_bandwidth_factor
        )
        angle_knots, angle_coeffs, grid_ang, dens_ang, U_ang_kde, _ = fit_angle_spline(
            all_angles, T=args.T, kB=args.kB, n_grid=args.spline_grid_points, bw_factor=args.kde_bandwidth_factor,
            theta_min=theta_min_spline, theta_max=theta_max_spline
        )
        dih_knots, dih_coeffs, grid_dih, dens_dih, U_dih_kde, _ = fit_dihedral_spline(
            all_dihedrals, T=args.T, kB=args.kB, n_grid=args.spline_grid_points, bw_factor=args.kde_bandwidth_factor
        )

        residue_specific_used = False
        angle_type_knots = None
        angle_type_coeffs = None
        angle_type_mask = None
        angle_type_counts = None
        angle_n_types = 0

        # Residue-specific angle splines (optional)
        if args.residue_specific_angles:
            if all_angle_central_species is None:
                logger.warning(
                    "residue_specific_angles requested, but no species arrays found in input datasets; "
                    "falling back to global angle spline."
                )
            else:
                if n_species <= 0:
                    n_species = int(np.max(all_angle_central_species)) + 1
                angle_n_types = int(n_species)
                (
                    _gk, _gc,
                    angle_type_knots,
                    angle_type_coeffs,
                    angle_type_mask,
                    angle_type_counts,
                ) = fit_angle_spline_by_type(
                    all_angles=all_angles,
                    all_central_species=all_angle_central_species,
                    n_types=angle_n_types,
                    T=args.T,
                    kB=args.kB,
                    n_grid=args.spline_grid_points,
                    bw_factor=args.kde_bandwidth_factor,
                    min_samples=args.angle_min_samples,
                    theta_min=theta_min_spline,
                    theta_max=theta_max_spline,
                )
                residue_specific_used = True
                n_own = int(np.sum(angle_type_mask))
                logger.info(
                    "Residue-specific angle splines: %d/%d types have dedicated splines (min_samples=%d)",
                    n_own, angle_n_types, int(args.angle_min_samples)
                )

        # Save spline payload for model runtime use.
        spline_out = Path(args.spline_out)
        spline_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bond_knots": np.asarray(bond_knots, dtype=np.float32),
            "bond_coeffs": np.asarray(bond_coeffs, dtype=np.float32),
            "angle_knots": np.asarray(angle_knots, dtype=np.float32),
            "angle_coeffs": np.asarray(angle_coeffs, dtype=np.float32),
            "dih_knots": np.asarray(dih_knots, dtype=np.float32),
            "dih_coeffs": np.asarray(dih_coeffs, dtype=np.float32),
            "temperature": np.asarray(float(args.T), dtype=np.float32),
            "kB": np.asarray(float(args.kB), dtype=np.float32),
            "grid_points": np.asarray(int(args.spline_grid_points), dtype=np.int32),
            "kde_bandwidth_factor": np.asarray(float(args.kde_bandwidth_factor), dtype=np.float32),
            "residue_specific_angles": np.asarray(bool(residue_specific_used)),
        }
        if residue_specific_used:
            payload.update({
                "angle_n_types": np.asarray(int(angle_n_types), dtype=np.int32),
                "angle_type_knots": np.asarray(angle_type_knots, dtype=np.float32),
                "angle_type_coeffs": np.asarray(angle_type_coeffs, dtype=np.float32),
                "angle_type_mask": np.asarray(angle_type_mask, dtype=np.int32),
                "angle_type_counts": np.asarray(angle_type_counts, dtype=np.int64),
            })
        np.savez(str(spline_out), **payload)
        logger.info(f"Wrote spline priors NPZ: {spline_out}")

        if not args.skip_plots:
            beta = 1.0 / (args.kB * args.T)

            # PMF overlay: histogram PMF vs KDE PMF vs exported spline PMF
            grid_eval_bond = np.linspace(grid_bond[0], grid_bond[-1], 1200)
            U_bond_spline = eval_piecewise_spline_numpy(grid_eval_bond, bond_knots, bond_coeffs)
            hist_bond, edges_bond = np.histogram(all_bonds, bins=120, range=(grid_bond[0], grid_bond[-1]), density=True)
            cent_bond = 0.5 * (edges_bond[:-1] + edges_bond[1:])
            U_bond_hist = _safe_probability_to_pmf(hist_bond, T=args.T, kB=args.kB)
            plt.figure()
            plt.plot(cent_bond, U_bond_hist, color="0.6", linewidth=1.2, label="Histogram PMF")
            plt.plot(grid_bond, U_bond_kde, color="C0", linewidth=2.0, label="KDE PMF")
            plt.plot(grid_eval_bond, U_bond_spline, "r--", linewidth=1.6, label="Spline PMF")
            plt.xlabel("r (Å)")
            plt.ylabel("U(r) [kcal/mol, shifted]")
            plt.title("Bond PMF: histogram vs KDE vs spline")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_bond_pmf_overlay.png")

            grid_eval_ang = np.linspace(grid_ang[0], grid_ang[-1], 1200)
            U_ang_spline = eval_piecewise_spline_numpy(grid_eval_ang, angle_knots, angle_coeffs)
            hist_ang, edges_ang = np.histogram(all_angles, bins=120, range=(grid_ang[0], grid_ang[-1]), density=True)
            cent_ang = 0.5 * (edges_ang[:-1] + edges_ang[1:])
            U_ang_hist = _safe_probability_to_pmf(hist_ang / np.clip(np.sin(cent_ang), 1e-8, None), T=args.T, kB=args.kB)
            plt.figure()
            plt.plot(cent_ang, U_ang_hist, color="0.6", linewidth=1.2, label="Histogram PMF (Jacobian corrected)")
            plt.plot(grid_ang, U_ang_kde, color="C0", linewidth=2.0, label="KDE PMF")
            plt.plot(grid_eval_ang, U_ang_spline, "r--", linewidth=1.6, label="Spline PMF")
            plt.xlabel("θ (rad)")
            plt.ylabel("U(θ) [kcal/mol, shifted]")
            plt.title("Angle PMF: histogram vs KDE vs spline")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_angle_pmf_overlay.png")

            grid_eval_dih = np.linspace(-np.pi, np.pi, 1400, endpoint=False)
            U_dih_spline = eval_piecewise_spline_numpy(grid_eval_dih, dih_knots, dih_coeffs, periodic=True)
            hist_dih, edges_dih = np.histogram(all_dihedrals, bins=120, range=(-np.pi, np.pi), density=True)
            cent_dih = 0.5 * (edges_dih[:-1] + edges_dih[1:])
            U_dih_hist = _safe_probability_to_pmf(hist_dih, T=args.T, kB=args.kB)
            plt.figure()
            plt.plot(cent_dih, U_dih_hist, color="0.6", linewidth=1.2, label="Histogram PMF")
            plt.plot(grid_dih, U_dih_kde, color="C0", linewidth=2.0, label="KDE PMF")
            plt.plot(grid_eval_dih, U_dih_spline, "r--", linewidth=1.6, label="Spline PMF")
            plt.xlabel("φ (rad)")
            plt.ylabel("U(φ) [kcal/mol, shifted]")
            plt.title("Dihedral PMF: histogram vs KDE vs spline")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_dihedral_pmf_overlay.png")

            # Implied distributions
            p_bond = np.exp(-beta * (U_bond_spline - np.min(U_bond_spline)))
            p_bond /= np.trapz(p_bond, grid_eval_bond)
            plt.figure()
            plt.plot(cent_bond, hist_bond, color="0.6", linewidth=1.2, label="Empirical P(r)")
            plt.plot(grid_eval_bond, p_bond, "r-", linewidth=2.0, label="Spline-implied P(r)")
            plt.xlabel("r (Å)")
            plt.ylabel("density")
            plt.title("Bond implied distribution")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_bond_implied_distribution.png")

            p_ang = np.sin(grid_eval_ang) * np.exp(-beta * (U_ang_spline - np.min(U_ang_spline)))
            p_ang /= np.trapz(p_ang, grid_eval_ang)
            plt.figure()
            plt.plot(cent_ang, hist_ang, color="0.6", linewidth=1.2, label="Empirical P(θ)")
            plt.plot(grid_eval_ang, p_ang, "r-", linewidth=2.0, label="Spline-implied P(θ)")
            plt.xlabel("θ (rad)")
            plt.ylabel("density")
            plt.title("Angle implied distribution (includes sinθ Jacobian)")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_angle_implied_distribution.png")

            p_dih = np.exp(-beta * (U_dih_spline - np.min(U_dih_spline)))
            p_dih /= np.trapz(p_dih, grid_eval_dih)
            plt.figure()
            plt.plot(cent_dih, hist_dih, color="0.6", linewidth=1.2, label="Empirical P(φ)")
            plt.plot(grid_eval_dih, p_dih, "r-", linewidth=2.0, label="Spline-implied P(φ)")
            plt.xlabel("φ (rad)")
            plt.ylabel("density")
            plt.title("Dihedral implied distribution")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_dihedral_implied_distribution.png")

            # Derivative sanity checks (numeric vs analytic from coeffs)
            dU_num_bond = np.gradient(U_bond_spline, grid_eval_bond)
            dU_ana_bond = eval_piecewise_spline_derivative_numpy(grid_eval_bond, bond_knots, bond_coeffs)
            plt.figure()
            plt.plot(grid_eval_bond, dU_num_bond, label="Numeric dU/dr")
            plt.plot(grid_eval_bond, dU_ana_bond, "--", label="Analytic dU/dr")
            plt.xlabel("r (Å)")
            plt.ylabel("dU/dr")
            plt.title("Bond spline derivative sanity check")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_bond_force_sanity.png")

            dU_num_ang = np.gradient(U_ang_spline, grid_eval_ang)
            dU_ana_ang = eval_piecewise_spline_derivative_numpy(grid_eval_ang, angle_knots, angle_coeffs)
            plt.figure()
            plt.plot(grid_eval_ang, dU_num_ang, label="Numeric dU/dθ")
            plt.plot(grid_eval_ang, dU_ana_ang, "--", label="Analytic dU/dθ")
            plt.xlabel("θ (rad)")
            plt.ylabel("dU/dθ")
            plt.title("Angle spline derivative sanity check")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_angle_force_sanity.png")

            dU_num_dih = np.gradient(U_dih_spline, grid_eval_dih)
            dU_ana_dih = eval_piecewise_spline_derivative_numpy(grid_eval_dih, dih_knots, dih_coeffs, periodic=True)
            plt.figure()
            plt.plot(grid_eval_dih, dU_num_dih, label="Numeric dU/dφ")
            plt.plot(grid_eval_dih, dU_ana_dih, "--", label="Analytic dU/dφ")
            plt.xlabel("φ (rad)")
            plt.ylabel("dU/dφ")
            plt.title("Dihedral spline derivative sanity check")
            plt.legend()
            savefig(Path(args.plots_dir) / "spline_dihedral_force_sanity.png")

            if residue_specific_used:
                ncols = 4
                nrows = int(np.ceil(angle_n_types / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
                for s in range(angle_n_types):
                    ax = axes[s // ncols, s % ncols]
                    U_global_eval = eval_piecewise_spline_numpy(grid_eval_ang, angle_knots, angle_coeffs)
                    U_type_eval = eval_piecewise_spline_numpy(
                        grid_eval_ang, angle_type_knots[s], angle_type_coeffs[s]
                    )
                    ax.plot(grid_eval_ang, U_global_eval, color="0.6", linewidth=1.2, label="global")
                    if int(angle_type_mask[s]) == 1:
                        ax.plot(grid_eval_ang, U_type_eval, color="C0", linewidth=1.4, label="typed")
                        title = f"type {s} (n={int(angle_type_counts[s])})"
                    else:
                        ax.plot(grid_eval_ang, U_type_eval, color="C3", linestyle="--", linewidth=1.2, label="fallback")
                        title = f"type {s} fallback (n={int(angle_type_counts[s])})"
                    ax.set_title(title, fontsize=9)
                    ax.set_xlim(grid_eval_ang[0], grid_eval_ang[-1])
                    ax.tick_params(labelsize=8)
                for s in range(angle_n_types, nrows * ncols):
                    axes[s // ncols, s % ncols].axis("off")
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right")
                fig.suptitle("Residue-specific angle spline diagnostics", y=1.01)
                fig.tight_layout()
                fig.savefig(Path(args.plots_dir) / "spline_angle_by_type_diagnostics.png", dpi=220, bbox_inches="tight")
                plt.close(fig)

    # --- Repulsion plots (verification only, no fit) ---
    if not args.skip_plots:
        if all_rep_dists is not None and all_rep_dists.size > 0:
            r = np.linspace(max(1e-3, np.min(all_rep_dists)), np.percentile(all_rep_dists, 99.5), 2000)
            rep = args.epsilon * (args.sigma / np.maximum(r, 1e-6)) ** args.rep_power

            plt.figure()
            plt.plot(r, rep)
            plt.xlabel("r (Å)")
            plt.ylabel(f"E_rep = eps*(sigma/r)^{args.rep_power} (kcal/mol)")
            plt.title("Repulsion energy curve (as configured)")
            savefig(Path(args.plots_dir) / "repulsion_energy_curve.png")

            # Show implied weights (just for intuition)
            beta = 1.0 / (args.kB * args.T)
            w = np.exp(-beta * (args.epsilon * (args.sigma / np.maximum(all_rep_dists, 1e-6)) ** args.rep_power))
            plt.figure()
            plt.scatter(all_rep_dists[:5000], w[:5000], s=3, alpha=0.3)
            plt.xlabel("r (Å) (sampled rep pairs)")
            plt.ylabel("exp(-beta E_rep)")
            plt.title("Repulsion weight vs distance (intuition)")
            savefig(Path(args.plots_dir) / "repulsion_weight_vs_distance.png")

    # --- Write YAML ---
    priors = {
        "r0": float(r0),
        "kr": float(kr),
        "a": [float(x) for x in np.asarray(a_coeff).ravel()],
        "b": [float(x) for x in np.asarray(b_coeff).ravel()],
        # Keep legacy keys and add config-native keys for compatibility.
        "k_dihedral": [float(k1), float(k2)],
        "gamma_dihedral": [float(g1), float(g2)],
        "k_dih": [float(k1), float(k2)],
        "gamma_dih": [float(g1), float(g2)],
        "epsilon": float(args.epsilon),
        "sigma": float(args.sigma),
    }

    out = {
        "model": {
            "priors": priors
        },
        "meta": {
            "datasets_used": dataset_paths,
            "T": float(args.T),
            "kB": float(args.kB),
            "angle_terms": int(args.angle_terms),
            "angle_bins": int(args.angle_bins),
            "dih_bins": int(args.dih_bins),
            "min_count": int(args.min_count),
            "ridge": float(args.ridge),
            "pseudocount": float(args.pseudocount),
            "spline_enabled": bool(args.spline),
            "spline_out": str(args.spline_out) if args.spline else None,
            "residue_specific_angles": bool(args.residue_specific_angles),
            "angle_min_samples": int(args.angle_min_samples),
            "kde_bandwidth_factor": float(args.kde_bandwidth_factor),
            "spline_grid_points": int(args.spline_grid_points),
        }
    }

    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    logger.info("=== Done ===")
    logger.info(f"Wrote YAML: {args.out_yaml}")
    if args.spline:
        logger.info(f"Wrote spline NPZ: {args.spline_out}")
    if not args.skip_plots:
        logger.info(f"Plots saved in: {args.plots_dir}")
    logger.info(f"YAML priors keys: {list(priors.keys())}")


if __name__ == "__main__":
    main()
