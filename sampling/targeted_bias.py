#!/usr/bin/env python3
"""Emit PLUMED input for the ddF-sensitivity TARGETED biases (channel 1 / channel 2).

This module builds only the two bias FIELDS. Every piece of PLUMED rendering is delegated to
`sampling.plumed_native`, which already does all of it correctly:

  tica_cv_block       DISTANCE + COMBINE, with bead -> AA atom translation through the mapping
                      (hand-rolling this produced ATOMS=1,2 instead of ATOMS=5,7 -- a run that
                      completes normally on the wrong atoms)
  padded_bounds       padding sized by PHYSICS, sigma = sqrt(kT/k), not a fixed cell count
  walls_block         the KAPPA = 0.5 * wall_k convention (PLUMED has no 1/2)
  external_block      the EXTERNAL action
  write_grid_from_fn  tabulates any `z -> (V, grad V)` CALLABLE

That last one is the important one. Tabulating a field and then finite-differencing it with
`np.gradient` gives PLUMED derivative columns that disagree with its own spline of the values,
and the disagreement is worst exactly where the field bends. Supplying an analytic callable
makes values and derivatives consistent BY CONSTRUCTION.

Which is why the field is handed over as a Nadaraya-Watson kernel interpolant rather than as a
grid: it is C^infinity, its gradient is exact for the function actually being tabulated, and it
requires no basis choice, no least-squares fit and no residual (unlike the earlier quadratic
committor-logit fit, which left an RMS residual of 2.32 in logit units).

Biases, derived in KB DESIGN/TARGETED_SAMPLING_BIASES.md:

  channel 1   V1(z) = kT ln P_i  inside basin i        (a CONSTANT per basin)
  channel 2   V2(z) = -2 kT ln(|grad q(z)|/max + eps)  (a well on the transition tube)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .plumed_native import (UNITS_HEADER, external_block, padded_bounds, tica_cv_block,
                            walls_block, write_grid_from_fn)
from .mapping import get_mapping

KT = 0.5921868690749673          # kcal/mol at 298 K


# ------------------------------------------------------------------------------ bias fields
def bias_channel1(labels, rho, keep, kt=KT):
    """V1 = kT ln P_i inside basin i, 0 elsewhere.

    Equalises basin occupancy and leaves the shape INSIDE each basin untouched. Needs only the
    integrated basin populations -- two scalars -- never the pointwise density.
    """
    V = np.zeros_like(rho, dtype=np.float64)
    pops = {}
    for i in keep:
        m = labels == i
        P = float(rho[m].sum())
        pops[i] = P
        V[m] = kt * np.log(max(P, 1e-300))
    return V, pops


def bias_channel2(q, eps=0.05, kt=KT):
    """V2 = -2 kT ln(g + eps), g = |grad q| / max|grad q|, zeroed at its minimum.

    eps regularises |grad q| -> 0 deep in the basins AND sets the effective cap: at g=0 the
    bias is -2kT ln(eps) ~ 6 kT for eps=0.05, so no separate clip is needed. A constant
    rescaling of |grad q| only shifts V2 by a constant, so grid spacing is irrelevant here.
    """
    def d(u, ax):
        g = np.empty_like(u)
        if ax == 0:
            g[1:-1] = (u[2:] - u[:-2]) / 2.0
            g[0] = u[1] - u[0]; g[-1] = u[-1] - u[-2]
        else:
            g[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
            g[:, 0] = u[:, 1] - u[:, 0]; g[:, -1] = u[:, -1] - u[:, -2]
        return g
    gm = np.sqrt(d(q, 0) ** 2 + d(q, 1) ** 2)
    g = gm / max(gm.max(), 1e-30)
    V = -2.0 * kt * np.log(g + eps)
    return V - V.min()


def field_callable(cx, cy, V, sigma):
    """Nadaraya-Watson kernel interpolant of a gridded field, with an ANALYTIC gradient.

    V(z) = sum_m w_m V_m / sum_m w_m,  w_m = exp(-|z - c_m|^2 / 2 sigma^2)

    so, differentiating the quotient,

    grad V(z) = sum_m w_m (V_m - V(z)) (-(z - c_m)/sigma^2) / sum_m w_m

    C^infinity, and the returned gradient is the exact derivative of the returned value -- the
    property `write_grid_from_fn` needs and that np.gradient over a pre-blurred array does not
    have. sigma also REPLACES the old pre-smoothing pass: the smoothing is now part of the
    definition of the bias rather than a separate step applied before differentiating.
    """
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    C = np.stack([X.ravel(), Y.ravel()], axis=-1)
    Vc = np.asarray(V, dtype=float).ravel()
    inv = 1.0 / (2.0 * sigma ** 2)

    def fn(z):
        z = np.asarray(z, dtype=float)
        dz = z[:, None, :] - C[None, :, :]
        e = -inv * np.einsum("nmk,nmk->nm", dz, dz)
        e -= e.max(axis=1, keepdims=True)          # stabilise before exp
        w = np.exp(e)
        S = w.sum(axis=1)
        Vz = (w @ Vc) / S
        coef = w * (Vc[None, :] - Vz[:, None])
        grad = np.einsum("nm,nmk->nk", coef, -dz / sigma ** 2) / S[:, None]
        return Vz, grad
    return fn


def _shim(pairs, mean, coefficients, cx, cy, wall_frac=0.05, kt=KT):
    """Minimal stand-in exposing the attributes plumed_native's helpers read.

    Wall stiffness is chosen so the thermal excursion sqrt(kT/k) is `wall_frac` of each axis
    range; padded_bounds then extends the grid 4 of those sigmas beyond, so EXTERNAL cannot be
    walked off before the walls turn the CV around.
    """
    lo = np.array([cx[0], cy[0]], dtype=float)
    hi = np.array([cx[-1], cy[-1]], dtype=float)
    wall_k = kt / (wall_frac * (hi - lo)) ** 2
    return SimpleNamespace(
        projection=SimpleNamespace(pairs=np.asarray(pairs), mean=np.asarray(mean),
                                   coefficients=np.asarray(coefficients)),
        bounds=np.stack([lo, hi], axis=1),
        wall_k_kcal_mol=wall_k,
        kbt_kcal_mol=kt,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sens", required=True, help="npz from df_sensitivity_map_tica.py")
    ap.add_argument("--channel", choices=["1", "2"], required=True)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=1.5,
                    help="kernel width in SOURCE CELLS (part of the bias definition)")
    ap.add_argument("--min-pop", type=float, default=0.01,
                    help="basins below this population share are NOT biased")
    ap.add_argument("--n-points", type=int, default=401,
                    help="grid points per axis; plumed_native measured 401 -> 2%% force error")
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--stride", type=int, default=500)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(a.sens, allow_pickle=True)
    ex, ey, rho, q, lab = z["ex"], z["ey"], z["rho"], z["q"], z["labels"]
    cx = 0.5 * (ex[1:] + ex[:-1]); cy = 0.5 * (ey[1:] + ey[:-1])

    allb = [int(v) for v in np.unique(lab) if v >= 0]
    pops_all = {i: float(rho[lab == i].sum()) for i in allb}
    keep = [i for i in allb if pops_all[i] >= a.min_pop]
    print(f"{len(allb)} watershed minima; {len(keep)} retained at population >= {a.min_pop} "
          f"(dropped {len(allb) - len(keep)} noise minima)")

    if a.channel == "1":
        V, pops = bias_channel1(lab, rho, keep)
        print("channel 1: V1 = kT ln P_i  (constant per basin)")
        for i, P in sorted(pops.items(), key=lambda kv: -kv[1]):
            print(f"  basin {i}: P = {P:.4f}   V1 = {KT * np.log(P):+.3f} kcal/mol")
        print(f"  after biasing all {len(pops)} basins carry equal weight -> "
              f"{100 / len(pops):.1f}% each")
    else:
        V = bias_channel2(q, eps=a.eps)
        print(f"channel 2: V2 = -2kT ln(|grad q|/max + {a.eps})")
        print(f"  range {V.min():.3f} .. {V.max():.3f} kcal/mol ({V.max() / KT:.2f} kT)")

    sigma = a.smooth * float(cx[1] - cx[0])
    fn = field_callable(cx, cy, V, sigma)
    shim = _shim(z["pairs"], z["tica_mean"], z["tica_coefficients"], cx, cy)
    mapping = get_mapping(a.mapping)

    lo_p, hi_p = padded_bounds(shim)
    grid = a.outdir / "bias.grid"
    gx, gy = write_grid_from_fn(fn, grid, lo_p, hi_p,
                                n_points=(a.n_points, a.n_points))
    print(f"\nkernel width sigma = {sigma:.4f} TIC units ({a.smooth} source cells)")
    print(f"data bounds  tic1 [{cx[0]:.3f}, {cx[-1]:.3f}]  tic2 [{cy[0]:.3f}, {cy[-1]:.3f}]")
    print(f"walls at     tic1 [{shim.bounds[0,0]:.3f}, {shim.bounds[0,1]:.3f}]  "
          f"tic2 [{shim.bounds[1,0]:.3f}, {shim.bounds[1,1]:.3f}]  "
          f"kappa={0.5*shim.wall_k_kcal_mol}")
    print(f"grid spans   tic1 [{gx[0]:.3f}, {gx[-1]:.3f}]  tic2 [{gy[0]:.3f}, {gy[-1]:.3f}]"
          f"  ({len(gx)}x{len(gy)} points)")

    plumed = (UNITS_HEADER
              + f"WHOLEMOLECULES ENTITY0={mapping.plumed_atom_selection()}\n"
              + tica_cv_block(shim, mapping)
              + external_block(grid.name)
              + walls_block(shim)
              + f"PRINT ARG=tic1,tic2,treg.bias,twall_lo.bias,twall_hi.bias"
                f" FILE=colvar.dat STRIDE={a.stride}\n")
    (a.outdir / "plumed.dat").write_text(plumed)
    np.savez_compressed(a.outdir / "bias_field.npz", cx=cx, cy=cy, V=V, rho=rho, q=q,
                        labels=lab, channel=a.channel, sigma=sigma, gx=gx, gy=gy)
    print(f"\nwrote {a.outdir/'plumed.dat'}\n      {grid}\n      {a.outdir/'bias_field.npz'}")


if __name__ == "__main__":
    main()
