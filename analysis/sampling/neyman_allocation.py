#!/usr/bin/env python3
"""Neyman allocation of labels over latent cells (MD-less, trajectory-only).

Stratified-sampling optimum: for a fixed label budget n, the allocation that
minimizes the variance of a stratified mean estimator is n_h ∝ N_h σ_h, where
N_h = reference frames in cell h and σ_h = within-cell standard deviation of
the quantity being estimated. Here that quantity is the instantaneous mapped
force (per bead/xyz component) read directly from the dataset NPZ — no new
simulation. Comparing f*_h to the current fraction N_h/N shows which cells are
under- or over-supplied, and the total-variance ratio quantifies what a
re-balanced dataset of the SAME size would buy.

Usage: python -m sampling.neyman_allocation --outdir <dir> [--n-frames 200001]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc
from analysis.sampling.diagnostics_common import load_latent


def _log(msg):
    print(msg, flush=True)


def neyman_fractions(N_h, sigma_h):
    """Optimal stratum fractions n_h/n ∝ N_h σ_h (zeros stay zero)."""
    w = N_h * sigma_h
    return np.where(w > 0, w / w.sum(), 0.0)


def stratified_variance(N_h, sigma_h, n_h):
    """Var of the stratified force-mean estimator for allocation n_h."""
    n_h = np.maximum(n_h, 1e-9)
    return float(np.sum((N_h * sigma_h) ** 2 / n_h))


def selfcheck():
    """2-stratum brute force: grid-searched optimal split == Neyman ratio."""
    N = np.array([1000.0, 100.0])
    s = np.array([1.0, 4.0])
    n_total = 200
    best_v, best_k = np.inf, None
    for k in range(0, n_total + 1):
        v = stratified_variance(N, s, np.array([k, n_total - k]))
        if v < best_v:
            best_v, best_k = v, k
    f = neyman_fractions(N, s)
    assert abs(best_k - f[0] * n_total) <= 1, (best_k, f)
    print(f"[selfcheck] brute-force argmin split {best_k} == "
          f"Neyman {f[0]*n_total:.1f}. OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    dc.add_latent_arguments(ap)
    ap.add_argument("--n-frames", type=int, default=200001)
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = dc.config_from_args(a)
    if a.selfcheck:
        selfcheck()

        a.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    raw = np.load(a.frames)
    R_all, F_all = np.asarray(raw["R"], np.float64), np.asarray(raw["F"], np.float64)
    idx = np.linspace(0, len(R_all) - 1, min(a.n_frames, len(R_all))).astype(int)
    R, F = R_all[idx], F_all[idx]
    Fc = F.reshape(len(F), -1)                       # (n, beads*3)

    lat = load_latent(a.frames, len(R), a.grid_lim, a.bins, config=cfg)
    inside, flat = lat["inside"], lat["flat"]
    ok = inside & np.isfinite(Fc).all(axis=1)
    bins = lat["bins"]
    n_cells = bins * bins
    _log(f"[load] {len(R)} frames ({ok.sum()} inside grid), {time.time()-t0:.0f}s")

    N_h = np.bincount(flat[ok], minlength=n_cells).astype(np.float64)
    cnt = np.bincount(flat[ok], minlength=n_cells)
    sum_ = np.zeros((n_cells, Fc.shape[1]))
    sq = np.zeros((n_cells, Fc.shape[1]))
    np.add.at(sum_, flat[ok], Fc[ok])
    np.add.at(sq, flat[ok], Fc[ok] ** 2)
    with np.errstate(invalid="ignore"):
        m_c = sum_ / np.maximum(cnt, 1)[:, None]
        v_c = sq / np.maximum(cnt, 1)[:, None] - m_c ** 2
    sigma_c = np.sqrt(np.nanmean(v_c, axis=1))       # RMS per-component sd
    valid = (cnt >= a.min_count) & np.isfinite(sigma_c) & (sigma_c > 0)

    f_cur = np.where(valid, N_h / max(N_h.sum(), 1), 0.0)
    f_ney = np.where(valid, neyman_fractions(N_h, sigma_c), 0.0)

    # headline: same-budget variance under current (proportional-to-N) vs Neyman
    n_tot = int(N_h.sum())
    n_cur = np.maximum(np.round(f_cur * n_tot), 1)
    n_opt = np.maximum(np.round(f_ney * n_tot), 1)
    V_cur = stratified_variance(N_h[valid], sigma_c[valid], n_cur[valid])
    V_ney = stratified_variance(N_h[valid], sigma_c[valid], n_opt[valid])

    ratio = np.where(valid & (f_cur > 0), np.log10((f_ney + 1e-12) / (f_cur + 1e-12)), np.nan)
    order = np.argsort(-np.abs(np.where(valid, ratio, 0.0)))[:15]
    centers = lat["centers"]
    _log(f"[headline] same-budget variance: uniform-by-frames {V_cur:.3f} -> "
         f"Neyman {V_ney:.3f} kcal^2/mol^2/comp  ({V_cur/V_ney:.2f}x reduction)")
    _log("[most misallocated cells: log10(f*_ney/f_cur)]")
    for i in order[:8]:
        ix, iy = divmod(int(i), bins)
        if not valid[i]:
            continue
        _log(f"  u=({centers[ix]:+.2f},{centers[iy]:+.2f}) N={int(cnt[i]):5d} "
             f"sigma={sigma_c[i]:6.2f} log-ratio={ratio[i]:+.2f}")

    reg = lat["region"][ok]
    basins = {}
    for b in ("beta", "alphaR", "alphaL", "other"):
        mb = reg == b
        if not mb.sum():
            continue
        cells = np.unique(flat[ok][mb])
        basins[b] = {"ref_pct": float(100.0 * mb.mean()),
                     "sigma_median": float(np.median(sigma_c[cells])),
                     "f_cur_pct": float(100 * f_cur[cells].sum()),
                     "f_ney_pct": float(100 * f_ney[cells].sum())}
        _log(f"[{b}] ref%={basins[b]['ref_pct']:.2f} sigma_med="
             f"{basins[b]['sigma_median']:.2f} f_cur={basins[b]['f_cur_pct']:.2f}% "
             f"f_ney={basins[b]['f_ney_pct']:.2f}%")

    summary = {"provenance": {"frames": a.frames, "n_frames": int(len(R)),
                              "bins": bins, "grid_lim": a.grid_lim,
                              "kT_kcal_mol": cfg.kT},
               "variance_uniform_by_frames": V_cur,
               "variance_neyman": V_ney,
               "variance_reduction_factor": V_cur / V_ney,
               "basins": basins,
               "misallocated_top": [dict(u1=float(centers[i // bins]),
                                         u2=float(centers[i % bins]),
                                         n=int(cnt[i]), sigma=float(sigma_c[i]),
                                         log10_ratio=float(ratio[i]))
                                    for i in order if valid[i]]}
    (a.outdir / "neyman.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(a.outdir / "neyman.npz",
                        count=N_h, sigma=sigma_c, f_cur=f_cur, f_ney=f_ney,
                        valid=valid, edges=lat["edges"])
    plot(a.outdir, lat["edges"], centers, bins, sigma_c, ratio, valid)
    _log(f"wrote {a.outdir}/neyman.json .npz .png")


def plot(outdir, edges, centers, bins, sigma_c, ratio, valid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    s = sigma_c.reshape(bins, bins)
    im0 = axes[0].pcolormesh(edges, edges, s.T, cmap="viridis")
    axes[0].set_title(r"within-cell force sd $\sigma_h$ (kcal/mol/$\AA$)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    r = np.where(valid, ratio, np.nan).reshape(bins, bins)
    vmax = np.nanpercentile(np.abs(r), 98) or 1.0
    im1 = axes[1].pcolormesh(edges, edges, r.T, cmap="RdBu_r",
                             norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    axes[1].set_title("$\\log_{10}(f^*_{\\rm Neyman}/f_{\\rm cur})$ "
                      "(red: needs MORE labels)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    m = valid.reshape(bins, bins)
    axes[2].pcolormesh(edges, edges, m.T, cmap="Greys")
    axes[2].set_title("cells with enough statistics")
    for ax in axes:
        ax.set_aspect("equal"), ax.set_xlabel("$u_1$"), ax.set_ylabel("$u_2$")
    fig.savefig(outdir / "neyman.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
