#!/usr/bin/env python3
"""ddF sensitivity map computed NATIVELY on the TICA grid (no Rama, no priors).

Same two channels as analysis/labels/df_sensitivity_map.py, but every step lives in the
space the bias actually acts in:

  * grid          : 2D TICA histogram (percentile edges), NOT periodic -> no-flux BC
  * basins        : discovered by watershed (steepest descent) on the smoothed TICA FES,
                    NOT taken from any hand-written phi/psi box
  * channel 1     : |p_A - p_B|, each normalised within its own basin
  * channel 2     : reactive flux rho|grad q|^2 from div(rho grad q)=0 with Neumann BC

Rama is used ONLY to label the discovered basins for human reading; it never enters the
computation. Removing it is what makes the method general to systems with no known 2 CVs.

Also recovers the affine map  Y = (X - mean) @ coefficients  by least squares and asserts
the residual is ~0, because that is exactly the form PLUMED reproduces with
DISTANCE + COMBINE, i.e. what lets the resulting grid be used as a bias.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping
from analysis.md.analyze_traj import build_features, fit_tica
from analysis.labels.df_sensitivity_map import committor, BASINS, KT


def watershed(F, occupied, connectivity8=True):
    """Assign every occupied cell to the local minimum it descends to (steepest descent
    + pointer jumping). Returns an integer label per cell (-1 where unoccupied)."""
    n, m = F.shape
    Fw = np.where(occupied, F, np.inf)
    flat = np.arange(n * m).reshape(n, m)
    best, bestF = flat.copy(), Fw.copy()
    offs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
            if (di, dj) != (0, 0) and (connectivity8 or di == 0 or dj == 0)]
    for di, dj in offs:
        Fs = np.full_like(Fw, np.inf); Is = flat.copy()
        ti = slice(max(0, -di), n - max(0, di)); si = slice(max(0, di), n - max(0, -di))
        tj = slice(max(0, -dj), m - max(0, dj)); sj = slice(max(0, dj), m - max(0, -dj))
        Fs[ti, tj] = Fw[si, sj]; Is[ti, tj] = flat[si, sj]
        upd = Fs < bestF
        bestF[upd] = Fs[upd]; best[upd] = Is[upd]
    root = best.ravel().copy()
    for _ in range(4 * (n + m)):                      # pointer jumping to fixed point
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    lab = root.reshape(n, m)
    lab = np.where(occupied, lab, -1)
    uniq = [u for u in np.unique(lab) if u >= 0]
    remap = {u: i for i, u in enumerate(uniq)}
    return np.vectorize(lambda v: remap.get(v, -1))(lab), len(uniq)


def gauss_blur(a, sigma):
    """Separable Gaussian blur with edge replication (no scipy dependency)."""
    if sigma <= 0:
        return a.copy()
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    out = a.astype(np.float64)
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[ax] = (r, r)
        p = np.pad(out, pad, mode="edge")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), ax, p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--lagtime", type=int, default=20)
    ap.add_argument("--smooth", type=float, default=1.2, help="blur sigma in CELLS, basin finding only")
    ap.add_argument("--min-pop", type=float, default=0.01, help="drop basins below this population share")
    ap.add_argument("--pair", nargs=2, type=int, default=[0, 1], help="basin ranks (0=most populated)")
    ap.add_argument("--committor-iters", type=int, default=4000000,
                    help="beta<->alphaR is stiff under Neumann BC; 400k sweeps is NOT enough")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)

    R = np.asarray(np.load(a.dataset)["R"], np.float32)
    nb = R.shape[1]
    pairs = np.array([(i, j) for i in range(nb) for j in range(i + 1, nb)], dtype=int)
    X = build_features(R, pairs)
    model, Y = fit_tica(X, a.lagtime)
    print(f"{len(R)} frames, {nb} beads -> {len(pairs)} pair distances, TICA lagtime {a.lagtime}")

    # ---- recover the affine map, because that is what PLUMED COMBINE can reproduce ----
    mu = X.mean(axis=0)
    coef, *_ = np.linalg.lstsq(X - mu, Y, rcond=None)
    resid = float(np.max(np.abs((X - mu) @ coef - Y)))
    scale = float(np.std(Y))
    rel = resid / max(scale, 1e-30)
    print(f"affine map residual max|(X-mean)@coef - Y| = {resid:.3e}  "
          f"(TIC spread {scale:.3f} -> RELATIVE {rel:.2e})")
    print("  TICA is linear in the pair distances, so DISTANCE+COMBINE reproduces it EXACTLY;")
    print("  a non-affine map would leave a residual of order the TIC spread itself, not 1e-5.")
    # Tolerance is RELATIVE: lstsq on 15 correlated distance features over 200k rows carries
    # float64 round-off ~1e-5 absolute. 1e-4 relative is still 4 orders below any nonlinearity.
    assert rel < 1e-4, (f"TICA projection is not affine in the features "
                        f"(relative residual {rel:.2e}); PLUMED COMBINE route invalid")

    # ---- grid (percentile edges; TICA is unbounded, so this is NOT periodic) ----------
    ex = np.linspace(*np.percentile(Y[:, 0], [0.2, 99.8]), a.bins + 1)
    ey = np.linspace(*np.percentile(Y[:, 1], [0.2, 99.8]), a.bins + 1)
    H, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], bins=[ex, ey])
    rho = H / H.sum()
    occ = rho > 0
    F = np.where(occ, -KT * np.log(np.maximum(rho, 1e-300)), np.nan)
    F -= np.nanmin(F)

    # ---- basins by watershed on the SMOOTHED FES (no phi/psi prior) ------------------
    rho_s = gauss_blur(np.where(occ, rho, 0.0), a.smooth)
    Fs = np.where(occ, -KT * np.log(np.maximum(rho_s, 1e-300)), np.inf)
    lab, nlab = watershed(Fs, occ)
    pops = np.array([rho[lab == i].sum() for i in range(nlab)])
    keep = np.argsort(-pops)[: max(2, int((pops >= a.min_pop).sum()))]
    print(f"\nwatershed found {nlab} minima; keeping {len(keep)} with population >= {a.min_pop}")

    ix = np.clip(np.digitize(Y[:, 0], ex) - 1, 0, a.bins - 1)
    iy = np.clip(np.digitize(Y[:, 1], ey) - 1, 0, a.bins - 1)
    m6 = get_mapping("ala2_backbone_cb_6")
    phi = m6.cvs["phi"].evaluate(R); psi = m6.cvs["psi"].evaluate(R)
    rama = {k: f(phi, psi) for k, f in BASINS.items()}
    fl = lab[ix, iy]
    print(f"{'rank':>4} {'cells':>6} {'pop':>8}   dominant Rama basin (DIAGNOSTIC ONLY)")
    names = {}
    for r, i in enumerate(keep):
        sel = fl == i
        share = {k: float(v[sel].mean()) for k, v in rama.items()}
        top = max(share, key=share.get)
        names[r] = top
        print(f"{r:>4} {int((lab==i).sum()):>6} {pops[i]:>8.3f}   "
              + "  ".join(f"{k} {100*v:.0f}%" for k, v in share.items()) + f"   -> {top}")

    A, B = [keep[r] for r in a.pair]
    nA, nB = names[a.pair[0]], names[a.pair[1]]
    print(f"\nchannel-2 pair: rank {a.pair[0]} ({nA})  <->  rank {a.pair[1]} ({nB})")
    mA, mB = (lab == A) & occ, (lab == B) & occ

    s1 = np.abs(np.where(mA, rho, 0)/max(rho[mA].sum(), 1e-30)
                - np.where(mB, rho, 0)/max(rho[mB].sum(), 1e-30))

    rho_c = np.maximum(rho, rho[occ].min() * 1e-3)
    q, iters, res = committor(rho_c, mA, mB, periodic=False, iters=a.committor_iters)
    print(f"committor CONVERGED (Neumann BC): residual {res:.2e} after {iters} sweeps")

    def d(u, ax):
        """Central difference inside, one-sided at the edges (no wrap)."""
        g = np.empty_like(u)
        if ax == 0:
            g[1:-1] = (u[2:] - u[:-2]) / 2.0
            g[0] = u[1] - u[0]; g[-1] = u[-1] - u[-2]
        else:
            g[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
            g[:, 0] = u[:, 1] - u[:, 0]; g[:, -1] = u[:, -1] - u[:, -2]
        return g
    flux = rho * (d(q, 0) ** 2 + d(q, 1) ** 2)
    s2 = flux / max(flux.sum(), 1e-30)
    s1n = s1 / max(s1.sum(), 1e-30); s2n = s2 / max(s2.sum(), 1e-30)
    s = 0.5 * (s1n + s2n)

    def corr(u, v):
        g = occ
        return float(np.corrcoef(np.log10(np.maximum(u[g], 1e-12)),
                                 np.log10(np.maximum(v[g], 1e-12)))[0, 1])
    print(f"\ncorr(log ch1, log density) = {corr(s1n, rho):+.3f}")
    print(f"corr(log ch2, log density) = {corr(s2n, rho):+.3f}")
    print(f"corr(log s,   log density) = {corr(s, rho):+.3f}")
    fw1, fw2 = s1n[ix, iy].sum(), s2n[ix, iy].sum()
    print(f"frame-weighted channel share: ch1 {100*fw1/(fw1+fw2):.1f}%  ch2 {100*fw2/(fw1+fw2):.1f}%")

    tag = f"{nA}_{nB}"
    np.savez_compressed(a.outdir / f"ddF_sensitivity_tica_{tag}.npz",
                        ex=ex, ey=ey, rho=rho, F=F, q=q, s1=s1n, s2=s2n, s=s,
                        labels=lab, pair_names=np.array([nA, nB]),
                        tica_mean=mu, tica_coefficients=coef, pairs=pairs,
                        lagtime=a.lagtime)

    ext = [ex[0], ex[-1], ey[0], ey[-1]]
    fig, ax = plt.subplots(1, 5, figsize=(25, 4.6))
    for k, (M, t, cm) in enumerate([
            (np.where(occ, F, np.nan), "TICA FES (kcal/mol)", "viridis"),
            (np.where(occ, lab.astype(float), np.nan), "watershed basins (no Rama prior)", "tab20"),
            (np.log10(np.maximum(s1n, 1e-12)), "log10 channel 1: |p_A - p_B|", "magma"),
            (np.log10(np.maximum(s2n, 1e-12)), "log10 channel 2: reactive flux", "inferno"),
            (np.where(occ, q, np.nan), f"committor q  ({nA} -> {nB})", "coolwarm")]):
        im = ax[k].imshow(M.T, origin="lower", extent=ext, aspect="auto", cmap=cm)
        ax[k].set_title(t, fontsize=10); ax[k].set_xlabel("TIC 1")
        if k == 0: ax[k].set_ylabel("TIC 2")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle(f"ddF sensitivity computed NATIVELY in TICA space ({nA} <-> {nB})", y=1.02)
    fig.tight_layout()
    kb = Path("/e/project1/cameo/schmidt36/KNOWLEDGE_BASE/P_cameo_cg/FIGURES")
    for p in (kb / f"2026-08-28_fig7_ddF_sensitivity_tica_{tag}.png",
              a.outdir / f"ddF_sensitivity_tica_{tag}.png"):
        fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\nwrote {kb / f'2026-08-28_fig7_ddF_sensitivity_tica_{tag}.png'}")
    print(f"      {a.outdir / f'ddF_sensitivity_tica_{tag}.npz'}")


if __name__ == "__main__":
    main()
