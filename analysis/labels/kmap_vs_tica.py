#!/usr/bin/env python3
"""Does TICA explain the K residual that 2D Rama density cannot?

The earlier control compared K against (phi,psi) density. But the flow/acquisition
bias lives in TICA space, and stencil directions are chosen there, so TICA is the
relevant comparison: TICA is built from ALL 15 pair distances and may absorb part
of the extra-dimensional structure that a (phi,psi) projection discards.

Same protocol as the Rama version so the numbers are directly comparable:
  * K-map = median K per cell, split-half reliability correction
  * regression of log K on log density
  * smoothing sweep, to separate real structure from ball-vs-point geometry
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features, fit_tica
from sampling.mapping import get_mapping


# NOTE: equal-OCCUPANCY (quantile) binning is WRONG here. It flattens the density
# range by construction, so log(density) becomes near-constant and the
# correlation collapses for reasons that have nothing to do with K
# (job 1504460: Rama r fell +0.874 -> +0.408, TICA -> -0.019). Equal-width
# bins preserve the density range; control median noise with MIN_QUERIES and
# occupancy-weighted regression instead.
QUANTILE_BINS = False
MIN_QUERIES = 15


def binmed(vals, px, py, edx, edy, nb, minc=None):
    H = np.full((nb, nb), np.nan)
    ix = np.clip(np.digitize(px, edx) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(py, edy) - 1, 0, nb - 1)
    for i in range(nb):
        for j in range(nb):
            m = (ix == i) & (iy == j)
            if m.sum() >= (MIN_QUERIES if minc is None else minc):
                H[i, j] = np.median(vals[m])
    return H


def smooth(A, sb):
    if sb <= 0:
        return A.copy()
    n = A.shape[0]
    k = np.arange(n); k = np.minimum(k, n - k)
    g = np.exp(-0.5 * (k / sb) ** 2); g /= g.sum()
    G = np.outer(g, g)
    M = np.isfinite(A).astype(float); B = np.where(np.isfinite(A), A, 0.0)
    num = np.real(np.fft.ifft2(np.fft.fft2(B) * np.fft.fft2(G)))
    den = np.real(np.fft.ifft2(np.fft.fft2(M) * np.fft.fft2(G)))
    return np.where(np.isfinite(A), np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan), np.nan)


def analyse(name, K, px, py, allx, ally, nb, rng, tag):
    if QUANTILE_BINS:
        # Equal-OCCUPANCY cells. TICA coords are far less uniform than angles, so
        # equal-width cells leave many cells at the 3-query minimum -> noisy
        # medians -> artificially low split-half reliability. Quantile bins make
        # the per-cell noise comparable between the two spaces.
        edx = np.quantile(allx, np.linspace(0, 1, nb + 1))
        edy = np.quantile(ally, np.linspace(0, 1, nb + 1))
        edx[0] -= 1e-6; edx[-1] += 1e-6; edy[0] -= 1e-6; edy[-1] += 1e-6
    else:
        lo_x, hi_x = np.percentile(allx, [0.2, 99.8]); lo_y, hi_y = np.percentile(ally, [0.2, 99.8])
        edx = np.linspace(lo_x, hi_x, nb + 1); edy = np.linspace(lo_y, hi_y, nb + 1)
    Kmap = binmed(K, px, py, edx, edy, nb)
    ixq = np.clip(np.digitize(px, edx) - 1, 0, nb - 1)
    iyq = np.clip(np.digitize(py, edy) - 1, 0, nb - 1)
    occ_map = np.bincount(ixq * nb + iyq, minlength=nb * nb).reshape(nb, nb).astype(float)
    dens, _, _ = np.histogram2d(allx, ally, bins=[edx, edy])
    dm = np.where(dens > 0, dens, np.nan)
    # split-half reliability = ceiling for ANY predictor
    h = rng.permutation(len(K)); hA, hB = h[:len(h)//2], h[len(h)//2:]
    KA = binmed(K[hA], px[hA], py[hA], edx, edy, nb)
    KB = binmed(K[hB], px[hB], py[hB], edx, edy, nb)
    gh = np.isfinite(KA) & np.isfinite(KB)
    rh = float(np.corrcoef(np.log10(np.maximum(KA[gh], .5)), np.log10(np.maximum(KB[gh], .5)))[0, 1])
    rel = 2 * rh / (1 + rh)
    best = (0.0, -1.0)
    for sb in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        S = smooth(dm, sb)
        g = np.isfinite(Kmap) & np.isfinite(S) & (S > 0)
        if g.sum() < 20: continue
        w = occ_map[g]
        x = np.log10(S[g]); y = np.log10(np.maximum(Kmap[g], .5))
        xm = np.average(x, weights=w); ym = np.average(y, weights=w)
        cov = np.average((x - xm) * (y - ym), weights=w)
        r = float(cov / np.sqrt(np.average((x-xm)**2, weights=w) * np.average((y-ym)**2, weights=w)))
        e = 100 * min(r * r / rel, 1.0)
        if sb == 0.0: r0, e0 = r, e
        if e > best[1]: best = (sb, e)
    print(f"\n=== {name} ===")
    ixq = np.clip(np.digitize(px, edx) - 1, 0, nb - 1)
    iyq = np.clip(np.digitize(py, edy) - 1, 0, nb - 1)
    occ = np.bincount(ixq * nb + iyq, minlength=nb * nb)
    occ = occ[occ > 0]
    print(f"  cells {int(np.isfinite(Kmap).sum())}   K-map reliability {rel:.3f}   "
          f"queries/cell p50 {np.median(occ):.0f} p10 {np.percentile(occ,10):.0f}")
    print(f"  corr(log dens, log K)            = {r0:+.3f}")
    print(f"  density explains RELIABLE K      = {e0:.1f}%")
    print(f"  BEST smoothed density explains   = {best[1]:.1f}%  (sigma {best[0]:.1f} bins)")
    print(f"  -> residual NOT explained        = {100-best[1]:.1f}%")
    return dict(space=name, cells=int(np.isfinite(Kmap).sum()), reliability=rel,
                corr=r0, explains_pct=e0, best_smoothed_pct=best[1],
                residual_pct=100 - best[1]), Kmap, dm, edx, edy


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--kmap", type=Path, required=True, help="ala2_kmap.npz from ala2_kmap.py")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--lagtime", type=int, default=20)
    ap.add_argument("--bins", type=int, default=48)
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    d = np.load(a.dataset); R = np.asarray(d["R"], np.float32)
    z = np.load(a.kmap); q = np.asarray(z["q"]); Kall = np.asarray(z["K"]); eps = np.asarray(z["eps"])
    ei = int(np.argmin(np.abs(eps - 0.12)))
    K = Kall[:, ei].astype(float)
    print(f"{len(R)} frames; {len(q)} queries; using eps={eps[ei]:.3f}")

    # MUST match analysis/md/analyze_fes_tica_vs_reference.py::all_pairs -- the
    # canonical construction the project's TICA FES uses. choose_pairs(..., mode=
    # "sequential") returns ONLY the n-1 consecutive-bead distances (5 for bb6),
    # i.e. stiff bond lengths with no slow-mode content: the resulting "TICA" has
    # no basins at all and every TICA number from it is meaningless.
    pairs = np.array([(i, j) for i in range(R.shape[1]) for j in range(i + 1, R.shape[1])],
                     dtype=int)
    assert len(pairs) == R.shape[1] * (R.shape[1] - 1) // 2
    X = build_features(R, pairs)
    model, Y = fit_tica(X, lagtime=a.lagtime)
    print(f"TICA fitted (lagtime={a.lagtime}) on {len(X)} frames, {len(pairs)} pair features "
          f"-> {Y.shape[1]} components")

    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(R); psi = m.cvs["psi"].evaluate(R)

    res_rama, Kr, dr, exr, eyr = analyse("RAMA (phi,psi)", K, phi[q], psi[q], phi, psi, a.bins, rng, "rama")
    res_tica, Kt, dt, ext, eyt = analyse("TICA (tic1,tic2)", K, Y[q, 0], Y[q, 1], Y[:, 0], Y[:, 1], a.bins, rng, "tica")

    print(f"\n>>> Rama leaves {res_rama['residual_pct']:.1f}% unexplained; "
          f"TICA leaves {res_tica['residual_pct']:.1f}%")
    if res_tica['residual_pct'] < res_rama['residual_pct'] - 3:
        print(">>> TICA ABSORBS part of the residual: the flow bias already sees some of it.")
    elif res_tica['residual_pct'] > res_rama['residual_pct'] + 3:
        print(">>> TICA is WORSE than Rama at explaining K.")
    else:
        print(">>> TICA is NO BETTER than Rama: the residual is invisible to the TICA bias too.")

    # ---- fig2-style figure for the TICA space -----------------------------
    KB = Path("/e/project1/cameo/schmidt36/KNOWLEDGE_BASE/P_cameo_cg/FIGURES")
    KB.mkdir(parents=True, exist_ok=True)
    fig, axf = plt.subplots(1, 4, figsize=(21.5, 4.8))
    gt = np.isfinite(Kt) & np.isfinite(dt)
    lkt = np.log10(np.maximum(Kt[gt], .5)); ldt = np.log10(dt[gt])
    slt, ict = np.polyfit(ldt, lkt, 1)
    a0 = axf[0]
    a0.scatter(ldt, lkt, s=9, alpha=0.35, color="C1")
    xx = np.linspace(ldt.min(), ldt.max(), 50)
    a0.plot(xx, slt*xx + ict, "k-", lw=2, label=f"fit r={res_tica['corr']:+.3f}")
    a0.set_xlabel("log10 TICA density"); a0.set_ylabel("log10 median K")
    a0.set_title(f"A. K vs TICA density\ndensity explains {res_tica['explains_pct']:.1f}% of RELIABLE K")
    a0.legend(fontsize=8); a0.grid(alpha=0.3)

    residt = np.full_like(Kt, np.nan); residt[gt] = lkt - (slt*ldt + ict)
    v = np.nanpercentile(np.abs(residt), 98)
    a1 = axf[1]
    im = a1.imshow(residt.T, origin="lower", extent=[ext[0], ext[-1], eyt[0], eyt[-1]],
                   aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
    a1.set_xlabel("TIC 1"); a1.set_ylabel("TIC 2")
    a1.set_title(f"B. log K residual in TICA space\n({res_tica['residual_pct']:.1f}% of reliable variance)")
    fig.colorbar(im, ax=a1, label="log10 K - density prediction")

    a2 = axf[2]
    im2 = a2.imshow(np.log10(np.maximum(Kt, .5)).T, origin="lower",
                    extent=[ext[0], ext[-1], eyt[0], eyt[-1]], aspect="auto", cmap="viridis")
    lowk = residt < -np.nanpercentile(np.abs(residt), 85)
    yy, xx2 = np.where(lowk.T)
    cx = 0.5*(ext[1:]+ext[:-1]); cy = 0.5*(eyt[1:]+eyt[:-1])
    a2.scatter(cx[xx2], cy[yy], s=18, facecolors="none", edgecolors="red", lw=1.2,
               label="fewer partners than density predicts")
    a2.set_xlabel("TIC 1"); a2.set_ylabel("TIC 2"); a2.legend(fontsize=8, loc="upper right")
    a2.set_title("C. K-map in TICA space")
    fig.colorbar(im2, ax=a2, label="log10 median K")

    a3 = axf[3]
    xs = ["Rama\n(phi,psi)", "TICA\n(tic1,tic2)"]
    a3.bar(xs, [res_rama["corr"], res_tica["corr"]], color=["C0", "C1"], alpha=0.85)
    for i, (r, rel_) in enumerate([(res_rama["corr"], res_rama["reliability"]),
                                   (res_tica["corr"], res_tica["reliability"])]):
        a3.text(i, r + 0.03, f"r={r:+.3f}\nreliab {rel_:.2f}", ha="center", fontsize=9)
    a3.set_ylim(0, 1.05); a3.set_ylabel("corr(log density, log K)")
    a3.set_title("D. Which coordinate organises K?\nTICA tracks partner availability far worse")
    a3.grid(alpha=0.3, axis="y")
    fig.suptitle("Does TICA explain the K residual? — ala2 bb6 (equal-width bins, occupancy-weighted)", y=1.03)
    fig.tight_layout()
    fig.savefig(KB / "2026-08-26_fig3_kmap_vs_tica.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {KB/'2026-08-26_fig3_kmap_vs_tica.png'}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for axx, (Km, dmm, ex, ey, lab) in zip(ax[:2], [(Kr, dr, exr, eyr, "Rama"), (Kt, dt, ext, eyt, "TICA")]):
        g = np.isfinite(Km) & np.isfinite(dmm)
        lk = np.log10(np.maximum(Km[g], .5)); ld = np.log10(dmm[g])
        sl, ic = np.polyfit(ld, lk, 1)
        resid = np.full_like(Km, np.nan); resid[g] = lk - (sl * ld + ic)
        v = np.nanpercentile(np.abs(resid), 98)
        im = axx.imshow(resid.T, origin="lower", extent=[ex[0], ex[-1], ey[0], ey[-1]],
                        aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
        axx.set_title(f"log K residual in {lab} space"); fig.colorbar(im, ax=axx)
    ax[2].bar(["Rama", "TICA"], [res_rama["residual_pct"], res_tica["residual_pct"]], color=["C0", "C1"])
    ax[2].set_ylabel("% of reliable K variance NOT explained")
    ax[2].set_title("Residual that density cannot explain")
    fig.tight_layout(); fig.savefig(a.outdir / "kmap_rama_vs_tica.png", dpi=150); plt.close(fig)
    (a.outdir / "kmap_vs_tica.json").write_text(json.dumps({"rama": res_rama, "tica": res_tica}, indent=2))
    print(f"\nwrote {a.outdir/'kmap_rama_vs_tica.png'}")


if __name__ == "__main__":
    main()
