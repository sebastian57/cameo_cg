#!/usr/bin/env python3
"""Where does a force error cost the most ddF? A sampling-acquisition map.

Relative basin free energies have TWO error channels under force matching:

  Channel 1 (basin shape).  dF_A depends on U inside A weighted by the NORMALISED
      Boltzmann density there:  d(ddF)/dU(x) = p_A(x) - p_B(x).

  Channel 2 (inter-basin offset).  Force matching fixes U only up to a constant, so the
      offset between basins is  int dF . dl  along a connecting path. The integral is
      path-INDEPENDENT (both fields are gradients), so it is dominated by regions that all
      connecting paths must cross -- bottlenecks. Quantified here by the committor q from a
      diffusive model on the reference FES: solve  div(rho grad q) = 0  with q=0 in A, q=1
      in B; the reactive flux density is  rho |grad q|^2, which concentrates at the saddle.

Neither channel is monotone in density, so NEITHER is expressible by alpha-tempering
(log p_temp = alpha log p_ref), which is why a tempered bias cannot target them.

Information gain per frame for ddF goes as s(x)^2 / sigma(x)^2; for fixed label noise the
optimal placement density for a fixed budget goes as s(x).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping

KT = 0.5921868690749673
BASINS = {
    "alphaR": lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
    "beta":   lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
    "alphaL": lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100),
}


def committor(rho, maskA, maskB, iters=400000, tol=1e-8, periodic=True):
    """Solve div(rho grad q)=0, q=0 on A, q=1 on B, periodic, by RED-BLACK SOR.

    NOTE: a vectorised SIMULTANEOUS (Jacobi) update with omega>1 is unstable and
    decouples the even/odd sublattices -- it produces a CHECKERBOARD that looks like
    a solution but is not one (seen in job 1517343). Red-black Gauss-Seidel updates
    the two sublattices alternately, which is both stable at omega~1.9 and immune to
    that mode. Convergence is asserted on the RESIDUAL, not the iteration count.
    """
    def sh(a, k, ax):
        """Shift by k along ax. periodic=wrap; otherwise REPLICATE the edge, which is
        the ghost-cell form of a no-flux (Neumann) boundary. TICA space is unbounded and
        NOT periodic, so wrapping there would connect unrelated ends of the FES."""
        out = np.roll(a, k, ax)
        if not periodic:
            i = 0 if k > 0 else -1
            sl = [slice(None)] * a.ndim
            sl[ax] = i
            out[tuple(sl)] = np.take(a, i, axis=ax)
        return out

    n, m = rho.shape
    q = np.full((n, m), 0.5)
    q[maskA] = 0.0; q[maskB] = 1.0
    fixed = maskA | maskB

    def face(a, b):
        return 2.0 * a * b / np.maximum(a + b, 1e-30)
    cE = face(rho, sh(rho, -1, 0)); cW = face(rho, sh(rho, 1, 0))
    cN = face(rho, sh(rho, -1, 1)); cS = face(rho, sh(rho, 1, 1))
    denom = np.maximum(cE + cW + cN + cS, 1e-30)

    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    red = ((ii + jj) % 2 == 0) & ~fixed
    black = ((ii + jj) % 2 == 1) & ~fixed
    omega = 1.9

    def sweep(sel):
        upd = (cE * sh(q, -1, 0) + cW * sh(q, 1, 0) +
               cN * sh(q, -1, 1) + cS * sh(q, 1, 1)) / denom
        q[sel] = np.clip(q[sel] + omega * (upd[sel] - q[sel]), 0.0, 1.0)

    res = np.inf
    for it in range(iters):
        sweep(red); sweep(black)
        if it % 200 == 0:
            lhs = (cE * sh(q, -1, 0) + cW * sh(q, 1, 0) +
                   cN * sh(q, -1, 1) + cS * sh(q, 1, 1)) - denom * q
            res = float(np.max(np.abs(lhs[~fixed] / denom[~fixed])))
            if res < tol:
                break
    if not np.isfinite(res) or res >= tol:
        raise RuntimeError(f"committor did NOT converge: residual {res:.3e} >= tol {tol:.1e} "
                           f"after {it+1} sweeps. Do not use the result.")
    # smoothness guard: a checkerboard has large cell-to-cell alternation
    alt = np.abs(q - 0.25*(sh(q,1,0)+sh(q,-1,0)+sh(q,1,1)+sh(q,-1,1)))
    if np.median(alt[~fixed]) > 0.05:
        raise RuntimeError(f"committor looks like a checkerboard "
                           f"(median local alternation {np.median(alt[~fixed]):.3f}); rejected.")
    return q, it + 1, res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--v4", default="local_work/input_data/ala2_bb6_v4_195k.npz")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--pair", nargs=2, default=["beta", "alphaR"])
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)

    R = np.asarray(np.load(a.dataset)["R"], np.float32)
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(R); psi = m.cvs["psi"].evaluate(R)
    ed = np.linspace(-180, 180, a.bins + 1)
    H, _, _ = np.histogram2d(phi, psi, bins=[ed, ed])
    rho = H / H.sum()
    ctr = 0.5 * (ed[1:] + ed[:-1])
    PX, PY = np.meshgrid(ctr, ctr, indexing="ij")
    F = np.where(rho > 0, -KT * np.log(np.maximum(rho, 1e-300)), np.nan)
    F -= np.nanmin(F)

    masks = {k: f(PX, PY) & (rho > 0) for k, f in BASINS.items()}
    A, B = a.pair
    print(f"basin cells: " + "  ".join(f"{k} {int(v.sum())}" for k, v in masks.items()))
    print(f"channel-2 pair: {A} <-> {B}")

    # ---- channel 1: |p_A - p_B|, normalised WITHIN each basin -------------
    pA = np.where(masks[A], rho, 0.0); pA /= max(pA.sum(), 1e-30)
    pB = np.where(masks[B], rho, 0.0); pB /= max(pB.sum(), 1e-30)
    s1 = np.abs(pA - pB)

    # ---- channel 2: reactive flux rho |grad q|^2 --------------------------
    rho_c = np.maximum(rho, rho[rho > 0].min() * 1e-3)   # keep the solve well-posed
    q, iters, res = committor(rho_c, masks[A], masks[B])
    dqx = (np.roll(q, -1, 0) - np.roll(q, 1, 0)) / 2.0
    dqy = (np.roll(q, -1, 1) - np.roll(q, 1, 1)) / 2.0
    flux = rho * (dqx ** 2 + dqy ** 2)
    s2 = flux / max(flux.sum(), 1e-30)
    print(f"committor CONVERGED: residual {res:.2e} after {iters} sweeps "
          f"(smoothness guard passed)")

    s1n = s1 / max(s1.sum(), 1e-30); s2n = s2 / max(s2.sum(), 1e-30)
    s = 0.5 * (s1n + s2n)                    # equal weight; both are normalised measures

    # ---- how different is this from density / tempered density? ----------
    good = rho > 0
    def corr(u, v):
        u = np.log10(np.maximum(u[good], 1e-12)); v = np.log10(np.maximum(v[good], 1e-12))
        return float(np.corrcoef(u, v)[0, 1])
    print(f"\ncorr(log sensitivity, log reference density) = {corr(s, rho):+.3f}")
    print(f"corr(log channel1, log density)              = {corr(s1n, rho):+.3f}")
    print(f"corr(log channel2, log density)              = {corr(s2n, rho):+.3f}")
    print("  (alpha-tempering is monotone in log density, so a |corr| well below 1 means")
    print("   NO tempering exponent can reproduce this map)")

    v4map = None
    if Path(a.v4).exists():
        R4 = np.asarray(np.load(a.v4)["R"], np.float32)
        p4 = m.cvs["phi"].evaluate(R4); s4 = m.cvs["psi"].evaluate(R4)
        h4, _, _ = np.histogram2d(p4, s4, bins=[ed, ed]); v4map = h4 / max(h4.sum(), 1)
        print(f"corr(log sensitivity, log v4 anchor density) = {corr(s, v4map):+.3f}")

    # where would the budget go?
    top = s >= np.nanpercentile(s[good], 90)
    print(f"\ntop-10% sensitivity cells hold {100*rho[top].sum():.1f}% of the equilibrium population")
    print(f"   -> a density-proportional scheme would give them {100*rho[top].sum():.1f}% of anchors;")
    print(f"      a sensitivity-proportional scheme gives them {100*s[top].sum():.1f}%")

    panels = [(F, "reference FES [kcal/mol]", "viridis_r", None),
              (q, f"committor q ({A}->{B})", "coolwarm", None),
              (np.log10(np.maximum(s1n, 1e-12)), "log10 channel 1: |p_A - p_B| (basin shape)", "magma", None),
              (np.log10(np.maximum(s2n, 1e-12)), "log10 channel 2: reactive flux (bottleneck)", "inferno", None),
              (np.log10(np.maximum(s, 1e-12)), "log10 COMBINED ddF sensitivity", "turbo", None),
              (np.log10(np.maximum(rho, 1e-12)), "log10 reference density (what tempering sees)", "cividis", None)]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    for axx, (M, t, cm, _) in zip(ax.ravel(), panels):
        Mm = np.where(good, M, np.nan)
        im = axx.imshow(Mm.T, origin="lower", extent=[-180, 180, -180, 180], aspect="auto", cmap=cm)
        axx.set_title(t, fontsize=10); axx.set_xlabel("phi"); axx.set_ylabel("psi")
        fig.colorbar(im, ax=axx)
    fig.suptitle("Where a force error costs the most ddF — ala2 bb6 "
                 f"({A} <-> {B}); neither channel is monotone in density", y=1.01)
    fig.tight_layout()
    out = Path("/e/project1/cameo/schmidt36/KNOWLEDGE_BASE/P_cameo_cg/FIGURES")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "2026-08-26_fig4_ddF_sensitivity_map.png", dpi=150, bbox_inches="tight")
    fig.savefig(a.outdir / "ddF_sensitivity_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(a.outdir / "ddF_sensitivity.npz", edges=ed, rho=rho, F=F, q=q,
                        s1=s1n, s2=s2n, s=s, pair=np.array([A, B]))
    print(f"\nwrote {out/'2026-08-26_fig4_ddF_sensitivity_map.png'}")
    print(f"      {a.outdir/'ddF_sensitivity.npz'}")


if __name__ == "__main__":
    main()
