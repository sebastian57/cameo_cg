#!/usr/bin/env python3
"""What does a TICA-density bias MISS, relative to the ddF sensitivity map?

Produces the decision plot: for a fixed anchor budget, what fraction of the total ddF
sensitivity does each placement scheme actually cover?

Schemes compared (all as per-FRAME selection weights on the reference trajectory):
  sensitivity-proportional   w ~ s(x)                     <- the proposal
  TICA-tempered              w ~ p_TICA(x)^alpha          <- what the current bias does
  Rama-tempered              w ~ p_rama(x)^alpha
  density-proportional       w ~ p(x)                     <- plain equilibrium
  uniform-over-cells         w ~ 1/p(x)                   <- flat coverage
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features, fit_tica
from sampling.mapping import get_mapping


def cellify(x, y, ed):
    nb = len(ed) - 1
    ix = np.clip(np.digitize(x, ed) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(y, ed) - 1, 0, nb - 1)
    return ix, iy, ix * nb + iy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sens", required=True, help="ddF_sensitivity.npz")
    ap.add_argument("--alpha", type=float, default=0.83, help="v4 tempering exponent")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--lagtime", type=int, default=20)
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(a.sens, allow_pickle=True)
    ed = z["edges"]; s_map = z["s"]; s1 = z["s1"]; s2 = z["s2"]; rho = z["rho"]
    pair = [str(x) for x in np.atleast_1d(z["pair"])]
    nb = len(ed) - 1

    R = np.asarray(np.load(a.dataset)["R"], np.float32)
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(R); psi = m.cvs["psi"].evaluate(R)
    npair = np.array([(i, j) for i in range(R.shape[1]) for j in range(i+1, R.shape[1])], int)
    Y = fit_tica(build_features(R, npair), a.lagtime)[1]

    ixr, iyr, _ = cellify(phi, psi, ed)
    s_f = s_map[ixr, iyr]                       # per-frame ddF sensitivity
    rho_f = rho[ixr, iyr]

    # TICA density on its own grid, then read back per frame
    edt_x = np.linspace(*np.percentile(Y[:, 0], [0.2, 99.8]), nb + 1)
    edt_y = np.linspace(*np.percentile(Y[:, 1], [0.2, 99.8]), nb + 1)
    itx = np.clip(np.digitize(Y[:, 0], edt_x) - 1, 0, nb - 1)
    ity = np.clip(np.digitize(Y[:, 1], edt_y) - 1, 0, nb - 1)
    Ht, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], bins=[edt_x, edt_y]); Ht /= Ht.sum()
    ptica_f = Ht[itx, ity]

    eps = 1e-12
    schemes = {
        "sensitivity  w~s(x)":        np.maximum(s_f, 0),
        f"TICA-tempered a={a.alpha}": np.maximum(ptica_f, eps) ** a.alpha,
        f"Rama-tempered a={a.alpha}": np.maximum(rho_f, eps) ** a.alpha,
        "density  w~p(x)":            np.maximum(rho_f, eps),
        "flat-in-cells  w~1/p(x)":    1.0 / np.maximum(rho_f, eps),
    }
    total_s = s_f.sum()
    # Anchors are DRAWN stochastically with probability ~ w, not taken top-N. Top-N is
    # invariant under any monotone transform of w, which makes tempering look identical to
    # plain density (it is not). Gumbel-top-k gives exact weighted sampling WITHOUT
    # replacement; average over several draws.
    rng = np.random.default_rng(0)
    N_DRAW = 12
    fracs = np.array([0.01, 0.05, 0.10, 0.25])
    print(f"pair {pair[0]} <-> {pair[1]};  {len(R)} frames\n")
    print(f"{'scheme':28s} " + " ".join(f"{f'top {q}%':>9}" for q in (1, 5, 10, 25)))
    curves = {}
    grid = np.unique(np.clip((np.logspace(-3.2, 0, 40) * len(s_f)).astype(int), 1, len(s_f)))
    for name, w in schemes.items():
        wn = np.maximum(w, 0) / max(np.sum(np.maximum(w, 0)), eps)
        acc = np.zeros(len(grid)); tab = np.zeros(len(fracs))
        for _ in range(N_DRAW):
            keys = np.log(np.maximum(wn, 1e-300)) + rng.gumbel(size=len(wn))
            order = np.argsort(-keys)                # weighted sample w/o replacement
            cum = np.cumsum(s_f[order]) / total_s
            acc += cum[grid - 1]
            tab += np.array([cum[int(f*len(order)) - 1] for f in fracs])
        curves[name] = (grid / len(s_f), acc / N_DRAW)
        print(f"{name:28s} " + " ".join(f"{100*v/N_DRAW:8.1f}%" for v in tab))
    ideal = np.cumsum(np.sort(s_f)[::-1]) / total_s
    curves["IDEAL (oracle, top-N)"] = (grid / len(s_f), ideal[grid - 1])
    print(f"{'IDEAL (oracle, top-N)':28s} " + " ".join(
        f"{100*ideal[int(f*len(s_f))-1]:8.1f}%" for f in fracs))
    print("\nNOTE: schemes are DRAWN ~ w (Gumbel top-k, 12 draws averaged). The oracle row is a\n"
          "top-N upper bound and is not achievable by stochastic sampling.")

    # --- alpha sweep: is ANY tempering exponent competitive with targeting s(x)? -------
    # w ~ p^alpha.  alpha=1 plain density, alpha=0 uniform-in-cells, alpha<0 flat-in-
    # population, alpha>1 sharpens onto the basin minima.
    def cov10(w):
        wn = np.maximum(w, 0) / max(np.sum(np.maximum(w, 0)), eps)
        v = 0.0
        for _ in range(N_DRAW):
            keys = np.log(np.maximum(wn, 1e-300)) + rng.gumbel(size=len(wn))
            order = np.argsort(-keys)
            v += (np.cumsum(s_f[order]) / total_s)[int(0.10 * len(order)) - 1]
        return 100.0 * v / N_DRAW

    print("\nalpha sweep, coverage at 10% budget (w ~ p^alpha), same Gumbel-top-k draws:")
    print(f"{'alpha':>7} {'Rama p^a':>10} {'TICA p^a':>10}")
    for al in (-0.5, 0.0, 0.5, 0.83, 1.0, 1.25, 1.5, 2.0, 3.0):
        print(f"{al:7.2f} {cov10(np.maximum(rho_f, eps) ** al):9.1f}% "
              f"{cov10(np.maximum(ptica_f, eps) ** al):9.1f}%")
    print(f"{'s(x)':>7} {cov10(np.maximum(s_f, 0)):9.1f}%  <- target the map directly")

    # --- per-channel: the combined metric is FRAME-weighted, so a cell contributes
    # population x s_map.  Channel 2 lives in low-population cells and can be nearly
    # invisible in the total even though the two maps carry equal mass by construction.
    s1_f, s2_f = z["s1"][ixr, iyr], z["s2"][ixr, iyr]
    print(f"\nframe-weighted share of total sensitivity:  ch1 {100*s1_f.sum()/(s1_f.sum()+s2_f.sum()):.1f}%"
          f"   ch2 {100*s2_f.sum()/(s1_f.sum()+s2_f.sum()):.1f}%")

    def cov10_of(target, w):
        tot = target.sum()
        wn = np.maximum(w, 0) / max(np.sum(np.maximum(w, 0)), eps)
        v = 0.0
        for _ in range(N_DRAW):
            keys = np.log(np.maximum(wn, 1e-300)) + rng.gumbel(size=len(wn))
            order = np.argsort(-keys)
            v += (np.cumsum(target[order]) / tot)[int(0.10 * len(order)) - 1]
        return 100.0 * v / N_DRAW

    print("\nPER-CHANNEL coverage at 10% budget (Rama density family):")
    print(f"{'scheme':>16} {'ch1 shape':>11} {'ch2 bottleneck':>15}")
    for nm, w in (("p^0.83 (v4)", np.maximum(rho_f, eps)**0.83),
                  ("p  (density)", np.maximum(rho_f, eps)),
                  ("p^3 (sharpen)", np.maximum(rho_f, eps)**3.0),
                  ("1/p (flat)",   1.0/np.maximum(rho_f, eps)),
                  ("s(x) combined", np.maximum(s_f, 0)),
                  ("s2 only",       np.maximum(s2_f, 0))):
        print(f"{nm:>16} {cov10_of(s1_f, w):10.1f}% {cov10_of(s2_f, w):14.1f}%")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    good = rho > 0
    def show(axx, M, t, cm="turbo"):
        Mm = np.where(good, M, np.nan)
        im = axx.imshow(Mm.T, origin="lower", extent=[-180, 180, -180, 180],
                        aspect="auto", cmap=cm)
        axx.set_title(t, fontsize=10); axx.set_xlabel("phi"); axx.set_ylabel("psi")
        fig.colorbar(im, ax=axx)

    show(ax[0, 0], np.log10(np.maximum(s_map, 1e-12)),
         f"A. ddF sensitivity s(x)  ({pair[0]}<->{pair[1]})")
    # what the TICA-tempered bias places, expressed back in Rama cells
    wt = np.maximum(ptica_f, eps) ** a.alpha; wt /= wt.sum()
    Wt = np.zeros((nb, nb)); np.add.at(Wt, (ixr, iyr), wt)
    show(ax[0, 1], np.log10(np.maximum(Wt, 1e-12)),
         f"B. what TICA-tempered bias places (a={a.alpha})", "cividis")
    sN = s_map / max(s_map.sum(), eps)
    miss = np.log10(np.maximum(sN, 1e-12)) - np.log10(np.maximum(Wt, 1e-12))
    v = np.nanpercentile(np.abs(np.where(good, miss, np.nan)), 98)
    Mm = np.where(good, miss, np.nan)
    im = ax[0, 2].imshow(Mm.T, origin="lower", extent=[-180,180,-180,180], aspect="auto",
                         cmap="coolwarm", vmin=-v, vmax=v)
    ax[0, 2].set_title("C. UNDER-COVERED by the bias (red = needs more)", fontsize=10)
    ax[0, 2].set_xlabel("phi"); ax[0, 2].set_ylabel("psi"); fig.colorbar(im, ax=ax[0, 2])

    show(ax[1, 0], np.log10(np.maximum(s1, 1e-12)), "D. channel 1: basin shape |p_A-p_B|", "magma")
    show(ax[1, 1], np.log10(np.maximum(s2, 1e-12)), "E. channel 2: bottleneck flux", "inferno")

    b = ax[1, 2]
    for name, (frac, cum) in curves.items():
        st = dict(lw=1.5, ls="--", color="k") if name.startswith("IDEAL") else dict(lw=2)
        b.plot(100*frac, 100*cum, label=name, **st)
    b.set_xscale("log"); b.set_xlim(0.1, 100); b.set_ylim(0, 101)
    b.set_xlabel("% of anchor budget (log)"); b.set_ylabel("% of total ddF sensitivity covered")
    b.set_title("F. Coverage efficiency — the decision plot", fontsize=10)
    b.legend(fontsize=7.5, loc="lower right"); b.grid(alpha=0.3, which="both")

    fig.suptitle(f"What a TICA-density bias misses vs the ddF sensitivity map — "
                 f"ala2 bb6, {pair[0]}<->{pair[1]}", y=1.01)
    fig.tight_layout()
    out = Path("/e/project1/cameo/schmidt36/KNOWLEDGE_BASE/P_cameo_cg/FIGURES")
    fig.savefig(out / f"2026-08-26_fig5_sensitivity_vs_bias_{pair[1]}.png", dpi=150, bbox_inches="tight")
    fig.savefig(a.outdir / "sensitivity_vs_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # ---- fig6: per-channel Pareto -------------------------------------------------
    names = ["p^0.83\n(v4 prod)", "p\n(density)", "p^3\n(sharpen)", "1/p\n(flat)",
             "s(x)\ncombined"]
    ws6 = [np.maximum(rho_f, eps)**0.83, np.maximum(rho_f, eps),
           np.maximum(rho_f, eps)**3.0, 1.0/np.maximum(rho_f, eps), np.maximum(s_f, 0)]
    c1 = [cov10_of(s1_f, w) for w in ws6]
    c2 = [cov10_of(s2_f, w) for w in ws6]
    f6, ax6 = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.arange(len(names))
    ax6[0].bar(xs - 0.2, c1, 0.4, label="channel 1: basin shape", color="#3b6ea5")
    ax6[0].bar(xs + 0.2, c2, 0.4, label="channel 2: bottleneck", color="#c1533c")
    ax6[0].set_xticks(xs); ax6[0].set_xticklabels(names, fontsize=8)
    ax6[0].set_ylabel("% of channel covered at 10% budget")
    ax6[0].set_title(f"{pair[0]} <-> {pair[1]}: tempering cannot reach both channels")
    ax6[0].legend(fontsize=8); ax6[0].grid(alpha=0.3, axis="y")
    ax6[1].scatter(c1, c2, s=90, c=["#888", "#888", "#888", "#888", "#c1533c"], zorder=3)
    for xi, yi, nm in zip(c1, c2, names):
        ax6[1].annotate(nm.replace("\n", " "), (xi, yi), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
    ax6[1].set_xlabel("channel 1 coverage (%)"); ax6[1].set_ylabel("channel 2 coverage (%)")
    ax6[1].set_title("Pareto view: s(x) is the only scheme off the density line")
    ax6[1].grid(alpha=0.3)
    f6.tight_layout()
    f6.savefig(out / f"2026-08-28_fig6_perchannel_coverage_{pair[1]}.png", dpi=150,
               bbox_inches="tight")
    f6.savefig(a.outdir / "perchannel_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(f6)
    print(f"wrote {out/f'2026-08-28_fig6_perchannel_coverage_{pair[1]}.png'}")

    print(f"\nwrote {out/f'2026-08-26_fig5_sensitivity_vs_bias_{pair[1]}.png'}")


if __name__ == "__main__":
    main()
