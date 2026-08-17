#!/usr/bin/env python3
"""Phase 2: build the flow-based acquisition bias and compare it to the incumbent KDE bias.

    python -m sampling.build_flow_bias --bias-npz <bias.npz> --reference <ref.npz> \
        --flow local_work/flow_sweep_final/flow_small_seed0.npz \
        --outdir local_work/flow_phase2

WHAT IS BEING ISOLATED
    `V = -kT log[(1-lambda) + lambda r(z)/p(z)]` with IDENTICAL lambda, r and kT for both
    routes. Only `p` differs: KDE vs flow. So every difference in the emitted bias is
    attributable to the density representation, which is the only claim Phase 2 needs to test.

    NOT asking "is the flow bias better at sampling" -- that needs MD (Phase 3). Asking whether
    swapping the density preserves the physical acquisition behaviour, and where it does not.

THE QUESTIONS (THEORY §26 "Bias reconstruction")
    * are the encouraged / discouraged regions the same?
    * is the force direction similar?
    * is the flow bias smoother?
    * does it behave better at the boundary and in unsupported regions?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bias-npz", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--flow", type=Path, required=True, help="flow_*.npz from the sweep")
    ap.add_argument("--flow-extra", type=Path, nargs="*", default=[],
                    help="further seeds of the SAME config, to show seed spread in the BIAS")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--grid", type=int, default=321)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--floor", choices=("kde", "constant", "none"), default="constant",
                   help="differentiable density floor: 'kde' (A, default) adds p_KDE; "
                        "'constant' (B) adds a calibrated constant and is KDE-FREE; "
                        "'none' is unprotected and for comparison only")
    ap.add_argument("--v-lim", type=float, default=1.5,
                   help="FIXED colour limit for V, kcal/mol. Deliberately not derived from the "
                        "data: a floating scale made the identical KDE panel look pale in one "
                        "run and saturated in the next, purely because the other panel changed.")
    ap.add_argument("--f-lim", type=float, default=30.0,
                   help="FIXED colour ceiling for |grad V|, same reason")
    ap.add_argument("--target-depth", type=float, default=1.77,
                   help="depth bound in kcal/mol used to calibrate the constant floor; the "
                        "default is the KDE's own structural bound on this artifact")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_bias import (AcquisitionBias, flow_density_fn, kde_density_fn,
                                    transition_weights)
    from sampling.flow_density import load_flow
    from sampling.plumed_native import padded_bounds

    bias = SmoothTICABias.load(a.bias_npz)
    kbt, lam = float(bias.kbt_kcal_mol), float(np.load(a.bias_npz)["lambda_value"])
    z_ref = np.asarray(bias.projection.transform(np.load(a.reference)["R"]), np.float64)[:, :2]
    r_w = transition_weights(a.bias_npz)
    centers, band = np.asarray(bias.centers, np.float64), np.asarray(bias.bandwidth, np.float64)
    print(f"lambda {lam}, kT {kbt:.4f}, r on {int((r_w > 0).sum())}/{len(r_w)} centres "
          f"(transition only, sparsity excluded)")

    # grid over the padded bounds -- the same window the EXTERNAL grid would be tabulated on
    lo, hi = padded_bounds(bias)
    gx = np.linspace(lo[0], hi[0], a.grid)
    gy = np.linspace(lo[1], hi[1], a.grid)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    H, _, _ = np.histogram2d(z_ref[:, 0], z_ref[:, 1], bins=[a.grid, a.grid],
                             range=[[gx[0], gx[-1]], [gy[0], gy[-1]]])
    solid, empty = H >= a.min_count, H == 0

    variants = {}

    def build(density, mode, floor_density=None, label=""):
        b = AcquisitionBias(density=density, centers=centers, r_weights=r_w, bandwidth=band,
                            lam=lam, kbt=kbt, floor_mode=mode, floor_density=floor_density,
                            target_depth_kcal=a.target_depth)
        pc = b.calibrate(pts)
        V, dV = b.energy_gradient(pts)
        variants[label] = dict(V=V.reshape(GX.shape), dV=dV.reshape(GX.shape + (2,)),
                               bound=b.depth_bound(pts), floor_constant=pc)
        return variants[label]

    dens_kde = kde_density_fn(bias)
    flows = [load_flow(a.flow)] + [load_flow(q) for q in a.flow_extra]
    dens_flow = flow_density_fn(*flows[0])

    build(dens_kde, "none", label="KDE")
    build(dens_flow, "none", label="flow-unprotected")
    build(dens_flow, "kde", floor_density=dens_kde, label="flow+A(KDE floor)")
    build(dens_flow, "constant", label="flow+B(const floor)")

    V_kde, dV_kde = variants["KDE"]["V"], variants["KDE"]["dV"]
    main = a.floor if a.floor != "none" else "flow-unprotected"
    key = {"kde": "flow+A(KDE floor)", "constant": "flow+B(const floor)",
           "none": "flow-unprotected"}[a.floor]
    V_flow, dV_flow = variants[key]["V"], variants[key]["dV"]
    built = [(variants[key]["V"], variants[key]["dV"])]
    for q, c in flows[1:]:
        d = flow_density_fn(q, c)
        r_ = build(d, a.floor, floor_density=dens_kde if a.floor == "kde" else None,
                   label=f"seed_{len(built)}")
        built.append((r_["V"], r_["dV"]))

    F_kde = np.linalg.norm(dV_kde, axis=-1)
    F_flow = np.linalg.norm(dV_flow, axis=-1)
    cos = ((dV_kde * dV_flow).sum(-1) /
           (F_kde * F_flow + 1e-12))

    # roughness: how much does the force change between neighbouring grid cells
    def roughness(dV):
        """Frobenius norm of the force Jacobian: how fast the bias FORCE changes with z.

        This is the quantity a smooth-looking FES can hide. An MD integrator feels the force;
        a rough force field means the bias is injecting structure on a scale the reference
        data does not resolve.
        """
        tot = np.zeros(dV.shape[:2])
        for k in (0, 1):
            d0, d1 = np.gradient(dV[..., k], gx, gy)
            tot += d0 ** 2 + d1 ** 2
        return np.sqrt(tot)

    R_kde, R_flow = roughness(dV_kde), roughness(dV_flow)

    def pct(x, m):
        return {f"p{q}": float(np.percentile(x[m], q)) for q in (50, 90, 99, 100)}

    # ---- WHICH density gradient is right? -------------------------------------------------
    # The cosine above measures AGREEMENT between the two biases, not correctness. Where they
    # disagree, adjudicate against the empirical histogram: the score field `grad log p` is the
    # part of the bias the density representation controls, and the histogram is the only
    # ground truth for it. Smoothed with a Gaussian narrower than the KDE bandwidth (which is
    # ~6.7 cells here) so it is differentiable without simply becoming another KDE.
    from scipy.ndimage import gaussian_filter
    adjudicate = {}
    for sig in (2.0, 3.0):
        ph = gaussian_filter(H, sig)
        ph = ph / (ph.sum() * (gx[1] - gx[0]) * (gy[1] - gy[0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(np.maximum(ph, 1e-300))
        g0, g1 = np.gradient(lg, gx, gy)
        g_hist = np.stack([g0, g1], -1)
        row = {}
        for name, dens in (("kde", kde_density_fn(bias)), ("flow", flow_density_fn(*flows[0]))):
            rho, grho = dens(pts)
            g = (grho / np.maximum(rho, 1e-300)[:, None]).reshape(GX.shape + (2,))
            num = (g * g_hist).sum(-1)
            den = np.linalg.norm(g, axis=-1) * np.linalg.norm(g_hist, axis=-1) + 1e-12
            c = (num / den)[solid]
            row[name] = dict(median_cosine=float(np.median(c)),
                             frac_above_0p9=float((c > 0.9).mean()),
                             median_mag_ratio=float(np.median(
                                 np.linalg.norm(g, axis=-1)[solid] /
                                 (np.linalg.norm(g_hist, axis=-1)[solid] + 1e-12))))
        adjudicate[f"hist_sigma_{sig}"] = row
    summary_adj = adjudicate

    # sign agreement: does the flow bias encourage where the KDE bias encourages?
    enc_kde, enc_flow = V_kde < -0.01, V_flow < -0.01
    agree = float((enc_kde[solid] == enc_flow[solid]).mean())

    summary = dict(
        lam=lam, kbt=kbt,
        cells=dict(solid=int(solid.sum()), empty=int(empty.sum()), total=int(H.size)),
        floor=dict(mode=a.floor, target_depth=a.target_depth,
                   constant=variants[key]["floor_constant"],
                   guaranteed_bound=variants[key]["bound"]),
        variants={k: dict(V_min=float(v["V"].min()), V_max=float(v["V"].max()),
                          cells_below_minus1=int((v["V"] < -1).sum()),
                          force_p99=float(np.percentile(np.linalg.norm(v["dV"], axis=-1), 99)),
                          force_max=float(np.linalg.norm(v["dV"], axis=-1).max()),
                          depth_bound=v["bound"])
                  for k, v in variants.items() if not k.startswith("seed_")},
        V_range=dict(kde=[float(V_kde.min()), float(V_kde.max())],
                     flow=[float(V_flow.min()), float(V_flow.max())]),
        V_on_support=dict(kde=pct(np.abs(V_kde), solid), flow=pct(np.abs(V_flow), solid)),
        force=dict(kde_solid=pct(F_kde, solid), flow_solid=pct(F_flow, solid),
                   kde_empty=pct(F_kde, empty), flow_empty=pct(F_flow, empty)),
        roughness=dict(kde_solid=pct(R_kde, solid), flow_solid=pct(R_flow, solid),
                       kde_empty=pct(R_kde, empty), flow_empty=pct(R_flow, empty)),
        force_direction_cosine=dict(median_solid=float(np.median(cos[solid])),
                                    frac_solid_above_0p9=float((cos[solid] > 0.9).mean())),
        encouraged_region_agreement=agree,
        score_vs_histogram=summary_adj,
    )
    if len(built) > 1:
        Vs = np.stack([b[0] for b in built])
        dVs = np.stack([b[1] for b in built])
        summary["seed_spread_in_bias"] = dict(
            V_rms_solid=float(np.sqrt(((Vs - Vs.mean(0)) ** 2).mean(0))[solid].mean()),
            force_rms_solid=float(np.sqrt((np.linalg.norm(dVs - dVs.mean(0), axis=-1) ** 2)
                                          .mean(0))[solid].mean()),
            n_seeds=len(built))
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    _figures(a, gx, gy, V_kde, V_flow, dV_kde, dV_flow, F_kde, F_flow, cos, solid, empty,
             R_kde, R_flow, built, variants)

    print(f"\n{'':22s}{'KDE':>12s}{'flow':>12s}")
    print(f"{'V range (kcal/mol)':22s}{f'{V_kde.min():.2f}..{V_kde.max():.2f}':>12s}"
          f"{f'{V_flow.min():.2f}..{V_flow.max():.2f}':>12s}")
    for lbl, d in (("|V| on support p50", summary["V_on_support"]),
                   ("|force| support p50", dict(kde=summary["force"]["kde_solid"],
                                                flow=summary["force"]["flow_solid"])),
                   ("|force| empty p99", dict(kde=summary["force"]["kde_empty"],
                                              flow=summary["force"]["flow_empty"]))):
        k = "p99" if "p99" in lbl else "p50"
        print(f"{lbl:22s}{d['kde'][k]:>12.3f}{d['flow'][k]:>12.3f}")
    print(f"{'roughness support p50':22s}{summary['roughness']['kde_solid']['p50']:>12.3f}"
          f"{summary['roughness']['flow_solid']['p50']:>12.3f}")
    print(f"{'roughness empty p99':22s}{summary['roughness']['kde_empty']['p99']:>12.3f}"
          f"{summary['roughness']['flow_empty']['p99']:>12.3f}")
    print(f"\nforce-direction cosine on support : median "
          f"{summary['force_direction_cosine']['median_solid']:.4f}, "
          f"{summary['force_direction_cosine']['frac_solid_above_0p9']*100:.1f}% above 0.9")
    print(f"encouraged-region agreement       : {agree*100:.1f}% of supported cells")
    print("\nWHICH score field matches the empirical histogram (supported cells)?")
    for k, row in summary_adj.items():
        print(f"  {k}:  " + "  ".join(
            f"{n} cos {v['median_cosine']:.3f} (>{0.9}: {v['frac_above_0p9']*100:4.1f}%) "
            f"|g|/|g_hist| {v['median_mag_ratio']:.2f}" for n, v in row.items()))
    print(f"\n{'variant':22s}{'V_min':>9}{'bound':>9}{'<-1 cells':>11}{'|F|p99':>9}{'|F|max':>9}")
    for k, v in summary["variants"].items():
        print(f"{k:22s}{v['V_min']:>9.2f}{v['depth_bound']:>9.2f}"
              f"{v['cells_below_minus1']:>11d}{v['force_p99']:>9.2f}{v['force_max']:>9.1f}")
    print(f"floor mode '{a.floor}'"
          + (f", p_c = {summary['floor']['constant']:.3e} "
             f"(target depth {a.target_depth} kcal/mol)" if a.floor == "constant" else ""))
    if "seed_spread_in_bias" in summary:
        s = summary["seed_spread_in_bias"]
        print(f"seed spread IN THE BIAS ({s['n_seeds']} seeds): V {s['V_rms_solid']:.4f} "
              f"kcal/mol, force {s['force_rms_solid']:.4f} kcal/mol/tic")
    print(f"wrote {a.outdir}/summary.json and figures")


def _figures(a, gx, gy, V_kde, V_flow, dV_kde, dV_flow, F_kde, F_flow, cos, solid, empty,
             R_kde, R_flow, built, variants) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    kw = dict(origin="lower", extent=ext, aspect="auto")
    sup = (~empty).T.astype(float)

    # --- fig1: the bias itself -----------------------------------------------------------
    lim = float(a.v_lim)          # fixed, so runs are directly comparable
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, (V, t) in enumerate([(V_kde, "KDE (incumbent)"), (V_flow, "flow")]):
        im = ax[k].imshow(V.T, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), **kw)
        ax[k].contour(gx, gy, sup, levels=[0.5], colors="k", linewidths=.8)
        ax[k].set_title(f"$V(z)$, {t}\n(blue = encouraged)")
        plt.colorbar(im, ax=ax[k], label="kcal/mol")
    d = V_flow - V_kde
    im = ax[2].imshow(d.T, cmap="PuOr_r", norm=TwoSlopeNorm(0, -lim, lim), **kw)
    ax[2].contour(gx, gy, sup, levels=[0.5], colors="k", linewidths=.8)
    ax[2].set_title("flow − KDE"); plt.colorbar(im, ax=ax[2], label="kcal/mol")
    for k in range(3):
        ax[k].set_xlabel("tic1")
    ax[0].set_ylabel("tic2")
    fig.tight_layout(); fig.savefig(a.outdir / "fig1_bias_comparison.png", dpi=130); plt.close(fig)

    # --- fig2: forces --------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    vmax = float(a.f_lim)         # fixed, see --f-lim
    for k, (F, t) in enumerate([(F_kde, "KDE"), (F_flow, "flow")]):
        im = ax[k].imshow(np.clip(F, 1e-3, vmax).T, cmap="magma",
                          norm=LogNorm(1e-2, vmax), **kw)
        ax[k].contour(gx, gy, sup, levels=[0.5], colors="cyan", linewidths=.8)
        ax[k].set_title(f"$|\\nabla V|$, {t}"); plt.colorbar(im, ax=ax[k], label="kcal/mol/tic")
    im = ax[2].imshow(cos.T, cmap="RdYlGn", vmin=-1, vmax=1, **kw)
    ax[2].contour(gx, gy, sup, levels=[0.5], colors="k", linewidths=.8)
    ax[2].set_title("force-direction cosine\n(green = same direction)")
    plt.colorbar(im, ax=ax[2])
    for k in range(3):
        ax[k].set_xlabel("tic1")
    ax[0].set_ylabel("tic2")
    fig.tight_layout(); fig.savefig(a.outdir / "fig2_force_comparison.png", dpi=130); plt.close(fig)

    # --- fig3: distributions + smoothness ------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    b = np.linspace(-lim, lim, 90)
    ax[0].hist(V_kde[solid], bins=b, alpha=.6, density=True, label="KDE")
    ax[0].hist(V_flow[solid], bins=b, alpha=.6, density=True, label="flow")
    ax[0].set_xlabel("V (kcal/mol)"); ax[0].set_title("bias on SUPPORTED cells"); ax[0].legend()

    lb = np.logspace(-3, 3, 80)
    ax[1].hist(F_kde[empty], bins=lb, alpha=.6, density=True, label="KDE, empty")
    ax[1].hist(F_flow[empty], bins=lb, alpha=.6, density=True, label="flow, empty")
    ax[1].set_xscale("log"); ax[1].set_xlabel("|grad V| (kcal/mol/tic)")
    ax[1].set_title("force in UNSUPPORTED cells\n(where interpolation is invented)")
    ax[1].legend()

    ax[2].hist(R_kde[solid], bins=np.logspace(-3, 4, 80), alpha=.6, density=True, label="KDE")
    ax[2].hist(R_flow[solid], bins=np.logspace(-3, 4, 80), alpha=.6, density=True, label="flow")
    ax[2].set_xscale("log"); ax[2].set_xlabel("|d(force)/dz|")
    ax[2].set_title("roughness of the force field\n(supported cells)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(a.outdir / "fig3_distributions.png", dpi=130); plt.close(fig)

    # --- fig4: seed spread in the emitted bias -------------------------------------------
    if len(built) > 1:
        Vs = np.stack([bb[0] for bb in built])
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
        sd = Vs.std(0)
        im = ax[0].imshow(sd.T, cmap="inferno", **kw)
        ax[0].contour(gx, gy, sup, levels=[0.5], colors="cyan", linewidths=.8)
        ax[0].set_title(f"seed spread of $V$ ({len(built)} flows)")
        plt.colorbar(im, ax=ax[0], label="kcal/mol")
        ax[1].hist(sd[solid], bins=60, alpha=.75, density=True)
        ax[1].set_xlabel("sd of V across seeds (kcal/mol)")
        ax[1].set_title("on supported cells")
        for k in range(2):
            ax[k].set_xlabel(ax[k].get_xlabel() or "tic1")
        fig.tight_layout(); fig.savefig(a.outdir / "fig4_bias_seed_spread.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
