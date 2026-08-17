#!/usr/bin/env python3
"""Phase 1: train and validate a normalizing-flow density over the frozen TICA projection.

    python -m sampling.build_flow_density \
        --bias-npz <reference_bias.npz> --reference <mapped-AA reference>.npz \
        --outdir local_work/flow_phase1 --seeds 3

WHAT THIS DECIDES
    Whether `p_theta` is a good enough stand-in for the KDE `p_ref` to carry a bias. The
    acceptance criterion is NOT density agreement -- it is the GRADIENT. The bias is
    `V = -kT log(q/p_theta)`, so what MD feels is `grad log p_theta`, and a flow can reproduce
    a density visually while carrying gradient oscillations that a histogram plot hides.

    Hence the central number here is **seed-to-seed gradient reproducibility**: train several
    independently initialised flows and ask whether their score fields agree. If
    `p_th1 ~ p_th2` but `grad log p_th1 != grad log p_th2`, the density-estimation problem is
    underdetermined at this capacity and the flow is not safe to bias with. That test is
    cheap here and expensive after a campaign has run.

NO MD IS RUN. Offline density experiment only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bias-npz", required=True,
                   help="existing artifact; supplies the FROZEN TICA projection and the KDE "
                        "reference density to compare against")
    p.add_argument("--reference", required=True, help="mapped-AA reference npz with R")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n-dims", type=int, default=2)
    p.add_argument("--seeds", type=int, default=3, help="independent fits; >=2 to test gradients")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grid", type=int, default=241, help="validation grid per axis")
    p.add_argument("--min-count", type=int, default=20,
                   help="histogram counts required before a cell enters the FES comparison")
    p.add_argument("--fes-vmax", type=float, default=8.0,
                   help="FES colour ceiling, kcal/mol. NOT a percentile of the supported cells: "
                        "those span <2 kcal/mol and saturate the plot, hiding exactly the "
                        "rare-region structure the comparison is about.")
    p.add_argument("--reuse", action="store_true",
                   help="load flow_seed*.npz from --outdir instead of retraining (replot)")
    return p.parse_args()


def main() -> None:
    a = _parse()
    a.outdir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import (FlowConfig, load_flow, log_prob, log_prob_and_grad,
                                       save_flow, to_latent, train_flow)

    bias = SmoothTICABias.load(a.bias_npz)
    kbt = float(bias.kbt_kcal_mol)
    ref = np.load(a.reference)
    R = ref["R"]
    z = np.asarray(bias.projection.transform(R), dtype=np.float64)[:, :a.n_dims]
    print(f"reference {R.shape[0]} frames -> z {z.shape}, "
          f"std {np.round(z.std(0), 3).tolist()}, kT {kbt:.4f} kcal/mol")

    # ---- train one flow per seed ------------------------------------------------------
    cfgs, params_all, hists = [], [], []
    for s in range(a.seeds):
        path = a.outdir / f"flow_seed{s}.npz"
        if a.reuse and path.exists():
            p, cfg = load_flow(path)
            h = dict(best_val_nll=float("nan"), best_step=-1, history=[])
            print(f"[seed {s}] reusing {path.name}")
        else:
            cfg = FlowConfig(n_dims=a.n_dims, n_layers=a.layers, n_bins=a.bins,
                             hidden=a.hidden, seed=s)
            print(f"\n[seed {s}]")
            p, h = train_flow(z, cfg, steps=a.steps, batch=a.batch, lr=a.lr,
                              report_every=max(1, a.steps // 50))
            save_flow(p, cfg, path,
                      tica_pairs=bias.projection.pairs, tica_mean=bias.projection.mean,
                      tica_coefficients=bias.projection.coefficients, kbt_kcal_mol=kbt)
        cfgs.append(cfg); params_all.append(p); hists.append(h)

    ev = evaluate_flows(bias, z, params_all, cfgs, grid=a.grid, min_count=a.min_count,
                        n_dims=a.n_dims)
    gx, gy, solid, empty = ev["gx"], ev["gy"], ev["solid"], ev["empty"]
    F_hist, F_kde, F_flow = ev["F_hist"], ev["F_kde"], ev["F_flow"]
    gnorm, spread = ev["gnorm"], ev["spread"]
    f_flow, f_kde, f_err = ev["f_flow"], ev["f_kde"], ev["f_err"]
    pcts = lambda x, m: {f"p{q}": float(np.percentile(x[m], q)) for q in (50, 90, 99, 100)}
    cos = ev["median_pairwise_cosine"]

    summary = dict(
        n_frames=int(len(z)), n_dims=int(a.n_dims), kbt_kcal_mol=kbt,
        flow=dict(layers=a.layers, bins=a.bins, hidden=a.hidden, steps=a.steps,
                  seeds=a.seeds, params_per_flow=int(sum(
                      w.size + b.size for p in [params_all[0]] for mlp in p["mlps"]
                      for w, b in mlp))),
        held_out_nll=[h["best_val_nll"] for h in hists],
        best_step=[h["best_step"] for h in hists],
        fes_vs_histogram=dict(kde=ev["fes_kde"], flow=ev["fes_flow"]),
        gradient_reproducibility=dict(
            units="kcal/mol per TICA unit",
            seed_force_error_solid=pcts(f_err, solid),
            seed_force_error_empty=pcts(f_err, empty),
            flow_force_solid=pcts(f_flow, solid), kde_force_solid=pcts(f_kde, solid),
            flow_force_empty=pcts(f_flow, empty), kde_force_empty=pcts(f_kde, empty),
            frac_solid_err_gt=ev["frac_solid_err_gt"],
            median_pairwise_cosine=cos),
        cells=ev["cells"],
        grid=int(a.grid), min_count=int(a.min_count),
    )
    (a.outdir / "history.json").write_text(json.dumps([h["history"] for h in hists], indent=1))

    # ---- latent ensemble (Phase 1b) -----------------------------------------------------
    u = np.asarray(to_latent(params_all[0], cfgs[0], jnp.asarray(z, jnp.float32)), np.float64)
    r = np.linalg.norm(u, axis=-1)
    # for d=2 a perfect fit gives |u| ~ Rayleigh: mean sqrt(pi/2)=1.2533, and u ~ N(0,I)
    summary["latent"] = dict(
        mean=u.mean(0).round(4).tolist(),
        cov=np.cov(u.T).round(4).tolist(),
        radial_mean=float(r.mean()), radial_mean_ideal=float(np.sqrt(np.pi / 2)),
        frac_beyond_3sigma=float((r > 3.0).mean()),
        frac_beyond_3sigma_ideal=float(np.exp(-4.5)),
    )
    np.save(a.outdir / "u_reference.npy", u.astype(np.float32))
    np.save(a.outdir / "z_reference.npy", z.astype(np.float32))

    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    _figures(a, gx, gy, F_hist, F_kde, F_flow, gnorm, spread, solid, u, z, bias,
             f_flow=f_flow, f_kde=f_kde, f_err=f_err, empty=empty, hists=hists)

    # ---- report --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"held-out NLL per seed : {[round(v, 4) for v in summary['held_out_nll']]}")
    print(f"FES vs histogram      : KDE  rmse {summary['fes_vs_histogram']['kde']['rmse_kcal']:.3f} "
          f"max {summary['fes_vs_histogram']['kde']['max_kcal']:.3f} kcal/mol")
    for i, f in enumerate(summary["fes_vs_histogram"]["flow"]):
        print(f"                        flow{i} rmse {f['rmse_kcal']:.3f} "
              f"max {f['max_kcal']:.3f} kcal/mol")
    gr = summary["gradient_reproducibility"]
    print("GRADIENT reproducibility (the acceptance metric) -- kcal/mol per TICA unit")
    print(f"{'':26s}{'median':>9s}{'p90':>9s}{'p99':>9s}{'max':>9s}")
    for name, key in (("flow |force| solid", "flow_force_solid"),
                      ("KDE  |force| solid", "kde_force_solid"),
                      ("SEED ERROR  solid", "seed_force_error_solid"),
                      ("SEED ERROR  empty", "seed_force_error_empty")):
        d = gr[key]
        print(f"  {name:24s}" + "".join(f"{d[k]:9.2f}" for k in ("p50", "p90", "p99", "p100")))
    print(f"  frac solid cells with seed error > 1.0 : "
          f"{gr['frac_solid_err_gt']['1.0']:.3f}")
    print(f"  median pairwise cosine      : {[round(c, 4) for c in gr['median_pairwise_cosine']]}")
    lat = summary["latent"]
    print(f"latent u: mean {lat['mean']}  cov {lat['cov']}")
    print(f"          <|u|> {lat['radial_mean']:.4f} vs ideal {lat['radial_mean_ideal']:.4f}; "
          f"P(|u|>3) {lat['frac_beyond_3sigma']:.5f} vs ideal {lat['frac_beyond_3sigma_ideal']:.5f}")
    print(f"wrote {a.outdir}/summary.json and figures")


def evaluate_flows(bias, z, params_all, cfgs, *, grid: int, min_count: int, n_dims: int):
    """Grid metrics shared by the single-run driver and the capacity sweep.

    Returns a dict of scalars plus the arrays the figures need. Keeping this in ONE place is
    what lets a sweep compare configurations on exactly the protocol Phase 1 was judged by --
    a sweep that quietly re-derives its own metric is not comparable to the run it follows.
    """
    import jax.numpy as jnp
    from sampling.flow_density import log_prob_and_grad

    kbt = float(bias.kbt_kcal_mol)
    # ---- validation grid --------------------------------------------------------------
    lo, hi = bias.bounds[:n_dims, 0], bias.bounds[:n_dims, 1]
    pad = 0.15 * (hi - lo)
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    pts_j = jnp.asarray(pts, jnp.float32)

    # empirical FES from a plain 2D histogram -- the ground truth for shape
    H, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=[grid, grid],
                             range=[[gx[0], gx[-1]], [gy[0], gy[-1]]])
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    p_hist = H / (H.sum() * cell)
    solid = H >= min_count                       # cells with enough counts to trust

    # KDE reference density AND its score, same grid (this is what the flow replaces)
    kde = [bias._log_density_and_gradient(q, bias.centers, bias.reference_weights,
                                          bias.bandwidth) for q in pts]
    lp_kde = np.array([k[0] for k in kde]).reshape(GX.shape)
    grad_kde = np.array([k[1] for k in kde]).reshape(GX.shape + (n_dims,))

    # flow densities and score fields, per seed
    lp_flow, grad_flow = [], []
    for p, cfg in zip(params_all, cfgs):
        lp, g = log_prob_and_grad(p, cfg, pts_j)
        lp_flow.append(np.asarray(lp, np.float64).reshape(GX.shape))
        grad_flow.append(np.asarray(g, np.float64).reshape(GX.shape + (n_dims,)))

    def fes(logp):                                  # free energy, min-shifted, kcal/mol
        F = -kbt * logp
        return F - F[solid].min()

    F_hist = np.where(p_hist > 0, -kbt * np.log(np.maximum(p_hist, 1e-300)), np.nan)
    F_hist = F_hist - np.nanmin(F_hist[solid])
    F_kde, F_flow = fes(lp_kde), [fes(l) for l in lp_flow]

    def dev(F):                                     # deviation from the histogram FES
        d = (F - F_hist)[solid]
        d = d - d.mean()                            # free energies are defined up to a constant
        return float(np.sqrt((d ** 2).mean())), float(np.abs(d).max())

    # ---- gradient reproducibility: THE acceptance metric --------------------------------
    # Reported in PHYSICAL units, kcal/mol per TICA unit, not as a relative spread. A relative
    # metric divides by |grad log p|, which VANISHES at every density maximum -- i.e. at the
    # basin cores -- so it reports 0/0 exactly where the fit is best. Measured: the naive
    # relative number was 0.148 and its map was a bright ridge straight through both basins.
    G = np.stack(grad_flow)                         # (seeds, nx, ny, d)
    gmean = G.mean(0)
    gnorm = np.linalg.norm(gmean, axis=-1)
    # RMS across seeds, NOT max-over-seeds. A max grows mechanically with the number of
    # seeds -- measured: the same configurations scored 1.84/0.57/1.48 on 3 seeds and
    # 2.89/1.11/2.24 on 6, purely from the extra draws -- so a max cannot be compared between
    # runs with different --seeds, which is exactly what a sweep needs to do. The RMS is a
    # population spread and is stable in n.
    spread = np.sqrt((np.linalg.norm(G - gmean, axis=-1) ** 2).mean(0))
    empty = H == 0
    f_flow = kbt * gnorm                            # |bias force| scale from the flow
    f_kde = kbt * np.linalg.norm(grad_kde, axis=-1)
    f_err = kbt * spread                            # force error attributable to seed choice

    def pcts(x, m):
        v = x[m]
        return {f"p{q}": float(np.percentile(v, q)) for q in (50, 90, 99, 100)}

    cos = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            num = (G[i] * G[j]).sum(-1)
            den = np.linalg.norm(G[i], axis=-1) * np.linalg.norm(G[j], axis=-1) + 1e-12
            cos.append(float(np.median((num / den)[solid])))


    def pcts(x, m):
        v = x[m]
        return {f"p{q}": float(np.percentile(v, q)) for q in (50, 90, 99, 100)}

    return dict(gx=gx, gy=gy, solid=solid, empty=empty, H=H,
                F_hist=F_hist, F_kde=F_kde, F_flow=F_flow,
                gnorm=gnorm, spread=spread, f_flow=f_flow, f_kde=f_kde, f_err=f_err,
                fes_kde=dict(zip(("rmse_kcal", "max_kcal"), dev(F_kde))),
                fes_flow=[dict(zip(("rmse_kcal", "max_kcal"), dev(F))) for F in F_flow],
                seed_err_solid=pcts(f_err, solid), seed_err_empty=pcts(f_err, empty),
                flow_force_solid=pcts(f_flow, solid), kde_force_solid=pcts(f_kde, solid),
                flow_force_empty=pcts(f_flow, empty), kde_force_empty=pcts(f_kde, empty),
                frac_solid_err_gt={str(t): float((f_err[solid] > t).mean())
                                   for t in (0.5, 1.0, 5.0)},
                median_pairwise_cosine=cos,
                cells=dict(solid=int(solid.sum()), empty=int(empty.sum()), total=int(H.size)))


def _figures(a, gx, gy, F_hist, F_kde, F_flow, gnorm, spread, solid, u, z, bias,
             *, f_flow, f_kde, f_err, empty, hists) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    kw = dict(origin="lower", extent=ext, aspect="auto")

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    vmax = float(a.fes_vmax)
    for k, (F, t) in enumerate([(F_hist, "histogram (truth)"), (F_kde, "KDE (current)"),
                                (F_flow[0], "flow seed 0"), (F_flow[-1], "flow seed last")]):
        im = ax[k].imshow(np.clip(F, 0, vmax).T, vmin=0, vmax=vmax, cmap="viridis", **kw)
        ax[k].contour(gx, gy, np.clip(np.nan_to_num(F, nan=vmax), 0, vmax).T,
                      levels=np.arange(1.0, vmax, 1.0), colors="w", linewidths=0.4, alpha=0.6)
        ax[k].set_title(f"F(z), {t}"); ax[k].set_xlabel("tic1")
        plt.colorbar(im, ax=ax[k], label="kcal/mol")
    ax[0].set_ylabel("tic2")
    fig.tight_layout(); fig.savefig(a.outdir / "fig1_fes_comparison.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    im = ax[0].imshow(gnorm.T, origin="lower", extent=ext, aspect="auto", cmap="magma")
    ax[0].set_title(r"$|\nabla \log p_\theta|$ (seed mean)"); plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(spread.T, origin="lower", extent=ext, aspect="auto", cmap="inferno")
    ax[1].set_title("worst seed-to-seed gradient disagreement"); plt.colorbar(im, ax=ax[1])
    rel = np.where(solid, spread / (gnorm + 1e-12), np.nan)
    im = ax[2].imshow(rel.T, origin="lower", extent=ext, aspect="auto", cmap="inferno",
                      vmin=0, vmax=1)
    ax[2].set_title("relative spread (supported cells)"); plt.colorbar(im, ax=ax[2])
    for k in range(3):
        ax[k].set_xlabel("tic1")
    ax[0].set_ylabel("tic2")
    fig.tight_layout(); fig.savefig(a.outdir / "fig2_gradient_field.png", dpi=130); plt.close(fig)

    # latent ensemble
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    sub = np.random.default_rng(0).choice(len(u), size=min(40000, len(u)), replace=False)
    ax[0].hexbin(u[sub, 0], u[sub, 1], gridsize=90, cmap="Blues", bins="log")
    th = np.linspace(0, 2 * np.pi, 200)
    for rr in (1, 2, 3):
        ax[0].plot(rr * np.cos(th), rr * np.sin(th), "r-", lw=0.8, alpha=0.7)
    ax[0].set_title("latent $u=f_\\theta(z)$ (red: 1,2,3$\\sigma$)")
    ax[0].set_xlabel("$u_1$"); ax[0].set_ylabel("$u_2$"); ax[0].set_aspect("equal")

    r = np.linalg.norm(u, axis=-1)
    rr = np.linspace(0, max(5.0, r.max()), 200)
    ax[1].hist(r, bins=160, density=True, alpha=0.65, label="observed")
    ax[1].plot(rr, rr * np.exp(-rr ** 2 / 2), "r-", lw=1.6, label="Rayleigh (ideal)")
    ax[1].set_xlabel("$|u|$"); ax[1].set_yscale("log"); ax[1].legend()
    ax[1].set_title("radial density: tails are where the flow is least constrained")

    sc = ax[2].scatter(u[sub, 0], u[sub, 1], c=z[sub, 0], s=1.5, cmap="Spectral", alpha=0.6)
    ax[2].set_title("latent coloured by tic1"); ax[2].set_aspect("equal")
    ax[2].set_xlabel("$u_1$"); plt.colorbar(sc, ax=ax[2], label="tic1")
    fig.tight_layout(); fig.savefig(a.outdir / "fig3_latent_ensemble.png", dpi=130); plt.close(fig)

    # ---- fig4: training curves -----------------------------------------------------------
    if any(h["history"] for h in hists):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for s, h in enumerate(hists):
            if not h["history"]:
                continue
            st = [r["step"] for r in h["history"]]
            ax[0].plot(st, [r["train_nll"] for r in h["history"]], alpha=0.55, label=f"seed {s}")
            ax[1].plot(st, [r["val_nll"] for r in h["history"]], marker="o", ms=3,
                       label=f"seed {s}")
            if h["best_step"] > 0:
                ax[1].axvline(h["best_step"], ls=":", lw=0.8, alpha=0.5)
        for k, t in enumerate(("training NLL (minibatch)", "held-out NLL  (dotted = best)")):
            ax[k].set_title(t); ax[k].set_xlabel("step"); ax[k].set_ylabel("NLL"); ax[k].legend()
        # Zoom on the plateau: the flow converges in a few hundred steps, so a full-range
        # y-axis compresses everything that matters into one flat line. The plateau SCATTER
        # is the interesting quantity -- "best val" is selected from it, and if that scatter
        # is comparable to the seed-to-seed difference then checkpoint choice is noise.
        allv = np.array([r["val_nll"] for h in hists for r in h["history"]])
        fin = allv[allv < np.percentile(allv, 60)]
        if len(fin):
            ax[0].set_ylim(fin.min() - 0.05, fin.max() + 0.05)
            ax[1].set_ylim(fin.min() - 0.004, fin.max() + 0.004)
            ax[1].set_title("held-out NLL, zoomed on plateau (dotted = best)")
        fig.tight_layout(); fig.savefig(a.outdir / "fig4_training_curves.png", dpi=130)
        plt.close(fig)

    # ---- fig5: bias-force error, the acceptance metric -----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))
    from matplotlib.colors import LogNorm
    m = np.maximum(f_err, 1e-3)
    im = ax[0].imshow(m.T, origin="lower", extent=ext, aspect="auto", cmap="inferno",
                      norm=LogNorm(vmin=1e-2, vmax=max(10.0, float(m.max()))))
    ax[0].contour(gx, gy, (~empty).T.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
    ax[0].set_title("seed-to-seed BIAS FORCE error\n(cyan: edge of sampled support)")
    plt.colorbar(im, ax=ax[0], label="kcal/mol per tic")

    im = ax[1].imshow((f_flow / np.maximum(f_kde, 1e-6)).T, origin="lower", extent=ext,
                      aspect="auto", cmap="coolwarm", norm=LogNorm(vmin=0.1, vmax=10.0))
    ax[1].contour(gx, gy, (~empty).T.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    ax[1].set_title("|force| flow / KDE\n(red: flow steeper)")
    plt.colorbar(im, ax=ax[1], label="ratio")
    for k in (0, 1):
        ax[k].set_xlabel("tic1")
    ax[0].set_ylabel("tic2")

    bins = np.logspace(-3, 3, 90)
    ax[2].hist(f_err[solid], bins=bins, density=True, alpha=0.6, label="seed error, supported")
    ax[2].hist(f_err[empty], bins=bins, density=True, alpha=0.45, label="seed error, empty")
    ax[2].axvline(np.median(f_kde[solid]), color="k", ls="--", lw=1.4,
                  label=f"KDE |force| median ({np.median(f_kde[solid]):.1f})")
    ax[2].axvline(np.median(f_flow[solid]), color="g", ls="--", lw=1.4,
                  label=f"flow |force| median ({np.median(f_flow[solid]):.1f})")
    ax[2].set_xscale("log"); ax[2].set_xlabel("kcal/mol per tic"); ax[2].legend(fontsize=8)
    ax[2].set_title("is the seed error small vs the force it perturbs?")
    fig.tight_layout(); fig.savefig(a.outdir / "fig5_force_error.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
