#!/usr/bin/env python3
"""Capacity/regularisation sweep for the TICA normalizing flow.

    python -m sampling.sweep_flow_capacity --bias-npz <bias.npz> --reference <ref.npz> \
        --outdir local_work/flow_sweep --seeds 3 --steps 1500

WHAT THIS IS FOR
    Phase 1 left one question open: is the score field `grad log p_theta` reproducible enough
    to carry an MD bias? The first answer was contaminated, because "best validation" was being
    selected from plateau noise -- two identical runs differed by ~20% on the acceptance
    statistic. Training now uses SWA over the plateau, so a sweep finally measures CAPACITY
    rather than selection noise.

THE TRADE-OFF BEING MEASURED
    More capacity fits the density better (lower FES error) but is free to place gradient
    structure the data does not constrain, which shows up as seed-to-seed DISAGREEMENT in the
    score field. The useful configuration is the one that minimises disagreement without
    giving away the density advantage over KDE -- not the one with the best likelihood.

    Reported in kcal/mol per TICA unit, i.e. the units of the bias force the flow would apply,
    and always against the incumbent KDE's own score magnitude. A relative metric is useless
    here: `|grad log p|` vanishes at every density maximum, so it reports 0/0 in the basins.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: (label, layers, bins, hidden, weight_decay). Ordered by parameter count, and deliberately
#: reaching BELOW the Phase 1 configuration -- the hypothesis is that Phase 1 was too big.
GRID = [
    ("tiny",    3,  4,  24, 1e-4),
    ("small",   4,  4,  32, 1e-4),
    ("small+",  4,  8,  32, 1e-4),
    ("base",    6,  8,  64, 1e-4),     # the Phase 1 configuration
    ("base-wd", 6,  8,  64, 1e-2),     # same capacity, stronger regularisation
    ("fine",    6, 16,  64, 1e-4),     # more spline bins
    ("deep",    8,  8,  64, 1e-4),
    ("wide",    6,  8, 128, 1e-4),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bias-npz", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-dims", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid", type=int, default=241)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--only", default=None, help="comma-separated labels to run")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.build_flow_density import evaluate_flows
    from sampling.flow_density import FlowConfig, save_flow, train_flow

    bias = SmoothTICABias.load(a.bias_npz)
    z = np.asarray(bias.projection.transform(np.load(a.reference)["R"]),
                   dtype=np.float64)[:, :a.n_dims]
    print(f"reference {len(z)} frames, d={a.n_dims}, kT={bias.kbt_kcal_mol:.4f}\n")

    wanted = set(a.only.split(",")) if a.only else None
    rows = []
    for label, layers, bins, hidden, wd in GRID:
        if wanted and label not in wanted:
            continue
        t0 = time.time()
        params_all, cfgs, nlls = [], [], []
        for s in range(a.seeds):
            cfg = FlowConfig(n_dims=a.n_dims, n_layers=layers, n_bins=bins,
                             hidden=hidden, seed=s)
            p, h = train_flow(z, cfg, steps=a.steps, batch=a.batch, lr=a.lr,
                              weight_decay=wd, report_every=a.steps,  # quiet
                              log=lambda _s: None)
            params_all.append(p); cfgs.append(cfg); nlls.append(h["swa_val_nll"])
            save_flow(p, cfg, a.outdir / f"flow_{label}_seed{s}.npz")
        ev = evaluate_flows(bias, z, params_all, cfgs, grid=a.grid,
                            min_count=a.min_count, n_dims=a.n_dims)
        nparams = int(sum(w.size + b.size for mlp in params_all[0]["mlps"] for w, b in mlp))
        rows.append(dict(
            label=label, layers=layers, bins=bins, hidden=hidden, weight_decay=wd,
            n_params=nparams, seconds=round(time.time() - t0, 1),
            val_nll=float(np.median(nlls)),
            fes_rmse=float(np.median([f["rmse_kcal"] for f in ev["fes_flow"]])),
            fes_max=float(np.median([f["max_kcal"] for f in ev["fes_flow"]])),
            seed_err_p50=ev["seed_err_solid"]["p50"], seed_err_p90=ev["seed_err_solid"]["p90"],
            seed_err_max=ev["seed_err_solid"]["p100"],
            seed_err_empty_p50=ev["seed_err_empty"]["p50"],
            seed_err_empty_max=ev["seed_err_empty"]["p100"],
            frac_gt1=ev["frac_solid_err_gt"]["1.0"],
            flow_force_p50=ev["flow_force_solid"]["p50"],
            kde_force_p50=ev["kde_force_solid"]["p50"],
            kde_fes_rmse=ev["fes_kde"]["rmse_kcal"]))
        r = rows[-1]
        print(f"{label:9s} params {nparams:6d}  FESrmse {r['fes_rmse']:.3f}  "
              f"seedErr p50 {r['seed_err_p50']:6.3f}  p90 {r['seed_err_p90']:6.3f}  "
              f"frac>1 {r['frac_gt1']:.3f}  ({r['seconds']:.0f}s)")

    (a.outdir / "sweep.json").write_text(json.dumps(rows, indent=1))
    _report(rows, a.outdir)


def _report(rows, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kde_force = rows[0]["kde_force_p50"]
    kde_fes = rows[0]["kde_fes_rmse"]
    order = np.argsort([r["n_params"] for r in rows])
    rows = [rows[i] for i in order]
    x = [r["n_params"] for r in rows]
    lab = [r["label"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    ax[0].plot(x, [r["seed_err_p50"] for r in rows], "o-", lw=2, label="median")
    ax[0].plot(x, [r["seed_err_p90"] for r in rows], "s--", lw=1.4, label="p90")
    ax[0].axhline(kde_force, color="k", ls=":", lw=1.6,
                  label=f"KDE |force| median ({kde_force:.1f})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("flow parameters"); ax[0].set_ylabel("kcal/mol per tic")
    ax[0].set_title("seed-to-seed BIAS FORCE error\n(lower = the score field is determined)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
    for xx, ll, r in zip(x, lab, rows):
        ax[0].annotate(ll, (xx, r["seed_err_p50"]), fontsize=7,
                       textcoords="offset points", xytext=(3, -10))

    ax[1].plot(x, [r["fes_rmse"] for r in rows], "o-", lw=2, color="tab:green")
    ax[1].axhline(kde_fes, color="k", ls=":", lw=1.6, label=f"KDE ({kde_fes:.3f})")
    ax[1].set_xscale("log"); ax[1].set_xlabel("flow parameters")
    ax[1].set_ylabel("FES RMSE vs histogram (kcal/mol)")
    ax[1].set_title("density fit\n(lower = better; all beat KDE)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    # the actual decision: reproducibility against density fit
    sc = ax[2].scatter([r["fes_rmse"] for r in rows], [r["seed_err_p50"] for r in rows],
                       c=np.log10(x), cmap="viridis", s=90, zorder=3)
    for r in rows:
        ax[2].annotate(r["label"], (r["fes_rmse"], r["seed_err_p50"]), fontsize=8,
                       textcoords="offset points", xytext=(6, 4))
    ax[2].axvline(kde_fes, color="k", ls=":", lw=1.4)
    ax[2].axhline(kde_force, color="k", ls="--", lw=1.4)
    ax[2].text(kde_fes * 1.02, ax[2].get_ylim()[1] * 0.95, "KDE density fit", fontsize=7,
               rotation=90, va="top")
    ax[2].set_xlabel("FES RMSE (kcal/mol)  -- better ->".replace("->", "←"))
    ax[2].set_ylabel("seed force error (kcal/mol per tic)")
    ax[2].set_title("the trade-off\nbottom-left is best")
    ax[2].set_yscale("log"); ax[2].grid(alpha=.3)
    plt.colorbar(sc, ax=ax[2], label="log10(parameters)")

    fig.tight_layout()
    fig.savefig(outdir / "fig_capacity_sweep.png", dpi=130)
    plt.close(fig)

    print(f"\n{'config':9s} {'params':>7s} {'valNLL':>8s} {'FESrmse':>8s} {'seedErr50':>10s} "
          f"{'seedErr90':>10s} {'frac>1':>7s} {'emptyMax':>9s}")
    for r in rows:
        print(f"{r['label']:9s} {r['n_params']:>7d} {r['val_nll']:>8.4f} {r['fes_rmse']:>8.3f} "
              f"{r['seed_err_p50']:>10.3f} {r['seed_err_p90']:>10.3f} {r['frac_gt1']:>7.3f} "
              f"{r['seed_err_empty_max']:>9.1f}")
    print(f"{'KDE':9s} {'-':>7s} {'-':>8s} {kde_fes:>8.3f} {'-':>10s} {'-':>10s} {'-':>7s} "
          f"{'-':>9s}   (|force| median {kde_force:.2f})")
    best = min(rows, key=lambda r: r["seed_err_p50"])
    print(f"\nmost reproducible score field: {best['label']} "
          f"({best['seed_err_p50']:.3f} kcal/mol/tic, FES rmse {best['fes_rmse']:.3f})")
    print(f"wrote {outdir}/fig_capacity_sweep.png")


if __name__ == "__main__":
    main()
