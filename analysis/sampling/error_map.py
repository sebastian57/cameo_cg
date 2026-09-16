#!/usr/bin/env python3
"""MD-less spatial error map: where does a model's energy error sit, and how big is it?

For each trained checkpoint, evaluate U(R) on equilibrium REFERENCE frames, form
dU(R) = U_model(R) - U_baseline(R), then aggregate into cells of the acquisition-flow
latent space u = f_theta(TICA(R)) (equal latent volume = equal reference probability,
see DESIGN/LATENT_ENSEMBLE_DIAGNOSTICS.md):

  dU_mean(cell)   first-order log-population shift of the cell  (~ -<dU>/kT)
  dF(cell)        -kT ln <exp(-beta dU)>   exact cell free-energy shift under the
                  assumption that reference frames sample the cell's density
  disagree(cell)  std of dU_mean across the supplied checkpoints (seed uncertainty)

No MD is involved: this localises energy error the way `delta` does (basin-level), but
per latent cell, so acquisition targets = high |dF| x high reference-mass cells can be
read off directly. All inputs derive from existing trajectories / trained models.

Usage (from cameo_cg root):
  python -m sampling.error_map --outdir <dir> \
      --model v4=<config.yaml>:<params.pkl> [--model v3=<...>:<...>] \
      [--baseline <config.yaml>:<params.pkl>] [--n-frames 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc
from analysis.sampling.diagnostics_common import (
    assign_regions, cell_stats, load_latent, parse_spec, selfcheck_cell_stats,
)


def _log(msg):
    print(msg, flush=True)


def _jsonable_path(value):
    """Convert argparse/path objects used in JSON provenance to strings."""
    return str(value) if isinstance(value, Path) else value


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--model", action="append", default=[],
                    help="LABEL=config.yaml:params.pkl (repeatable)")
    ap.add_argument("--baseline", required=True,
                    help="config.yaml:params.pkl of the reference baseline model")
    dc.add_latent_arguments(ap)
    ap.add_argument("--n-frames", type=int, default=20000)
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = dc.config_from_args(a)
    if a.selfcheck:
        selfcheck_cell_stats()
        if not a.model:
            return
    if not a.model:
        raise SystemExit("need at least one --model (or --selfcheck alone)")

    import jax
    import jax.numpy as jnp
    from analysis.md.analyze_model_residuals_by_region import _load_model          # noqa: E402
    from sampling.mapping import get_mapping                           # noqa: E402

    a.outdir.mkdir(parents=True, exist_ok=True)

    # ---- frames -> TICA -> latent u ----------------------------------------------------
    lat = load_latent(a.frames, a.n_frames, a.grid_lim, a.bins, config=cfg)
    R, u, reg = lat["R"], lat["u"], lat["region"]
    inside, flat = lat["inside"], lat["flat"]
    edges, centers, bins = lat["edges"], lat["centers"], lat["bins"]
    mapping = lat["mapping"]
    n_cells = bins * bins

    # ---- model energies -----------------------------------------------------------------
    def eval_U(spec):
        cfg, par = parse_spec(spec)
        t0 = time.time()
        model, params, mask0, species0 = _load_model(cfg, par, a.frames, mapping.n_beads)
        ev = jax.jit(jax.vmap(lambda r: model.compute_energy(params, r, mask0, species0)))
        Rf = R.astype(np.float32)[inside]
        vals = [np.asarray(ev(jnp.asarray(Rf[i:i + 512])), np.float64)
                for i in range(0, len(Rf), 512)]
        full = np.full(len(R), np.nan)
        full[inside] = np.concatenate(vals)
        _log(f"[eval] {spec.split('/')[-1]} {time.time()-t0:.1f}s")
        return full

    U_base = eval_U(a.baseline)
    U_models, labels = {}, []
    for spec in a.model:
        lab = str(spec).partition("=")[0]
        labels.append(lab)
        U_models[lab] = eval_U(str(spec).partition("=")[2])

    # ---- per-cell maps -------------------------------------------------------------------
    res, du_means, du_centred = {}, {}, {}
    for lab in labels:
        ok_all = np.isfinite(U_models[lab]) & np.isfinite(U_base)
        dU = (U_models[lab] - U_base)
        # FM constrains gradients only: the absolute energy intercept is unidentifiable
        # and drifts freely between checkpoints (measured: -236 kcal/mol between two
        # bb6 models on identical frames). A constant cancels out of the Boltzmann
        # distribution, so centre it away or the whole map is one meaningless offset.
        dU = dU - dU[inside].mean()
        du_centred[lab] = dU
        ok = ok_all & inside
        dm, df, n_c = cell_stats(dU[ok], flat[ok], n_cells, a.min_count, kT=cfg.kT)
        du_means[lab], res[lab] = dm, dict(dU_mean=dm, dF=df, count=n_c)

    # regression tie-in with the established delta screen (basin-level, same frames).
    # Uses the SAME centred dU as the maps so table and figures are consistent.
    basins = {}
    for b in ("beta", "alphaR", "alphaL", "other"):
        mb = ok & (reg == b)
        basins[b] = {"ref_pct": float(100.0 * (reg == b).mean()), "n": int(mb.sum())}
        for lab in labels:
            dUb = du_centred[lab][mb]
            basins[b][f"dU_mean_{lab}"] = float(dUb.mean())
            basins[b][f"dF_{lab}"] = float(-cfg.kT * np.log(np.exp(-dUb / cfg.kT).mean()))
    if {"beta", "alphaR"} <= set(basins):
        for lab in labels:
            basins[f"delta_{lab}"] = (basins["alphaR"][f"dU_mean_{lab}"]
                                      - basins["beta"][f"dU_mean_{lab}"])

    stack = np.stack([du_means[l] for l in labels])
    disagree = (stack.std(axis=0, ddof=0) if len(labels) > 1
                else np.full(n_cells, np.nan))

    # top candidate cells: |dF| weighted by reference mass (acquisition targets)
    mass = res[labels[0]]["count"] / max(res[labels[0]]["count"].sum(), 1)
    score = np.abs(res[labels[0]]["dF"]) * mass
    score[np.isnan(score)] = -1
    order = np.argsort(score)[::-1][:20]
    cx, cy = np.meshgrid(centers, centers, indexing="ij")
    top = [dict(u1=float(cx.flat[i]), u2=float(cy.flat[i]), n=int(res[labels[0]]["count"][i]),
                dF=float(res[labels[0]]["dF"][i]),
                dU_mean=float(du_means[labels[0]][i])) for i in order if score[i] > 0]

    summary = {
        "provenance": {
            "frames": _jsonable_path(a.frames), "n_frames": int(len(R)), "inside_grid": int(inside.sum()),
            "baseline": parse_spec(a.baseline), "models": {l: parse_spec(s) for s, l in
                                                      zip(a.model, labels)},
            "flow": str(cfg.flow_path()), "bias_npz": _jsonable_path(cfg.bias_npz),
            "kT_kcal_mol": cfg.kT, "bins": a.bins, "grid_lim": a.grid_lim,
        },
        "basin_table": basins,
        "top20_cells_by_absdF_x_mass_label_" + labels[0]: top,
    }
    (a.outdir / "error_map.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(a.outdir / "error_map.npz", edges=edges,
                        u=u[inside], region=reg[inside],
                        **{f"{l}_{k}": v for l in labels
                           for k, v in res[l].items()},
                        disagree=disagree)
    _log(f"[delta-screen cross-check] " +
         " ".join(f"{l}: {basins.get('delta_' + l, float('nan')):+.3f}" for l in labels))
    _log(f"[top cells ({labels[0]}): highest |dF| x ref-mass]")
    for t in top[:8]:
        _log(f"  u=({t['u1']:+.2f},{t['u2']:+.2f}) n={t['n']:5d} "
             f"dF={t['dF']:+7.3f} kcal/mol")

    plot(a.outdir, edges, centers,
         [(l, res[l]["dF"]) for l in labels], disagree, reg[inside], u[inside])
    _log(f"wrote {a.outdir}/error_map.json .npz .png")


def plot(outdir, edges, centers, maps, disagree, reg, u):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_panels = len(maps) + (1 if np.isfinite(disagree).any() else 0)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(4.6 * max(n_panels, 1), 4.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    lim = edges[-1]
    fin_maps = [(l, m[np.isfinite(m)]) for l, m in maps]
    vmax = max(np.nanpercentile(np.abs(m), 98) for _, m in fin_maps) or 1.0
    for b, mk in (("beta", "o"), ("alphaR", "s"), ("alphaL", "^")):
        m = reg == b
        if m.sum():
            for ax in axes:
                ax.plot(u[m][:, 0].mean(), u[m][:, 1].mean(), mk, ms=9, mec="w", mew=1.2)
            axes[0].annotate(b, (u[m][:, 0].mean(), u[m][:, 1].mean()), fontsize=8,
                             xytext=(3, 3), textcoords="offset points")
    im = None
    for ax, (lab, dF) in zip(axes, maps):
        g = dF.reshape(len(centers), len(centers))
        im = ax.pcolormesh(edges, edges, g.T, cmap="RdBu_r",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                           rasterized=True)
        ax.set_title(f"{lab}: per-cell $\\Delta F$")
    if np.isfinite(disagree).any():
        axes[len(maps)].pcolormesh(edges, edges,
                                   disagree.reshape(len(centers), len(centers)).T,
                                   cmap="viridis", rasterized=True)
        axes[len(maps)].set_title("model disagreement (std of $\\langle dU\\rangle$)")
    fig.colorbar(im, ax=axes, label="kcal/mol", shrink=0.85)
    for ax in axes:
        ax.set_aspect("equal"), ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_xlabel("$u_1$"), ax.set_ylabel("$u_2$")
    fig.suptitle("$\\Delta F(\\mathrm{cell}) = -kT\\ln\\langle e^{-\\beta\\,dU}\\rangle$, "
                 "energy-intercept-centred, on reference frames", fontsize=10)
    fig.savefig(outdir / "error_map.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
