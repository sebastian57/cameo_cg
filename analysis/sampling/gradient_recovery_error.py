#!/usr/bin/env python3
"""Zienkiewicz-Zhu style gradient-recovery error indicator (MD-less, baseline-free).

For every reference frame the model force F_i = -grad_R U(R_i) is evaluated.
F_i is then compared against the k-nearest-neighbour mean of the OTHER frames'
forces in acquisition-flow latent space. Frames whose force disagrees with
their latent neighbours carry a large indicator |F_i - <F>_kNN| — the
scattered-data analogue of the FEM gradient-recovery estimator. No baseline
model is involved, so the map answers "where is this model internally
inconsistent", not "where does it differ from another model".

Usage:
  python -m sampling.gradient_recovery_error --outdir <dir> \
      --model v4=<config.yaml>:<params.pkl> [--model ...] [--n-frames 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc
from analysis.sampling.diagnostics_common import load_latent, pair_distance_features, parse_spec


def _log(msg):
    print(msg, flush=True)


def _jsonable_path(value):
    """Convert argparse/path objects used in JSON provenance to strings."""
    return str(value) if isinstance(value, Path) else value


def knn_indicator(u, F, k=16):
    """Per-frame |F - mean_kNN(F)| / sqrt(dof) among latent points u."""
    from scipy.spatial import cKDTree
    dof = F.shape[1]
    tree = cKDTree(u)
    _, idx = tree.query(u, k=k + 1)
    nb = idx[:, 1:]                                   # exclude self
    F_smooth = F[nb].mean(axis=1)
    return np.linalg.norm(F - F_smooth, axis=1) / np.sqrt(dof), F_smooth


def selfcheck():
    """Corrupted force vectors must rank top; smooth field stays low."""
    rng = np.random.default_rng(0)
    n = 4000
    u = rng.normal(0, 1, (n, 2))
    F = np.concatenate([u, -u], axis=1)              # smooth linear "force" field
    bad = rng.choice(n, 5, replace=False)
    F[bad] += rng.normal(0, 10, (5, 4))
    ind, _ = knn_indicator(u, F, k=16)
    assert set(np.argsort(ind)[::-1][:5]) == set(bad), "corrupted frames not flagged"
    med = float(np.median(np.delete(ind, bad)))
    assert med < 0.05 * ind[bad].mean(), (med, ind[bad].mean())
    print(f"[selfcheck] corrupted frames flagged; background median {med:.4f}. OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--model", action="append", default=[],
                    help="LABEL=config.yaml:params.pkl (repeatable)")
    dc.add_latent_arguments(ap)
    ap.add_argument("--n-frames", type=int, default=20000)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--nn-space", choices=("latent", "dist"), default="latent",
                    help="neighbour space for smoothing: latent (TICA-flow u) or "
                         "dist (standardized bead-pair distances)")
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    diagnostic_cfg = dc.config_from_args(a)
    if a.selfcheck:
        selfcheck()
        if not a.model:
            return
    if not a.model:
        raise SystemExit("need at least one --model (or --selfcheck alone)")

    import jax
    import jax.numpy as jnp
    from analysis.md.analyze_model_residuals_by_region import _load_model          # noqa: E402

    a.outdir.mkdir(parents=True, exist_ok=True)

    lat = load_latent(a.frames, a.n_frames, a.grid_lim, a.bins, config=diagnostic_cfg)
    inside, flat = lat["inside"], lat["flat"]
    bins, centers = lat["bins"], lat["centers"]
    n_cells = bins * bins
    R_in64 = lat["R"][inside]
    R_in = R_in64.astype(np.float32)
    tree_pts = u_in = lat["u"][inside]
    if a.nn_space == "dist":
        tree_pts = pair_distance_features(R_in64)

    maps, per_frame = {}, {}
    for spec in a.model:
        lab = str(spec).partition("=")[0]
        model_cfg, par = parse_spec(str(spec).partition("=")[2])
        t0 = time.time()
        model, params, mask0, species0 = _load_model(model_cfg, par, a.frames,
                                                     lat["mapping"].n_beads)
        gfn = jax.jit(jax.vmap(jax.grad(
            lambda r: model.compute_energy(params, r, mask0, species0))))
        Fs = [np.asarray(gfn(jnp.asarray(R_in[i:i + 512])), np.float64)
              for i in range(0, len(R_in), 512)]
        Fm = np.concatenate(Fs).reshape(len(R_in), -1)
        ind, _ = knn_indicator(tree_pts, Fm, k=a.knn)
        per_frame[lab] = ind
        m_c = np.full(n_cells, np.nan)
        cnt = np.bincount(flat[inside], minlength=n_cells)
        s_c = np.bincount(flat[inside], weights=ind, minlength=n_cells)
        m_c = np.where(cnt >= a.min_count, s_c / np.maximum(cnt, 1), np.nan)
        maps[lab] = dict(cell_median=m_c, mean=float(ind.mean()))
        _log(f"[{lab}] forces+indicator in {time.time()-t0:.0f}s; "
             f"mean {ind.mean():.4f} kcal/mol/A")

    labs = list(maps)
    mass = np.bincount(flat[inside], minlength=n_cells).astype(float)
    mass /= max(mass.sum(), 1)
    score = np.abs(maps[labs[0]]["cell_median"]) * mass
    score[np.isnan(score)] = -1
    order = np.argsort(score)[::-1][:10]
    _log(f"[top cells by |indicator| x mass ({labs[0]})]")
    for i in order[:6]:
        if score[i] <= 0:
            continue
        ix, iy = i // bins, i % bins
        _log(f"  u=({centers[ix]:+.2f},{centers[iy]:+.2f}) "
             f"indicator={maps[labs[0]]['cell_median'][i]:.4f}")

    summary = {"provenance": {"frames": _jsonable_path(a.frames), "n_frames": int(len(lat["R"])),
                              "knn_k": a.knn, "bins": bins, "grid_lim": a.grid_lim,
                              "models": {l: parse_spec(s) for s, l in
                                         zip(a.model, labs)},
                              "kT_kcal_mol": diagnostic_cfg.kT},
               "mean_indicator": {l: maps[l]["mean"] for l in labs}}
    if len(labs) > 1:
        fin = np.isfinite(maps[labs[0]]["cell_median"])
        cors = {l: float(np.corrcoef(maps[labs[0]]["cell_median"][fin],
                                     maps[l]["cell_median"][fin])[0, 1])
                for l in labs[1:]}
        summary["cell_map_correlation_vs_" + labs[0]] = cors
        _log(f"[compare] cell-map corr vs {labs[0]}: "
             + " ".join(f"{l}: {c:+.3f}" for l, c in cors.items()))
    (a.outdir / "gradient_recovery.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(a.outdir / "gradient_recovery.npz",
                        edges=lat["edges"],
                        **{f"{l}_cell_median": maps[l]["cell_median"] for l in labs},
                        **{f"{l}_frame_indicator": per_frame[l] for l in labs})
    plot(a.outdir, lat["edges"], centers, bins, maps, labs)
    _log(f"wrote {a.outdir}/gradient_recovery.json .npz .png")


def plot(outdir, edges, centers, bins, maps, labs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labs)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = max(np.nanpercentile(maps[l]["cell_median"], 98) for l in labs)
    for ax, l in zip(axes, labs):
        g = maps[l]["cell_median"].reshape(bins, bins)
        im = ax.pcolormesh(edges, edges, g.T, cmap="viridis",
                           vmin=0, vmax=vmax, rasterized=True)
        ax.set_title(f"{l}\nZZ indicator (median/cell)")
        ax.set_aspect("equal"), ax.set_xlabel("$u_1$")
    axes[0].set_ylabel("$u_2$")
    fig.colorbar(im, ax=axes, label="kcal/mol/$\\AA$", shrink=0.85)
    fig.savefig(outdir / "gradient_recovery.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
