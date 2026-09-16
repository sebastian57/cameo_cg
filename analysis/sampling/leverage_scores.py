#!/usr/bin/env python3
"""Exact per-frame leverage scores on reference frames (MD-less, model-free).

l_i = x_i^T (X^T X)^{-1} x_i over a feature matrix X built from every frame.
High l_i = frame least interpolatable from all others (irreplaceable); low =
redundant. Two bases are computed so their rankings can be compared:

  dist  : unique bead-pair distances (the coordinates an FM potential consumes)
  tica  : frozen-TICA projection (the ensemble-relevant coordinates)

At ala2 scale (n=2e5, d<=15) exact scores are cheaper than Nyström
approximations, so none are used.

Usage: python -m sampling.leverage_scores --outdir <dir> [--n-frames 200000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc
from analysis.sampling.diagnostics_common import assign_regions


def _log(msg):
    print(msg, flush=True)


def leverage_exact(X):
    """diag(X (X^T X)^-1 X^T) with a tiny ridge for numerical safety."""
    G = X.T @ X
    G += 1e-10 * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    sol = np.linalg.solve(L, X.T)                    # (d, n)
    return np.einsum("in,in->n", sol, sol)


def selfcheck():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 3))
    X[-1] += 25.0                                    # one far outlier
    l = leverage_exact(X)
    assert int(np.argmax(l)) == len(X) - 1, "outlier must be max leverage"
    rank = np.linalg.matrix_rank(X)
    assert abs(l.sum() - rank) < 1e-8, f"sum(l)={l.sum()} != rank={rank}"
    print(f"[selfcheck] outlier flagged, sum(l)={l.sum():.6f} == rank {rank}. OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    dc.add_latent_arguments(ap)
    ap.add_argument("--n-frames", type=int, default=200001)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = dc.config_from_args(a)
    if a.selfcheck:
        selfcheck()

        from sampling.biases.tica_regional import SmoothTICABias
    from sampling.mapping import get_mapping

    a.outdir.mkdir(parents=True, exist_ok=True)

    # ---- frames ---------------------------------------------------------------------
    R_all = np.asarray(np.load(a.frames)["R"], np.float64)
    idx = np.linspace(0, len(R_all) - 1, min(a.n_frames, len(R_all))).astype(int)
    R = R_all[idx]
    _log(f"[load] {len(R)} frames from {a.frames}")

    bases = {}
    # pair-distance basis
    n_beads = R.shape[1]
    iu, ju = np.triu_indices(n_beads, k=1)
    D = np.linalg.norm(R[:, iu, :] - R[:, ju, :], axis=-1)
    bases["dist"] = (D - D.mean(0)) / D.std(0)
    # frozen-TICA basis
    bias = SmoothTICABias.load(cfg.bias_npz)
    Zt = np.asarray(bias.projection.transform(R), np.float64)[:, :2]
    bases["tica"] = (Zt - Zt.mean(0)) / Zt.std(0)

    scores = {}
    t0 = time.time()
    for name, X in bases.items():
        scores[name] = leverage_exact(X.astype(np.float64))
        _log(f"[scores] {name}: d={X.shape[1]}, range "
             f"[{scores[name].min():.3f},{scores[name].max():.3f}], "
             f"mean {scores[name].mean():.4f} ({time.time()-t0:.1f}s)")
        t0 = time.time()

    # basin composition of the top-k most leveraged frames
    mapping = get_mapping(cfg.mapping_name)
    reg, _, _ = assign_regions(R, mapping)
    top = {}
    summary = {"provenance": {"frames": a.frames, "n_frames": int(len(R)),
                              "features": {k: int(v.shape[1]) for k, v in bases.items()}},
               "basins_topk": {}}
    for name in bases:
        order = np.argsort(scores[name])[::-1][:a.top_k]
        labs, cnts = np.unique(reg[order], return_counts=True)
        top[name] = dict(zip(labs.tolist(), cnts.tolist()))
        _log(f"[top-{a.top_k}] {name}: {dict(zip(labs.tolist(), [int(c) for c in cnts]))}")
    summary["basins_topk"] = {
        "topk": a.top_k,
        **{name: {str(k): int(v) for k, v in top[name].items()} for name in top},
    }

    from scipy.stats import spearmanr
    sub = np.linspace(0, len(R) - 1, 20000).astype(int)
    rho = float(spearmanr(scores["dist"][sub], scores["tica"][sub]).statistic)
    summary["spearman_dist_vs_tica"] = rho
    _log(f"[compare] spearman(dist, tica) = {rho:+.3f}")

    np.savez_compressed(a.outdir / "leverage.npz",
                        frame_index=idx, region=reg.astype(str),
                        **{f"score_{k}": v for k, v in scores.items()})
    (a.outdir / "leverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(a.outdir, idx, reg, scores, bases)
    _log(f"wrote {a.outdir}/leverage.json .npz .png")


def plot(outdir, idx, reg, scores, bases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    colors = {"beta": "#2a78d6", "alphaR": "#eb6834", "alphaL": "#1baf7a",
              "other": "#9a988e"}
    for ax, name in zip(axes[:2], scores):
        s = scores[name]
        ax.hist(s, bins=120, color="#3987e5")
        ax.set_yscale("log"), ax.set_xlabel(f"leverage ({name})")
        ax.set_ylabel("frames")
        ax.set_title(f"leverage, {name} basis (d={bases[name].shape[1]})")
    ax = axes[2]
    sub = np.linspace(0, len(reg) - 1, 20000).astype(int)
    for b in ("beta", "alphaR", "alphaL", "other"):
        m = reg[sub] == b
        if m.sum():
            xs = np.sort(scores["dist"][sub][m])[::-1][:200]
            ax.plot(xs, label=b, color=colors[b], lw=1.8)
    ax.set_xscale("log"), ax.legend(), ax.set_title("top leverage by basin (dist)")
    fig.savefig(outdir / "leverage.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
