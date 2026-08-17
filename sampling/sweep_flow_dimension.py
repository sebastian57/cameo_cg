#!/usr/bin/env python3
"""Does the flow beat the KDE as the TICA dimension grows? The experiment that decides.

    python -m sampling.sweep_flow_dimension --bias-npz <b.npz> --reference <ref.npz> \
        --outdir local_work/flow_dscaling --dims 2 3 4 6 --seeds 3

WHY THIS IS THE DECIDING EXPERIMENT
    At d=2 the flow is an EQUAL-performing replacement for the KDE: Phase 3 measured identical
    coverage (85.6% vs 85.4%), transition enrichment (1.246x vs 1.270x) and realised bias
    (-1.32 vs -1.34 kcal/mol). Its better density fit did not convert into better sampling.

    The case for the flow therefore rests entirely on higher dimension. And the reason is
    STATISTICAL, not computational -- a correction to an earlier claim of mine:

      * "the grid dies at n^d" is about TABULATION and hits both methods equally;
      * neither method actually needs a grid. A KDE over ~470 centres is ~3k flops per
        evaluation and the small flow ~5k; both are differentiable, history-free JAX
        functions that export to MLIR through the same CG_BIAS MODEL= route.

    What does NOT scale is KDE density ESTIMATION. A KDE places one kernel per observation
    (or per occupied cell); in 6D with 200k frames there is no meaningful occupancy structure,
    and no bandwidth is simultaneously smooth enough and sharp enough. A flow shares ~6k
    parameters across the whole space instead.

METRICS THAT WORK IN ANY DIMENSION
    No grids and no histograms -- both die with d, which is the point. Everything is evaluated
    at HELD-OUT DATA POINTS:
      * held-out NLL, per dimension, KDE vs flow at the same d -- the direct measure;
      * seed-to-seed score reproducibility |grad log p|, at held-out points.

    The KDE is given its BEST SHOT: its bandwidth scale is optimised on held-out likelihood at
    every d rather than fixed by a rule of thumb. A rigged comparison would prove nothing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def kde_logpdf(query, centers, log_w, h):
    """log sum_k w_k N(query; c_k, diag(h^2)), chunked. Returns (n_query,)."""
    from scipy.special import logsumexp
    d = query.shape[1]
    const = -0.5 * d * np.log(2 * np.pi) - np.sum(np.log(h))
    out = np.empty(len(query))
    for s in range(0, len(query), 512):
        q = query[s:s + 512]
        m = -0.5 * np.sum(((q[:, None, :] - centers[None, :, :]) / h) ** 2, axis=2)
        out[s:s + 512] = logsumexp(m + log_w[None, :], axis=1) + const
    return out


def kde_score(query, centers, log_w, h):
    """grad_x log p for the same KDE (soft-assignment weighted mean direction)."""
    from scipy.special import logsumexp
    out = np.empty_like(query)
    for s in range(0, len(query), 512):
        q = query[s:s + 512]
        delta = q[:, None, :] - centers[None, :, :]
        m = -0.5 * np.sum((delta / h) ** 2, axis=2) + log_w[None, :]
        resp = np.exp(m - logsumexp(m, axis=1)[:, None])
        out[s:s + 512] = np.einsum("nk,nkd->nd", resp, -delta / h ** 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bias-npz", required=True, help="only for the frozen pair-distance features")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lagtime", type=int, default=20)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-centers", type=int, default=8000, help="KDE centres (subsampled)")
    ap.add_argument("--n-eval", type=int, default=4000, help="held-out points for all metrics")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    from deeptime.decomposition import TICA
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import FlowConfig, log_prob_and_grad, train_flow

    bias = SmoothTICABias.load(a.bias_npz)
    R = np.load(a.reference)["R"]
    # same frozen pair-distance featurisation the production TICA uses
    pairs = np.asarray(bias.projection.pairs)
    X = np.linalg.norm(R[:, pairs[:, 0], :] - R[:, pairs[:, 1], :], axis=-1).astype(np.float64)
    dmax = max(a.dims)
    tica = TICA(lagtime=a.lagtime, dim=dmax).fit(X).fetch_model()
    Z_all = np.asarray(tica.transform(X), dtype=np.float64)
    print(f"{len(X)} frames, {X.shape[1]} pair distances -> TICA {Z_all.shape} "
          f"(lagtime {a.lagtime})")
    # HOW MANY DIMENSIONS ARE REAL? A TICA eigenvalue near zero -- or negative, which is
    # unphysical for a reversible process -- is a NOISE direction, and a density comparison
    # on noise directions says nothing about the physics. Reported so the caveat travels with
    # the numbers. Measured on ala2: [0.730, 0.236, 0.012, -0.008, ...] i.e. TWO real modes.
    sv = np.asarray(tica.singular_values)[:dmax]
    ts = np.asarray(tica.timescales(lagtime=a.lagtime))[:dmax]
    print(f"  TICA eigenvalues : {np.round(sv, 4).tolist()}")
    print(f"  timescales (fr)  : {np.round(ts, 2).tolist()}")
    n_real = int(np.sum(sv > 0.05))
    print(f"  -> {n_real} dimension(s) carry real kinetic signal (eigenvalue > 0.05); "
          f"beyond that the comparison is on NOISE directions")
    print(f"  per-TIC std: {np.round(Z_all.std(0), 3).tolist()}\n")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Z_all))
    n_val = max(a.n_eval, int(0.1 * len(Z_all)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    ev_idx = val_idx[:a.n_eval]

    rows = []
    for d in a.dims:
        Z = Z_all[:, :d]
        Ztr, Zev = Z[train_idx], Z[ev_idx]

        # ---- KDE, bandwidth scale optimised on held-out likelihood ----------------------
        t0 = time.time()
        cen = Ztr[rng.choice(len(Ztr), size=min(a.n_centers, len(Ztr)), replace=False)]
        log_w = np.full(len(cen), -np.log(len(cen)))
        # Scott's rule as the centre of the scan; the scan is what makes the comparison fair
        h0 = Ztr.std(0) * len(cen) ** (-1.0 / (d + 4))
        best = (np.inf, None)
        # scan must not hit its own boundary; measured, 0.40 sat on the old lower edge
        for s in (0.10, 0.15, 0.22, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.7, 2.2):
            nll = -kde_logpdf(Zev[:1000], cen, log_w, h0 * s).mean()
            if nll < best[0]:
                best = (nll, s)
        s_star = best[1]
        kde_nll = float(-kde_logpdf(Zev, cen, log_w, h0 * s_star).mean())
        kde_t = time.time() - t0

        # ---- flow, the `small` configuration from the capacity sweep --------------------
        t0 = time.time()
        f_nll, scores = [], []
        for seed in range(a.seeds):
            cfg = FlowConfig(n_dims=d, n_layers=4, n_bins=4, hidden=32, seed=seed)
            p, hist = train_flow(Ztr, cfg, steps=a.steps, batch=4096, lr=1e-3,
                                 report_every=a.steps, log=lambda _s: None)
            f_nll.append(hist["swa_val_nll"])
            import jax.numpy as jnp
            _, g = log_prob_and_grad(p, cfg, jnp.asarray(Zev, jnp.float32))
            scores.append(np.asarray(g, np.float64))
        flow_nll = float(np.median(f_nll))
        flow_t = (time.time() - t0) / a.seeds

        # ---- seed reproducibility of the score, at held-out points ----------------------
        G = np.stack(scores)
        spread = np.sqrt((np.linalg.norm(G - G.mean(0), axis=-1) ** 2).mean(0))
        gnorm = np.linalg.norm(G.mean(0), axis=-1)
        kde_g = np.linalg.norm(kde_score(Zev, cen, log_w, h0 * s_star), axis=-1)

        rows.append(dict(
            d=d, kde_nll=kde_nll, flow_nll=flow_nll, gap=kde_nll - flow_nll,
            kde_nll_per_dim=kde_nll / d, flow_nll_per_dim=flow_nll / d,
            kde_bandwidth_scale=s_star, kde_h=(h0 * s_star).tolist(),
            flow_score_median=float(np.median(gnorm)),
            kde_score_median=float(np.median(kde_g)),
            flow_seed_spread_median=float(np.median(spread)),
            flow_seed_spread_rel=float(np.median(spread / (gnorm + 1e-12))),
            kde_seconds=round(kde_t, 1), flow_seconds=round(flow_t, 1)))
        r = rows[-1]
        print(f"d={d}:  KDE nll {kde_nll:8.4f} (h scale {s_star:.2f})   "
              f"flow nll {flow_nll:8.4f}   GAP {r['gap']:+7.4f}"
              f"   flow seed spread {r['flow_seed_spread_median']:.3f}")

    (a.outdir / "dscaling.json").write_text(json.dumps(
        dict(rows=rows, tica_eigenvalues=sv.tolist(), tica_timescales=ts.tolist(),
             n_real_dims=n_real, lagtime=a.lagtime), indent=1))
    _report(rows, a.outdir, n_real=n_real, sv=sv)


def _report(rows, outdir: Path, n_real: int = 0, sv=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = [r["d"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax[0].plot(d, [r["kde_nll"] for r in rows], "s--", lw=2, label="KDE (tuned bandwidth)")
    ax[0].plot(d, [r["flow_nll"] for r in rows], "o-", lw=2, label="flow")
    ax[0].set_xlabel("TICA dimensions"); ax[0].set_ylabel("held-out NLL (nats)")
    ax[0].set_title("density quality\n(lower = better)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xticks(d)

    gap = [r["gap"] for r in rows]
    ax[1].axhline(0, color="k", lw=1)
    ax[1].bar([str(x) for x in d], gap,
              color=["tab:green" if g > 0 else "tab:red" for g in gap])
    ax[1].set_xlabel("TICA dimensions"); ax[1].set_ylabel("KDE NLL − flow NLL (nats)")
    ax[1].set_title("flow advantage\n(positive = flow better)"); ax[1].grid(alpha=.3, axis="y")
    for x, g in zip([str(x) for x in d], gap):
        ax[1].annotate(f"{g:+.2f}", (x, g), ha="center",
                       va="bottom" if g > 0 else "top", fontsize=9)

    ax[2].plot(d, [r["flow_seed_spread_rel"] for r in rows], "o-", lw=2, label="flow")
    ax[2].set_xlabel("TICA dimensions"); ax[2].set_xticks(d)
    ax[2].set_ylabel("median seed spread / |score|")
    ax[2].set_title("score reproducibility\n(at held-out points)")
    ax[2].grid(alpha=.3); ax[2].legend()
    if n_real:
        for k in range(3):
            ax[k].axvspan(n_real + 0.5, max(d) + 0.5, color="red", alpha=.08)
        ax[0].text(n_real + 0.6, ax[0].get_ylim()[1],
                   f" noise dims\n (TICA eig < 0.05)", va="top", fontsize=8, color="darkred")

    fig.tight_layout(); fig.savefig(outdir / "fig_dimension_scaling.png", dpi=130)
    plt.close(fig)

    print(f"\n{'d':>3}{'KDE nll':>10}{'flow nll':>10}{'gap':>9}{'KDE/dim':>9}{'flow/dim':>9}"
          f"{'h scale':>9}{'seed spread':>13}")
    for r in rows:
        print(f"{r['d']:>3}{r['kde_nll']:>10.4f}{r['flow_nll']:>10.4f}{r['gap']:>+9.4f}"
              f"{r['kde_nll_per_dim']:>9.4f}{r['flow_nll_per_dim']:>9.4f}"
              f"{r['kde_bandwidth_scale']:>9.2f}{r['flow_seed_spread_rel']:>13.3f}")
    print(f"\nwrote {outdir}/fig_dimension_scaling.png")


if __name__ == "__main__":
    main()
