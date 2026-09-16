#!/usr/bin/env python3
"""Stationarity test: does the model PRESERVE the reference equilibrium?

Replicas are started from frames drawn from the reference equilibrium
distribution. If pi_model == pi_ref the ensemble must STAY there, so any
systematic drift in basin populations is model error -- and it appears without
needing a barrier crossing, which is what makes this usable with short replicas.

Unlike analyze_fes_tica_vs_reference.py this keeps the TIME axis: that script
discards the first 20% of every replica and pools, which destroys the signal.

    python -m analysis.md.analyze_stationarity_drift \\
        --npz local_work/md_runs/<run>/traj_*_rep*.npz \\
        --reference <mapped-AA reference>.npz \\
        --outdir local_work/md_analysis/<run> --prefix <run>
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampling.mapping import get_mapping


# Basin definitions copied verbatim from
# analysis/md/analyze_fes_tica_vs_reference.py:237-239 so numbers are comparable.
def basin_masks(phi, psi):
    return {
        "alphaR": (phi > -180) & (phi < 0) & (psi > -120) & (psi < 50),
        "alphaL": (phi > 0) & (phi < 120) & (psi > -50) & (psi < 100),
        "phi_positive": phi > 0,
    }


def js_bits(P, Q):
    """Jensen-Shannon divergence in bits between two normalised histograms."""
    P = P / max(P.sum(), 1e-30); Q = Q / max(Q.sum(), 1e-30)
    M = 0.5 * (P + Q)
    def kl(A, B):
        m = A > 0
        return float(np.sum(A[m] * np.log2(A[m] / np.maximum(B[m], 1e-30))))
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def mcnemar(n01, n10):
    """Exact two-sided McNemar on discordant pairs; n01 = entered, n10 = left."""
    n = n01 + n10
    if n == 0:
        return 1.0
    from math import comb
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--mapping", type=str, default="ala2_backbone_cb_6")
    ap.add_argument("--frame-ps", type=float, default=0.2)
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--burnin-ps", type=float, default=0.0,
                    help="Ignore the first X ps when fitting the slow drift "
                         "(mapped-AA starts relax in stiff DOF first).")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    m = get_mapping(a.mapping)

    # ---- load: (n_rep, n_frames) angle arrays -----------------------------
    PHI, PSI = [], []
    for f in sorted(a.npz):
        R = np.load(f)["R"]                       # (T, N, 3)
        PHI.append(m.cvs["phi"].evaluate(R)); PSI.append(m.cvs["psi"].evaluate(R))
    n_len = min(len(x) for x in PHI)
    phi = np.stack([x[:n_len] for x in PHI]); psi = np.stack([x[:n_len] for x in PSI])
    n_rep, n_t = phi.shape
    t_ps = np.arange(n_t) * a.frame_ps
    print(f"{n_rep} replicas x {n_t} frames ({t_ps[-1]:.1f} ps)")

    ref = np.load(a.reference)["R"]
    rphi = m.cvs["phi"].evaluate(ref); rpsi = m.cvs["psi"].evaluate(ref)
    ref_masks = basin_masks(rphi, rpsi)
    ref_pct = {k: float(v.mean() * 100) for k, v in ref_masks.items()}

    # ---- p_basin(t) with binomial CI --------------------------------------
    masks = basin_masks(phi, psi)                 # each (n_rep, n_t)
    out = {"n_replicas": n_rep, "n_frames": n_t, "total_ps": float(t_ps[-1]),
           "reference_pct": ref_pct, "basins": {}}

    fig, axes = plt.subplots(len(masks), 1, figsize=(9, 3.0 * len(masks)), sharex=True)
    for ax, (name, M) in zip(np.atleast_1d(axes), masks.items()):
        p = M.mean(axis=0) * 100
        se = np.sqrt(np.maximum(p / 100 * (1 - p / 100), 0) / n_rep) * 100
        ax.axhline(ref_pct[name], color="k", ls="--", lw=1.2,
                   label=f"reference {ref_pct[name]:.2f}%")
        ax.fill_between(t_ps, p - 1.96 * se, p + 1.96 * se, alpha=0.25, color="C0")
        ax.plot(t_ps, p, color="C0", lw=1.0, label="ensemble")
        ax.set_ylabel(f"{name}  [%]"); ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        # paired start-vs-end test (McNemar): same replicas, so far more
        # sensitive than comparing two independent population estimates.
        start, end = M[:, 0], M[:, -1]
        n01 = int((~start & end).sum()); n10 = int((start & ~end).sum())
        pv = mcnemar(n01, n10)
        out["basins"][name] = {
            "p0_pct": float(p[0]), "pT_pct": float(p[-1]),
            "drift_pct_points": float(p[-1] - p[0]),
            "reference_pct": ref_pct[name],
            "entered": n01, "left": n10, "mcnemar_p": pv,
            "n_at_start": int(start.sum()),
        }
        print(f"{name:14s} {p[0]:6.2f}% -> {p[-1]:6.2f}%  (ref {ref_pct[name]:6.2f}%)  "
              f"entered {n01:3d} left {n10:3d}  McNemar p={pv:.4g}")
    np.atleast_1d(axes)[-1].set_xlabel("time [ps]")
    fig.suptitle(f"{a.prefix}: basin population vs time (start = reference equilibrium)")
    fig.tight_layout()
    f1 = a.outdir / f"{a.prefix}_stationarity_populations.png"
    fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- whole-distribution drift: JS(t) vs t=0 and vs reference ----------
    edges = np.linspace(-180, 180, a.bins + 1)
    def hist(p, q):
        H, _, _ = np.histogram2d(p, q, bins=[edges, edges]); return H
    H_ref = hist(rphi, rpsi)
    H0 = hist(phi[:, 0], psi[:, 0])
    js_t0, js_ref = [], []
    for k in range(n_t):
        H = hist(phi[:, k], psi[:, k])
        js_t0.append(js_bits(H, H0)); js_ref.append(js_bits(H, H_ref))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_ps, js_t0, label="JS vs own t=0", color="C1")
    ax.plot(t_ps, js_ref, label="JS vs reference equilibrium", color="C2")
    ax.set_xlabel("time [ps]"); ax.set_ylabel("JS divergence [bits]")
    ax.set_title(f"{a.prefix}: ensemble drift away from the starting equilibrium")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    f2 = a.outdir / f"{a.prefix}_stationarity_js_drift.png"
    fig.savefig(f2, dpi=150); plt.close(fig)

    out["js_vs_t0_final"] = float(js_t0[-1])
    out["js_vs_reference_start"] = float(js_ref[0])
    out["js_vs_reference_final"] = float(js_ref[-1])
    # note: JS vs reference at t=0 is NOT zero -- it is the finite-sample floor
    # of n_rep frames against 200k, and is the baseline any drift must beat.
    out["interpretation"] = (
        "js_vs_reference_start is the finite-sample floor (n_rep frames vs the full "
        "reference); drift is only meaningful where js_vs_reference_final exceeds it.")
    (a.outdir / f"{a.prefix}_stationarity.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {f1}\n      {f2}\n      {a.outdir / (a.prefix + '_stationarity.json')}")


if __name__ == "__main__":
    main()
