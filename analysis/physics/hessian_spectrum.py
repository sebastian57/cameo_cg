"""Eigen-spectrum of the projected PMF Hessian, from the existing stencil data alone.

Each anchor carries central differences along 6 internal directions, so the 6x6 projection of
the Hessian onto that subspace is directly measurable:

    v_i   = (R_i^+ - R_i^-) / ||R_i^+ - R_i^-||          (REALIZED displacement)
    H v_i = -(F_i^+ - F_i^-) / ||R_i^+ - R_i^-||         (F = -grad A)
    M_ij  = v_i . (H v_j),   symmetrised (measured asymmetry is ~2%)

Question this answers: how many directions must a stencil probe to capture most of the local
curvature?  If the spectrum is dominated by k << 6 eigenvalues, the production stencil can drop
to k directions aligned with the top eigenvectors -- a direct cost reduction with no new physics.

No model, no GPU, no training. ~1 minute on a login node.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_input, resolve_output
from analysis.common.provenance import write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["inner", "outer"], default="inner")
    ap.add_argument("--meanforce", type=Path, required=True)
    ap.add_argument("--stencil", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    add_project_root_argument(ap)
    a = ap.parse_args()
    project_root = repo_root(a.project_root)
    a.outdir = resolve_output(a.outdir, base=project_root)
    meanforce = resolve_input(a.meanforce, base=project_root, label="mean-force labels")
    stencil = resolve_input(a.stencil, base=project_root, label="stencil states")
    mm, st = np.load(meanforce), np.load(stencil)
    state = mm["state"]
    slot, direction = state // 25, st["direction"][state]
    mult = np.round(st["multiplier"][state], 2)
    R, F = np.asarray(mm["R"], np.float64), np.asarray(mm["F"], np.float64)
    n_slots = int(slot.max()) + 1
    lvl = 1.0 if a.layer == "inner" else 2.0

    pos, neg = {}, {}
    for i in range(len(state)):
        d = int(direction[i])
        if d < 0 or abs(mult[i]) != lvl:
            continue
        (pos if mult[i] > 0 else neg)[(int(slot[i]), d)] = i

    V = np.zeros((n_slots, 6, 18))
    HV = np.zeros((n_slots, 6, 18))
    ok = np.zeros((n_slots, 6), bool)
    for s in range(n_slots):
        for d in range(6):
            ip, im = pos.get((s, d)), neg.get((s, d))
            if ip is None or im is None:
                continue
            dr = (R[ip] - R[im]).ravel()
            n = np.linalg.norm(dr)
            if n == 0:
                continue
            V[s, d] = dr / n
            HV[s, d] = -(F[ip] - F[im]).ravel() / n
            ok[s, d] = True

    full = ok.all(1)
    print(f"layer={a.layer}   anchors with all 6 directions: {full.sum():,}/{n_slots:,}")

    M = np.einsum("sik,sjk->sij", V[full], HV[full])
    asym = np.abs(M - np.transpose(M, (0, 2, 1))).max((1, 2)) / np.abs(M).max((1, 2))
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    print(f"asymmetry |M-M^T|/|M|  median {np.median(asym):.4f}  p95 {np.percentile(asym,95):.4f}"
          "   (a pure measurement check: a true Hessian is symmetric)")

    # non-orthogonality of the sampled directions (they are only approximately orthonormal)
    G = np.einsum("sik,sjk->sij", V[full], V[full])
    offdiag = np.abs(G - np.eye(6)).max((1, 2))
    print(f"direction non-orthogonality max|G-I|  median {np.median(offdiag):.4f}")

    w = np.linalg.eigvalsh(M)[:, ::-1]                 # descending
    aw = np.abs(w)
    frac = np.cumsum(aw, 1) / aw.sum(1, keepdims=True)

    print(f"\n=== eigenvalue spectrum of the 6x6 projected Hessian ({full.sum():,} anchors) ===")
    print(f"{'rank':>4s} {'median':>10s} {'p25':>10s} {'p75':>10s} {'|w|/|w1| med':>13s}")
    for k in range(6):
        r = np.median(aw[:, k] / aw[:, 0])
        print(f"{k+1:4d} {np.median(w[:,k]):10.1f} {np.percentile(w[:,k],25):10.1f} "
              f"{np.percentile(w[:,k],75):10.1f} {r:13.3f}")

    print(f"\n=== cumulative |eigenvalue| mass captured by the top-k directions ===")
    for k in range(6):
        print(f"  top-{k+1}: {100*np.median(frac[:,k]):5.1f}%  "
              f"(p25 {100*np.percentile(frac[:,k],25):.1f}, p95 {100*np.percentile(frac[:,k],95):.1f})")

    pr = aw.sum(1) ** 2 / (aw ** 2).sum(1)
    print(f"\nparticipation ratio (effective number of active directions): "
          f"median {np.median(pr):.2f} of 6   p5/p95 {np.percentile(pr,5):.2f}/{np.percentile(pr,95):.2f}")
    print(f"  1.0 = one direction carries everything;  6.0 = perfectly isotropic")

    nsign = (w > 0).sum(1)
    print(f"\npositive eigenvalues per anchor: " +
          "  ".join(f"{k}:{100*(nsign==k).mean():.1f}%" for k in range(7)))
    print("  (6 = locally convex; fewer = saddle directions, i.e. the anchor sits on a ridge)")

    summary = a.outdir / f"spectrum_{a.layer}.npz"
    np.savez_compressed(summary, eig=w, M=M, slots=np.flatnonzero(full),
                        participation=pr)
    manifest = write_manifest(
        a.outdir, inputs={"meanforce": meanforce, "stencil": stencil},
        parameters=vars(a), module="analysis.physics.hessian_spectrum",
        extra={"summary": str(summary)})
    print(f"\nwrote {summary} and {manifest}")


if __name__ == "__main__":
    main()
