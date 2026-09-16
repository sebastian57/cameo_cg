#!/usr/bin/env python3
"""Close the Stage-2 gap: does a LOCAL-covariance Mahalanobis metric beat the TICA cell?

Stage 2 found globally-whitened Mahalanobis (Spearman 0.204) no better than plain Euclidean
(0.201), while the TICA cell scored 0.489. The diagnosis was that the GLOBAL descriptor
covariance is dominated by between-basin dihedral flips and isolates no local stiffness. This
tests the per-anchor LOCAL covariance, estimated from that anchor's k nearest reference frames.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("local_work/stage1_shots"))
    ap.add_argument("--sens", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--k", type=int, default=400, help="reference neighbours for the local cov")
    a = ap.parse_args()

    R = np.load(a.root / "bead_coords.npz")["R"]
    rows = json.loads((a.root / "shots.json").read_text())
    n_shot, n_frame = R.shape[0], R.shape[1]
    flat = R.reshape(-1, 6, 3)
    d = np.linalg.norm(flat[:, PAIRS[:, 0], :] - flat[:, PAIRS[:, 1], :],
                       axis=-1).reshape(n_shot, n_frame, -1)

    z = np.load(a.sens, allow_pickle=True)
    Rref = np.asarray(np.load(a.reference)["R"], np.float64)
    dref = np.linalg.norm(Rref[:, PAIRS[:, 0], :] - Rref[:, PAIRS[:, 1], :], axis=-1)
    Yref = (dref - z["tica_mean"]) @ z["tica_coefficients"]
    ex, ey, rho = z["ex"], z["ey"], z["rho"]
    ix = lambda v, e: np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)

    anch = np.array([f'{r["stratum"]}/{r["anchor"]:02d}' for r in rows])
    uniq = sorted(set(anch))
    a0 = d[:, 0, :]
    Y0 = (a0 - z["tica_mean"]) @ z["tica_coefficients"]
    pr = rho[ix(Y0[:, 0], ex), ix(Y0[:, 1], ey)]

    print(f"{len(uniq)} anchors; local covariance from k={a.k} nearest reference frames")
    resid_local, resid_glob = [], []
    Cg = np.linalg.inv(np.cov(dref.T) + 1e-9 * np.eye(dref.shape[1]))
    for u in uniq:
        s = np.flatnonzero(anch == u)
        anchor_desc = a0[s[0]]
        nn = np.argpartition(np.linalg.norm(dref - anchor_desc, axis=1), a.k)[:a.k]
        C = np.cov(dref[nn].T) + 1e-6 * np.eye(dref.shape[1])
        Ci = np.linalg.inv(C)
        dd = d[s] - anchor_desc[None, None, :]
        resid_local.append(-np.sqrt(np.einsum("sfi,ij,sfj->sf", dd, Ci, dd)).mean())
        resid_glob.append(-np.sqrt(np.einsum("sfi,ij,sfj->sf", dd, Cg, dd)).mean())
    prA = np.array([pr[anch == u][0] for u in uniq]); ok = prA > 0
    lp = np.log(prA[ok])
    print(f"\n{'metric':34s} {'Spearman vs log p_ref':>22s}")
    print(f"{'Mahalanobis, LOCAL covariance':34s} "
          f"{spearmanr(np.array(resid_local)[ok], lp).statistic:22.3f}")
    print(f"{'Mahalanobis, GLOBAL (Stage 2)':34s} "
          f"{spearmanr(np.array(resid_glob)[ok], lp).statistic:22.3f}")
    print(f"{'(TICA cell, Stage 2 winner)':34s} {0.489:22.3f}")


if __name__ == "__main__":
    main()
