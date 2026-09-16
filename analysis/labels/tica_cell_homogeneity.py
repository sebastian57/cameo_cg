#!/usr/bin/env python3
"""Do structures at the SAME TICA point resemble each other globally?

TICA is a 2D LINEAR projection of a ~12-dimensional internal space, so each TICA point
carries a ~10-dimensional fibre. The question is not whether that fibre exists (it must)
but how WIDE it is: are same-cell structures nearly identical, or arbitrarily different
in the 10 orthogonal directions?

If TICA cells are structurally broad, a bias defined on TICA density is steering by a
coordinate that does not pin the configuration -- which would make anchor placement
"incomplete" in exactly the way the user suspects.

Method: equal-OCCUPANCY cells (so TICA and Rama cells hold the same number of frames and
the comparison is fair), then the proper-rotation Kabsch RMSD between random within-cell
pairs. Baselines: same-Rama-cell pairs, and fully random pairs.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features, fit_tica
from sampling.mapping import get_mapping


def kabsch_rmsd(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    H = Q.T @ P
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = S.copy(); S[2] *= d
    msd = max((P**2).sum() + (Q**2).sum() - 2.0 * S.sum(), 0.0) / len(P)
    return float(np.sqrt(msd))


def cell_ids(x, y, nb):
    ex = np.quantile(x, np.linspace(0, 1, nb + 1)); ey = np.quantile(y, np.linspace(0, 1, nb + 1))
    ex[0] -= 1e-6; ex[-1] += 1e-6; ey[0] -= 1e-6; ey[-1] += 1e-6
    ix = np.clip(np.digitize(x, ex) - 1, 0, nb - 1); iy = np.clip(np.digitize(y, ey) - 1, 0, nb - 1)
    return ix * nb + iy


def within_cell_rmsd(R, cid, rng, n_pairs=4000):
    out = []
    uc, inv = np.unique(cid, return_inverse=True)
    for _ in range(n_pairs):
        c = rng.integers(0, len(uc))
        mem = np.flatnonzero(inv == c)
        if len(mem) < 2: continue
        i, j = rng.choice(mem, 2, replace=False)
        out.append(kabsch_rmsd(R[i], R[j]))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-frames", type=int, default=20000)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--lagtime", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    R = np.asarray(np.load(a.dataset)["R"], np.float32)
    idx = np.linspace(0, len(R) - 1, min(a.n_frames, len(R))).astype(int)
    Rs = R[idx]
    nb_at = Rs.shape[1]
    pairs = np.array([(i, j) for i in range(nb_at) for j in range(i + 1, nb_at)], dtype=int)
    Y = fit_tica(build_features(R, pairs), a.lagtime)[1][idx]     # canonical all-pairs TICA
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(Rs); psi = m.cvs["psi"].evaluate(Rs)

    print(f"{len(Rs)} frames, {a.bins}x{a.bins} EQUAL-OCCUPANCY cells "
          f"(~{len(Rs)//(a.bins**2)} frames/cell)\n")
    res = {}
    res["same TICA cell"] = within_cell_rmsd(Rs, cell_ids(Y[:, 0], Y[:, 1], a.bins), rng)
    res["same Rama cell"] = within_cell_rmsd(Rs, cell_ids(phi, psi, a.bins), rng)
    ri = rng.integers(0, len(Rs), (4000, 2))
    res["random pairs"] = np.array([kabsch_rmsd(Rs[i], Rs[j]) for i, j in ri])

    print(f"{'pair set':22s} {'median RMSD':>12} {'p90':>8} {'vs random':>10}")
    rnd = np.median(res["random pairs"])
    for k, v in res.items():
        print(f"{k:22s} {np.median(v):12.3f} {np.percentile(v,90):8.3f} {np.median(v)/rnd:10.2f}")
    t, r = np.median(res["same TICA cell"]), np.median(res["same Rama cell"])
    print(f"\nTICA cells are {t/r:.2f}x as structurally broad as Rama cells "
          f"({t:.3f} vs {r:.3f} A median RMSD).")
    print(f"A value near 1.0 vs random ({rnd:.3f}) would mean the coordinate does not pin")
    print("structure at all; near 0 would mean it pins it completely.")


if __name__ == "__main__":
    main()
