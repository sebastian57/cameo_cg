#!/usr/bin/env python3
"""Would a DIHEDRAL-based TICA featurisation beat pair distances?

Pair-distance TICA is a LINEAR map and cannot represent dihedral periodicity, which is why
it needed ~8 components to reach 0.106 A while two angles (phi,psi) reach 0.074 A. But
phi/psi are system-specific knowledge we will not have for a general protein.

A GENERAL, periodicity-aware alternative: sin/cos of every bead-quadruple pseudo-dihedral.
No system-specific choice is needed -- for a 1-bead-per-residue protein these are exactly
the backbone pseudo-dihedrals. Tested here against the same kNN structural-tightness metric.
"""
from __future__ import annotations
import argparse, sys, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features
from sampling.mapping import get_mapping
from analysis.labels.tica_dim_homogeneity import kabsch_rmsd_many, knn_rmsd


def dihedrals(R, quads):
    out = np.empty((len(R), len(quads)), np.float64)
    for k, (a, b, c, d) in enumerate(quads):
        b0 = R[:, a] - R[:, b]; b1 = R[:, c] - R[:, b]; b2 = R[:, d] - R[:, c]
        b1n = b1 / np.maximum(np.linalg.norm(b1, axis=1, keepdims=True), 1e-9)
        v = b0 - (b0 * b1n).sum(1, keepdims=True) * b1n
        w = b2 - (b2 * b1n).sum(1, keepdims=True) * b1n
        out[:, k] = np.arctan2((np.cross(b1n, v) * w).sum(1), (v * w).sum(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-frames", type=int, default=8000)
    ap.add_argument("--n-probe", type=int, default=1500)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min-sep", type=int, default=20)
    ap.add_argument("--lagtime", type=int, default=20)
    ap.add_argument("--max-dim", type=int, default=10)
    a = ap.parse_args()
    from deeptime.decomposition import TICA
    rng = np.random.default_rng(0)

    R = np.asarray(np.load(a.dataset)["R"], np.float32)
    idx = np.linspace(0, len(R) - 1, min(a.n_frames, len(R))).astype(int)
    Rs = R[idx]; n = R.shape[1]
    pairs = np.array(list(itertools.combinations(range(n), 2)), int)
    quads = np.array(list(itertools.permutations(range(n), 4)), int)
    quads = np.array([q for q in quads if q[0] < q[3]], int)      # drop reversals
    print(f"{n} beads -> {len(pairs)} pair distances, {len(quads)} quadruple dihedrals")

    Xd = build_features(R, pairs)
    th = dihedrals(R, quads)
    Xdih = np.concatenate([np.sin(th), np.cos(th)], 1)
    Xboth = np.concatenate([Xd, Xdih], 1)

    def tica(X, dim):
        model = TICA(lagtime=a.lagtime, dim=dim).fit(X).fetch_model()
        Y = np.asarray(model.transform(X))[idx]
        return (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-12), model

    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(Rs); psi = m.cvs["psi"].evaluate(Rs)
    Xrama = np.stack([np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)),
                      np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi))], 1)
    dpair = np.linalg.norm(Rs[:, pairs[:, 0]] - Rs[:, pairs[:, 1]], axis=-1)
    probe = rng.choice(len(Rs), min(a.n_probe, len(Rs)), replace=False)
    ri = rng.integers(0, len(Rs), (2000, 2))
    floor = float(np.median([kabsch_rmsd_many(Rs[i], Rs[j][None])[0] for i, j in ri]))
    ceil = float(np.median(knn_rmsd(Rs, dpair, a.k, probe, a.min_sep)))

    print(f"\n{'featurisation / space':34s} {'2 comps':>9} {'4 comps':>9} {'8 comps':>9}")
    for name, X in [("TICA on pair distances", Xd),
                    ("TICA on dihedral sin/cos", Xdih),
                    ("TICA on distances+dihedrals", Xboth)]:
        Y, model = tica(X, a.max_dim)
        ts = np.atleast_1d(model.timescales(lagtime=a.lagtime))[:4]
        vals = [float(np.median(knn_rmsd(Rs, Y[:, :d], a.k, probe, a.min_sep))) for d in (2, 4, 8)]
        print(f"{name:34s} " + " ".join(f"{v:9.3f}" for v in vals) +
              f"   timescales {' '.join(f'{t:.1f}' for t in ts)}")
    print(f"\n{'Rama (phi,psi) [system-specific]':34s} "
          f"{float(np.median(knn_rmsd(Rs, Xrama, a.k, probe, a.min_sep))):9.3f}")
    print(f"{'full pair-distance descriptor':34s} {ceil:9.3f}   <- ceiling")
    print(f"{'random pairs':34s} {floor:9.3f}   <- floor")


if __name__ == "__main__":
    main()
