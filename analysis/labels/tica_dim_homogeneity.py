#!/usr/bin/env python3
"""How structurally tight is a TICA neighbourhood as a function of DIMENSIONALITY?

The 2D test (job 1517293) found TICA cells 2.24x structurally broader than Rama cells.
That could mean TICA is a poor structural coordinate, OR simply that 2 components cannot
pin a ~12-dimensional internal space. This separates the two by sweeping the number of
components used.

kNN formulation rather than binning: 6D equal-occupancy grids are hopeless with 10^4
frames. For each query, take its k nearest neighbours in the chosen coordinate space and
measure the proper-rotation Kabsch RMSD to them.

Baselines:
  full 16-D descriptor  -> the CEILING (best structural match achievable by any coordinate)
  Rama (phi,psi)        -> the 2D comparison
  random pairs          -> the FLOOR (no coordinate at all)

Each TIC is standardised to unit variance so components are weighted equally.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features
from sampling.mapping import get_mapping


def kabsch_rmsd_many(P, Qs):
    P = P - P.mean(0)
    out = np.empty(len(Qs))
    for i, Q in enumerate(Qs):
        Q = Q - Q.mean(0)
        U, S, Vt = np.linalg.svd(Q.T @ P)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        S = S.copy(); S[2] *= d
        out[i] = np.sqrt(max((P**2).sum() + (Q**2).sum() - 2.0*S.sum(), 0.0) / len(P))
    return out


def knn_rmsd(R, X, k, probe, min_sep, chunk=256):
    """median Kabsch RMSD from each probe frame to its k nearest neighbours in space X."""
    vals = []
    n = len(X)
    for s in range(0, len(probe), chunk):
        qi = probe[s:s+chunk]
        d = np.sqrt(np.maximum(((X[qi][:, None, :] - X[None, :, :])**2).sum(-1), 0.0))
        sep = np.abs(qi[:, None] - np.arange(n)[None, :])
        d[sep < min_sep] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        for r, gi in enumerate(qi):
            vals.append(np.median(kabsch_rmsd_many(R[gi], R[nn[r]])))
    return np.asarray(vals)


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
    Rs = R[idx]
    nb = Rs.shape[1]
    pairs = np.array([(i, j) for i in range(nb) for j in range(i+1, nb)], dtype=int)
    Xfull_all = build_features(R, pairs)                      # fit TICA on ALL frames
    model = TICA(lagtime=a.lagtime, dim=a.max_dim).fit(Xfull_all).fetch_model()
    Y = np.asarray(model.transform(Xfull_all))[idx]
    Y = (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-12)         # equal weight per component

    ts = getattr(model, "timescales", None)
    try:
        ts = model.timescales(lagtime=a.lagtime)
    except Exception:
        ts = None
    if ts is not None:
        print("TICA timescales (frames): " + "  ".join(f"{t:.1f}" for t in np.atleast_1d(ts)[:a.max_dim]))

    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(Rs); psi = m.cvs["psi"].evaluate(Rs)
    # periodic angles -> sin/cos so the metric wraps correctly
    Xrama = np.stack([np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)),
                      np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi))], 1)
    v = np.einsum("ni,ni->n", np.cross(Rs[:,1]-Rs[:,0], Rs[:,2]-Rs[:,0]), Rs[:,4]-Rs[:,0])/10.0
    dpair = np.linalg.norm(Rs[:, pairs[:,0]] - Rs[:, pairs[:,1]], axis=-1)
    Xdesc = np.concatenate([dpair, v[:, None]], 1)

    probe = rng.choice(len(Rs), min(a.n_probe, len(Rs)), replace=False)
    ri = rng.integers(0, len(Rs), (2000, 2))
    rand = np.array([kabsch_rmsd_many(Rs[i], Rs[j][None])[0] for i, j in ri])
    floor = float(np.median(rand))

    print(f"\n{len(Rs)} frames, k={a.k} nearest neighbours, {len(probe)} probes")
    print(f"{'coordinate space':28s} {'median RMSD':>12} {'vs random':>10} {'vs descriptor':>14}")
    ceil_v = knn_rmsd(Rs, Xdesc, a.k, probe, a.min_sep)
    ceil = float(np.median(ceil_v))
    rows = []
    for n in [1, 2, 3, 4, 6, 8, a.max_dim]:
        if n > a.max_dim: continue
        v_ = knn_rmsd(Rs, Y[:, :n], a.k, probe, a.min_sep)
        rows.append((f"TICA first {n}", float(np.median(v_))))
    rows.append(("Rama (phi,psi)", float(np.median(knn_rmsd(Rs, Xrama, a.k, probe, a.min_sep)))))
    rows.append(("full 16-D descriptor", ceil))
    rows.append(("random pairs", floor))
    for name, val in rows:
        print(f"{name:28s} {val:12.3f} {val/floor:10.2f} {val/ceil:14.2f}")
    print(f"\nceiling = full descriptor ({ceil:.3f} A); floor = random ({floor:.3f} A).")
    print("If TICA-first-6 approaches the ceiling, 2 components were simply too few.")


if __name__ == "__main__":
    main()
