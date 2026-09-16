#!/usr/bin/env python3
"""Is a region's reference sampling STRUCTURALLY saturated, or would more frames still add?

The "visits, not frames" rule is about estimating ENSEMBLE AVERAGES, where correlated frames
are worthless. Force matching is different: it fits U(x) pointwise, and two frames from the
same visit are distinct configurations, not duplicates. So the right question for a training
set is whether a region's frames still cover NEW structure as you add more.

Measured here by the median nearest-neighbour distance in the full pair-distance descriptor
(the space the model actually sees; KB DESIGN/DDF_SENSITIVITY_PLACEMENT.md establishes it as
the structural ceiling) as a function of subsample size N. If the curve has flattened by the
region's full count, more frames add little new structure; if it is still falling as ~N^(-1/d),
they do.

A pure power law is the signature of unsaturated coverage; a plateau is saturation.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--lagtime", type=int, default=20)
    ap.add_argument("--probe", type=int, default=1500, help="query points per (region, N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--origin-filter", type=int, default=None,
                    help="restrict to one source: 0=reference, or None for all")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    _z = np.load(a.dataset)
    R = np.asarray(_z["R"], np.float32)
    if a.origin_filter is not None and "origin" in _z:
        keep = _z["origin"] == a.origin_filter
        R = R[keep]
        print(f"origin filter {a.origin_filter}: {len(R)} frames")
    nb = R.shape[1]
    pairs = np.array([(i, j) for i in range(nb) for j in range(i + 1, nb)], dtype=int)
    X = build_features(R, pairs)
    z = np.load(a.sens, allow_pickle=True)
    # Project onto the REFERENCE TICA stored in the sensitivity npz. Refitting TICA per dataset
    # (as this did) yields different axes, so the watershed labels -- which were built on the
    # reference TICA -- get applied to the wrong coordinates. That silently reported alphaL as
    # 95 frames for the assembled set when it holds ~47,000.
    Y = (X - z["tica_mean"]) @ z["tica_coefficients"]
    ex, ey, lab, rho = z["ex"], z["ey"], z["labels"], z["rho"]
    keep = [int(i) for i in np.unique(lab) if i >= 0 and float(rho[lab == i].sum()) >= 0.01]
    order = sorted(keep, key=lambda i: -float(rho[lab == i].sum()))
    names = {order[0]: "beta", order[1]: "alphaR", order[2]: "alphaL"}
    ix = np.clip(np.digitize(Y[:, 0], ex) - 1, 0, len(ex) - 2)
    iy = np.clip(np.digitize(Y[:, 1], ey) - 1, 0, len(ey) - 2)
    reg = lab[ix, iy]

    groups = {v: np.where(reg == k)[0] for k, v in names.items()}
    groups["transition"] = np.where(~np.isin(reg, list(names)))[0]

    print(f"{len(R)} frames, descriptor dim {X.shape[1]}")
    print(f"\nmedian nearest-neighbour descriptor distance vs subsample size N")
    print(f"(falling = still covering new structure; flat = saturated)\n")
    sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    hdr = f"{'region':11s} {'total':>8s} " + " ".join(f"{n:>7d}" for n in sizes)
    print(hdr); print("-" * len(hdr))
    for name, idx in groups.items():
        if len(idx) < 500:
            print(f"{name:11s} {len(idx):8d}   (too few frames)"); continue
        row = f"{name:11s} {len(idx):8d} "
        for N in sizes:
            if N > len(idx):
                row += f"{'-':>7s} "; continue
            sub = rng.choice(idx, N, replace=False)
            Xi = X[sub]
            q = rng.choice(len(sub), min(a.probe, len(sub)), replace=False)
            d = np.linalg.norm(Xi[q][:, None, :] - Xi[None, :, :], axis=-1)
            d[np.arange(len(q)), q] = np.inf
            row += f"{np.median(d.min(axis=1)):7.4f} "
        print(row)
    print("\nA halving of N should raise the NN distance by ~2^(1/d_eff) if unsaturated;")
    print("no change means the region's structural space is already filled.")


if __name__ == "__main__":
    main()
