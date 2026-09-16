#!/usr/bin/env python3
"""Measure INDEPENDENT VISITS per nanosecond from pilot colvar files.

KB DESIGN/CG_ACQUISITION_AND_ASSEMBLY.md: visits, not frames, set the statistical floor
(SE(F) = sigma/sqrt(N_visits), and sigma_F is flat across regions). A visit is counted only
when residence in a region lasts >= `--min-res` ps -- the SAME >=5 ps criterion used for the
reference's 236 alphaL visits, so the numbers are directly comparable. Without that threshold
a trajectory skittering across a boundary inflates the count (the 8.6x -> 2x correction).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REF_VISITS = {"beta": 3710, "alphaR": 4429, "alphaL": 236, "transition": 3411}
REF_FRAMES = {"beta": 130400, "alphaR": 59945, "alphaL": 5576, "transition": 4080}


def colvar(p):
    rows = [l.split() for l in Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return np.array(rows, dtype=float)


def visits(labels, times, min_res):
    """Count maximal runs of a constant label lasting >= min_res ps."""
    out, n = {}, len(labels)
    if n == 0:
        return out
    dt = float(np.median(np.diff(times))) if n > 1 else 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        if (j - i + 1) * dt >= min_res:
            out[labels[i]] = out.get(labels[i], 0) + 1
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--min-res", type=float, default=5.0, help="ps; matches the reference")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    z = np.load(a.sens, allow_pickle=True)
    ex, ey, lab, rho = z["ex"], z["ey"], z["labels"], z["rho"]
    allb = [int(v) for v in np.unique(lab) if v >= 0]
    keep = [i for i in allb if float(rho[lab == i].sum()) >= 0.01]
    order = sorted(keep, key=lambda i: -float(rho[lab == i].sum()))
    name = {order[0]: "beta", order[1]: "alphaR", order[2]: "alphaL"}

    tot_v, tot_f, tot_ns = {}, {}, 0.0
    for d in a.dirs:
        C = colvar(Path(d) / "colvar.dat")
        t, zz = C[:, 0], C[:, 1:3]
        ix = np.clip(np.digitize(zz[:, 0], ex) - 1, 0, len(ex) - 2)
        iy = np.clip(np.digitize(zz[:, 1], ey) - 1, 0, len(ey) - 2)
        raw = lab[ix, iy]
        lb = np.array([name.get(int(v), "transition") if v in name else "transition"
                       for v in raw])
        tot_ns += (t[-1] - t[0]) / 1000.0
        for k, v in visits(lb, t, a.min_res).items():
            tot_v[k] = tot_v.get(k, 0) + v
        for k in set(lb):
            tot_f[k] = tot_f.get(k, 0) + int((lb == k).sum())

    print(f"\n=== {a.label or 'pilot'} : {len(a.dirs)} replicas, {tot_ns:.2f} ns total, "
          f"visit threshold >= {a.min_res} ps ===")
    print(f"{'region':11s} {'occ%':>7s} {'frames':>8s} {'visits':>7s} {'vis/ns':>8s} "
          f"{'fr/visit':>9s} {'ns for REF-parity':>18s}")
    nf = sum(tot_f.values())
    for k in ["beta", "alphaR", "alphaL", "transition"]:
        f, v = tot_f.get(k, 0), tot_v.get(k, 0)
        vpn = v / tot_ns if tot_ns else 0.0
        need = (REF_VISITS[k] / vpn) if vpn > 0 else float("inf")
        print(f"{k:11s} {100*f/max(nf,1):6.2f}% {f:8d} {v:7d} {vpn:8.1f} "
              f"{(f/v if v else float('nan')):9.1f} {need:18.1f}")
    print(f"\nreference for comparison: " +
          "  ".join(f"{k} {REF_VISITS[k]}v/{REF_FRAMES[k]}f" for k in REF_VISITS))


if __name__ == "__main__":
    main()
