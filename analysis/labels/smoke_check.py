#!/usr/bin/env python3
"""Validate a live GROMACS+PLUMED run of the targeted bias.

Three checks that only a real MD run can make:
  1. LIVE bias consistency -- PLUMED's printed treg.bias vs our field evaluated at the tic1/tic2
     PLUMED printed on the same line. This exercises the whole chain (atoms -> DISTANCE ->
     COMBINE -> EXTERNAL grid) inside mdrun, not in `plumed driver`.
  2. Domain safety -- how close the CVs came to the grid edge; EXTERNAL aborts if they leave it.
  3. Ensemble effect -- basin occupancy vs an unbiased control on identical settings.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.targeted_bias import field_callable

KT = 0.5921868690749673


def colvar(p):
    rows = [l.split() for l in Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--control", type=Path, default=None)
    ap.add_argument("--sens", required=True)
    a = ap.parse_args()

    B = np.load(a.dir / "bias_field.npz")
    C = colvar(a.dir / "colvar.dat")
    t, z = C[:, 0], C[:, 1:3]
    print(f"{len(C)} colvar rows, {t[0]:.1f} -> {t[-1]:.1f} ps")

    # 1. live bias consistency
    fn = field_callable(B["cx"], B["cy"], B["V"], float(B["sigma"]))
    Vours = fn(z)[0]
    d = np.abs(C[:, 3] - Vours)
    rng = float(B["V"].max() - B["V"].min())
    print(f"\n1. LIVE bias consistency (PLUMED treg.bias vs our field at PLUMED's own CVs)")
    print(f"   max |dV| {d.max():.3e}  median {np.median(d):.3e} kcal/mol"
          f"   ({d.max()/KT:.4f} kT max, bias range {rng:.3f})")

    # 2. domain safety
    gx, gy = B["gx"], B["gy"]
    m1 = min(z[:, 0].min() - gx[0], gx[-1] - z[:, 0].max())
    m2 = min(z[:, 1].min() - gy[0], gy[-1] - z[:, 1].max())
    print(f"\n2. DOMAIN safety (EXTERNAL aborts if a CV leaves the grid)")
    print(f"   tic1 sampled [{z[:,0].min():+.3f}, {z[:,0].max():+.3f}]  "
          f"grid [{gx[0]:+.3f}, {gx[-1]:+.3f}]  margin {m1:+.3f}")
    print(f"   tic2 sampled [{z[:,1].min():+.3f}, {z[:,1].max():+.3f}]  "
          f"grid [{gy[0]:+.3f}, {gy[-1]:+.3f}]  margin {m2:+.3f}")
    if C.shape[1] > 5:
        w = C[:, 4] + C[:, 5]
        print(f"   wall energy: nonzero on {int((w>1e-9).sum())}/{len(C)} frames, "
              f"max {w.max():.3f} kcal/mol")

    # 3. ensemble effect
    z0 = np.load(a.sens, allow_pickle=True)
    ex, ey, lab, rho = z0["ex"], z0["ey"], z0["labels"], z0["rho"]
    allb = [int(v) for v in np.unique(lab) if v >= 0]
    keep = [i for i in allb if float(rho[lab == i].sum()) >= 0.01]
    names = {int(k): v for k, v in zip(keep, ["beta", "alphaR", "alphaL"])}

    def occ(zz):
        ix = np.clip(np.digitize(zz[:, 0], ex) - 1, 0, len(ex) - 2)
        iy = np.clip(np.digitize(zz[:, 1], ey) - 1, 0, len(ey) - 2)
        l = lab[ix, iy]
        return {i: float((l == i).mean()) for i in keep}, float((l < 0).mean())

    ob, unassigned = occ(z)
    print(f"\n3. ENSEMBLE effect (200 ps -- far too short to equilibrate; mechanics only)")
    hdr = f"   {'basin':10s} {'reference':>10s} {'biased':>9s}"
    oc = None
    if a.control and (a.control / "colvar.dat").exists():
        oc, _ = occ(colvar(a.control / "colvar.dat")[:, 1:3])
        hdr += f" {'control':>9s}"
    print(hdr)
    for i in keep:
        ref = float(rho[lab == i].sum())
        line = f"   {names.get(i,str(i)):10s} {100*ref:9.2f}% {100*ob[i]:8.2f}%"
        if oc is not None:
            line += f" {100*oc[i]:8.2f}%"
        print(line)
    print(f"   outside any basin: {100*unassigned:.2f}%")


if __name__ == "__main__":
    main()
