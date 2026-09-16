#!/usr/bin/env python3
"""Compare PLUMED's atomic forces against the ANALYTIC gradient of the tabulated field.

The finite-difference test compares PLUMED's force to the derivative of PLUMED's own energy.
That cannot say WHICH of the two is wrong. This compares PLUMED's force to the exact analytic
force implied by the field we handed it:

    F_i = -sum_t (dV/dtic_t) * (d tic_t / d r_i),
    d tic_t / d r_i = sum_p coef[p,t] * d d_p / d r_i,   d d_p/d r_i = +-(r_i - r_j)/d_p

with dV/dtic from `field_callable`, i.e. the very function `write_grid_from_fn` tabulated.
No finite difference and no PLUMED energy enter the comparison, so a disagreement localises
squarely in PLUMED's grid DERIVATIVE output (or in the tabulation of it).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.targeted_bias import field_callable
from sampling.mapping import get_mapping


def read_xyz(path):
    L = Path(path).read_text().splitlines()
    frames, k = [], 0
    while k < len(L):
        n = int(L[k].split()[0]); k += 2
        frames.append([[float(v) for v in L[k + i].split()[1:4]] for i in range(n)])
        k += n
    return np.array(frames, dtype=float)


def read_forces(path, nat):
    rows = [l.split()[1:] for l in Path(path).read_text().splitlines()
            if l.strip() and not l.startswith("#") and l.split()[0] == "X"]
    return np.array(rows, dtype=float).reshape(-1, nat, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    a = ap.parse_args()

    z = np.load(a.sens, allow_pickle=True)
    B = np.load(a.dir / "bias_field.npz")
    pairs = np.asarray(z["pairs"]); mean = np.asarray(z["tica_mean"])
    coef = np.asarray(z["tica_coefficients"])
    m = get_mapping(a.mapping)
    aa = [int(i) - 1 for i in m.aa_atom_indices_1based]

    R = read_xyz(a.dir / "frames.xyz")
    nat = R.shape[1]
    Fp = read_forces(a.dir / "forces.dat", nat)
    n = min(len(R), len(Fp)); R, Fp = R[:n], Fp[:n]
    beads = R[:, aa, :]

    diff = beads[:, pairs[:, 0], :] - beads[:, pairs[:, 1], :]
    d = np.linalg.norm(diff, axis=-1)
    u = diff / d[..., None]
    zc = (d - mean) @ coef

    fn = field_callable(B["cx"], B["cy"], B["V"], float(B["sigma"]))
    Vz, dVdz = fn(zc)

    acoef = dVdz @ coef.T                       # (n, P): dV/dd_p
    G = np.zeros_like(beads)
    for p in range(len(pairs)):
        c = acoef[:, p, None] * u[:, p, :]
        G[:, int(pairs[p, 0]), :] += c
        G[:, int(pairs[p, 1]), :] -= c
    Fa = -G                                     # analytic force on the beads

    Fpb = Fp[:, aa, :]
    # Best-fit scalar between the two force sets. `plumed driver --dump-forces` writes ENERGY
    # in PLUMED-internal kJ/mol even when the input says UNITS ENERGY=kcal/mol (UNITS governs
    # parsing and PRINT, not this file), so a clean 4.184 here means the bias is correct and
    # the HARNESS was wrong.
    num = float((Fpb * Fa).sum()); den = float((Fa * Fa).sum())
    k = num / max(den, 1e-30)
    print(f"best-fit scale  F_plumed / F_analytic = {k:.6f}   "
          f"(kcal->kJ is 4.184; 1/0.1 nm per A is 10)")
    # Report against the EXACT constant. A fitted factor would absorb genuine grid error.
    resid = Fpb / 4.184 - Fa
    print(f"after dividing by the EXACT 4.184: max |dF| {np.abs(resid).max():.3e}  "
          f"median {np.median(np.abs(resid)):.3e} kcal/mol/A")
    rel_k = np.abs(resid).max(axis=(1,2)) / np.maximum(np.abs(Fa).max(axis=(1,2)), 1e-12)
    print(f"  relative: max {rel_k.max():.3e}  median {np.median(rel_k):.3e}   "
          f"within 2%: {int((rel_k<0.02).sum())}/{n}")

    filler = np.setdiff1d(np.arange(nat), aa)
    print(f"frames {n}, {nat} atoms, beads at {[i+1 for i in aa]}")
    print(f"force on FILLER atoms (must be 0): max {np.abs(Fp[:, filler, :]).max():.3e}")
    err = np.abs(Fpb - Fa).max(axis=(1, 2))
    scale = np.abs(Fa).max(axis=(1, 2))
    rel = err / np.maximum(scale, 1e-12)
    print(f"|F_plumed - F_analytic|  max {err.max():.3e}  median {np.median(err):.3e} kcal/mol/A")
    print(f"relative to |F_analytic| max {rel.max():.3e}  median {np.median(rel):.3e}")
    ok = int((rel < 0.02).sum())
    print(f"frames within 2% : {ok}/{n}   within 10%: {int((rel<0.10).sum())}/{n}")
    w = np.argsort(-rel)[:6]
    print(f"\nworst frames: idx {w.tolist()}")
    print(f"  rel        {np.round(rel[w], 4).tolist()}")
    print(f"  |F_analytic| {np.round(scale[w], 4).tolist()}")
    print(f"  tic1       {np.round(zc[w,0], 4).tolist()}")
    print(f"  tic2       {np.round(zc[w,1], 4).tolist()}")
    print(f"\ndata bounds tic1 [{B['cx'][0]:.3f}, {B['cx'][-1]:.3f}]  "
          f"tic2 [{B['cy'][0]:.3f}, {B['cy'][-1]:.3f}]")
    print(f"grid  bounds tic1 [{B['gx'][0]:.3f}, {B['gx'][-1]:.3f}]  "
          f"tic2 [{B['gy'][0]:.3f}, {B['gy'][-1]:.3f}]")
    inside = ((zc[:,0] > B['cx'][0]) & (zc[:,0] < B['cx'][-1]) &
              (zc[:,1] > B['cy'][0]) & (zc[:,1] < B['cy'][-1]))
    print(f"frames inside the DATA region: {int(inside.sum())}/{n}")
    print(f"  rel error there : max {rel[inside].max():.3e}  median {np.median(rel[inside]):.3e}")
    if (~inside).any():
        print(f"  rel error OUTSIDE: max {rel[~inside].max():.3e}  median {np.median(rel[~inside]):.3e}")


if __name__ == "__main__":
    main()
