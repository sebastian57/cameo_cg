#!/usr/bin/env python3
"""Equivalence test for the emitted PLUMED bias, using `plumed driver` as the oracle.

Checks the two things that can silently be wrong:
  1. the CVs -- PLUMED's DISTANCE+COMBINE tic1/tic2 vs the numpy projection (X-mean)@coef
  2. the bias -- PLUMED's ext.bias vs bilinear interpolation of the grid we wrote

mode=write emits frames.xyz + ref.npz; mode=check reads colvar.dat and compares.
Split because PLUMED and the JAX venv live in CONFLICTING module stacks.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

KCAL_PER_KJ = 4.184

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.md.analyze_traj import build_features
from sampling.targeted_bias import field_callable
from sampling.mapping import get_mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["write", "check"], required=True)
    ap.add_argument("--dataset"); ap.add_argument("--sens")
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--h", type=float, default=1e-3, help="FD step in Angstrom")
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    a = ap.parse_args()

    if a.mode == "write":
        R = np.asarray(np.load(a.dataset)["R"], np.float32)
        z = np.load(a.sens, allow_pickle=True)
        idx = np.linspace(0, len(R) - 1, a.n).astype(int)
        Rs = R[idx]
        X = build_features(Rs, z["pairs"])
        Y = (X - z["tica_mean"]) @ z["tica_coefficients"]

        # The emitted plumed.dat addresses ALL-ATOM indices (a CG bead is one AA atom, e.g.
        # ala2_backbone_cb_6 -> 5,7,9,11,15,17), so the test trajectory must have at least
        # that many atoms with the beads at those positions. Writing a bare 6-bead xyz would
        # force the emitter to use identity indices -- which is exactly how the wrong-atom bug
        # hid. Filler atoms are parked far away and are referenced by nothing.
        m = get_mapping(a.mapping)
        aa = [int(i) - 1 for i in m.aa_atom_indices_1based]
        nat = max(aa) + 1
        full = np.zeros((len(Rs), nat, 3), dtype=float)
        full[:, :, 0] = 500.0 + 3.0 * np.arange(nat)[None, :]
        full[:, aa, :] = Rs
        Rs = full
        print(f"padded to {nat} atoms; beads at 1-based AA indices "
              f"{list(m.aa_atom_indices_1based)}")
        lines = []
        for fr in Rs:
            lines.append(str(fr.shape[0]))
            lines.append("1000.0 1000.0 1000.0")   # plumed driver requires a box here;
            # irrelevant because every DISTANCE is emitted NOPBC and the solute is whole
            for p in fr:
                lines.append(f"X {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        (a.dir / "frames.xyz").write_text("\n".join(lines) + "\n")

        # FORCE TEST: displace every bead along a fixed random unit direction by +-h and
        # emit two more trajectories. PLUMED's dumped forces must satisfy
        #     -sum_i F_i . d_i  =  dV/dlambda  ~=  (V(+h) - V(-h)) / 2h
        # This exercises the WHOLE chain: grid derivatives -> COMBINE chain rule -> atomic
        # forces. Checking the bias ENERGY alone (as the first version did) says nothing
        # about the forces, and the forces are what actually drive the MD.
        rng = np.random.default_rng(0)
        D = rng.normal(size=Rs.shape); D /= np.linalg.norm(D, axis=(1, 2), keepdims=True)
        for tag, sgn in (("plus", +1.0), ("minus", -1.0)):
            ls = []
            for fr, dd in zip(Rs + sgn * a.h * D, D):
                ls.append(str(fr.shape[0])); ls.append("1000.0 1000.0 1000.0")
                for q in fr:
                    ls.append(f"X {q[0]:.9f} {q[1]:.9f} {q[2]:.9f}")
            (a.dir / f"frames_{tag}.xyz").write_text("\n".join(ls) + "\n")
        np.savez(a.dir / "ref.npz", Y=Y, D=D, h=a.h)
        print(f"wrote {a.n} frames -> {a.dir/'frames.xyz'}; TIC1 range "
              f"{Y[:,0].min():.3f}..{Y[:,0].max():.3f}, TIC2 {Y[:,1].min():.3f}..{Y[:,1].max():.3f}")
        return

    Y = np.load(a.dir / "ref.npz")["Y"]
    B = np.load(a.dir / "bias_field.npz")
    rows = [l.split() for l in (a.dir / "colvar.dat").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    C = np.array(rows, dtype=float)
    n = min(len(C), len(Y))
    dt1 = np.abs(C[:n, 1] - Y[:n, 0]).max()
    dt2 = np.abs(C[:n, 2] - Y[:n, 1]).max()
    scale = float(np.std(Y))
    print(f"frames compared: {n}")
    print(f"CV agreement   : max|tic1_plumed - tic1_numpy| = {dt1:.3e}   "
          f"({dt1/scale:.2e} relative to TIC spread {scale:.3f})")
    print(f"                 max|tic2_plumed - tic2_numpy| = {dt2:.3e}   ({dt2/scale:.2e})")
    # Ground truth is the EXACT callable that was tabulated, not a bilinear proxy: any
    # difference is now PLUMED's spline reconstruction error and nothing else.
    fn = field_callable(B["cx"], B["cy"], B["V"], float(B["sigma"]))
    ours = fn(Y[:n])[0]
    db = np.abs(C[:n, 3] - ours)
    rng = float(B["V"].max() - B["V"].min())
    print(f"bias agreement : max|ext.bias_plumed - grid_interp| = {db.max():.3e} kcal/mol "
          f"({db.max()/max(rng,1e-30):.2e} of the {rng:.3f} kcal/mol bias range)")
    print(f"                 median {np.median(db):.3e}")
    # DIAGNOSTIC: PLUMED EXTERNAL interpolates with a SPLINE using the derivative columns;
    # our reference here is BILINEAR. The two differ most where the field bends sharply. If
    # the residual concentrates in high-|grad V| cells the cause is the interpolation scheme,
    # not a corrupted grid. Test that rather than assume it.
    gx, gy = np.gradient(B["V"], B["cx"][1]-B["cx"][0], B["cy"][1]-B["cy"][0])
    gmag = np.sqrt(gx**2 + gy**2)
    jx = np.clip(np.searchsorted(B["cx"], Y[:n, 0]) - 1, 0, len(B["cx"]) - 2)
    jy = np.clip(np.searchsorted(B["cy"], Y[:n, 1]) - 1, 0, len(B["cy"]) - 2)
    gf = gmag[jx, jy]
    hi = gf >= np.quantile(gf, 0.90)
    def _m(v):
        return "n/a" if v.size == 0 else f"max {v.max():.3e}  median {np.median(v):.3e}"
    print(f"  residual, top-decile |grad V| cells: {_m(db[hi])}")
    print(f"  residual, all other cells          : {_m(db[~hi])}")
    KT_ = 0.5921868690749673
    print(f"  max residual = {db.max()/KT_:.4f} kT")
    # ABSOLUTE physical criterion: this test exists to catch "PLUMED is not reading the grid we
    # wrote", which appears as O(bias-range) error EVERYWHERE, not 0.1 kT confined to edges.
    ok = ((dt1 / scale < 1e-4) and (dt2 / scale < 1e-4)
          and (db.max() < 0.1 * KT_) and (np.median(db) < 1e-3 * KT_))
    # ---- FORCE TEST -------------------------------------------------------------------
    fok = True
    ref = np.load(a.dir / "ref.npz")
    fpath = a.dir / "forces.dat"
    if not fpath.exists():
        print("\nFORCE TEST: SKIPPED (no forces.dat) -- forces are UNVERIFIED")
        fok = False
    else:
        D, h = ref["D"], float(ref["h"])
        def bias_col(name):
            r = [l.split() for l in (a.dir / name).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
            A = np.array(r, dtype=float)
            return A[:, 3:].sum(axis=1)   # every printed bias component, walls included or not
        # ALIGNMENT: the three colvar files come from three SEPARATE plumed driver runs. If
        # any run emits a different row count, frame i is compared against frame j, and a few
        # fast-varying frames blow up while the median stays tiny -- exactly the signature
        # seen here. Displacing by h=1e-3 A must move a CV by ~1e-3, never more.
        def cvs(name):
            r = [l.split() for l in (a.dir / name).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
            return np.array(r, dtype=float)
        C0, Cp, Cm = cvs("colvar.dat"), cvs("colvar_plus.dat"), cvs("colvar_minus.dat")
        print(f"  rows: colvar {len(C0)}  plus {len(Cp)}  minus {len(Cm)}")
        na = min(len(C0), len(Cp), len(Cm))
        sh = max(np.abs(Cp[:na, 1] - C0[:na, 1]).max(), np.abs(Cm[:na, 1] - C0[:na, 1]).max())
        print(f"  max |tic1(displaced) - tic1(base)| = {sh:.3e}  (should be ~1e-3; a large "
              f"value means the runs are MISALIGNED)")
        Vp, Vm = bias_col("colvar_plus.dat"), bias_col("colvar_minus.dat")
        nf = min(len(Vp), len(Vm), n)
        fd = (Vp[:nf] - Vm[:nf]) / (2 * h)
        # plumed --dump-forces writes an xyz-like block per frame: a natoms line, a
        # virial/box-derivative line (3 numbers, NO name column), then one "X fx fy fz" per
        # atom. Keep only the named atom rows.
        rows = [l.split()[1:] for l in fpath.read_text().splitlines()
                if l.strip() and not l.startswith("#") and l.split()[0] == "X"]
        nat = D.shape[1]
        F = np.array(rows, dtype=float).reshape(-1, nat, 3)[:nf]
        # UNITS TRAP: `plumed driver --dump-forces` writes ENERGY in PLUMED-internal kJ/mol
        # even when the input sets UNITS ENERGY=kcal/mol -- UNITS governs parsing and PRINT,
        # not this file. Lengths DO follow --length-units A. So forces come out in
        # kJ/mol/A and must be divided by 4.184. Missing this shows up as a relative error of
        # exactly 4.184 - 1 = 3.184 on every frame where the gradient is not ~0, which is the
        # constant "max 3.184" that appeared in every run before it was spotted.
        F = F / KCAL_PER_KJ
        ana = -np.einsum("fij,fij->f", F, D[:nf])
        # translational invariance: the bias depends only on interatomic distances
        net = np.abs(F.sum(axis=1)).max()
        sc = max(np.abs(fd).max(), 1e-12)
        err = np.abs(ana - fd)
        print(f"\nFORCE TEST ({nf} frames, FD step {h} A)")
        print(f"  net force on system (must be ~0): max {net:.3e} kcal/mol/A")
        print(f"  -sum F.d  vs  dV/dlambda : max err {err.max():.3e}, "
              f"median {np.median(err):.3e}  (|dV/dlambda| up to {sc:.3f})")
        print(f"  relative: max {err.max()/sc:.3e}, median {np.median(err)/sc:.3e}")
        # localisation: are the worst frames the ones the walls are acting on?
        wall = (C0[:nf, 4] + C0[:nf, 5]) > 1e-9
        # Both sides of this comparison come from PLUMED ITSELF, so a mismatch means the
        # grid's der_ columns are inconsistent with its values -- most likely at the
        # data/padding junction, where edge replication puts a KINK in the field that a
        # cubic spline cannot represent consistently.
        cxd, cyd = B["cx"], B["cy"]
        din = np.minimum.reduce([Y[:nf,0]-cxd[0], cxd[-1]-Y[:nf,0],
                                 Y[:nf,1]-cyd[0], cyd[-1]-Y[:nf,1]])
        dcell = min(cxd[1]-cxd[0], cyd[1]-cyd[0])
        near = din < 3 * dcell        # within 3 cells of the data-region edge
        print(f"  within 3 cells of data-region edge: {near.sum()}/{nf}; "
              f"max err there {err[near].max() if near.any() else float('nan'):.3e}, "
              f"max err interior {err[~near].max() if (~near).any() else float('nan'):.3e}")
        worst = np.argsort(-err)[:5]
        print(f"  worst 5 frames: err {np.round(err[worst],3)}  "
              f"dist-to-edge(cells) {np.round(din[worst]/dcell,2)}")
        if wall.any():
            print(f"  frames inside a wall: {wall.sum()}/{nf}; "
                  f"max err there {err[wall].max():.3e}, "
                  f"max err elsewhere {err[~wall].max():.3e}")
        fok = (net < 1e-4 * max(np.abs(F).max(), 1e-12) + 1e-6) and (err.max() / sc < 0.05)
        print(f"  FORCE VERDICT: {'PASS' if fok else 'FAIL'}")

    print("\nVERDICT:", "PASS" if (ok and fok) else "FAIL",
          "(CVs < 1e-4 rel; bias max < 0.1 kT, median < 1e-3 kT; forces < 5% and net ~ 0)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
