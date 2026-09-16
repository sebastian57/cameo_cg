#!/usr/bin/env python3
"""DHH-v3 Step 4: does UNBIASED dynamics agree the harvested frames are transition states?

Step 3's harvest is 53.94% "transition", but that is the Rama-box complement of the three basins
AND the frames were held there by an umbrella at kappa=450. A restraint can hold a configuration
that unbiased dynamics leaves instantly, so box-membership is an upper bound on the transition
content that Step 5's anchors can actually use.

This screens the harvested candidates with 8 unbiased shots x 2 ps each and cross-tabulates the
committor class against the Rama box the candidate started in. The diagonal is agreement; the
off-diagonal "outside a box but immediately committed" cell is the over-count.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping
from analysis.labels._pbc import unwrap_beads

BASINS = {"alphaR": lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
          "beta":   lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
          "alphaL": lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100)}
BEADS0 = [4, 6, 8, 10, 14, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("local_work/step4_shots"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dt-ps", type=float, default=0.2)
    a = ap.parse_args()

    cache = a.root / "bead_coords.npz"
    rows = json.loads((a.root / "shots.json").read_text())
    if cache.exists():
        R = np.load(cache)["R"]
    else:
        import mdtraj as md
        out = []
        for n, r in enumerate(rows):
            t = md.load(str(a.root / r["case"] / "biased.xtc"),
                        top=str(a.root / r["case"] / "seed.gro"))
            out.append(unwrap_beads(t.xyz[:, BEADS0, :] * 10.0,
                                    t.unitcell_vectors * 10.0))
            if n % 4000 == 0: print(f"  {n}/{len(rows)}", flush=True)
        nmin = min(len(x) for x in out)
        R = np.stack([x[:nmin] for x in out]); np.savez_compressed(cache, R=R.astype(np.float32))
    n_shot, n_frame = R.shape[0], R.shape[1]
    m = get_mapping("ala2_backbone_cb_6")
    flat = R.reshape(-1, 6, 3)
    phi = m.cvs["phi"].evaluate(flat).reshape(n_shot, n_frame)
    psi = m.cvs["psi"].evaluate(flat).reshape(n_shot, n_frame)
    bas = np.full((n_shot, n_frame), "other", dtype=object)
    for k, f in BASINS.items(): bas[f(phi, psi)] = k

    anch = np.array([r["anchor"] for r in rows])
    block = np.array([r["block"] for r in rows])
    uniq = np.array(sorted(set(anch)))
    first = np.full(n_shot, "none", dtype=object)
    for s in range(n_shot):
        w = np.flatnonzero((bas[s] == "beta") | (bas[s] == "alphaR"))
        if len(w): first[s] = bas[s, w[0]]
    pB = np.full(len(uniq), np.nan); blk = np.empty(len(uniq), dtype=object)
    for i, u in enumerate(uniq):
        k = anch == u; v = first[k]; g = v != "none"
        if g.any(): pB[i] = np.mean(v[g] == "alphaR")
        blk[i] = block[k][0]
    cls = np.where(np.isnan(pB), 3, np.where(pB < 0.2, 0, np.where(pB > 0.8, 2, 1)))
    start = bas[np.searchsorted(anch, uniq), 0]      # Rama box the candidate started in

    names = {0: "committed beta", 1: "TRANSITION", 2: "committed alphaR", 3: "no commitment"}
    print(f"\n{len(uniq)} harvested candidates, 8 shots x {a.dt_ps*(n_frame-1):.0f} ps "
          f"({n_shot*a.dt_ps*(n_frame-1)/1000:.1f} ns)")

    print(f"\n=== committor class x starting Rama box ===")
    print(f"{'committor class':22s} {'n':>6s} {'share':>8s} | " +
          " ".join(f"{b:>9s}" for b in ("beta", "alphaR", "alphaL", "other")))
    print("-" * 76)
    for c in (0, 2, 1, 3):
        k = cls == c
        row = f"{names[c]:22s} {k.sum():6d} {100*k.mean():7.2f}% | "
        for b in ("beta", "alphaR", "alphaL", "other"):
            row += f"{int((k & (start == b)).sum()):9d} "
        print(row)

    out_box = start == "other"
    print(f"\n=== the key number: frames OUTSIDE every basin box ===")
    print(f"  candidates outside a box (the '53.94% transition' population): "
          f"{int(out_box.sum())} = {100*out_box.mean():.2f}%")
    if out_box.any():
        agree = float(np.mean(cls[out_box] == 1))
        print(f"  of those, committor says TRANSITION : {100*agree:.2f}%")
        print(f"  of those, immediately COMMITTED     : "
              f"{100*np.mean(np.isin(cls[out_box], (0, 2))):.2f}%   <- box over-count")
        print(f"  of those, no commitment in 2 ps     : "
              f"{100*np.mean(cls[out_box] == 3):.2f}%")

    print(f"\n=== per harvest block ===")
    print(f"{'block':14s} {'n':>6s} {'TRANSITION':>12s} {'committed':>11s} {'no-commit':>11s}")
    for b in ("transition", "coverage"):
        k = blk == b
        if not k.any(): continue
        print(f"{b:14s} {int(k.sum()):6d} {100*np.mean(cls[k]==1):11.2f}% "
              f"{100*np.mean(np.isin(cls[k],(0,2))):10.2f}% {100*np.mean(cls[k]==3):10.2f}%")
    nt = int((cls == 1).sum())
    print(f"\nVERIFIED transition anchors: {nt}/{len(uniq)} = {100*nt/len(uniq):.2f}%")
    print(f"  Step-2 discovery screen : 8.46%")
    print(f"  tempered anchor draw    : 0.45%")
    np.savez_compressed(a.out, cand=uniq, p_alphaR=pB, cls=cls,
                        block=blk.astype(str), start=start.astype(str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
