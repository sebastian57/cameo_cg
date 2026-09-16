#!/usr/bin/env python3
"""DHH-v3 Step 3 yield: did the transition-block windows MANUFACTURE transition frames?

The whole Step-3 design rests on this. Half the 384 windows were placed on configurations the
Step-2 committor screen classified as TRANSITION; the other half on coverage. If the transition
block does not hold the system near the dividing surface, the harvest inherits the same
starvation the tempered anchor draw had (9 transition anchors in 1,998).

Compares, per block, the fraction of frames outside every basin box -- the operational
definition of "transition" used throughout this project.
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
    ap.add_argument("--camp", type=Path, default=Path("local_work/dhh3_stage3_harvest"))
    ap.add_argument("--centres", type=Path, default=Path("local_work/step3_centres.npz"))
    ap.add_argument("--discard-ps", type=float, default=20.0)
    a = ap.parse_args()
    import mdtraj as md

    kind = np.load(a.centres, allow_pickle=True)["kind"].astype(str)
    cases = sorted(p for p in a.camp.glob("case_*") if p.is_dir())
    print(f"{len(cases)} cases, centres: {int((kind=='transition').sum())} transition / "
          f"{int((kind=='coverage').sum())} coverage")
    m = get_mapping("ala2_backbone_cb_6")

    tot = {k: {b: 0 for b in list(BASINS) + ["transition"]} for k in ("transition", "coverage")}
    per_case_trans = {"transition": [], "coverage": []}
    for ci, c in enumerate(cases):
        t = md.load(str(c / "biased.xtc"), top=str(c / "seed.gro"))
        n0 = int(a.discard_ps / (t.time[1] - t.time[0])) if len(t) > 1 else 0
        b = unwrap_beads(t.xyz[n0:, BEADS0, :] * 10.0, t.unitcell_vectors[n0:] * 10.0)
        phi = m.cvs["phi"].evaluate(b); psi = m.cvs["psi"].evaluate(b)
        lab = np.full(len(phi), "transition", dtype=object)
        for k, f in BASINS.items(): lab[f(phi, psi)] = k
        kd = kind[ci] if ci < len(kind) else "coverage"
        for b_ in tot[kd]: tot[kd][b_] += int((lab == b_).sum())
        per_case_trans[kd].append(float((lab == "transition").mean()))
        if ci % 64 == 0: print(f"  {ci}/{len(cases)}", flush=True)

    print(f"\n{'block':12s} {'frames':>9s} " +
          " ".join(f"{b:>11s}" for b in ("beta", "alphaR", "alphaL", "transition")))
    print("-" * 62)
    for kd in ("transition", "coverage"):
        n = sum(tot[kd].values())
        row = f"{kd:12s} {n:9d} "
        for b_ in ("beta", "alphaR", "alphaL", "transition"):
            row += f"{100*tot[kd][b_]/max(n,1):10.2f}% "
        print(row)
    nt = sum(tot[k]["transition"] for k in tot)
    N = sum(sum(tot[k].values()) for k in tot)
    print(f"\nTOTAL transition frames: {nt} of {N} = {100*nt/N:.2f}%")
    print(f"  reference (same definition, TICA watershed): 651 of 200,001 = 0.33%")
    print(f"  campaign_targeted_v1: 2,438 of 116,040 = 2.10%")
    for kd in ("transition", "coverage"):
        v = np.array(per_case_trans[kd])
        print(f"  per-window transition fraction, {kd:10s}: median {np.median(v):.3f} "
              f"p90 {np.percentile(v,90):.3f} max {v.max():.3f}")


if __name__ == "__main__":
    main()
