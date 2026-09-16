#!/usr/bin/env python3
"""DHH-v3 Step 2b: classify the screened discovery candidates and build the window-placement map.

Answers the question Step 3 depends on: **does the MetaD discovery pool actually contain
transition configurations, at a useful rate?** The tempered anchor draw of `stencil_armM` held
only 0.5% (9 of 1,998). If the discovery pool is no better, transition anchors must be
MANUFACTURED by umbrella windows rather than found by screening.

Outputs a per-TICA-cell transition density, which is the allocation input for Step 3:
    windows_per_cell = floor + surplus * (transition density)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping

BASINS = {"alphaR": lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
          "beta":   lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
          "alphaL": lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100)}
BEADS0 = [4, 6, 8, 10, 14, 16]
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("local_work/step2_shots"))
    ap.add_argument("--cands", type=Path, default=Path("local_work/step2_candidates"))
    ap.add_argument("--sens", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dt-ps", type=float, default=0.2)
    a = ap.parse_args()

    cache = a.root / "bead_coords.npz"
    if cache.exists():
        R = np.load(cache)["R"]
    else:
        import mdtraj as md
        rr = json.loads((a.root / "shots.json").read_text())
        out = []
        for n, r in enumerate(rr):
            t = md.load(str(a.root / r["case"] / "biased.xtc"),
                        top=str(a.root / r["case"] / "seed.gro"))
            out.append(t.xyz[:, BEADS0, :] * 10.0)
            if n % 4000 == 0: print(f"  {n}/{len(rr)}", flush=True)
        nmin = min(len(x) for x in out)
        R = np.stack([x[:nmin] for x in out]); np.savez_compressed(cache, R=R.astype(np.float32))
    rows = json.loads((a.root / "shots.json").read_text())
    n_shot, n_frame = R.shape[0], R.shape[1]
    m = get_mapping("ala2_backbone_cb_6")
    flat = R.reshape(-1, 6, 3)
    phi = m.cvs["phi"].evaluate(flat).reshape(n_shot, n_frame)
    psi = m.cvs["psi"].evaluate(flat).reshape(n_shot, n_frame)
    bas = np.full((n_shot, n_frame), "other", dtype=object)
    for k, f in BASINS.items(): bas[f(phi, psi)] = k

    anch = np.array([r["anchor"] for r in rows]); uniq = np.array(sorted(set(anch)))
    first = np.full(n_shot, "none", dtype=object)
    for s in range(n_shot):
        w = np.flatnonzero((bas[s] == "beta") | (bas[s] == "alphaR"))
        if len(w): first[s] = bas[s, w[0]]
    pB = np.full(len(uniq), np.nan)
    for i, u in enumerate(uniq):
        v = first[anch == u]; g = v != "none"
        if g.any(): pB[i] = np.mean(v[g] == "alphaR")
    cls = np.where(np.isnan(pB), 3, np.where(pB < 0.2, 0, np.where(pB > 0.8, 2, 1)))

    # where does each candidate START, and in which TICA cell
    start = R[:, 0, :, :][np.searchsorted(anch, uniq)]
    d0 = np.linalg.norm(start[:, PAIRS[:, 0], :] - start[:, PAIRS[:, 1], :], axis=-1)
    z = np.load(a.sens, allow_pickle=True)
    Y0 = (d0 - z["tica_mean"]) @ z["tica_coefficients"]
    ex, ey = z["ex"], z["ey"]
    ix = np.clip(np.digitize(Y0[:, 0], ex) - 1, 0, len(ex) - 2)
    iy = np.clip(np.digitize(Y0[:, 1], ey) - 1, 0, len(ey) - 2)
    p0 = m.cvs["phi"].evaluate(start); s0 = m.cvs["psi"].evaluate(start)
    where = np.full(len(uniq), "other", dtype=object)
    for k, f in BASINS.items(): where[f(p0, s0)] = k

    names = {0: "committed beta", 1: "TRANSITION", 2: "committed alphaR", 3: "no commitment"}
    print(f"\n{len(uniq)} screened discovery candidates, 8 shots x "
          f"{a.dt_ps*(n_frame-1):.0f} ps ({n_shot*a.dt_ps*(n_frame-1)/1000:.1f} ns)")
    print(f"\n{'class':22s} {'n':>6s} {'share':>8s} | " +
          " ".join(f"{b:>8s}" for b in ("beta", "alphaR", "alphaL", "other")))
    print("-" * 74)
    for c in (0, 2, 1, 3):
        k = cls == c
        row = f"{names[c]:22s} {k.sum():6d} {100*k.mean():7.2f}% | "
        for b in ("beta", "alphaR", "alphaL", "other"):
            row += f"{int((k & (where == b)).sum()):8d} "
        print(row)
    nt = int((cls == 1).sum())
    print(f"\nTRANSITION yield: {nt}/{len(uniq)} = {100*nt/len(uniq):.2f}%")
    print(f"  vs the tempered stencil_armM anchor draw: 9/1998 = 0.45%")
    print(f"  -> discovery pool is {100*nt/len(uniq)/0.45:.1f}x richer in transition anchors")

    # per-cell transition density -> Step 3 allocation input
    cellid = ix * (len(ey) - 1) + iy
    cells = np.unique(cellid)
    dens = np.array([np.mean(cls[cellid == c] == 1) for c in cells])
    ncand = np.array([int((cellid == c).sum()) for c in cells])
    hot = dens > 0
    print(f"\n{len(cells)} cells hold candidates; {int(hot.sum())} contain >=1 transition "
          f"candidate ({100*hot.mean():.1f}%)")
    print(f"  transition density in those cells: median {np.median(dens[hot]):.2f}, "
          f"max {dens.max():.2f}")
    np.savez_compressed(a.out, cand=uniq, p_alphaR=pB, cls=cls, where=where.astype(str),
                        cell=cellid, tic=Y0, cells=cells, trans_density=dens, n_cand=ncand)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
