#!/usr/bin/env python3
"""Out-of-sample test of the screening recipe.

The 92.8% agreement for `4 shots x 2 ps` was measured on the SAME 200 anchors used to pick the
recipe, drawn from 4 deliberately stratified CV boxes. These 100 anchors were drawn with NO box
(farthest-point over the whole map), i.e. a different distribution, and the recipe was fixed
before they were run. Anything here is generalisation, not fit.
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
classify = lambda pB: np.where(pB < 0.2, 0, np.where(pB > 0.8, 2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dt-ps", type=float, default=0.2)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cache = a.root / "bead_coords.npz"
    if cache.exists():
        R = np.load(cache)["R"]
    else:
        import mdtraj as md
        rows0 = json.loads((a.root / "shots.json").read_text())
        out = []
        for n, r in enumerate(rows0):
            t = md.load(str(a.root / r["case"] / "biased.xtc"),
                        top=str(a.root / r["case"] / "seed.gro"))
            out.append(t.xyz[:, [4, 6, 8, 10, 14, 16], :] * 10.0)
            if n % 200 == 0: print(f"  {n}/{len(rows0)}", flush=True)
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
    anch = np.array([r["anchor"] for r in rows]); uniq = sorted(set(anch))
    first_f = np.full(n_shot, -1); first_id = np.full(n_shot, "none", dtype=object)
    for s in range(n_shot):
        w = np.flatnonzero((bas[s] == "beta") | (bas[s] == "alphaR"))
        if len(w): first_f[s] = w[0]; first_id[s] = bas[s, w[0]]

    def pB_at(nf, sel=None):
        idc = np.where((first_f >= 0) & (first_f < nf), first_id, "none")
        o = np.empty(len(uniq))
        for i, u in enumerate(uniq):
            k = (anch == u) if sel is None else ((anch == u) & sel)
            v = idc[k]; g = v != "none"
            o[i] = np.mean(v[g] == "alphaR") if g.any() else np.nan
        return o

    full = pB_at(n_frame); ref = classify(full)
    print(f"\n{len(uniq)} OUT-OF-SAMPLE anchors (no CV box), {n_shot} shots, "
          f"{a.dt_ps*(n_frame-1):.0f} ps")
    print(f"  true transition fraction: {100*np.mean(ref == 1):.1f}%")
    print(f"  committed within 40 ps  : {100*np.mean(np.isfinite(full)):.1f}%")
    print(f"\n{'recipe':>18} {'cost/anchor':>12} {'agreement':>11}")
    for ns, tp in ((4, 2.0), (4, 5.0), (3, 2.0), (6, 2.0), (8, 2.0)):
        ag = []
        for _ in range(300):
            sel = np.zeros(n_shot, bool)
            for u in uniq:
                idx = np.flatnonzero(anch == u)
                sel[rng.choice(idx, ns, replace=False)] = True
            p = pB_at(int(tp / a.dt_ps) + 1, sel)
            g = np.isfinite(p) & np.isfinite(full)
            ag.append(np.mean(classify(p)[g] == ref[g]))
        star = "  <- the recipe" if (ns, tp) == (4, 2.0) else ""
        print(f"{ns} shots x {tp:4.1f} ps {ns*tp:11.0f}ps {100*np.mean(ag):10.1f}%{star}")


if __name__ == "__main__":
    main()
