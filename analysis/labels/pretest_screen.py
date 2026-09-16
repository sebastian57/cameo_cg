#!/usr/bin/env python3
"""Pre-test: what would the committor screen have said about an EXISTING Stage-3 anchor set?

Screens the 1,998 anchors of `stencil_armM` (built by the v3 recipe with pref-frac 0.5 on the
campaign pool) and reports:

  * the composition of that anchor set by committor class;
  * how much of its ~23 node-hours went to each class;
  * whether the FREE modelled-q pre-filter reproduces the screen, since that is what decides
    whether a full campaign needs to shoot every candidate or only an uncertain band.

The screen does NOT simply reject committed anchors -- a stencil campaign needs basin-interior
coverage (channel 1) as well as transition anchors (channel 2). What it provides is the ability
to allocate the two DELIBERATELY instead of accepting whatever the anchor draw happens to give.
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
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)])
STENCIL_PS = 13 * 3.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("local_work/pretest_shots"))
    ap.add_argument("--camp", type=Path, default=Path("local_work/stencil_armM_campaign"))
    ap.add_argument("--sens", required=True)
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
            out.append(t.xyz[:, [4, 6, 8, 10, 14, 16], :] * 10.0)
            if n % 2000 == 0: print(f"  {n}/{len(rr)}", flush=True)
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
    pB, reach = np.empty(len(uniq)), np.empty(len(uniq))
    for i, u in enumerate(uniq):
        v = first[anch == u]; g = v != "none"
        reach[i] = g.mean()
        pB[i] = np.mean(v[g] == "alphaR") if g.any() else np.nan
    cls = np.where(np.isnan(pB), 3, np.where(pB < 0.2, 0, np.where(pB > 0.8, 2, 1)))
    names = {0: "committed beta", 1: "TRANSITION", 2: "committed alphaR",
             3: "no commitment (alphaL-like)"}

    print(f"\n{len(uniq)} anchors of stencil_armM screened, 8 shots x "
          f"{a.dt_ps*(n_frame-1):.0f} ps each ({n_shot*a.dt_ps*(n_frame-1)/1000:.1f} ns total)")
    print(f"\n=== COMPOSITION of the existing Stage-3 anchor set ===")
    print(f"{'class':30s} {'n':>6s} {'share':>8s} {'stencil ps spent':>18s}")
    for c in (0, 2, 1, 3):
        k = cls == c
        print(f"{names[c]:30s} {k.sum():6d} {100*k.mean():7.1f}% "
              f"{k.sum()*STENCIL_PS:17.0f}")
    tot = len(uniq) * STENCIL_PS
    print(f"{'TOTAL':30s} {len(uniq):6d} {100.0:7.1f}% {tot:17.0f}")

    # free pre-filter: modelled q at the anchor
    T = np.load(a.camp / "targets.npy")
    A = T[::25][:len(uniq)]
    d = np.linalg.norm(A[:, PAIRS[:, 0], :] - A[:, PAIRS[:, 1], :], axis=-1)
    z = np.load(a.sens, allow_pickle=True)
    Y = (d - z["tica_mean"]) @ z["tica_coefficients"]
    ix = lambda v, e: np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    q = z["q"][ix(Y[:, 0], z["ex"]), ix(Y[:, 1], z["ey"])]
    band = (q > 0.1) & (q < 0.9)
    committed_true = np.isin(cls, (0, 2))
    print(f"\n=== FREE modelled-q pre-filter on this set ===")
    print(f"  uncertain band 0.1<q<0.9         : {100*band.mean():5.1f}% of anchors")
    print(f"  of anchors OUTSIDE the band, truly committed: "
          f"{100*committed_true[~band].mean():5.1f}%   <- pre-filter precision")
    print(f"  TRANSITION anchors missed by the band       : "
          f"{100*np.mean((cls == 1) & ~band)/max(np.mean(cls == 1),1e-9):5.1f}%")
    scr = band.mean() * 16.0
    print(f"\n=== WHAT A SCREENED REBUILD WOULD COST ===")
    print(f"  screen only the band: {100*band.mean():.1f}% x 16 ps = {scr:.1f} ps/anchor "
          f"= {scr/STENCIL_PS:.3f}x the stencil")
    print(f"  for {len(uniq)} anchors: {len(uniq)*scr/1000:.2f} ns screen vs "
          f"{tot/1000:.1f} ns of stencil MD ({100*len(uniq)*scr/tot:.1f}%)")
    np.savez_compressed(a.root / "pretest.npz", anchor=uniq, p_alphaR=pB, reached=reach,
                        cls=cls, q=q, band=band)
    print(f"\nwrote {a.root/'pretest.npz'}")


if __name__ == "__main__":
    main()
