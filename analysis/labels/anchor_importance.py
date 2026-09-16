#!/usr/bin/env python3
"""Stages 2-4 of ANCHOR_IMPORTANCE: which anchors matter, measured from unbiased AA shots.

Stage 2  which metric defines "near"? Four candidates scored by how well the residence they
         imply tracks the reference's own local density p_ref.
Stage 3  empirical committor by shooting: of the 8 shots from an anchor, what fraction commit
         to alphaR before beta. Assumption-free -- no diffusive model, no 2D projection.
Stage 4  does the MODELLED s(x) survive? Correlate the empirical committor against the PDE
         committor q, and empirical residence against p_ref.

Reads the shot trajectories once and caches the CG bead coordinates, because 1,600 xtc files
is the expensive part and every stage reuses them.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping

BEADS0 = [4, 6, 8, 10, 14, 16]
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)])
BASINS = {
    "alphaR": lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
    "beta":   lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
    "alphaL": lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100),
}


def load_shots(root: Path, cache: Path):
    if cache.exists():
        z = np.load(cache); print(f"bead coords from cache {cache}")
        return z["R"], json.loads((root / "shots.json").read_text())
    import mdtraj as md
    rows = json.loads((root / "shots.json").read_text())
    out = []
    for n, r in enumerate(rows):
        t = md.load(str(root / r["case"] / "biased.xtc"),
                    top=str(root / r["case"] / "seed.gro"))
        out.append(t.xyz[:, BEADS0, :] * 10.0)          # nm -> A
        if n % 200 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    n_min = min(len(a) for a in out)
    R = np.stack([a[:n_min] for a in out])
    np.savez_compressed(cache, R=R.astype(np.float32))
    print(f"cached {R.shape} to {cache}")
    return R, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("local_work/stage1_shots"))
    ap.add_argument("--sens", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    R, rows = load_shots(a.root, a.root / "bead_coords.npz")
    n_shot, n_frame = R.shape[0], R.shape[1]
    print(f"\n{n_shot} shots x {n_frame} frames ({0.2*(n_frame-1):.1f} ps)")

    m = get_mapping("ala2_backbone_cb_6")
    flat = R.reshape(-1, 6, 3)
    phi = m.cvs["phi"].evaluate(flat).reshape(n_shot, n_frame)
    psi = m.cvs["psi"].evaluate(flat).reshape(n_shot, n_frame)
    d = np.linalg.norm(flat[:, PAIRS[:, 0], :] - flat[:, PAIRS[:, 1], :], axis=-1)

    z = np.load(a.sens, allow_pickle=True)
    Y = ((d - z["tica_mean"]) @ z["tica_coefficients"]).reshape(n_shot, n_frame, 2)
    d = d.reshape(n_shot, n_frame, -1)

    # reference: local density and the descriptor covariance for the whitened metric
    Rref = np.asarray(np.load(a.reference)["R"], np.float64)
    dref = np.linalg.norm(Rref[:, PAIRS[:, 0], :] - Rref[:, PAIRS[:, 1], :], axis=-1)
    Yref = (dref - z["tica_mean"]) @ z["tica_coefficients"]
    ex, ey, rho = z["ex"], z["ey"], z["rho"]
    Cinv = np.linalg.inv(np.cov(dref.T) + 1e-9 * np.eye(dref.shape[1]))

    strat = np.array([r["stratum"] for r in rows])
    anch = np.array([f'{r["stratum"]}/{r["anchor"]:02d}' for r in rows])
    uniq = sorted(set(anch))
    print(f"{len(uniq)} anchors, {n_shot // len(uniq)} shots each")

    # ---------- Stage 2: four notions of "near" ----------
    a0 = d[:, 0, :]                                     # anchor descriptor per shot
    dd = d - a0[:, None, :]
    eucl = np.linalg.norm(dd, axis=-1)
    maha = np.sqrt(np.einsum("sfi,ij,sfj->sf", dd, Cinv, dd))
    ix = lambda v, e: np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    cell_same = ((ix(Y[..., 0], ex) == ix(Y[:, :1, 0], ex)) &
                 (ix(Y[..., 1], ey) == ix(Y[:, :1, 1], ey)))
    def basin_of(p, s):
        out = np.full(p.shape, "other", dtype=object)
        for k, f in BASINS.items():
            out[f(p, s)] = k
        return out
    bas = basin_of(phi, psi)
    basin_same = bas == bas[:, :1]

    # p_ref at each anchor, from the reference TICA histogram
    pr = rho[ix(Y[:, 0, 0], ex), ix(Y[:, 0, 1], ey)]

    print("\n=== STAGE 2: which metric's residence tracks p_ref? ===")
    print(f"{'metric':28s} {'residence (mean)':>17s} {'Spearman vs log p_ref':>23s}")
    from scipy.stats import spearmanr
    res = {}
    for name, val in (("Euclidean descriptor", -eucl.mean(axis=1)),
                      ("Mahalanobis (whitened)", -maha.mean(axis=1)),
                      ("same TICA cell", cell_same.mean(axis=1)),
                      ("same Rama basin", basin_same.mean(axis=1))):
        per = np.array([val[anch == u].mean() for u in uniq])
        prA = np.array([pr[anch == u][0] for u in uniq])
        ok = prA > 0
        rs = spearmanr(per[ok], np.log(prA[ok])).statistic
        res[name] = per
        print(f"{name:28s} {per.mean():17.4f} {rs:23.3f}")

    # ---------- Stage 3: empirical committor ----------
    print("\n=== STAGE 3: empirical committor (first commitment to alphaR vs beta) ===")
    first = np.full(n_shot, "none", dtype=object)
    for s in range(n_shot):
        for f in range(n_frame):
            if bas[s, f] in ("beta", "alphaR"):
                first[s] = bas[s, f]; break
    pB = np.array([np.mean(first[anch == u] == "alphaR") for u in uniq])
    reached = np.array([np.mean(first[anch == u] != "none") for u in uniq])
    ust = np.array([u.split("/")[0] for u in uniq])
    print(f"{'stratum':14s} {'n':>4s} {'mean p_alphaR':>14s} {'committed':>11s} "
          f"{'0.2<p<0.8':>11s}")
    for s in ("beta_core", "alphaR_core", "corridor_bR", "alphaL"):
        k = ust == s
        print(f"{s:14s} {k.sum():4d} {pB[k].mean():14.3f} {reached[k].mean():11.2f} "
              f"{np.mean((pB[k] > 0.2) & (pB[k] < 0.8)):11.2f}")

    # ---------- Stage 4: does the modelled map agree? ----------
    print("\n=== STAGE 4: empirical vs MODELLED ===")
    q = z["q"]
    qA = q[ix(Y[:, 0, 0], ex), ix(Y[:, 0, 1], ey)]
    qU = np.array([qA[anch == u][0] for u in uniq])
    good = np.isfinite(qU) & np.isfinite(pB)
    print(f"empirical committor p_alphaR vs PDE committor q : "
          f"Spearman {spearmanr(pB[good], qU[good]).statistic:+.3f}  (n={good.sum()})")
    s2 = z["s2"][ix(Y[:, 0, 0], ex), ix(Y[:, 0, 1], ey)]
    s2U = np.array([s2[anch == u][0] for u in uniq])
    mid = (pB > 0.2) & (pB < 0.8)
    print(f"channel-2 flux at EMPIRICAL transition anchors  : {np.median(s2U[mid]):.3e}")
    print(f"channel-2 flux at committed anchors             : {np.median(s2U[~mid]):.3e}")
    if mid.any() and (~mid).any():
        print(f"  ratio {np.median(s2U[mid])/max(np.median(s2U[~mid]),1e-30):.2f}x "
              f"({mid.sum()} transition / {(~mid).sum()} committed)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, anchor=np.array(uniq), stratum=ust, p_alphaR=pB,
                        reached=reached, p_ref=np.array([pr[anch == u][0] for u in uniq]),
                        q_pde=qU, s2=s2U, **{k.replace(" ", "_"): v for k, v in res.items()})
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
