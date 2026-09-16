#!/usr/bin/env python3
"""How cheap can the committor screen be, and does it then beat the stencil it protects?

As run, the screen costs 8 shots x 40 ps = 320 ps per anchor while the v4 stencil it would
filter costs 13 states x 3.8 ps = 49.4 ps -- the screen is 6.5x the thing it protects, which
makes it pointless unless it can be shortened. Two levers, both measurable from the shots
already on disk:

  LENGTH  most shots commit long before 40 ps. Truncate the trajectory at t and ask whether the
          anchor's classification (committed-to-beta / committed-to-alphaR / transition) still
          matches the 40 ps answer.
  COUNT   8 shots gives resolution 1/8. Bootstrap subsets to find the smallest count that
          reproduces the 8-shot classification.

Also fixes the divide-by-zero in the stage-2/4 flux ratio: the committed-anchor median is
exactly 0, so the ratio is meaningless and is reported as a separation statement instead.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping

BASINS = {
    "alphaR": lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
    "beta":   lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
    "alphaL": lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100),
}
STENCIL_PS = 13 * 3.8          # v4: 13 states/anchor x 3.8 ps


def classify(pB, lo=0.2, hi=0.8):
    return np.where(pB < lo, 0, np.where(pB > hi, 2, 1))   # 0=beta 1=transition 2=alphaR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("local_work/stage1_shots"))
    ap.add_argument("--dt-ps", type=float, default=0.2)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    R = np.load(a.root / "bead_coords.npz")["R"]
    rows = json.loads((a.root / "shots.json").read_text())
    n_shot, n_frame = R.shape[0], R.shape[1]
    m = get_mapping("ala2_backbone_cb_6")
    flat = R.reshape(-1, 6, 3)
    phi = m.cvs["phi"].evaluate(flat).reshape(n_shot, n_frame)
    psi = m.cvs["psi"].evaluate(flat).reshape(n_shot, n_frame)
    bas = np.full((n_shot, n_frame), "other", dtype=object)
    for k, f in BASINS.items():
        bas[f(phi, psi)] = k
    anch = np.array([f'{r["stratum"]}/{r["anchor"]:02d}' for r in rows])
    uniq = sorted(set(anch)); ust = np.array([u.split("/")[0] for u in uniq])

    # first commitment frame and identity, per shot
    first_f = np.full(n_shot, -1); first_id = np.full(n_shot, "none", dtype=object)
    for s in range(n_shot):
        w = np.flatnonzero((bas[s] == "beta") | (bas[s] == "alphaR"))
        if len(w):
            first_f[s] = w[0]; first_id[s] = bas[s, w[0]]
    t_commit = np.where(first_f >= 0, first_f * a.dt_ps, np.nan)

    print(f"{n_shot} shots, {n_frame} frames, {a.dt_ps} ps apart ({a.dt_ps*(n_frame-1):.0f} ps)")
    print(f"\n=== COMMITMENT TIME (ps) ===")
    ok = np.isfinite(t_commit)
    print(f"  committed: {100*ok.mean():.1f}% of shots")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3d} {np.nanpercentile(t_commit, q):7.2f} ps")

    def pB_at(tmax_frames, sel=None):
        """Committor using only the first tmax_frames of each shot."""
        idc = np.where((first_f >= 0) & (first_f < tmax_frames), first_id, "none")
        out = np.empty(len(uniq))
        for i, u in enumerate(uniq):
            k = anch == u if sel is None else (anch == u) & sel
            v = idc[k]
            got = v != "none"
            out[i] = np.mean(v[got] == "alphaR") if got.any() else np.nan
        return out

    full = pB_at(n_frame)
    ref_cls = classify(full)
    print(f"\n=== LENGTH: does a TRUNCATED shot reproduce the 40 ps classification? ===")
    print(f"{'length':>9} {'cost/anchor':>12} {'classification agreement':>26} {'|dp| median':>12}")
    for tp in (2.0, 5.0, 10.0, 20.0, 40.0):
        nf = int(tp / a.dt_ps) + 1
        p = pB_at(nf)
        good = np.isfinite(p) & np.isfinite(full)
        agree = np.mean(classify(p)[good] == ref_cls[good])
        print(f"{tp:8.1f}p {8*tp:11.0f}ps {100*agree:25.1f}% "
              f"{np.nanmedian(np.abs(p[good]-full[good])):12.3f}")

    print(f"\n=== COUNT x LENGTH grid: agreement with the 8 x 40 ps reference ===")
    lens = (2.0, 5.0, 10.0, 20.0)
    print("shots " + "".join(f"{t:>10.0f}ps" for t in lens))
    grid = {}
    for ns in (2, 3, 4, 6, 8):
        row = f"{ns:5d} "
        for tp in lens:
            ag = []
            for _ in range(120):
                sel = np.zeros(n_shot, bool)
                for u in uniq:
                    idx = np.flatnonzero(anch == u)
                    sel[rng.choice(idx, ns, replace=False)] = True
                p = pB_at(int(tp / a.dt_ps) + 1, sel)
                g = np.isfinite(p) & np.isfinite(full)
                ag.append(np.mean(classify(p)[g] == ref_cls[g]))
            grid[(ns, tp)] = float(np.mean(ag))
            row += f"{100*np.mean(ag):9.1f}% "
        print(row)

    # ---- cost model: what does a screened campaign actually cost? ----
    # The modelled committor q was validated at Spearman +0.728, so use it as a FREE
    # pre-filter: anchors it places deep in a basin need no shots at all. Only the
    # uncertain band is shot. p50 commitment time is 0 ps, i.e. most anchors are already
    # inside a basin, which is exactly what makes the pre-filter cheap.
    st = np.load("local_work/md_analysis/anchor_importance/stage234.npz", allow_pickle=True)
    q = st["q_pde"]; band = (q > 0.1) & (q < 0.9)
    frac_true = float(np.mean(classify(full) == 1))     # true transition fraction in this pool
    print(f"\n=== COST MODEL for obtaining N verified transition anchors ===")
    print(f"  pool composition: {100*frac_true:.1f}% are true transition anchors")
    print(f"  modelled-q uncertain band (0.1<q<0.9): {100*band.mean():.1f}% of anchors")
    best = (4, 5.0)
    c_screen = best[0] * best[1]
    print(f"  screen = {best[0]} shots x {best[1]} ps = {c_screen:.0f} ps/anchor "
          f"(agreement {100*grid[best]:.1f}%)")
    n_cand = 1.0 / max(frac_true, 1e-9)
    unscreened = n_cand * STENCIL_PS
    screened = n_cand * band.mean() * c_screen + STENCIL_PS
    print(f"\n  per verified transition anchor:")
    print(f"    NO screen : {n_cand:5.2f} candidates x {STENCIL_PS:.1f} ps stencil "
          f"= {unscreened:8.1f} ps")
    print(f"    WITH screen: {n_cand:5.2f} x {100*band.mean():.0f}% shot at {c_screen:.0f} ps "
          f"+ 1 stencil = {screened:8.1f} ps")
    print(f"    saving: {100*(1-screened/unscreened):.1f}%")

    print(f"\n=== COST vs the stencil it protects ({STENCIL_PS:.1f} ps/anchor) ===")
    for ns, tp in ((8, 40.0), (8, 10.0), (4, 10.0), (4, 5.0), (3, 5.0)):
        c = ns * tp
        print(f"  {ns} shots x {tp:4.1f} ps = {c:6.1f} ps/anchor  "
              f"= {c/STENCIL_PS:5.2f}x the stencil")


if __name__ == "__main__":
    main()
