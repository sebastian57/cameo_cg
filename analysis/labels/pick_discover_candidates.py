#!/usr/bin/env python3
"""DHH-v3 Step 2a: draw screening candidates from the Stage-1 discovery frames.

Covers ALL occupied TICA cells, not just the high-occupancy ones: the point of screening is to
learn the importance of cells a density- or occupancy-driven draw would never probe. Allocation
per cell is a FLOOR plus a share proportional to occupancy, so rare discovered cells are
represented without letting beta swamp the set.

Writes AA .gro directly from the discovery xtc with mdtraj -- no gmx, no scan of a 17.7 GB trr
(extracting 100 frames from the reference trr cost 54 minutes).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping

PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)])
BEADS0 = [4, 6, 8, 10, 14, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", type=Path, required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--floor", type=int, default=3, help="minimum candidates per occupied cell")
    ap.add_argument("--stride", type=int, default=5, help="subsample frames when scanning")
    ap.add_argument("--min-visits", type=int, default=5,
                    help="drop cells below this strided frame count (fly-throughs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    import mdtraj as md

    reps = sorted(p for p in a.discover.glob("replica_*") if p.is_dir())
    print(f"{len(reps)} replicas")
    Y, keep_ref = [], []
    z = np.load(a.sens, allow_pickle=True)
    for n, r in enumerate(reps):
        t = md.load(str(r / "biased.xtc"), top=str(r / "biased.gro"), stride=a.stride)
        b = t.xyz[:, BEADS0, :] * 10.0
        d = np.linalg.norm(b[:, PAIRS[:, 0], :] - b[:, PAIRS[:, 1], :], axis=-1)
        Y.append((d - z["tica_mean"]) @ z["tica_coefficients"])
        keep_ref += [(n, int(f)) for f in range(len(b))]
        if n % 16 == 0: print(f"  {n}/{len(reps)}", flush=True)
    Y = np.concatenate(Y); keep_ref = np.array(keep_ref)
    print(f"{len(Y)} frames scanned (stride {a.stride})")

    ex, ey = z["ex"], z["ey"]
    ix = np.clip(np.digitize(Y[:, 0], ex) - 1, 0, len(ex) - 2)
    iy = np.clip(np.digitize(Y[:, 1], ey) - 1, 0, len(ey) - 2)
    cell = ix * (len(ey) - 1) + iy
    cells, counts = np.unique(cell, return_counts=True)
    print(f"{len(cells)} occupied TICA cells (any occupancy)")
    # KB DISCOVER_HARVEST_HARVEST.md stage-1 check: "a cell touched by one frame is a
    # fly-through, not a discovery". Screening those spends 16 ps to learn what the visit
    # count already says, so require real residence before a cell earns candidates.
    keep = counts >= a.min_visits
    print(f"{keep.sum()} cells with >= {a.min_visits} frames (stride {a.stride}, "
          f"i.e. >= {a.min_visits*a.stride} raw); dropped {(~keep).sum()} fly-through cells "
          f"holding {counts[~keep].sum()} frames")
    cells, counts = cells[keep], counts[keep]

    # floor per cell + occupancy-proportional surplus
    quota = np.full(len(cells), a.floor, dtype=int)
    surplus = a.n - quota.sum()
    if surplus > 0:
        w = counts / counts.sum()
        quota += np.floor(surplus * w).astype(int)
    quota = np.minimum(quota, counts)
    print(f"quota: {quota.sum()} candidates, per-cell min {quota.min()} max {quota.max()}")

    picks = []
    for c, q in zip(cells, quota):
        idx = np.flatnonzero(cell == c)
        picks.append(rng.choice(idx, int(q), replace=False))
    picks = np.concatenate(picks)
    print(f"selected {len(picks)}")

    a.out.mkdir(parents=True, exist_ok=True)
    recs = []
    by_rep = {}
    for gi, p in enumerate(picks):
        rep, fr = int(keep_ref[p, 0]), int(keep_ref[p, 1])
        by_rep.setdefault(rep, []).append((gi, fr))
    for rep, items in sorted(by_rep.items()):
        t = md.load(str(reps[rep] / "biased.xtc"), top=str(reps[rep] / "biased.gro"),
                    stride=a.stride)
        for gi, fr in items:
            t[fr].save_gro(str(a.out / f"cand_{gi:05d}.gro"))
            recs.append({"cand": gi, "replica": reps[rep].name, "frame_strided": fr,
                         "cell": int(cell[picks[gi]]),
                         "tic": [float(Y[picks[gi], 0]), float(Y[picks[gi], 1])]})
        print(f"  wrote {len(items)} from {reps[rep].name}", flush=True)
    (a.out / "candidates.json").write_text(json.dumps(recs))
    print(f"\nwrote {len(recs)} candidate .gro to {a.out}")


if __name__ == "__main__":
    main()
