#!/usr/bin/env python3
"""v5 Stage B extras: corridor + alphaL-rim anchor ids (reference frames only).

Corridor : nearest `--per-cell` reference frames to each TPT-lite path cell.
Rim      : frames whose latent cell is within BFS distance <= 2 of any
           alphaL-basin cell (8-connectivity), sampled without replacement.
Output   : npz with anchor_source=0 (SRC_REFERENCE), anchor_index into the
           full 200,001-frame reference npz. Consumed by
           build_stencil_states_v4.py --extra-anchor-npz.

Usage: python sampling/v5_stage_b_extras.py --outdir local_work/v5_stageB
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tpt-json", required=True)
    ap.add_argument("--outdir", required=True)
    dc.add_latent_arguments(ap)
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--n-rim", type=int, default=6000)
    ap.add_argument("--rim-bfs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()
    cfg = dc.config_from_args(a)
    rng = np.random.default_rng(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    lat = dc.load_latent(a.frames, 200001, config=cfg)          # FULL frame set: index == row id
    u, inside, flat = lat["u"], lat["inside"], lat["flat"]
    bins, centers = lat["bins"], lat["centers"]
    region = lat["region"]

    # ---- corridor -----------------------------------------------------------------
    tp = json.load(open(a.tpt_json))
    path_cells = sorted({c for spec in tp["paths"].values() for c in spec["cells"]})
    pos_in_cell, idx_in_cell = [], []
    for c in range(bins * bins):
        m = flat[inside] == c
        pos_in_cell.append(u[inside][m])
        idx_in_cell.append(np.flatnonzero(inside)[m])

    def nearest_frames(cell, k):
        target = np.array([centers[cell // bins], centers[cell % bins]])
        P, I = pos_in_cell[cell], idx_in_cell[cell]
        if len(P) == 0:
            return []
        d = np.linalg.norm(P - target, axis=1)
        return I[np.argsort(d)[:k]].tolist()

    corr = [i for cell in path_cells for i in nearest_frames(cell, a.per_cell)]
    print(f"[corridor] {len(path_cells)} cells -> {len(corr)} anchors")

    # ---- alphaL rim (BFS <= rim_bfs around alphaL basin cells) --------------------
    alphaL_cells = set(np.unique(flat[inside & (region == "alphaL")]).tolist())
    frontier = set(alphaL_cells)
    rim = set(alphaL_cells)
    for _ in range(a.rim_bfs):
        nxt = set()
        for c in frontier:
            ix, iy = divmod(c, bins)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    j = (ix + dx) * bins + (iy + dy)
                    if 0 <= ix + dx < bins and 0 <= iy + dy < bins and j not in rim:
                        nxt.add(j)
        rim |= nxt
        frontier = nxt
    rim_frames = np.concatenate([idx_in_cell[c] for c in sorted(rim)
                                 if len(idx_in_cell[c]) > 0])
    pick = rng.choice(len(rim_frames), size=min(a.n_rim, len(rim_frames)),
                      replace=False)
    rim_ids = rim_frames[pick].tolist()
    n_alphaL_rim = int((region[rim_ids] == "alphaL").sum())
    print(f"[rim] {len(rim)} cells (bfs<={a.rim_bfs}), {len(rim_frames)} frames "
          f"-> sampled {len(rim_ids)} ({n_alphaL_rim} inside alphaL)")

    ids = np.asarray(corr + rim_ids, np.int64)
    np.savez_compressed(out / "extra_anchors.npz",
                        anchor_source=np.zeros(len(ids), np.int8),
                        anchor_index=ids)
    meta = dict(corridor_cells=len(path_cells), corridor_anchors=len(corr),
                per_cell=a.per_cell, rim_cells=len(rim), rim_anchors=len(rim_ids),
                rim_alphaL_fraction=n_alphaL_rim / max(len(rim_ids), 1),
                total_extras=int(len(ids)))
    (out / "extras_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[total] extras: {len(ids)} -> {out/'extra_anchors.npz'}")


if __name__ == "__main__":
    main()
