#!/usr/bin/env python3
"""TPT-lite corridor extraction: minimum-free-energy paths on the latent grid.

Builds the empirical free-energy surface F(cell) = -kT ln(N_cell/N) on the
acquisition-flow latent grid (floor cost for empty cells so corridors stay
traversable), then runs Dijkstra between basin-core cells. This is a
trajectory-only surrogate for transition-path theory: good enough to place
corridor labels, not a claim about true MFEPs (cells with no reference mass
have floor-dominated costs).

Usage: python -m sampling.tpt_paths --outdir <dir> [--n-frames 200001]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc
from analysis.sampling.diagnostics_common import KT, load_latent

BASINS = ("beta", "alphaR", "alphaL")
PAIRS = [("beta", "alphaR"), ("beta", "alphaL"), ("alphaR", "alphaL")]


def _log(msg):
    print(msg, flush=True)


def fes_grid(counts, n_frames, kT=KT, empty_floor=0.5):
    """Per-cell free energy; empty cells get the `empty_floor`-frame cost."""
    p = np.maximum(counts, empty_floor) / n_frames
    return -kT * np.log(p)


def min_fes_path(counts, src, dst, kT=KT, empty_floor=0.5):
    """Dijkstra shortest path under pairwise-mean FES edge costs (8-connectivity)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    bins_side = int(round(np.sqrt(counts.size)))
    F = fes_grid(counts, counts.sum(), kT, empty_floor)

    def idx(ix, iy):
        return ix * bins_side + iy

    rows, cols, weights = [], [], []
    for ix in range(bins_side):
        for iy in range(bins_side):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < bins_side and 0 <= jy < bins_side:
                        rows.append(idx(ix, iy))
                        cols.append(idx(jx, jy))
                        weights.append(0.5 * (F[idx(ix, iy)] + F[idx(jx, jy)]))
    G = csr_matrix((weights, (rows, cols)), shape=(counts.size, counts.size))
    _, predecessors = dijkstra(G, indices=src, return_predecessors=True)
    if not np.isfinite(predecessors[dst]) and predecessors[dst] < 0:
        return None, np.inf
    path, node = [dst], dst
    while node != src:
        node = predecessors[node]
        path.append(int(node))
    return path[::-1], float(sum(
        0.5 * (F[a] + F[b]) for a, b in zip(path[:-1], path[1:])))


def selfcheck():
    """Two deep wells joined only through a gap in a zero wall."""
    c = np.zeros(20 * 20)
    well_a = [r * 20 + col for r in range(2, 8) for col in range(2, 8)]
    well_b = [r * 20 + col for r in range(12, 18) for col in range(12, 18)]
    c[well_a] = 500
    c[well_b] = 500
    gap = [10 * 20 + col for col in range(8, 12)]
    c[gap] = 300                                   # the only crossing
    blocked = [10 * 20 + col for col in range(0, 8)]
    c[blocked] = 300                               # decoys left of the wall
    src = int(np.argmax(c))
    dst = len(c) - 1 - int(np.argmax(c[::-1]))
    path, _ = min_fes_path(c.astype(float), src, dst)
    assert path is not None, "no path found"
    cols_on_path = {p % 20 for p in path}
    rows_on_path = {p // 20 for p in path}
    assert any(p in gap for p in path), "path must cross the gap"
    assert max(rows_on_path) >= 10 and min(rows_on_path) <= 9
    print(f"[selfcheck] path crosses the gap cells (cols {sorted(cols_on_path)}). OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    dc.add_latent_arguments(ap)
    ap.add_argument("--n-frames", type=int, default=200001)
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--states-per-cell", type=int, default=4,
                    help="planned frozen-state budget per path cell")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = dc.config_from_args(a)
    if a.selfcheck:
        selfcheck()

        a.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lat = load_latent(a.frames, a.n_frames, a.grid_lim, a.bins, config=cfg)
    inside, flat, bins = lat["inside"], lat["flat"], lat["bins"]
    counts = np.bincount(flat[inside], minlength=bins * bins).astype(float)
    centers = lat["centers"]
    _log(f"[load] {int(inside.sum())} frames on grid, {time.time()-t0:.0f}s")

    # basin-core cells: densest cell among that basin's frames
    core_cells = {}
    for b in BASINS:
        m = inside & (lat["region"] == b)
        cc = np.bincount(flat[m], minlength=bins * bins)
        core_cells[b] = int(np.argmax(cc))
        ix, iy = divmod(core_cells[b], bins)
        _log(f"[core] {b}: cell ({centers[ix]:+.3f},{centers[iy]:+.3f}), "
             f"N={int(cc[core_cells[b]])}")

    paths = {}
    for b1, b2 in PAIRS:
        p, cost = min_fes_path(counts, core_cells[b1], core_cells[b2], kT=cfg.kT)
        if p is None:
            _log(f"[path] {b1}->{b2}: NONE")
            continue
        cells = [c for c in p]
        paths[f"{b1}-{b2}"] = {
            "cells": cells,
            "u": [[float(centers[c // bins]), float(centers[c % bins])] for c in cells],
            "barrier_kcal_mol": cost,
            "planned_states": len(cells) * a.states_per_cell,
            "n_empty_cells": int((counts[cells] == 0).sum()),
        }
        _log(f"[path] {b1}->{b2}: {len(cells)} cells "
             f"({paths[f'{b1}-{b2}']['n_empty_cells']} empty), "
             f"path-integrated FES {cost:.1f} kcal/mol, "
             f"planned {len(cells)*a.states_per_cell} states")

    total_planned = sum(v["planned_states"] for v in paths.values())
    summary = {"provenance": {"frames": a.frames, "n_frames": int(len(lat["R"])),
                              "bins": bins, "grid_lim": a.grid_lim,
                              "states_per_cell": a.states_per_cell},
               "core_cells": core_cells, "paths": paths,
               "total_planned_corridor_states": total_planned}
    (a.outdir / "tpt_paths.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(a.outdir / "tpt_paths.npz",
                        counts=counts.reshape(bins, bins),
                        edges=lat["edges"],
                        **{f"path_{k}": np.asarray(v["cells"])
                           for k, v in paths.items()})
    plot(a.outdir, lat["edges"], centers, bins, counts, paths)
    _log(f"[total] planned corridor states: {total_planned}")
    _log(f"wrote {a.outdir}/tpt_paths.json .npz .png")


def plot(outdir, edges, centers, bins, counts, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = fes_grid(counts.ravel(), counts.sum())
    Fg = np.where(counts.ravel() > 0, F, np.nan).reshape(bins, bins)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.pcolormesh(edges, edges, Fg.T, cmap="viridis_r")
    colors = {"beta-alphaR": "#eb6834", "beta-alphaL": "#1baf7a",
              "alphaR-alphaL": "#2a78d6"}
    for name, spec in paths.items():
        u = np.asarray(spec["u"])
        ax.plot(u[:, 0], u[:, 1], "-", color=colors.get(name, "k"), lw=2.2,
                label=name)
    ax.set_xlabel("$u_1$"), ax.set_ylabel("$u_2$")
    ax.set_title("Minimum-FES paths between basin cores (empirical grid)")
    fig.colorbar(im, ax=ax, label="FES (kcal/mol)")
    ax.legend()
    fig.savefig(outdir / "tpt_paths.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
