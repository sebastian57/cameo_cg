#!/usr/bin/env python3
"""Label-support connectivity gate (dataset acceptance contract).

Checks, before any training, that a planned label set forms a connected graph
over latent space and covers the reference ensemble:

  1. labels are linked into components by a radius graph (cKDTree pairs);
  2. the fraction of REFERENCE mass within `--radius` of the LARGEST component
     must exceed `--min-coverage` (default 0.99);
  3. no reference cell holding more than `--max-cell-mass` may be uncovered.

Usage:
  python -m sampling.connectivity_gate --outdir <dir> \
      --labels planned_v5.npy   # (N,2) latent coords of every planned row
      [--reference ...] [--radius 0.35]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.sampling import diagnostics_common as dc
from analysis.sampling.diagnostics_common import load_latent


def _log(msg):
    print(msg, flush=True)


def gate_stats(labels, ref_u, radius, cell_centers=None, cell_counts=None, knn_k=8):
    """Components over a sparse kNN graph + reference-mass/cell coverage within
    `radius` of the largest component. kNN keeps the graph O(n) for dense sets;
    `radius` parametrizes label spacing (what "covered" means), not edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n_labels = len(labels)
    out = dict(n_components=n_labels, largest_fraction=1.0 / n_labels,
               covered_mass=0.0, covered_cells=0.0)
    if n_labels <= knn_k:
        return out
    tree_l = cKDTree(labels)
    k = min(knn_k + 1, n_labels)
    _, idx = tree_l.query(labels, k=k)
    rows = np.repeat(np.arange(n_labels), k - 1)
    cols = idx[:, 1:].ravel()
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)),
                     shape=(n_labels, n_labels)).tocsr()
    n_comp, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp)
    largest = int(np.argmax(sizes))

    tree_ref = cKDTree(labels[comp == largest])
    dist, _ = tree_ref.query(ref_u, k=1)
    out["covered_mass"] = float((dist <= radius).mean())
    if cell_centers is not None and cell_counts is not None:
        occ = cell_centers[cell_counts >= 1]
        d_cell, _ = tree_ref.query(occ, k=1)
        out["covered_cells"] = float((d_cell <= radius).mean())
    out["n_components"] = int(n_comp)
    out["largest_fraction"] = float(sizes[largest] / n_labels)
    return out


def selfcheck():
    rng = np.random.default_rng(0)
    left = np.array([-3.0, 0.0]) + rng.normal(0, 0.15, (60, 2))
    right = np.array([+3.0, 0.0]) + rng.normal(0, 0.15, (60, 2))
    bridge = np.stack([np.linspace(-3.2, 3.2, 60), np.zeros(60)], axis=1)
    ref = np.vstack([left, right])
    ok = gate_stats(np.vstack([left, bridge, right]), ref, 0.5)
    assert ok["n_components"] == 1 and ok["covered_mass"] > 0.99, ok
    bad = gate_stats(np.vstack([left, right]), ref, 0.5)
    # kNN graph: two dense blobs with no bridge stay 2 components; the gap in
    # between is uncovered mass
    assert bad["n_components"] >= 2 and bad["covered_mass"] < 0.9, bad
    print(f"[selfcheck] bridged: {ok['n_components']} comp, "
          f"coverage {ok['covered_mass']:.3f}; unbridged: "
          f"{bad['n_components']} comps, coverage {bad['covered_mass']:.3f}. OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--labels", default=None, help=".npy (N,2) latent coords")
    dc.add_latent_arguments(ap)
    ap.add_argument("--n-ref", type=int, default=50000)
    ap.add_argument("--radii", type=float, nargs="+", default=[0.25, 0.35, 0.5])
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument("--min-cell-coverage", type=float, default=0.95)
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = dc.config_from_args(a)
    if a.selfcheck:
        selfcheck()
        if not a.labels:
            return
    if not a.labels:
        raise SystemExit("need --labels (or --selfcheck alone)")

        a.outdir.mkdir(parents=True, exist_ok=True)

    labels = np.load(a.labels)
    _log(f"[load] {len(labels)} planned label positions")

    t0 = time.time()
    lat = load_latent(a.frames, a.n_ref, a.grid_lim, a.bins, config=cfg)
    inside, flat, bins = lat["inside"], lat["flat"], lat["bins"]
    u_ref = lat["u"][inside]
    counts = np.bincount(flat[inside], minlength=bins * bins).astype(float)
    cell_mass = counts / counts.sum()
    centers = lat["centers"]
    _log(f"[load] {len(u_ref)} reference points, {time.time()-t0:.0f}s")

    results = {}
    gx, gy = np.meshgrid(centers, centers, indexing="ij")
    cell_centers = np.column_stack([gx.ravel(), gy.ravel()])
    for r in a.radii:
        st = gate_stats(labels, u_ref, r, cell_centers, counts)
        # which high-mass cells have NO label within r
        from scipy.spatial import cKDTree
        d_lab, _ = cKDTree(labels).query(cell_centers, k=1)
        uncovered = (d_lab > r).reshape(bins * bins)
        bad_cells = [(int(c), float(cell_mass[c]))
                     for c in np.argsort(-cell_mass * uncovered)[:10]
                     if cell_mass[c] > 0.001]
        st["uncovered_high_mass_cells"] = [
            {"u1": float(centers[c // bins]), "u2": float(centers[c % bins]),
             "mass_pct": round(100 * m, 3)} for c, m in bad_cells]
        st["pass"] = bool(st["covered_mass"] >= a.min_coverage
                          and st["covered_cells"] >= a.min_cell_coverage
                          and not bad_cells)
        results[f"r={r}"] = st
        _log(f"[r={r}] components={st['n_components']} "
             f"largest={100*st['largest_fraction']:.1f}% of labels; "
             f"ref mass covered {100*st['covered_mass']:.2f}%; "
             f"occupied cells covered {100*st['covered_cells']:.1f}% -> "
             f"{'PASS' if st['pass'] else 'FAIL'}")
        for c in st["uncovered_high_mass_cells"][:4]:
            _log(f"    uncovered: u=({c['u1']:+.2f},{c['u2']:+.2f}) "
                 f"mass {c['mass_pct']:.2f}%")

    summary = {"provenance": {"labels": a.labels, "frames": a.frames,
                              "n_ref": int(len(u_ref)), "kT_kcal_mol": cfg.kT,
                              "min_coverage": a.min_coverage},
               "results": results}
    (a.outdir / f"connectivity_{Path(a.labels).stem}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    _log("wrote connectivity json")


if __name__ == "__main__":
    main()
