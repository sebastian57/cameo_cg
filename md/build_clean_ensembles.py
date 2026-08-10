"""Reduce CG MD runs to their USABLE frames, one npz per run, for FES/TICA comparison.

Two corrections are applied, and both change the answer:

1. TRUNCATE AT DISSOCIATION, don't filter it. Dissociation is an absorbing event -- once a
   replica breaks up it never re-forms -- so the usable data is each replica's PREFIX up to
   its first bad frame. Selecting `frames where every bond < max_bond` instead would splice
   together stretches from before and after the break and present them as one ensemble.

2. DISCARD EQUILIBRATION PER REPLICA, not per concatenated array. Every replica starts from
   the same frame (frame_idx 0), so the opening stretch is correlated across all of them
   and over-weights that basin. Discarding a fraction of the concatenated array would
   instead delete whole leading replicas.

The result is NOT an unbiased sample when a run dissociated: those trajectories are cut
exactly where the model starts to misbehave, so surviving frames are conditioned on
not-yet-having-failed, and runs have unequal usable lengths. Read such rows as a ranking.
Runs with 0% dissociation (e.g. anything at dt=1fs so far) carry no such caveat.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from sampling.mapping import get_mapping

warnings.filterwarnings("ignore")


def build(run_dir: Path, out: Path, mapping_name: str, max_bond: float,
          equil_frac: float) -> None:
    m = get_mapping(mapping_name)
    chunks, per_rep = [], []
    for f in sorted(run_dir.glob("*rep??.npz")):
        R = np.load(f)["R"].astype(np.float64)
        bl = np.stack([np.linalg.norm(R[:, i] - R[:, j], axis=-1) for i, j in m.bonds], axis=1)
        bad = ~np.isfinite(bl).all(axis=1) | (bl.max(axis=1) > max_bond)
        cut = int(np.argmax(bad)) if bad.any() else len(R)
        start = int(equil_frac * cut)
        if cut - start > 0:
            chunks.append(R[start:cut].astype(np.float32))
        per_rep.append((f.name.split("_rep")[-1][:2], cut - start, len(R), bad.mean() * 100))
    if not chunks:
        raise SystemExit(f"{run_dir}: no clean frames")
    allR = np.concatenate(chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, R=allR,
                        species=np.tile(np.arange(m.n_beads, dtype=np.int32), (len(allR), 1)))
    total = sum(t for _, _, t, _ in per_rep)
    n_diss = sum(1 for _, _, _, d in per_rep if d > 0)
    print(f"{out.stem:26s} {len(allR):7d} of {total:7d} frames "
          f"({100*len(allR)/total:5.1f}%), {len(allR)*0.2/1000:6.2f} ns, "
          f"{n_diss}/{len(per_rep)} replicas dissociated")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="LABEL=path/to/md_run_dir, repeatable")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--equil-frac", type=float, default=0.10)
    a = ap.parse_args()
    for spec in a.run:
        label, _, path = str(spec).partition("=")
        build(Path(path), a.outdir / f"{label}.npz", a.mapping, a.max_bond, a.equil_frac)


if __name__ == "__main__":
    main()
