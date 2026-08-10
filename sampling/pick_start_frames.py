#!/usr/bin/env python3
"""Pick N conformationally diverse starting structures from the AA reference run.

Independent velocity seeds alone leave every replica starting in the same basin, so
they all sample the same neighbourhood for the first part of the run. Distinct
starting *conformations* are what spread replicas across the FES.

Frames are chosen by k-means++-style farthest-point selection on the periodic
(phi, psi) of the CG mapping, so the set spans the Ramachandran map rather than
clustering wherever the reference spent most of its time.

    python -m sampling.pick_start_frames --n 12 --out <dir>

Writes `start_<i>.gro` plus `start_frames.json`. NOTE: the reference trajectory lives
on /p, which is NOT mounted on compute nodes -- run this on the login node and keep
the outputs under /e.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .mapping import get_mapping

REF_TRR = "/p/project1/cameo/edelkoetter2/ala2/constrained/production/md_ala2.trr"
REF_CG = ("/e/project1/cameo/edelkoetter2/work/cameo_cg/local_work/input_data/"
          "ala2_cg_backbone_CB_aggforce.npz")
GMX_MODULES = "Stages/2025 GCC/13.3.0 ParaStationMPI/5.11.0-1 GROMACS/2024.3-PLUMED-2.9.3"


def _farthest_point(phi: np.ndarray, psi: np.ndarray, n: int, seed: int = 0):
    """Farthest-point selection under periodic angular distance."""
    rng = np.random.default_rng(seed)
    pts = np.stack([phi, psi], axis=1)
    chosen = [int(rng.integers(len(pts)))]
    d = np.full(len(pts), np.inf)
    for _ in range(n - 1):
        last = pts[chosen[-1]]
        dd = np.abs(pts - last)
        dd = np.minimum(dd, 360.0 - dd)          # periodic
        d = np.minimum(d, np.hypot(dd[:, 0], dd[:, 1]))
        chosen.append(int(d.argmax()))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--mapping", type=str, default="ala2_backbone_cb_6")
    ap.add_argument("--trr", type=str, default=REF_TRR)
    ap.add_argument("--tpr", type=str, required=True,
                    help="a tpr matching the reference system (for trjconv)")
    ap.add_argument("--cg-npz", type=str, default=REF_CG)
    ap.add_argument("--frame-stride-ps", type=float, default=5.0,
                    help="time between reference CG frames (5 ps at nstfout=2500, dt=2fs)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cv-box", action="append", default=None, metavar="cv:lo:hi",
                    help="restrict the candidate pool to a CV box before farthest-point "
                         "selection, repeatable. Use to seed replicas INSIDE a specific "
                         "basin (e.g. alphaL) rather than spread over the whole map.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = get_mapping(args.mapping)
    R = np.load(args.cg_npz)["R"]
    # only the first half has a known 1:1 correspondence with the reference trajectory
    n_ref = min(len(R), 100001)
    R = R[:n_ref]
    phi = m.cvs["phi"].evaluate(R)
    psi = m.cvs["psi"].evaluate(R)

    pool = np.arange(len(phi))
    if args.cv_box:
        keep = np.ones(len(phi), dtype=bool)
        for spec in args.cv_box:
            cv, lo, hi = spec.split(":")
            v = m.cvs[cv].evaluate(R)
            keep &= (v >= float(lo)) & (v <= float(hi))
        pool = np.flatnonzero(keep)
        if len(pool) < args.n:
            raise SystemExit(f"only {len(pool)} frames in the CV box, need {args.n}")
        print(f"CV box: {len(pool)} candidate frames")
    sub = _farthest_point(phi[pool], psi[pool], args.n, args.seed)
    idx = [int(pool[i]) for i in sub]
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for r, fi in enumerate(idx):
        t_ps = fi * args.frame_stride_ps
        gro = args.out / f"start_{r:02d}.gro"
        # Resumable: every -dump rescans the whole (9.5 GB) reference, so a restart that
        # redid finished structures would cost ~1 min each for nothing. Selection is
        # deterministic given --seed, so an existing file is the right file.
        if gro.exists() and gro.stat().st_size > 0:
            records.append({"replica": r, "frame": int(fi), "time_ps": float(t_ps),
                            "phi": float(phi[fi]), "psi": float(psi[fi]),
                            "gro": str(gro), "reused": True})
            print(f"  replica {r:02d}: reusing existing {gro.name}")
            continue
        cmd = (f"module --force purge >/dev/null 2>&1; "
               f"module load {GMX_MODULES} >/dev/null 2>&1; "
               f"gmx trjconv -f {args.trr} -s {args.tpr} -dump {t_ps} -pbc whole -o {gro}")
        p = subprocess.run(["bash", "-lc", cmd], input="System\n", text=True,
                           capture_output=True)
        if p.returncode != 0 or not gro.exists():
            raise RuntimeError(f"trjconv failed for frame {fi} (t={t_ps} ps)\n{p.stderr[-1200:]}")
        records.append({"replica": r, "frame": int(fi), "time_ps": float(t_ps),
                        "phi": float(phi[fi]), "psi": float(psi[fi]),
                        "gro": str(gro)})
        print(f"  replica {r:02d}: frame {fi:6d}  t={t_ps:9.1f} ps  "
              f"phi={phi[fi]:+7.1f}  psi={psi[fi]:+7.1f}")

    (args.out / "start_frames.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} structures to {args.out}")


if __name__ == "__main__":
    main()
