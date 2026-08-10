"""Assemble a dataset with a FIXED FRAME QUOTA PER REGION, ignoring equilibrium weight.

This inverts the 2026-08-05 assembly rule on purpose, and the justification is the
2026-08-08 result that REM corrects populations DOWNWARD well and UPWARD badly:

    REM400   alphaL 71.9% -> 3.21%   (25x over -> correct)   beta 1.09 -> 61.76 too
    REM-500  alphaL  0.08% -> 0.78%  (35x under -> only 10x of the needed 35x)

REM's gradient is <dU/dtheta>_ref - <dU/dtheta>_model. The model-side term needs the model
to actually VISIT the region: over-populated means plenty of samples and a strong
correction; under-populated means no samples and only the reference side does work.

So the protocol is: **over-populate every region, then let REM tune down.** The FM stage's
job is only to get the forces locally right everywhere -- it is explicitly NOT asked to get
the ensemble weights right. Equal quotas are the simplest way to guarantee "enough frames
everywhere", which is the measured failure mode of the alphaL branch.

Sources are consumed in PRIORITY ORDER per region: the first source that can supply a
region is used first. Pass the equilibrium reference first so the main basins are filled
with true equilibrium frames ("full reference accuracy") and only the regions the reference
cannot cover fall through to enhanced-sampling campaigns.

THE OUTPUT IS NOT AN EQUILIBRIUM DATASET AND MUST NOT BE EVALUATED AS ONE. A model trained
on it is expected to badly over-populate the rare regions in CG MD; that is the input to
the REM stage, not a failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sampling.mapping import get_mapping


def _regions(phi, psi, boxes):
    """Region label per frame; anything outside every box is 'transition'."""
    lab = np.full(len(phi), "transition", dtype=object)
    for name, (p0, p1, s0, s1) in boxes.items():
        lab[(phi >= p0) & (phi <= p1) & (psi >= s0) & (psi <= s1)] = name
    return lab


def _even(n, k):
    return np.linspace(0, n - 1, k).astype(int) if k < n else np.arange(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="LABEL=path.npz, repeatable. PRIORITY ORDER: earlier sources are "
                         "consumed first, so pass the equilibrium reference first.")
    ap.add_argument("--region", action="append", required=True,
                    help="LABEL=phi0:phi1:psi0:psi1, repeatable. 'transition' is implicit "
                         "(the complement) and does not need declaring.")
    ap.add_argument("--n-per-region", type=int, required=True)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from sampling.mapping import dihedral_deg, wrap_deg
    m = get_mapping(a.mapping)
    boxes = {}
    for spec in a.region:
        lab, _, box = str(spec).partition("=")
        boxes[lab] = tuple(float(x) for x in box.split(":"))
    wanted = list(boxes) + ["transition"]

    pools = []
    for spec in a.source:
        lab, _, path = str(spec).partition("=")
        d = np.load(path)
        R, F, S = d["R"].astype(np.float32), d["F"].astype(np.float32), d["species"]
        bl = np.stack([np.linalg.norm(R[:, i].astype(np.float64) - R[:, j].astype(np.float64),
                                      axis=-1) for i, j in m.bonds], axis=1)
        ok = bl.max(axis=1) <= a.max_bond
        R, F, S = R[ok], F[ok], S[ok]
        cv = lambda n: wrap_deg(dihedral_deg(R.astype(np.float64), m.cvs[n].bead_indices)
                                + m.cvs[n].shift_deg)
        pools.append((lab, R, F, S, _regions(cv("phi"), cv("psi"), boxes)))
        print(f"source {lab:22s} {len(R):7d} frames  " +
              "  ".join(f"{r}:{(pools[-1][4]==r).sum()}" for r in wanted))

    Rs, Fs, Ss, prov = [], [], [], []
    print()
    for reg in wanted:
        need = a.n_per_region
        for lab, R, F, S, labels in pools:
            if need <= 0:
                break
            idx = np.flatnonzero(labels == reg)
            if len(idx) == 0:
                continue
            take = idx[_even(len(idx), min(need, len(idx)))]
            Rs.append(R[take]); Fs.append(F[take]); Ss.append(S[take])
            prov.append({"region": reg, "source": lab, "taken": int(len(take)),
                         "available": int(len(idx))})
            print(f"  {reg:12s} <- {lab:22s} {len(take):6d} of {len(idx):7d} available")
            need -= len(take)
        if need > 0:
            print(f"  {reg:12s} !! SHORT by {need} frames -- no source can supply them")

    R = np.concatenate(Rs); F = np.concatenate(Fs); S = np.concatenate(Ss)
    order = np.random.default_rng(a.seed).permutation(len(R))
    R, F, S = R[order], F[order], S[order]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, R=R, F=F, species=S.astype(np.int32),
                        mask=np.ones(R.shape[:2], dtype=np.float32))

    cv = lambda n: wrap_deg(dihedral_deg(R.astype(np.float64), m.cvs[n].bead_indices)
                            + m.cvs[n].shift_deg)
    lab = _regions(cv("phi"), cv("psi"), boxes)
    summary = {"out": str(a.out.resolve()), "total_frames": int(len(R)),
               "n_per_region": a.n_per_region, "regions": {k: list(v) for k, v in boxes.items()},
               "seed": a.seed, "sources": prov,
               "final_composition_pct": {r: float((lab == r).mean() * 100) for r in wanted},
               "F_std_kcal_mol_A": float(F.std()),
               "WARNING": "region-balanced, NOT equilibrium-weighted. Expect heavy "
                          "over-population of rare regions in CG MD; REM finetuning is "
                          "the intended next stage, not optional."}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out}  ({len(R)} frames)")
    print(json.dumps({k: summary[k] for k in
                      ("total_frames", "final_composition_pct", "F_std_kcal_mol_A")}, indent=2))


if __name__ == "__main__":
    main()
