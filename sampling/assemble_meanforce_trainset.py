#!/usr/bin/env python3
"""Assemble an FM training set from a base set plus the stage-3 mean-force block.

    python -m sampling.assemble_meanforce_trainset \
        --base local_work/input_data/ala2_bb6_harvest265k.npz \
        --meanforce local_work/dhh_stage3_meanforce/meanforce_dataset.npz \
        --replicate 64 --out local_work/input_data/ala2_bb6_dhh300k.npz

WHY REPLICATION, AND WHY IT IS NOT THE THING collect_meanforce FORBIDS
    `collect_meanforce` refuses to emit 1,751 rows for one state, because those frames are
    1,751 correlated samples of ONE configuration and emitting them would multiply that
    region's weight by the sampling effort rather than by the information gained.

    Replicating the finished MEAN is a different operation. The row is a single label with a
    known precision, and repeating it k times in an unweighted MSE is exactly importance
    weighting by k -- the standard way to express "trust this label more" when the loss has no
    per-sample weight hook. (`force_loss_weights` in the trainer normalises the MSE by beads
    per structure; it is not a precision weight and is rebuilt per batch after tiling.)

CHOOSING k
    Statistically, the correct weight relative to a single-frame label is
    `(sigma_ref / SE)^2` -- with sigma_ref = 18.9 and SE = 0.513 that is **1,356**, which
    would make 512 geometries ~78% of the training set. That is right for estimating a force
    field at those points and wrong for a ~10^5-parameter potential that has to generalise
    away from them.

    The default k = 64 gives the new block ~11% of the assembled set, matching the share the
    34,656-frame alphaL harvest block held in `harvest265k`. It is an engineering compromise
    between the two failure modes, and it is recorded in the output so it can be varied.

UNITS ARE CHECKED, NOT ASSUMED. Both files must already be in Angstrom / kcal/mol/A; the
bond-length and force-scale guards below abort rather than silently train on a unit mismatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True,
                    help="base training set; also the source of species/mask")
    ap.add_argument("--base-frames", type=int, default=None,
                    help="randomly subsample the base to this many rows before assembling")
    ap.add_argument("--base-fraction", type=float, default=1.0,
                    help="0.0 keeps NO base rows -- a mean-force-only ablation set. species and "
                         "mask are still read from --base, so it must still be given.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meanforce", type=Path, required=True)
    ap.add_argument("--replicate", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bond-tol", type=float, default=0.05,
                    help="max Angstrom between the two sets' mean bead0-bead1 distance")
    a = ap.parse_args()

    base = np.load(a.base)
    mf = np.load(a.meanforce)
    rng = np.random.default_rng(a.seed)

    # species/mask always come from the base, even when no base ROWS are kept, so the two
    # blocks can never disagree about atom typing.
    spec0, mask0 = base["species"][0], base["mask"][0]
    if not (base["species"] == spec0).all():
        raise SystemExit("base species vary per frame; cannot broadcast to the new block")

    keep = np.arange(len(base["R"]))
    if a.base_fraction <= 0.0:
        keep = keep[:0]
    elif a.base_frames is not None and a.base_frames < len(keep):
        keep = rng.choice(len(keep), a.base_frames, replace=False)
    elif a.base_fraction < 1.0:
        keep = rng.choice(len(keep), int(len(keep) * a.base_fraction), replace=False)
    Rb, Fb = base["R"][keep], base["F"][keep]
    Rn, Fn = mf["R"].astype(Rb.dtype), mf["F"].astype(Fb.dtype)

    # --- unit guards -------------------------------------------------------------------
    bond = lambda R: float(np.linalg.norm(R[:, 0] - R[:, 1], axis=-1).mean())
    dn = bond(Rn)
    db = bond(Rb) if len(Rb) else dn          # nothing to cross-check against an empty base
    if len(Rb) and abs(db - dn) > a.bond_tol:
        raise SystemExit(f"bond length mismatch: base {db:.3f} A vs meanforce {dn:.3f} A "
                         f"-- one of these is in nm, refusing to assemble")
    if Rn.shape[1] != Rb.shape[1]:
        raise SystemExit(f"bead count mismatch: {Rb.shape[1]} vs {Rn.shape[1]}")

    # The mean-force block SHOULD have a much smaller force spread than the base: averaging
    # removes the instantaneous thermal component. Check the decomposition holds, because if
    # the two were on different force scales this is where it would show.
    sd_n = float(Fn.std())
    sd_b = float(Fb.std()) if len(Fb) else float("nan")
    implied_noise = (float(np.sqrt(max(sd_b ** 2 - sd_n ** 2, 0.0))) if len(Fb)
                     else float("nan"))

    n_rep = max(1, a.replicate)
    Rn_r = np.repeat(Rn, n_rep, axis=0)
    Fn_r = np.repeat(Fn, n_rep, axis=0)

    Sn = np.repeat(spec0[None, :], len(Rn_r), axis=0).astype(base["species"].dtype)
    Mn = np.repeat(mask0[None, :], len(Rn_r), axis=0).astype(base["mask"].dtype)

    R = np.concatenate([Rb, Rn_r]); F = np.concatenate([Fb, Fn_r])
    S = np.concatenate([base["species"][keep], Sn])
    M = np.concatenate([base["mask"][keep], Mn])

    # provenance: which rows came from where, so a later analysis can separate them
    origin = np.concatenate([np.zeros(len(Rb), np.int8), np.ones(len(Rn_r), np.int8)])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, R=R, F=F, species=S, mask=M, origin=origin)

    share = 100.0 * len(Rn_r) / len(R)
    prov = dict(base=str(a.base), n_base=int(len(Rb)),
                meanforce=str(a.meanforce), n_states=int(len(Rn)),
                replicate=n_rep, n_meanforce_rows=int(len(Rn_r)),
                n_total=int(len(R)), meanforce_share_pct=round(share, 2),
                base_force_sd=round(sd_b, 3), meanforce_force_sd=round(sd_n, 3),
                implied_single_frame_noise=round(implied_noise, 3),
                median_SE=float(np.median(mf["SE"])),
                information_optimal_replicate=(
                    int(round((implied_noise / float(np.median(mf["SE"]))) ** 2))
                    if len(Fb) else None))
    Path(str(a.out).replace(".npz", "_provenance.json")).write_text(json.dumps(prov, indent=1))

    print(f"base       {len(Rb):>8,} rows  force sd {sd_b:.3f}")
    print(f"meanforce  {len(Rn):>8,} states x {n_rep} = {len(Rn_r):,} rows  "
          f"force sd {sd_n:.3f}")
    if len(Fb):
        # The decomposition sigma_single^2 = sigma_mean^2 + sigma_noise^2 only holds if both
        # blocks sample the SAME configurations. Farthest-point selection does not: the v2
        # states sit in reference-density regions 41.5x lower than typical reference frames
        # (79% below the reference's 10th percentile), i.e. on slopes rather than in minima,
        # so their mean-force spread legitimately EXCEEDS the reference's single-frame spread.
        # Only report the decomposition when it is actually applicable.
        if sd_n < sd_b:
            print(f"  the base's extra spread implies single-frame noise "
                  f"{implied_noise:.2f} kcal/mol/A")
            print(f"  information-optimal replicate would be "
                  f"{prov['information_optimal_replicate']:,} (using {n_rep})")
        else:
            print(f"  NOTE sd(mean-force) {sd_n:.2f} > sd(base) {sd_b:.2f}: the two blocks do "
                  f"not sample the same\n       configurations, so the noise decomposition "
                  f"does not apply. Check the selection's\n       density profile instead.")
    print(f"TOTAL      {len(R):>8,} rows; new block is {share:.1f}%")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
