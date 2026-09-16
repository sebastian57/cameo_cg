"""Build the Week-1 stencil-ablation datasets from the existing v3 250k harvest.

Every variant is a SUBSET of `v3_stencil_meanforce.npz` — no new AA sampling. Variants are
specified as (n_anchors, directions, layers) and are compared at FIXED EXPENSIVE-LABEL BUDGET
(= row count = number of frozen-AA states), so "more anchors" and "more directions per anchor"
trade against each other at constant cost.

Anchor sets are NESTED: one fixed permutation, first N taken. A variant with fewer anchors uses
a subset of the anchors of every larger variant, which removes anchor-draw variance from the
comparisons.

Directions: 0,1 are the TICA-gradient directions; 2..5 are the random orthonormal complement.
The saturation ladder (d1/d2/d3) draws only from the complement {2,3,4,5} so it is homogeneous;
`all6` necessarily includes the TICA pair (it is the production design).

Usage:
    python sampling/build_ablation_datasets.py [--outdir local_work/input_data] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MEANFORCE = "local_work/v3_stencil_meanforce.npz"
STENCIL = "local_work/v3_stage3_stencil/stencil_states.npz"
REFERENCE = "local_work/input_data/ala2_cg_backbone_cb_6bead_200k.npz"
PREFIX = "ala2_bb6_v3abl"
ANCHOR_PERM_SEED = 20260819
DIR_SEED = 20260819
COMPLEMENT = (2, 3, 4, 5)          # non-TICA directions
TICA_DIRS = (0, 1)
ALL_DIRS = (0, 1, 2, 3, 4, 5)

# name -> (n_anchors, dirs, layers)
#   dirs:   "none" | ("rand", k) | "tica2" | "all6"
#   layers: "both" (+/-eps and +/-2eps) | "inner" (+/-eps) | "outer" (+/-2eps)
VARIANTS = {
    # ---- budget 10,000 states: does curvature beat coverage at equal cost? ----
    "b10_d0":      (10000, "none",       "both"),
    "b10_d1":      (2000,  ("rand", 1),  "both"),
    "b10_d3":      (769,   ("rand", 3),  "both"),
    "b10_d6":      (400,   "all6",       "both"),
    # ---- budget 50,000 states: direction saturation ----
    "b50_d1":      (10000, ("rand", 1),  "both"),
    "b50_d2":      (5555,  ("rand", 2),  "both"),
    "b50_d3":      (3846,  ("rand", 3),  "both"),
    "b50_d6":      (2000,  "all6",       "both"),
    # ---- budget 50,000 states: which displacements, at identical rows/anchor ----
    "b50_inner":   (3846,  "all6",       "inner"),
    "b50_outer":   (3846,  "all6",       "outer"),
    "b50_d2tica":  (5555,  "tica2",      "both"),
}


def rows_per_anchor(dirs, layers) -> int:
    if dirs == "none":
        return 1
    n = len(TICA_DIRS) if dirs == "tica2" else (len(ALL_DIRS) if dirs == "all6" else dirs[1])
    return 1 + n * (4 if layers == "both" else 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("local_work/input_data"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    mm = np.load(MEANFORCE)
    st = np.load(STENCIL)
    state = mm["state"]
    slot = state // 25
    direction = st["direction"][state]
    multiplier = np.round(st["multiplier"][state], 2)
    R, F = mm["R"], mm["F"]
    n_slots = int(slot.max()) + 1

    base = np.load(REFERENCE)
    species1, mask1 = base["species"][:1], base["mask"][:1]

    perm = np.random.default_rng(ANCHOR_PERM_SEED).permutation(n_slots)
    is_anchor = direction == -1
    inner = np.isin(np.abs(multiplier), [1.0])
    outer = np.isin(np.abs(multiplier), [2.0])

    print(f"source: {len(state):,} states, {n_slots:,} anchors\n")
    print(f"{'variant':12s} {'anchors':>8s} {'rows/anc':>9s} {'rows':>9s} {'dirs':>16s} {'layers':>7s}")

    for name, (n_anc, dirs, layers) in VARIANTS.items():
        keep_slots = np.zeros(n_slots, bool)
        keep_slots[perm[:n_anc]] = True
        in_set = keep_slots[slot]

        if dirs == "none":
            sel = in_set & is_anchor
        else:
            if layers == "both":
                lay = inner | outer
            elif layers == "inner":
                lay = inner
            else:
                lay = outer
            if dirs == "all6":
                dmask = np.isin(direction, ALL_DIRS)
            elif dirs == "tica2":
                dmask = np.isin(direction, TICA_DIRS)
            else:
                # per-anchor random draw from the non-TICA complement
                rng = np.random.default_rng(DIR_SEED)
                chosen = np.full((n_slots, len(COMPLEMENT)), False)
                for s_ in perm[:n_anc]:
                    pick = rng.choice(len(COMPLEMENT), dirs[1], replace=False)
                    chosen[s_, pick] = True
                dmask = np.zeros(len(state), bool)
                for j, d_ in enumerate(COMPLEMENT):
                    dmask |= (direction == d_) & chosen[slot, j]
            sel = in_set & (is_anchor | (dmask & lay))

        n = int(sel.sum())
        exp = n_anc * rows_per_anchor(dirs, layers)
        dl = "none" if dirs == "none" else ("all6" if dirs == "all6" else
                                            ("tica{0,1}" if dirs == "tica2" else f"rand{dirs[1]}/4"))
        flag = "" if n == exp else f"  !! expected {exp:,}"
        print(f"{name:12s} {n_anc:8,d} {rows_per_anchor(dirs, layers):9d} {n:9,d} {dl:>16s} {layers:>7s}{flag}")
        if a.dry_run:
            continue

        out = a.outdir / f"{PREFIX}_{name}.npz"
        np.savez_compressed(
            out, R=R[sel], F=F[sel],
            species=np.repeat(species1, n, 0), mask=np.repeat(mask1, n, 0),
            origin=np.ones(n, np.int8), anchor_slot=slot[sel].astype(np.int32),
            direction=direction[sel].astype(np.int8), multiplier=multiplier[sel].astype(np.int8))

    if a.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    main()
