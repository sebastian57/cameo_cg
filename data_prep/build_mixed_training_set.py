#!/usr/bin/env python3
"""Assemble a mixed CG training set from the reference plus enhanced-sampling campaigns.

    python -m data_prep.build_mixed_training_set \
        --reference .../ala2_cg_backbone_CB_aggforce.npz --n-reference 120000 \
        --enhanced .../ala2_bb6_iso_inversion/enhanced_frames.npz:50000:priority \
        --enhanced .../ala2_bb6_iso_tica/enhanced_frames.npz:30000 \
        --out local_work/input_data/ala2_bb6_mixed200k.npz

Why mixing non-equilibrium data is sound
----------------------------------------
Force-matching labels are pointwise: F(x) is the correct force at x no matter what
biased process produced x. Unlike reweighting-based objectives, a non-equilibrium
sampling distribution therefore does NOT bias the learned force field -- it only
decides where model capacity is spent. That is what makes it legitimate to fold flat
-in-chi umbrella windows into an otherwise equilibrium set.

The labels must nevertheless share ONE force-mapping convention. Every input here is
expected to have been produced with the same persisted aggforce matrix; see
KB DESIGN/CG_FORCE_MAPPING.md. Mixing conventions is silent and ruinous.

`priority` selection
--------------------
`path:n:priority` keeps the frames that are scarcest first -- for the inversion ladder,
everything near the planar transition region -- and fills the remainder by even stride.
Those frames are the entire reason the campaign was run and the reference contains none
of them, so a blind stride would dilute them back out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sampling.mapping import get_mapping, normalized_signed_volume  # noqa: E402


def _even_indices(n_available: int, n_take: int) -> np.ndarray:
    """Evenly spaced selection; frames within a replica are strongly correlated."""
    if n_take >= n_available:
        return np.arange(n_available)
    return np.unique(np.linspace(0, n_available - 1, n_take).astype(np.int64))


def _transition_indices(R: np.ndarray, mapping, chi_window: float,
                        max_basin_frac: float, basin: str = "alphaL") -> np.ndarray:
    """Transition-region frames ONLY, stratified so they do not import a basin skew.

    Rationale (2026-08-05): the previous mix took 50k inversion frames by |chi| priority
    plus stride. Those frames are 34.5% alphaL, so the set carried alphaL at 16.0% against
    a reference truth of 2.84% -- and the trained model then over-populated alphaL by ~27x
    in MD (Rama JS 0.70). Force-matching labels are pointwise-correct, but at R^2=0.875 the
    residual error is distributed by training density, so density still shapes the FES.

    So: keep the frames that teach the BARRIER (the reference has none below |chi|=0.383),
    and cap the fraction of them sitting in an over-represented basin. The reference
    supplies basin structure; the enhanced data supplies only the transition.
    """
    c, nb = next(iter(mapping.inversion_centers().items()))
    chi = np.abs(normalized_signed_volume(R, c, nb))
    phi = mapping.cvs["phi"].evaluate(R)
    psi = mapping.cvs["psi"].evaluate(R)
    if basin == "alphaL":
        in_basin = (phi > 0) & (phi < 120) & (psi > -50) & (psi < 100)
    else:
        raise ValueError(f"unknown basin {basin!r}")

    trans = np.flatnonzero(chi < chi_window)
    t_basin = trans[in_basin[trans]]
    t_other = trans[~in_basin[trans]]
    # keep every non-basin transition frame; admit basin ones only up to the cap
    if max_basin_frac >= 1.0:
        keep_basin = t_basin
    else:
        n_allowed = int(len(t_other) * max_basin_frac / max(1e-9, 1.0 - max_basin_frac))
        keep_basin = t_basin[_even_indices(len(t_basin), min(n_allowed, len(t_basin)))]
    out = np.sort(np.concatenate([t_other, keep_basin]))
    print(f"    transition pool |chi|<{chi_window}: {len(trans)} frames "
          f"({len(t_basin)} in {basin}); kept {len(out)} "
          f"({100 * len(keep_basin) / max(1, len(out)):.1f}% {basin})")
    return out


def _outside_basins_indices(R, mapping, boxes, n_take):
    """Keep only frames OUTSIDE every declared basin core: the transition frames.

    The complement of the basins is exactly the data the reference is poorest in
    (10.6% of reference frames vs 35% of the attractor+MetaD run) and the data whose
    forces point outward into the basins, so enriching it does not distort basin
    ratios. Basin frames are dropped rather than reweighted: the reference already
    covers them at the correct Boltzmann weight, so the enhanced copies add density
    where density IS the physics -- the 2026-08-05 failure mode.

    These boxes must be WIDE, and are NOT the conservative cores used for reactive-path
    extraction in sampling/build_transition_map.py. The two uses pull in opposite
    directions and confusing them silently changes the result:

      * path extraction wants SMALL cores, so a reactive segment has room to exist
        between them (generous cores -> 89% in-core -> every "transition" is a 1-frame hop)
      * assembly wants WIDE boxes, so basin PERIPHERY frames are dropped too. A frame just
        outside a tight core is still basin-like, and the reference already covers it at
        the right weight; keeping it re-adds density the reference has.

    Measured on the same campaign (2026-08-05): conservative cores leave alphaL at 4.48%
    of the assembled set, wide boxes 2.74%, against a reference truth of 2.79%.
    """
    phi = mapping.cvs["phi"].evaluate(R)
    psi = mapping.cvs["psi"].evaluate(R)
    inside = np.zeros(len(R), dtype=bool)
    for label, (p0, p1, s0, s1) in boxes.items():
        m = (phi >= p0) & (phi <= p1) & (psi >= s0) & (psi <= s1)
        inside |= m
        print(f"    basin {label}: {int(m.sum())} frames excluded")
    out = np.flatnonzero(~inside)
    print(f"    intermediate (kept): {len(out)} frames "
          f"({100 * len(out) / len(R):.1f}% of the run), phi>0 among them "
          f"{100 * (phi[out] > 0).mean():.1f}%")
    if n_take > 0 and len(out) > n_take:
        out = out[_even_indices(len(out), n_take)]
    return out


def _inside_basins_indices(R: np.ndarray, mapping, boxes, n_take: int) -> np.ndarray:
    """The COMPLEMENT of `outside_basins`: keep only frames inside a declared box.

    Used to deliberately over-represent a basin the model fails to reproduce. This
    knowingly violates the usual rule that basin populations must match the reference
    (DESIGN/CG_ACQUISITION_AND_ASSEMBLY.md), and that is the point: measured 2026-08-06,
    training at the reference's own 2.79% alphaL yields 0.04-0.17% in CG MD, and a replica
    started inside alphaL is expelled in 0.2-2.0 ps against an AA median residence of 85 ps.
    Force matching at correct weight does not reproduce this basin at all.

    So the over-representation here is a CORRECTION for a known systematic under-population,
    not a coverage grab. It is still dangerous in exactly the way 2026-08-05 documented --
    mixed200k's 16% alphaL became 77.5% in MD -- so the added fraction must be modest and
    the resulting MD alphaL must be checked against 2.79%, not merely be non-zero.
    """
    phi = mapping.cvs["phi"].evaluate(R)
    psi = mapping.cvs["psi"].evaluate(R)
    inside = np.zeros(len(R), dtype=bool)
    for label, (p0, p1, s0, s1) in boxes.items():
        msk = (phi >= p0) & (phi <= p1) & (psi >= s0) & (psi <= s1)
        inside |= msk
        print(f"    basin {label}: {int(msk.sum())} frames available")
    idx = np.flatnonzero(inside)
    if len(idx) == 0:
        raise SystemExit("inside_basins selected nothing -- check --basin-box")
    if n_take > 0 and len(idx) > n_take:
        idx = idx[_even_indices(len(idx), n_take)]
    print(f"    inside_basins kept {len(idx)} frames")
    return idx


def _cap_stretch_indices(R: np.ndarray, F: np.ndarray, mapping, n_take: int,
                         d_min: float, d_max: float, n_bins: int) -> np.ndarray:
    """Frames whose TERMINAL CAP bonds are stretched past the reference's support edge.

    The bb6 models dissociate at dt=4fs because training support for the cap bonds ends at
    1.451 A while a 4fs excursion reaches ~1.65-1.70 A, where the force is pure
    extrapolation. These frames close that gap.

    Selection is UNIFORM IN CAP DISTANCE, not an even stride over the campaign. An umbrella
    campaign spends most of its frames near each window centre, so a stride would pile up
    at the window distances and leave the space between them thin -- the opposite of what
    a support-filling set needs.

    Frames below `d_min` are dropped outright: the reference already covers that range at
    correct Boltzmann weight, and re-adding it is the density error of 2026-08-05 in a new
    guise. Frames above `d_max` are dropped because the force there grows fast enough to
    dominate the objective -- measured: a frame at 1.85-2.00 A carries ~84x the
    squared-force loss of a reference frame, so a few thousand of them would outweigh the
    entire equilibrium set and the model would optimise bond rescue at the FES's expense.
    """
    caps = [(i, j) for i, j in mapping.bonds
            if len(mapping.neighbors(i)) == 1 or len(mapping.neighbors(j)) == 1]
    d = np.stack([np.linalg.norm(R[:, i] - R[:, j], axis=-1) for i, j in caps]).max(axis=0)
    edges = np.linspace(d_min, d_max, n_bins + 1)
    per_bin = max(1, n_take // n_bins)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((d >= lo) & (d < hi))
        if len(idx) == 0:
            print(f"    cap bin {lo:.3f}-{hi:.3f} A: EMPTY -- support gap not covered here")
            continue
        take = idx[_even_indices(len(idx), min(per_bin, len(idx)))]
        picked.append(take)
        print(f"    cap bin {lo:.3f}-{hi:.3f} A: {len(idx):6d} available, took {len(take):5d}, "
              f"|F| rms {np.sqrt((F[take] ** 2).sum(-1).mean()):.1f}")
    if not picked:
        raise SystemExit("cap_stretch selected nothing -- check --cap-stretch-min/max")
    out = np.concatenate(picked)
    print(f"    cap_stretch total {len(out)} frames spanning {d[out].min():.3f}-{d[out].max():.3f} A")
    return out


def _priority_indices(R: np.ndarray, n_take: int, mapping, chi_window: float) -> np.ndarray:
    """All near-planar frames first, then an even stride over the rest."""
    centers = mapping.inversion_centers()
    if len(centers) != 1:
        raise SystemExit(f"priority selection needs exactly one inversion center, got {centers}")
    c, nb = next(iter(centers.items()))
    chi = normalized_signed_volume(R, c, nb)
    near = np.flatnonzero(np.abs(chi) < chi_window)
    if len(near) >= n_take:
        return near[_even_indices(len(near), n_take)]
    rest = np.flatnonzero(np.abs(chi) >= chi_window)
    fill = rest[_even_indices(len(rest), n_take - len(near))]
    return np.sort(np.concatenate([near, fill]))


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    for key in ("R", "F"):
        if key not in d.files:
            raise SystemExit(f"{path}: missing '{key}'")
    R = np.asarray(d["R"], dtype=np.float64)
    F = np.asarray(d["F"], dtype=np.float64)
    if R.shape != F.shape:
        raise SystemExit(f"{path}: R {R.shape} != F {F.shape}")
    species = (np.asarray(d["species"], dtype=np.int64) if "species" in d.files
               else np.tile(np.arange(R.shape[1], dtype=np.int64), (len(R), 1)))
    return R, F, species


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--n-reference", type=int, required=True)
    ap.add_argument("--enhanced", action="append", default=[],
                    help="path:n[:priority], repeatable")
    ap.add_argument("--mapping", type=str, default="ala2_backbone_cb_6")
    ap.add_argument("--chi-window", type=float, default=0.15,
                    help="|chi| below this counts as transition region for priority picks")
    ap.add_argument("--basin-box", action="append", default=None,
                    metavar="LABEL=phi0:phi1:psi0:psi1",
                    help="basin core for selection mode 'outside_basins', repeatable")
    ap.add_argument("--enhanced-basin-frac", type=float, default=0.10,
                    help="max fraction of selected transition frames allowed to sit in the "
                         "alphaL basin (selection mode 'transition'); the reference, not the "
                         "enhanced data, should set basin populations")
    ap.add_argument("--cap-stretch-min", type=float, default=1.451,
                    help="lower cap-bond distance for selection mode 'cap_stretch' (A). "
                         "Default is the bb6 reference's max cap bond -- below it the "
                         "reference already has correct-weight coverage")
    ap.add_argument("--cap-stretch-max", type=float, default=1.85,
                    help="upper cap-bond distance (A). Beyond this the squared-force loss "
                         "per frame exceeds ~60x a reference frame and starts to dominate")
    ap.add_argument("--cap-stretch-bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mapping = get_mapping(args.mapping)
    args.basin_boxes = {}
    for spec in (args.basin_box or []):
        label, box = spec.split("=", 1)
        args.basin_boxes[label] = tuple(float(x) for x in box.split(":"))
    Rs: List[np.ndarray] = []
    Fs: List[np.ndarray] = []
    Ss: List[np.ndarray] = []
    prov = []

    R, F, S = _load(args.reference)
    idx = _even_indices(len(R), args.n_reference)
    Rs.append(R[idx]); Fs.append(F[idx]); Ss.append(S[idx])
    prov.append({"source": str(args.reference), "role": "reference",
                 "available": int(len(R)), "taken": int(len(idx)), "selection": "even"})
    print(f"reference : {len(idx):>7d} / {len(R)} (even)")

    for spec in args.enhanced:
        parts = spec.split(":")
        if len(parts) < 2:
            raise SystemExit(f"--enhanced expects path:n[:priority], got {spec!r}")
        path, n_take = Path(parts[0]), int(parts[1])
        mode = parts[2] if len(parts) > 2 else "even"
        R, F, S = _load(path)
        if R.shape[1] != mapping.n_beads:
            raise SystemExit(f"{path}: {R.shape[1]} beads, mapping has {mapping.n_beads}")
        if mode == "outside_basins":
            idx = _outside_basins_indices(R, mapping, args.basin_boxes, n_take)
        elif mode == "transition":
            idx = _transition_indices(R, mapping, args.chi_window, args.enhanced_basin_frac)
            if n_take > 0 and len(idx) > n_take:
                idx = idx[_even_indices(len(idx), n_take)]
        elif mode == "inside_basins":
            idx = _inside_basins_indices(R, mapping, args.basin_boxes, n_take)
        elif mode == "cap_stretch":
            idx = _cap_stretch_indices(R, F, mapping, n_take, args.cap_stretch_min,
                                       args.cap_stretch_max, args.cap_stretch_bins)
        elif mode == "priority":
            idx = _priority_indices(R, n_take, mapping, args.chi_window)
        elif mode == "even":
            idx = _even_indices(len(R), n_take)
        else:
            raise SystemExit(f"unknown selection mode {mode!r}")
        chi = normalized_signed_volume(R[idx], 2, (1, 3, 4)) if mapping.n_beads == 6 else None
        n_near = int((np.abs(chi) < args.chi_window).sum()) if chi is not None else 0
        Rs.append(R[idx]); Fs.append(F[idx]); Ss.append(S[idx])
        entry = {"source": str(path), "role": "enhanced", "available": int(len(R)),
                 "requested": int(n_take), "taken": int(len(idx)), "selection": mode,
                 "transition_frames": n_near}
        # The boxes ARE the definition of `outside_basins`; recording only the mode name
        # makes the dataset unreproducible. Measured 2026-08-05: the same campaign and the
        # same mode give alphaL 4.48% with the conservative path-extraction cores and
        # 2.74% with wide basin boxes -- a 2x swing in the quantity the whole
        # acquisition/assembly split exists to control.
        if mode == "outside_basins":
            entry["basin_boxes"] = {k: list(v) for k, v in args.basin_boxes.items()}
        prov.append(entry)
        if n_take > 0 and len(idx) < n_take:
            print(f"    WARNING: requested {n_take} frames, only {len(idx)} available "
                  f"after selection -- the dataset is smaller than asked for")
        print(f"{path.parent.name[:20]:<20}: {len(idx):>7d} / {len(R)} ({mode}, "
              f"{n_near} with |chi|<{args.chi_window})")

    R = np.concatenate(Rs); F = np.concatenate(Fs); S = np.concatenate(Ss)

    # Pre-shuffle. The trainer shuffles before its sequential train/val split, but that
    # path is conditional (skipped under cross-fit); an unshuffled concatenation would
    # otherwise put entire source campaigns wholly inside validation.
    order = np.random.default_rng(args.seed).permutation(len(R))
    R, F, S = R[order], F[order], S[order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        R=R.astype(np.float32), F=F.astype(np.float32),
        species=S.astype(np.int32),
        mask=np.ones(R.shape[:2], dtype=np.float32),
    )

    chi = normalized_signed_volume(R, 2, (1, 3, 4)) if mapping.n_beads == 6 else None
    phi = mapping.cvs["phi"].evaluate(R)
    psi = mapping.cvs["psi"].evaluate(R)
    summary = {
        "out": str(args.out.resolve()), "total_frames": int(len(R)),
        "n_beads": int(R.shape[1]), "seed": args.seed, "shuffled": True,
        "units": "R in Angstrom, F in kcal/mol/A",
        "force_map": "aggforce persisted matrix (all sources share one convention)",
        "sources": prov,
        "alphaL_percent": float(((phi > 0) & (phi < 120) & (psi > -50) & (psi < 100)).mean() * 100),
        "phi_positive_percent": float((phi > 0).mean() * 100),
        "F_std_kcal_mol_A": float(F.std()),
    }
    if chi is not None:
        summary["chi"] = {"min": float(chi.min()), "max": float(chi.max()),
                          "transition_frames": int((np.abs(chi) < args.chi_window).sum())}
    Path(str(args.out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}  ({len(R)} frames x {R.shape[1]} beads)")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
