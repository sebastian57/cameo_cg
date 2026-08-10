#!/usr/bin/env python3
"""Reduce finished sampling cases to CG beads, applying the reference force map.

At 0.1 ps output a 1 ns solvated replica writes ~640 MB of TRR, twice over (biased +
unbiased rerun). Twelve replicas is ~15 GB of which we need six mapped beads.

    python -m sampling.collect --campaign <dir> --mapping ala2_backbone_cb_6 \
        --weights .../ala2_bb6_aggforce_weight_matrix.npz [--delete-trr]

Two things here are easy to get wrong and silent when wrong:

1. **Coordinates come from the BIASED trajectory** (the configurations actually
   visited); **forces from the UNBIASED rerun** (the physical labels). Training on
   biased forces teaches the model the sampler.
2. **Bead forces are NOT the AA forces of the bead atoms.** The reference dataset
   was built with an aggforce-fitted linear map spreading weight over all AA atoms
   (e.g. bead C carries 1.072x the force of its O). Slicing or group-summing instead
   gives plausible-looking labels in a different convention.
   See KB DESIGN/CG_FORCE_MAPPING.md.

Do not re-fit the map on these frames -- `project_forces` fits whatever it is given
and returns a *different* map (measured: 4.7e-01 vs 4.2e-05 max error against the
reference). Apply the stored matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .mapping import get_mapping

NM_TO_A = 10.0
KJ_NM_TO_KCAL_A = 1.0 / 4.184 / 10.0   # kJ/mol/nm -> kcal/mol/A


# GROMACS lives in the 2025 module stack and the project venv in the 2026 one; loading
# either purges the other, and calling the gmx binary by absolute path fails (127) since
# it needs its module environment. So gmx is invoked in its own login shell.
GMX_MODULES = "Stages/2025 GCC/13.3.0 ParaStationMPI/5.11.0-1 GROMACS/2024.3-PLUMED-2.9.3"
DEFAULT_GMX = "gmx"


def _discover_cases(campaign: Path) -> List[Path]:
    """Case subdirectories of a campaign, whatever the generator named them.

    `sampling/cases.py` emits `replica_NN`; `sampling/build_bond_stretch_campaign.py`
    emits `case_NN_<bond>_<target>A_rN`, because for an umbrella campaign the window is
    the identity of the run and folding it into an opaque replica index loses it. Both
    layouts are otherwise identical (biased.trr + unbiased_forces.trr), so collection
    accepts either rather than forcing one generator to lie about its cases.
    """
    dirs = sorted(p for p in campaign.iterdir()
                  if p.is_dir() and (p.name.startswith("replica_") or p.name.startswith("case_")))
    return dirs


def _gmx_traj(case_dir: Path, trr: str, flag: str, xvg_name: str, n_aa: int,
              gmx: str, group: str = "Protein") -> Tuple[np.ndarray, np.ndarray]:
    """Run `gmx traj` and parse the xvg -> (times_ps, (n, n_aa, 3)).

    Everything goes through GROMACS rather than mdtraj: mdtraj exposes no TRR force
    block (`_forces` is None in 1.11.1) and cannot read a .tpr as topology, so one
    code path for coords and forces avoids two sets of failure modes and guarantees
    the two arrays are frame-aligned.
    """
    xvg = case_dir / xvg_name
    if not xvg.exists():
        cmd = (f"module --force purge >/dev/null 2>&1; "
               f"module load {GMX_MODULES} >/dev/null 2>&1; "
               f"{gmx} traj -f {case_dir / trr} -s {case_dir / 'biased.tpr'} "
               f"{flag} {xvg} -xvg none")
        proc = subprocess.run(["bash", "-lc", cmd], input=f"{group}\n", text=True,
                              capture_output=True)
        if proc.returncode != 0 or not xvg.exists():
            raise RuntimeError(
                f"{case_dir.name}: gmx traj failed (rc={proc.returncode})\n"
                f"{proc.stderr[-1500:]}"
            )
    raw = np.loadtxt(xvg)
    if raw.ndim == 1:
        raw = raw[None, :]
    v = raw[:, 1:]
    if v.shape[1] != 3 * n_aa:
        raise ValueError(
            f"{case_dir.name}: expected {3 * n_aa} components for {n_aa} atoms, got "
            f"{v.shape[1]} -- is '{group}' the right index group?"
        )
    return raw[:, 0], v.reshape(len(v), n_aa, 3)


def _make_whole(case_dir: Path, gmx: str) -> None:
    """Write whole.trr with molecules unbroken across the periodic boundary."""
    whole = case_dir / "whole.trr"
    if whole.exists():
        return
    cmd = (f"module --force purge >/dev/null 2>&1; "
           f"module load {GMX_MODULES} >/dev/null 2>&1; "
           f"{gmx} trjconv -f {case_dir / 'biased.trr'} -s {case_dir / 'biased.tpr'} "
           f"-pbc whole -o {whole}")
    proc = subprocess.run(["bash", "-lc", cmd], input="System\n", text=True,
                          capture_output=True)
    if proc.returncode != 0 or not whole.exists():
        raise RuntimeError(f"{case_dir.name}: trjconv -pbc whole failed\n{proc.stderr[-1500:]}")


def _load_case(case_dir: Path, bead_atoms0: List[int], W: np.ndarray,
               discard_ps: float, gmx: str, force_cap: float,
               mapping) -> Tuple[np.ndarray, np.ndarray]:
    for name in ("biased.trr", "unbiased_forces.trr", "biased.tpr"):
        if not (case_dir / name).exists():
            raise FileNotFoundError(f"{case_dir.name}: missing {name}")

    n_aa = W.shape[1]
    # Coords from the BIASED run (configurations visited). They must be made whole
    # first: raw TRR coords are wrapped into the box, so on any frame where the
    # molecule straddles a boundary the dihedrals are garbage. Measured on
    # replica_00: 12.4% of frames, CA-CB up to 44 A, inflating apparent phi>0 from
    # 0.5% to 26%. Forces need no such treatment.
    _make_whole(case_dir, gmx)
    times, R_aa = _gmx_traj(case_dir, "whole.trr", "-ox", "aa_coords.xvg", n_aa, gmx)
    # forces from the BIAS-FREE rerun (physical labels)
    tf, F_aa = _gmx_traj(case_dir, "unbiased_forces.trr", "-of", "aa_forces.xvg", n_aa, gmx)

    if len(F_aa) != len(R_aa):
        raise ValueError(
            f"{case_dir.name}: {len(R_aa)} coordinate frames but {len(F_aa)} force frames"
        )
    if not np.allclose(times, tf, atol=1e-6):
        raise ValueError(f"{case_dir.name}: coord/force frame times differ")

    R = R_aa[:, bead_atoms0, :] * NM_TO_A
    F = np.einsum("bn,fnd->fbd", W, F_aa * KJ_NM_TO_KCAL_A)   # the reference force map

    # Molecule must be whole on EVERY frame, checked over EVERY mapped covalent bond
    # rather than one hardcoded pair. Check the worst frame, not the median: a 12.4%
    # broken minority leaves the median at a healthy 1.63 A while scrambling the
    # dihedrals of one frame in eight.
    if not mapping.bonds:
        raise ValueError(
            f"mapping {mapping.name} declares no bonds, so PBC integrity cannot be "
            f"verified; add `bonds` to the CGMapping before collecting"
        )
    for i, j in mapping.bonds:
        d = np.linalg.norm(R[:, i, :] - R[:, j, :], axis=-1)
        bad = (d < 0.8) | (d > 3.0)
        if bad.any():
            raise ValueError(
                f"{case_dir.name}: {bad.mean() * 100:.1f}% of frames have bond "
                f"{mapping.bead_labels[i]}-{mapping.bead_labels[j]} outside 0.8-3.0 A "
                f"(max {d.max():.1f}) -- molecule broken across PBC despite "
                f"trjconv -pbc whole"
            )

    # Transition-region AA forces are legitimately large, but beyond ~1000 kcal/mol/A
    # the structure is broken rather than informative. Reject rather than train on it.
    fmax = np.abs(F).max(axis=(1, 2))
    keep = (times >= (times[0] + discard_ps)) & (fmax < force_cap)
    n_capped = int(((times >= (times[0] + discard_ps)) & (fmax >= force_cap)).sum())
    if n_capped:
        print(f"    {case_dir.name}: dropped {n_capped} frames with |F| >= "
              f"{force_cap} kcal/mol/A")
    return R[keep], F[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", type=Path, default=None,
                     help="a single replica directory")
    src.add_argument("--campaign", type=Path, default=None,
                     help="a campaign directory containing replica_* subdirectories")
    ap.add_argument("--mapping", type=str, required=True)
    ap.add_argument("--coords-only", action="store_true",
                    help="extract bead COORDINATES only, skipping the bias-free force "
                         "rerun and the aggforce map. For kinetics/pathway runs written "
                         "with nstfout=0 there are no forces to map; requiring --weights "
                         "there would demand a force convention for data that has none.")
    ap.add_argument("--weights", type=Path, default=None,
                    help="NPZ holding the aggforce weight matrix W (n_beads x n_aa)")
    ap.add_argument("--discard-ps", type=float, default=None,
                    help="drop the equilibration + ramp window. Default: the campaign's "
                         "own required_discard_ps (written by cases.py from the bias "
                         "schedules), else 20 ps")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gmx", type=str, default=DEFAULT_GMX,
                    help="gmx binary; the 2026 Python stack purges the GROMACS module, "
                         "so an absolute path is needed when both are in play")
    ap.add_argument("--force-cap", type=float, default=1000.0,
                    help="drop frames whose max |F| exceeds this (kcal/mol/A); large "
                         "transition forces are signal, but beyond this the structure "
                         "is broken (default 1000)")
    ap.add_argument("--delete-trr", action="store_true",
                    help="delete biased.trr, unbiased_forces.trr, whole.trr and the "
                         "intermediate xvgs after extraction -- ~1.5 GB per replica")
    args = ap.parse_args()

    mapping = get_mapping(args.mapping)
    bead_atoms0 = [i - 1 for i in mapping.aa_atom_indices_1based]

    if args.coords_only:
        n_aa = 22
        Rs, per_case = [], []
        cases = ([args.case] if args.case
                 else _discover_cases(args.campaign))
        skipped = []
        for case in cases:
            try:
                _make_whole(case, args.gmx)
                times, R_aa = _gmx_traj(case, "whole.trr", "-ox", "aa_coords.xvg",
                                        n_aa, args.gmx)
            except Exception as exc:
                # A crashed replica (e.g. LINCS blow-up) leaves a truncated trr. Skipping
                # it is right for a shooting campaign, where individual failures are
                # expected -- aborting the batch would discard 44 good trajectories.
                skipped.append(case.name)
                print(f"  !! {case.name}: SKIPPED ({type(exc).__name__}: {str(exc)[:90]})")
                continue
            R = R_aa[:, bead_atoms0, :] * NM_TO_A
            for i, j in mapping.bonds:
                d = np.linalg.norm(R[:, i, :] - R[:, j, :], axis=-1)
                if ((d < 0.8) | (d > 3.0)).any():
                    raise ValueError(f"{case.name}: PBC-broken frames remain")
            np.savez_compressed(case / "cg_coords.npz", R=R.astype(np.float32),
                                time_ps=times.astype(np.float64))
            Rs.append(R); per_case.append({"case": case.name, "frames": int(len(R))})
            print(f"  {case.name}: {len(R)} frames")
        if skipped:
            print(f"  skipped {len(skipped)} replica(s): {skipped}")
        R = np.concatenate(Rs)
        out = args.out or ((args.campaign / "cg_coords_all.npz") if args.campaign
                           else (args.case / "cg_coords_all.npz"))
        np.savez_compressed(out, R=R.astype(np.float32),
                            per_case=np.array([c["frames"] for c in per_case]))
        print(f"wrote {out}  ({len(R)} frames x {R.shape[1]} beads, coords only)")
        return

    if args.weights is None:
        raise SystemExit("--weights is required unless --coords-only is given")
    wz = np.load(args.weights)
    W = wz["W"]
    if W.shape[0] != mapping.n_beads:
        raise SystemExit(
            f"weight matrix has {W.shape[0]} beads, mapping {mapping.name} has "
            f"{mapping.n_beads}"
        )
    stored = wz.get("retained_indices_0based")
    if stored is not None and list(np.asarray(stored)) != bead_atoms0:
        raise SystemExit(
            f"weight matrix was fitted for AA atoms {list(np.asarray(stored))} but "
            f"mapping {mapping.name} selects {bead_atoms0}"
        )
    print(f"force map: {W.shape[0]} beads <- {W.shape[1]} AA atoms "
          f"(residual at fit {float(wz.get('max_abs_residual', np.nan)):.2e})")

    # A campaign's ramp length is a property of that campaign, not of whoever runs
    # the collector: the inversion ladder equilibrates 10 ps then ramps 20 ps, so a
    # 20 ps CLI default would silently admit 10 ps of steered, non-stationary frames
    # into a dataset documented as fixed-window.
    discard_ps, discard_source = args.discard_ps, "cli"
    if discard_ps is None:
        meta = (args.campaign / "campaign.json") if args.campaign else None
        required = None
        if meta is not None and meta.exists():
            required = json.loads(meta.read_text()).get("required_discard_ps")
        if required is not None:
            discard_ps, discard_source = float(required), "campaign.json"
        else:
            discard_ps, discard_source = 20.0, "fallback default"
    print(f"discarding first {discard_ps:g} ps per replica (source: {discard_source})")

    cases = ([args.case] if args.case
             else _discover_cases(args.campaign))
    if not cases:
        raise SystemExit("no cases found")

    Rs, Fs, per_case = [], [], []
    for case in cases:
        R, F = _load_case(case, bead_atoms0, W, discard_ps, args.gmx, args.force_cap,
                          mapping)
        Rs.append(R); Fs.append(F)
        per_case.append({"case": case.name, "frames": int(len(R))})
        np.savez_compressed(case / "cg_frames.npz", R=R.astype(np.float32),
                            F=F.astype(np.float32))
        print(f"  {case.name}: {len(R)} frames  |F| std {F.std():.2f} kcal/mol/A")

    R = np.concatenate(Rs); F = np.concatenate(Fs)
    n, nb = R.shape[0], R.shape[1]
    out = args.out or ((args.campaign / "enhanced_frames.npz") if args.campaign
                       else (args.case / "cg_frames_dataset.npz"))
    np.savez_compressed(
        out,
        R=R.astype(np.float64), F=F.astype(np.float64),
        species=np.tile(np.arange(nb, dtype=np.int64), (n, 1)),
    )
    (Path(out).parent / "collect_summary.json").write_text(json.dumps(
        {"mapping": mapping.name, "n_beads": nb, "total_frames": int(n),
         "discard_ps": discard_ps, "discard_source": discard_source,
         "force_cap": args.force_cap, "per_case": per_case,
         "weights": str(args.weights),
         "coords_from": "biased.trr", "forces_from": "unbiased_forces.trr (bias-free)",
         "force_map": "aggforce weight matrix applied (NOT slice/group-sum)"},
        indent=2))
    print(f"wrote {out}  ({n} frames x {nb} beads)")

    # Deletion happens ONLY here, after every replica loaded and the aggregate was
    # written. Deleting inside the loop meant a failure on replica k destroyed the
    # raw trajectories of 0..k-1 while no combined dataset existed -- unrecoverable
    # except by re-running the MD.
    if args.delete_trr:
        freed = 0
        for case in cases:
            for name in ("biased.trr", "unbiased_forces.trr", "whole.trr",
                         "aa_coords.xvg", "aa_forces.xvg"):
                p = case / name
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
        print(f"deleted raw trajectories: {freed / 2**30:.1f} GiB freed")


if __name__ == "__main__":
    main()
