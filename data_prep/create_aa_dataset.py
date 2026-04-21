#!/usr/bin/env python3
"""
All-atom NPZ dataset creator from mdCATH H5 files.

Extracts all protein atoms (no water) from an mdCATH H5 file across all
requested temperature groups and concatenates into a single NPZ file ready
for training an all-atom Allegro model.

Species assignment: unique atomic numbers Z are sorted ascending and mapped
to 0-indexed integers (e.g. H(1)→0, C(6)→1, N(7)→2, O(8)→3, S(16)→4).

Required NPZ fields produced:
  R        (N_frames, N_atoms, 3) float32  coordinates in Å
  F        (N_frames, N_atoms, 3) float32  forces in kcal/mol/Å (or eV/Å)
  species  (N_atoms,)             int32    element-type index (0-based)
  Z        (N_atoms,)             int32    atomic numbers
  mask     (N_frames, N_atoms)    float32  all-ones (no padding for single protein)
  aa_to_id pickled dict                    element symbol → species index

Optional fields also saved for bookkeeping:
  box       (N_frames, 3, 3) float32   simulation box matrix per frame
  element   (N_atoms,)                 element symbol strings
  resid     (N_atoms,)       int32     residue IDs
  resname   (N_atoms,)                 residue name strings
  frame_ids (N_frames,)               "temp/run/frame" labels

Usage:
    python data_prep/create_aa_dataset.py \\
        --h5   data_prep/datasets/all_atom/mdcath_dataset_4zohB01.h5 \\
        --protein 4zohB01 \\
        --nframes 500 \\
        --out_dir data_prep/datasets/1pro_4zohB01_alltemp_aa

    # Select specific temperatures:
    python data_prep/create_aa_dataset.py ... --temps 320 348

    # Convert forces to eV/Å:
    python data_prep/create_aa_dataset.py ... --convert_to_ev
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

KCAL_MOL_TO_EV = 0.04336425351090843

ATOMIC_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S", 34: "Se"}

logger = logging.getLogger("CreateAADataset")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _sorted_group_keys(group):
    return sorted(group.keys(), key=lambda k: (0, int(k)) if k.isdigit() else (1, k))


def build_species_map(z_arr: np.ndarray):
    """Map atomic numbers → sorted 0-indexed species IDs.

    Returns:
        species  (N_atoms,) int32 — species index per atom
        elem_to_id  dict[str, int] — element symbol → species index
        unique_z  list[int] — sorted unique atomic numbers
    """
    unique_z = sorted(int(v) for v in np.unique(z_arr))
    z_to_idx = {zv: i for i, zv in enumerate(unique_z)}
    species = np.array([z_to_idx[int(zv)] for zv in z_arr], dtype=np.int32)
    elem_to_id = {
        ATOMIC_SYMBOLS.get(zv, f"Z{zv}"): i for zv, i in z_to_idx.items()
    }
    return species, elem_to_id, unique_z


def build_dataset(
    h5_path: Path,
    protein_id: str,
    out_path: Path,
    nframes: int,
    temp_groups,
    convert_to_ev: bool = False,
):
    """Extract all-atom frames from h5 and write all-atom NPZ."""
    logger.info("=== All-Atom Dataset Extraction ===")
    logger.info(f"H5 file:       {h5_path}")
    logger.info(f"Protein:       {protein_id}")
    logger.info(f"Frames/run:    {nframes}")
    logger.info(f"Temperatures:  {temp_groups if temp_groups != 'all' else 'ALL'}")
    logger.info(f"Force units:   {'eV/Å' if convert_to_ev else 'kcal/mol/Å'}")

    all_R, all_F, all_box = [], [], []
    frame_ids = []

    with h5py.File(h5_path, "r") as h5f:
        prot = h5f[protein_id]

        # Topology (same for all frames)
        Z = prot["z"][()].astype(np.int64)
        element = prot["element"][()].astype(str)
        resid = prot["resid"][()].astype(np.int64)
        resname = prot["resname"][()].astype(str)

        # Loop temperature conditions
        for cond_name in _sorted_group_keys(prot):
            if not cond_name.isdigit():
                continue
            if temp_groups != "all" and cond_name not in temp_groups:
                continue

            logger.info(f"  Temperature {cond_name} K ...")
            cond = prot[cond_name]

            for run_name in _sorted_group_keys(cond):
                run = cond[run_name]
                coords = run["coords"][()]   # (n_frames, n_atoms, 3)
                forces = run["forces"][()]   # (n_frames, n_atoms, 3)
                box_mat = run["box"][()]     # (3, 3)

                if convert_to_ev:
                    forces = forces * KCAL_MOL_TO_EV

                n_take = min(nframes, coords.shape[0])
                coords = coords[:n_take].astype(np.float32)
                forces = forces[:n_take].astype(np.float32)
                box_rep = np.repeat(box_mat[None], n_take, axis=0).astype(np.float32)

                all_R.append(coords)
                all_F.append(forces)
                all_box.append(box_rep)
                for i in range(n_take):
                    frame_ids.append(f"{cond_name}/{run_name}/frame{i}")

    R = np.concatenate(all_R, axis=0)  # (N_frames, N_atoms, 3)
    F = np.concatenate(all_F, axis=0)
    box = np.concatenate(all_box, axis=0)
    mask = np.ones(R.shape[:2], dtype=np.float32)

    species, elem_to_id, unique_z = build_species_map(Z)

    logger.info(f"Total frames: {R.shape[0]}")
    logger.info(f"Atoms:        {R.shape[1]}")
    logger.info(f"Species map:  {elem_to_id}  ({len(elem_to_id)} types)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        R=R,
        F=F,
        mask=mask,
        species=species,
        Z=Z.astype(np.int32),
        box=box,
        element=element,
        resid=resid.astype(np.int32),
        resname=resname,
        aa_to_id=np.array(elem_to_id, dtype=object),
        frame_ids=np.array(frame_ids, dtype=object),
    )

    logger.info(f"Saved: {out_path}")
    logger.info(f"  R={R.shape}, F={F.shape}, mask={mask.shape}, species={species.shape}")
    for sym, idx in sorted(elem_to_id.items(), key=lambda x: x[1]):
        zval = unique_z[idx]
        count = int(np.sum(Z == zval))
        logger.info(f"    species {idx}: {sym} (Z={zval}), {count} atoms")


def main():
    parser = argparse.ArgumentParser(
        description="Create all-atom NPZ dataset from mdCATH H5 file."
    )
    parser.add_argument("--h5", required=True, help="Input H5 file (mdcath_dataset_*.h5)")
    parser.add_argument("--protein", required=True, help="Protein key inside H5 (e.g. 4zohB01)")
    parser.add_argument("--nframes", type=int, required=True, help="Frames per run to extract")
    parser.add_argument("--out_dir", required=True, help="Output directory; NPZ saved as <protein>_alltemp_aa.npz")
    parser.add_argument(
        "--temps",
        nargs="+",
        default=["all"],
        help="Temperature groups to include (e.g. 320 348 379) or 'all' (default)",
    )
    parser.add_argument("--out_name", default=None, help="Override output filename (default: <protein>_alltemp_aa.npz)")
    parser.add_argument("--convert_to_ev", action="store_true", default=False,
                        help="Convert forces from kcal/mol/Å to eV/Å")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    temp_groups = args.temps
    if len(temp_groups) == 1 and temp_groups[0].lower() == "all":
        temp_groups = "all"

    out_dir = Path(args.out_dir)
    out_name = args.out_name or f"{args.protein}_alltemp_aa.npz"
    out_path = out_dir / out_name

    build_dataset(
        h5_path=Path(args.h5),
        protein_id=args.protein,
        out_path=out_path,
        nframes=args.nframes,
        temp_groups=temp_groups,
        convert_to_ev=args.convert_to_ev,
    )


if __name__ == "__main__":
    main()
