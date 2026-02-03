import argparse
import h5py
import numpy as np
import logging
import sys
from pathlib import Path

KCAL_MOL_TO_EV = 0.04336425351090843

# Logger — matches clean_code_base [Name] format from utils/logging.py
logger = logging.getLogger("H5Transform")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def iter_groups(root_group):
    """Yield (name, group) for all subgroups under the protein root."""
    for cond_name in sorted(root_group.keys()):
        cond = root_group[cond_name]
        if isinstance(cond, h5py.Group):
            yield cond_name, cond


def iter_runs(cond_group):
    for run_name in sorted(cond_group.keys()):
        run = cond_group[run_name]
        if isinstance(run, h5py.Group):
            yield run_name, run


def safe_decode(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    if isinstance(obj, np.ndarray) and obj.shape == ():
        item = obj.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(obj)


def select_frames(coords, forces, box, nframes):
    """Select first nframes frames."""
    total_frames = coords.shape[0]
    n = min(nframes, total_frames)

    R_list = coords[:n]
    F_list = forces[:n]
    box_list = np.repeat(box[None, :, :], n, axis=0)

    return R_list, F_list, box_list


def build_dataset(
    h5_path,
    protein_id,
    out_file,
    nframes,
    temp_groups,
    convert_to_ev=False,
):
    h5_path = Path(h5_path)
    assert h5_path.exists(), f"HDF5 file not found: {h5_path}"

    logger.info("=== Extracting frames ===")
    logger.info(f"Input H5 file:      {h5_path}")
    logger.info(f"Protein ID:         {protein_id}")
    logger.info(f"Frames per run:     {nframes}")
    logger.info(f"Temperature groups: {temp_groups if temp_groups != 'all' else 'ALL AVAILABLE'}")
    if convert_to_ev:
        logger.info("Force units:        kcal/mol -> eV")

    all_R = []
    all_F = []
    all_box = []
    frame_origins = []

    with h5py.File(h5_path, "r") as f:
        protein = f[protein_id]

        # Topology-level data
        chain = protein["chain"][()].astype(str)
        element = protein["element"][()].astype(str)
        resid = protein["resid"][()].astype(np.int64)
        resname = protein["resname"][()].astype(str)
        Z = protein["z"][()].astype(np.int64)

        pdb = safe_decode(protein["pdb"][()])
        pdbProteinAtoms = safe_decode(protein["pdbProteinAtoms"][()])

        # Loop through condition groups (e.g., 320, 348...)
        for cond_name, cond_group in iter_groups(protein):

            # Temperature filtering
            if temp_groups != "all":
                if cond_name not in temp_groups:
                    continue

            logger.info(f"Processing temperature group: {cond_name}")

            for run_name, run_group in iter_runs(cond_group):

                coords = run_group["coords"][()]   # (frames, n_atoms, 3)
                forces = run_group["forces"][()]   # (frames, n_atoms, 3)
                box = run_group["box"][()]         # (3, 3)

                if convert_to_ev:
                    forces = forces * KCAL_MOL_TO_EV

                R_list, F_list, box_list = select_frames(coords, forces, box, nframes)

                all_R.append(R_list.astype(np.float32))
                all_F.append(F_list.astype(np.float32))
                all_box.append(box_list.astype(np.float32))

                for i in range(len(R_list)):
                    frame_origins.append(f"{cond_name}/{run_name}/frame{i}")

    R = np.concatenate(all_R, axis=0)
    F = np.concatenate(all_F, axis=0)
    box = np.concatenate(all_box, axis=0)

    logger.info(f"Collected {len(R)} total frames.")

    np.savez(
        out_file,
        R=R,
        F=F,
        box=box,
        Z=Z,
        chain=chain,
        element=element,
        resid=resid,
        resname=resname,
        pdb=pdb,
        pdbProteinAtoms=pdbProteinAtoms,
        frame_ids=np.array(frame_origins, dtype=object),
    )

    logger.info(f"Saved dataset to {out_file}")
    logger.info(f"  R={R.shape}, F={F.shape}, box={box.shape}, Z={Z.shape}")
    logger.debug("Frame origins:")
    for o in frame_origins:
        logger.debug(f"  - {o}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from MDCath HDF5 dataset.")
    parser.add_argument("--h5", required=True, help="Input HDF5 dataset file")
    parser.add_argument("--protein", required=True, help="Protein ID inside the HDF5 file")
    parser.add_argument("--nframes", type=int, required=True, help="Number of frames per run")
    parser.add_argument("--out", required=True, help="Output NPZ file")
    parser.add_argument(
        "--temps",
        nargs="+",
        default="all",
        help=(
            "Temperature groups to include (e.g. 320 348 379), "
            "or 'all' to include every group."
        )
    )
    parser.add_argument("--convert_to_ev", action="store_true", default=False,
                        help="Convert forces from kcal/mol to eV before saving.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Handle "all" case
    temp_groups = args.temps
    if isinstance(temp_groups, list) and len(temp_groups) == 1 and temp_groups[0].lower() == "all":
        temp_groups = "all"
    elif isinstance(temp_groups, list):
        temp_groups = [t.strip() for t in temp_groups]

    build_dataset(
        h5_path=args.h5,
        protein_id=args.protein,
        out_file=args.out,
        nframes=args.nframes,
        temp_groups=temp_groups,
        convert_to_ev=args.convert_to_ev,
    )


if __name__ == "__main__":
    main()
