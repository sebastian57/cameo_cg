#!/usr/bin/env python3
import argparse
import logging
import sys
from collections import OrderedDict

import numpy as np

from cg_1bead import load_npz, per_type_force_normalization


logger = logging.getLogger("CGBackboneCB")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


BACKBONE_ATOM_TYPES = ("N", "CA", "C", "O")
SIDECHAIN_TYPE_PREFIX = "CB"


def _parse_pdb_atoms(pdbProteinAtoms):
    records = []
    for line in str(pdbProteinAtoms).splitlines():
        if not line.startswith("ATOM"):
            continue
        fields = line.split()
        if len(fields) >= 6:
            atom_name = fields[2]
            resname = fields[3]
            chain = fields[4]
            resid = int(fields[5])
        else:
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resid = int(line[22:26])
        records.append(
            {
                "atom_index": len(records),
                "atom_name": atom_name,
                "resname": resname,
                "chain": chain,
                "resid": resid,
            }
        )
    if not records:
        raise ValueError("No ATOM records found in pdbProteinAtoms. Check dataset integrity.")
    return records


def _selection_from_pdb(pdbProteinAtoms):
    records = _parse_pdb_atoms(pdbProteinAtoms)
    residues = OrderedDict()
    for rec in records:
        key = (rec["chain"], rec["resid"], rec["resname"])
        residues.setdefault(key, {})[rec["atom_name"]] = rec

    selected_indices = []
    atom_names = []
    type_labels = []
    missing = []

    for (_chain, _resid, resname), atoms in residues.items():
        for atom_name in BACKBONE_ATOM_TYPES:
            rec = atoms.get(atom_name)
            if rec is None:
                missing.append((resname, atom_name))
                continue

            selected_indices.append(rec["atom_index"])
            atom_names.append(atom_name)
            if resname == "GLY" and atom_name == "CA":
                type_labels.append(f"{SIDECHAIN_TYPE_PREFIX}_GLY")
            else:
                type_labels.append(atom_name)

        if resname != "GLY":
            rec = atoms.get("CB")
            if rec is None:
                missing.append((resname, "CB"))
                continue
            selected_indices.append(rec["atom_index"])
            atom_names.append("CB")
            type_labels.append(f"{SIDECHAIN_TYPE_PREFIX}_{resname}")

    if missing:
        preview = ", ".join(f"{res}:{atom}" for res, atom in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        logger.warning(f"Missing expected retained atoms ({len(missing)}): {preview}{suffix}")

    if not selected_indices:
        raise ValueError("No retained atoms selected for backbone+CB mapping.")

    return (
        np.asarray(selected_indices, dtype=np.int64),
        np.asarray(atom_names, dtype=object),
        np.asarray(type_labels, dtype=object),
    )


def _project_forces_to_selected_atoms(forces, coords, selected_indices):
    try:
        from aggforce import LinearMap, guess_pairwise_constraints, project_forces
    except ImportError as exc:
        raise ImportError(
            "aggforce is required when use_aggforce=True. "
            "Install aggforce or rerun with --no_aggforce for local CPU smoke data."
        ) from exc

    cmap = LinearMap([[int(i)] for i in selected_indices], n_fg_sites=coords.shape[1])
    constraints = guess_pairwise_constraints(coords[0:100], threshold=5e-3)
    return project_forces(
        coords=coords,
        forces=forces,
        coord_map=cmap,
        constrained_inds=constraints,
    )


def _build_type_mapping(type_labels):
    unique_labels = sorted(set(str(label) for label in type_labels))
    aa_to_id = {label: i for i, label in enumerate(unique_labels)}
    species = np.asarray([aa_to_id[str(label)] for label in type_labels], dtype=np.int64)
    return species, aa_to_id


def build_cg_dataset(npz_in, npz_out, use_aggforce=True, normalize_forces=False):
    dataset, general = load_npz(npz_in)

    logger.info(f"Loaded NPZ dataset: {npz_in}")
    selected_indices, atom_names, type_labels = _selection_from_pdb(general["pdbProteinAtoms"])

    logger.info(
        f"Retained {len(selected_indices)} atoms "
        f"({len(set(general['resid'][selected_indices]))} residues)."
    )
    logger.debug(
        f"  retained indices: {selected_indices[:20]}"
        f"{' ...' if len(selected_indices) > 20 else ''}"
    )

    R_cg = dataset["R"][:, selected_indices, :]
    F_cg = dataset["F"][:, selected_indices, :]
    Z_cg = general["Z"][selected_indices]
    resid_cg = general["resid"][selected_indices]
    resname_cg = general["resname"][selected_indices]
    species, aa_to_id = _build_type_mapping(type_labels)

    logger.info(f"Atom type mapping ({len(aa_to_id)} types): {aa_to_id}")

    aggforce_weight_matrix = None
    if use_aggforce:
        optimal_mapping = _project_forces_to_selected_atoms(
            dataset["F"], dataset["R"], selected_indices
        )
        F_out = optimal_mapping["mapped_forces"]
        logger.info("Force projection: aggforce optimal mapping to retained atoms")
        try:
            aggforce_weight_matrix = np.asarray(
                optimal_mapping["tmap"].force_map.standard_matrix, dtype=np.float64
            )
            logger.info(f"Extracted aggforce weight matrix: shape {aggforce_weight_matrix.shape}")
        except Exception as exc:
            logger.warning(f"Could not extract aggforce weight matrix: {exc}")
    else:
        F_out = F_cg
        logger.info("Force projection: retained-atom sliced (aggforce disabled)")

    if normalize_forces:
        F_out, sigmas, counts = per_type_force_normalization(
            F_out, species, n_types=len(aa_to_id)
        )
        logger.info(f"Per-type force normalization applied (n_types={len(aa_to_id)})")
        logger.debug(f"  sigmas: {sigmas}")
        logger.debug(f"  counts: {counts}")

    n_beads = R_cg.shape[1]
    mask = np.ones(n_beads, dtype=np.float32)

    save_kwargs = dict(
        R=R_cg,
        F=F_out,
        Z=Z_cg,
        resid=resid_cg,
        resname=resname_cg,
        atom_name=atom_names,
        type_label=type_labels,
        species=np.tile(species[None, :], (R_cg.shape[0], 1)),
        aa_to_id=np.array([aa_to_id], dtype=object),
        mask=np.tile(mask[None, :], (R_cg.shape[0], 1)),
        box=dataset["box"] * 10,
        retained_indices=selected_indices,
        N_max=np.array([n_beads], dtype=np.int32),
    )
    if aggforce_weight_matrix is not None:
        save_kwargs["aggforce_weight_matrix"] = aggforce_weight_matrix

    np.savez(f"{npz_out}", **save_kwargs)
    logger.info(f"Saved CG dataset to: {npz_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Coarse-grain NPZ dataset to retained N/CA/C/O/CB atoms."
    )
    parser.add_argument("--infile", required=True, help="Input NPZ file path")
    parser.add_argument("--outfile", required=True, help="Output NPZ file path")
    parser.add_argument(
        "--use_aggforce",
        action="store_true",
        default=True,
        help="Use aggforce optimal force mapping (default: True).",
    )
    parser.add_argument(
        "--no_aggforce",
        dest="use_aggforce",
        action="store_false",
        help="Disable aggforce; save retained-atom sliced forces.",
    )
    parser.add_argument(
        "--normalize_forces",
        action="store_true",
        default=False,
        help="Apply per-type force normalization to output forces.",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    build_cg_dataset(
        args.infile,
        args.outfile,
        use_aggforce=args.use_aggforce,
        normalize_forces=args.normalize_forces,
    )


if __name__ == "__main__":
    main()
