#!/usr/bin/env python3
import argparse
import numpy as np
import jax.numpy as jnp

from aggforce import (LinearMap, 
                      guess_pairwise_constraints, 
		      project_forces, 
		      constraint_aware_uni_map,
		      )
import re
import mdtraj as md
import tempfile
import os

def generate_optim_forcematch(forces, coords, pdb_string):

    inds = []

    topology = pdb_string_to_topology(str(pdb_string))
    atomlist = list(topology.atoms)

    for ind, a in enumerate(atomlist):
        if re.search(r"CA$", str(a)):
            inds.append([ind])


    cmap = LinearMap(inds, n_fg_sites=coords.shape[1])

    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    basic_results = project_forces(
        coords=coords,
        forces=forces,
        coord_map=cmap,
        constrained_inds=constraints,
        method=constraint_aware_uni_map,
    )

    optim_results = project_forces(
        coords=coords,
        forces=forces, 
        coord_map=cmap,
        constrained_inds=constraints
    )
    
    return optim_results

def pdb_string_to_topology(pdb_string):

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp.write(pdb_string)
        tmp_path = tmp.name
    
    try:
        pdb_structure = md.load(tmp_path)
        topology = pdb_structure.topology
    finally:
        os.unlink(tmp_path)
    
    return topology

def per_type_force_normalization(F_cg: np.ndarray, species: np.ndarray, n_types: int = 4, eps: float = 1e-12):

    F_cg = np.asarray(F_cg, dtype=np.float32)
    species = np.asarray(species)

    if F_cg.ndim != 3 or F_cg.shape[-1] != 3:
        raise ValueError(f"Expected F_cg shape (N_frames, n_beads, 3), got {F_cg.shape}")
    if species.ndim != 1 or species.shape[0] != F_cg.shape[1]:
        raise ValueError(f"Expected species shape (n_beads,), got {species.shape} for n_beads={F_cg.shape[1]}")

    F_norm = np.empty_like(F_cg)
    sigmas = np.zeros((n_types,), dtype=np.float32)
    counts = np.zeros((n_types,), dtype=np.int64)

    for t in range(n_types):
        mask = (species == t)              
        counts[t] = int(mask.sum())

        if counts[t] == 0:
            sigmas[t] = 1.0
            continue

        F_t = F_cg[:, mask, :]

        sigma_t = np.sqrt(np.mean(F_t * F_t))
        sigma_t = float(max(sigma_t, eps))
        sigmas[t] = sigma_t

        F_norm[:, mask, :] = F_t / sigma_t

    return F_norm, sigmas, counts


def group_amino_acids_4way(resname):
    """
    Group 18 amino acids into 4 categories based on charge/polarity.
    
    Returns:
        species: Array of group IDs (0-3)
        group_names: List of group names
        aa_to_group: Dict mapping AA name to group ID
    """
    
    POSITIVE = {
        'LYS',  # Lysine
        'ARG',  # Arginine
        'HIS',  # Histidine (generic)
        'HSP',  # Histidine protonated
        'HSD',  # Histidine neutral
        'HSE',  # Histidine epsilon-protonated
    }
    
    NEGATIVE = {
        'GLU',  # Glutamic acid - negatively charged
        'ASP',  # Aspartic acid - negatively charged
    }
    
    POLAR_UNCHARGED = {
        'SER',  # Serine
        'THR',  # Threonine
        'ASN',  # Asparagine
        'GLN',  # Glutamine
        'TYR',  # Tyrosine (aromatic but polar)
        'CYS',  # Cysteine (thiol, weakly polar)
    }
    
    NONPOLAR = {
        'ALA',  # Alanine
        'VAL',  # Valine
        'LEU',  # Leucine
        'ILE',  # Isoleucine
        'MET',  # Methionine (thioether, hydrophobic)
        'PHE',  # Phenylalanine
        'TRP',  # Tryptophan (aromatic, mostly hydrophobic)
        'PRO',  # Proline
        'GLY',  # Glycine
    }

    
    aa_to_group = {}
    for aa in POSITIVE:
        aa_to_group[aa] = 0
    for aa in NEGATIVE:
        aa_to_group[aa] = 1
    for aa in POLAR_UNCHARGED:
        aa_to_group[aa] = 2
    for aa in NONPOLAR:
        aa_to_group[aa] = 3
    
    group_names = [
        "Positive (Basic)",
        "Negative (Acidic)", 
        "Polar Uncharged",
        "Nonpolar (Hydrophobic)"
    ]
    
    species = jnp.array([aa_to_group[aa] for aa in resname])
    
    return species, group_names, aa_to_group

def load_npz(path):
    data = np.load(path, allow_pickle=True)

    protein_specific = {
        "R": data["R"].astype(np.float32),     # (N_frames, n_atoms, 3)
        "F": data["F"].astype(np.float32),     # (N_frames, n_atoms, 3)
        "box": data["box"].astype(np.float32)  # (N_frames, 3, 3)
    }

    general = {
        "Z": data["Z"].astype(np.int64),
        "resid": data["resid"].astype(np.int64),
        "resname": data["resname"],
        "chain": data["chain"],
        "element": data["element"],
        "pdb": data["pdb"],
        "pdbProteinAtoms": data["pdbProteinAtoms"],
    }

    return protein_specific, general


def extract_ca_indices(pdbProteinAtoms):
    lines = str(pdbProteinAtoms).split("\n")

    atom_names = []
    for line in lines:
        if line.startswith("ATOM"):
            # PDB atom name columns (13–16)
            atom_name = line[12:16].strip()
            atom_names.append(atom_name)

    atom_names = np.array(atom_names, dtype=object)
    ca_indices = np.where(atom_names == "CA")[0]

    if len(ca_indices) == 0:
        raise ValueError("No CA atoms found in pdbProteinAtoms. Check dataset integrity.")

    return ca_indices


def build_cg_dataset(npz_in, npz_out):
    dataset, general = load_npz(npz_in)

    print(f" Loaded NPZ dataset: {npz_in}")
    print(f"  R shape = {dataset['R'].shape}")

    ca_indices = extract_ca_indices(general["pdbProteinAtoms"])

    print(f" Found {len(ca_indices)} CA atoms (residues).")
    print(f" CA indices: {ca_indices[:10]}{' ...' if len(ca_indices) > 10 else ''}")

    R_cg = dataset["R"][:, ca_indices, :]
    F_cg = dataset["F"][:, ca_indices, :]
    Z_cg = general["Z"][ca_indices]
    resid_cg = general["resid"][ca_indices]
    resname_cg = general["resname"][ca_indices]

    # species, group_names, aa_to_group = group_amino_acids_4way(resname_cg)
    # species = np.asarray(species, dtype=np.int64)

    unique_aas = set(resname_cg)
    aa_to_id = {aa: i for i, aa in enumerate(sorted(unique_aas))}
    species = jnp.array([aa_to_id[aa] for aa in resname_cg])

    # Normalization
    # sigma_F = np.sqrt(np.mean(F_cg**2))
    # F_cg_norm = F_cg / sigma_F
    F_cg_norm, sigmas, counts = per_type_force_normalization(F_cg, species, n_types=4)

    print("Per-type force sigmas:", sigmas)
    print("Per-type bead counts: ", counts)

    optimal_mapping = generate_optim_forcematch(dataset["F"],dataset["R"],general["pdb"])

    F_CG_optimal = optimal_mapping["mapped_forces"]
    R_CG_optimal = optimal_mapping["mapped_coords"]
    constraints_optimal = optimal_mapping["constraints"]

    n_beads = R_cg.shape[1]
    mask = jnp.ones(n_beads, dtype=jnp.float32)
    N_max = n_beads

    print(f"  F_CG_optimal contents = {list(optimal_mapping.keys())}")

    # Aggforce vs base comparison
    print(f"Max Aggforce: {np.max(np.abs(F_CG_optimal))}")
    print(f"Max base: {np.max(np.abs(F_cg))}")

    print(f"Mean Aggforce: {np.mean(np.abs(F_CG_optimal))}")
    print(f"Mean base: {np.mean(np.abs(F_cg))}")


    print(f"Max Aggforce coords: {np.max(np.abs(R_CG_optimal))}")
    print(f"Max base coords: {np.max(np.abs(R_cg))}")

    print(f"Mean Aggforce coords: {np.mean(np.abs(R_CG_optimal))}")
    print(f"Mean base coods: {np.mean(np.abs(R_cg))}")

    print(F"Aggforce constraints: {constraints_optimal}")

    print(f" Coarse-grained shapes:")
    print(f"  R_cg = {R_cg.shape}")
    print(f"  F_cg = {F_cg.shape}")
    print(f"  Z_cg = {Z_cg.shape}")

    print(f"box: {dataset['box'].shape}")


    np.savez(
        f"{npz_out}",
        R=R_cg,
        F=F_CG_optimal, # F_cg_norm, # 
        Z=Z_cg,
        resid=resid_cg,
        resname=resname_cg,
        species=[species]*R_cg.shape[0],
        aa_to_id=aa_to_id,
        mask=[mask]*R_cg.shape[0],
        box=dataset["box"]*10,
        ca_indices=ca_indices,
        N_max=[n_beads]*10,
    )

    print(f"\n Saved CG dataset to: {npz_out}")


def main():
    parser = argparse.ArgumentParser(description="Coarse-grain NPZ dataset to CA atoms.")
    parser.add_argument("--infile", required=True, help="Input NPZ file path")
    parser.add_argument("--outfile", required=True, help="Output NPZ file path for CG dataset")
    args = parser.parse_args()

    build_cg_dataset(args.infile, args.outfile)


if __name__ == "__main__":
    main()

