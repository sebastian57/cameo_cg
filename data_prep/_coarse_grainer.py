import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    
    protein_specific = {
        "R": data["R"].astype(np.float32),   # (N, natoms, 3)
        "F": data["F"].astype(np.float32),   # (N, natoms, 3)
        "box": data["box"].astype(np.float32)
    }

    general = {"Z": data["Z"].astype(np.int64), 
    "resid": data["resid"].astype(np.int64), 
    "resname": data["resname"], 
    "chain": data["chain"], 
    "element": data["element"], 
    "pdb": data["pdb"], 
    "pdbProteinAtoms": data["pdbProteinAtoms"]}
    
    return protein_specific, general 

path = "5k39B02_singleframes.npz"
dataset, general = load_npz(path)

lines = str(general["pdb"]).split("\n")

atom_names = []
for line in lines:
    if line.startswith("ATOM"):
        atom_name = line[12:16].strip()
        atom_names.append(atom_name)

atom_names = np.array(atom_names, dtype=object)
ca_indices = np.where(atom_names == "CA")[0]

print("Number of residues:", len(ca_indices))
print("CA indices:", ca_indices)

R_cg = dataset["R"][:, ca_indices, :]
F_cg = dataset["F"][:, ca_indices, :]
Z_cg = general["Z"][ca_indices]

print(dataset['R'].shape)
print(R_cg.shape)
print(Z_cg.shape)







