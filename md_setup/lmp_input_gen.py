import numpy as np

#data = np.load("2gy5A01_320K_kcalmol_1bead_notnorm_aggforce.npz")
data = np.load("4zohB01_320K_kcalmol_1bead_notnorm_aggforce.npz")

i = 2234

pos = data["R"][i]    


resnames = data["resname"] 

unique_aa = sorted(set(resnames))
aa_to_id = {aa: i for i, aa in enumerate(unique_aa)}

print(f"\nAmino acid type mapping:")
for aa, id in aa_to_id.items():
    print(f"  {aa} -> species {id}")

species = np.array([aa_to_id[aa] for aa in resnames], dtype=np.int32)

N_species = len(unique_aa)
print(f"\nTotal unique amino acid types: {N_species}")
print(f"Species array: {species}")
print(f"Species range: {np.min(species)} to {np.max(species)}")

N = len(pos)

xlo, xhi = -400.0, 400.0
ylo, yhi = -400.0, 400.0
zlo, zhi = -400.0, 400.0

with open(f"config_{i}.data", "w") as f:
    # Header
    f.write("LAMMPS data file via write_data\n\n")
    f.write(f"{N} atoms\n")
    f.write(f"{N_species} atom types\n\n")

    # Box
    f.write(f"{xlo:.1f} {xhi:.1f} xlo xhi\n")
    f.write(f"{ylo:.1f} {yhi:.1f} ylo yhi\n")
    f.write(f"{zlo:.1f} {zhi:.1f} zlo zhi\n\n")

    # Atoms section
    f.write("Atoms\n\n")

    assert len(pos) == N
    assert len(species) == N

    for idx in range(N):
        species_lammps = int(species[idx]) + 1
        x, y, z = pos[idx]
        f.write(f"{idx+1} {species_lammps} {x:.8f} {y:.8f} {z:.8f}\n")

print(f"\nSuccessfully wrote {N} atoms to config.data")
