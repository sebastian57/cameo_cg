import numpy as np
from pathlib import Path

def dump_to_npz(dump_path: Path, npz_path: Path):
    with open(dump_path, "r") as f:
        lines = f.readlines()

    # 1) Find number of atoms
    for i, line in enumerate(lines):
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            n_atoms = int(lines[i + 1].strip())
            break
    else:
        raise ValueError("NUMBER OF ATOMS not found")

    # 2) Find ATOMS section
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            header = line.strip().split()[2:]
            start = i + 1
            break
    else:
        raise ValueError("ITEM: ATOMS section not found")

    # 3) Column indices
    ix = header.index("x")
    iy = header.index("y")
    iz = header.index("z")

    # 4) Read atom lines
    atom_lines = lines[start : start + n_atoms]
    data = np.array([l.split() for l in atom_lines], dtype=np.float32)

    R = data[:, [ix, iy, iz]]  # (N, 3)

    # 5) Save in training-compatible format
    np.savez(npz_path, R=R.astype(np.float32))

def main():
	dump_path = "datasets/single_problem_frame_1.out"
	npz_path = "datasets/single_problem_frame_1.npz"
	dump_to_npz(dump_path, npz_path)

if __name__ == "__main__":
    main()

