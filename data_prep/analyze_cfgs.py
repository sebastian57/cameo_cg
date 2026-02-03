#!/usr/bin/env python3
import argparse
import numpy as np


def load_npz(path):
    data = np.load(path, allow_pickle=True)

    protein_specific = {
        "R": data["R"].astype(np.float32),     # (N_frames, n_atoms, 3)
        "F": data["F"].astype(np.float32),     # (N_frames, n_atoms, 3)
        "box": data["box"].astype(np.float32)  # (N_frames, 3, 3)
    }

    general = {
        "Z": data["Z"].astype(np.int64),
    }

    return protein_specific, general




specific, general = load_npz("datasets/4zohB01_320K_1bead.npz")

print("forces")
forces = specific["F"][0]
print(np.max(forces))
print(np.max(np.abs(forces)))
print(np.min(forces))
print(np.min(np.abs(forces)))

print("positions")
pos = specific["R"][0]
print(np.max(pos))
print(np.max(np.abs(pos)))
print(np.min(pos))
print(np.min(np.abs(pos)))

print("box")
box = specific["box"][0]
print(np.max(box))





