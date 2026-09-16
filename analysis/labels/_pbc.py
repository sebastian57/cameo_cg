"""Unwrap a small solute's beads across the periodic boundary.

Raw GROMACS xtc/trr frames store molecules broken across the boundary — the same hazard
`sampling/build_stencil_campaign_v4.py` documents ("a bead-bead distance of 31.8 A in a 33.2 A
box"). The production collectors handle it with `trjconv -pbc whole` / `mdtraj.image_molecules`;
the DHH screening scripts read bead coordinates straight out of the trajectory and need this.

The boxes here are rhombic dodecahedra (angles 60/60/90), so a per-axis `round(d/L)` is wrong.
Fractional-space rounding against the full box matrix is exact whenever every intra-solute
separation stays inside the cell's inscribed sphere (~11.9 A here, against a ~5 A solute).
"""
from __future__ import annotations
import numpy as np


def unwrap_beads(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Place every bead at its minimum image relative to bead 0.

    x: (n_frames, n_beads, 3) coordinates; H: (n_frames, 3, 3) box vectors as ROWS.
    Both in the same length unit. Returns an array of the same shape.
    """
    x = np.asarray(x, np.float64)
    H = np.asarray(H, np.float64)
    d = x - x[:, :1, :]
    f = np.einsum("fbi,fij->fbj", d, np.linalg.inv(H))
    return x[:, :1, :] + np.einsum("fbi,fij->fbj", f - np.round(f), H)
