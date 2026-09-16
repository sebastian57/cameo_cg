import numpy as np
from pathlib import Path
from typing import List, Callable, Type, Union, Optional, Tuple
import matplotlib.pyplot as plt


def wrap_degrees(angle: float) -> float:
    """Wrap angles in degrees to the interval [-180, 180)."""
    
    return ((angle + 180.0) % 360.0) - 180.0




def compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the signed dihedral angle in degrees for four 3D points."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0:
        raise ValueError("Cannot compute dihedral because two consecutive points are identical.")

    b1_unit = b1 / b1_norm
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return np.degrees(np.arctan2(y, x))


def compute_ramachandran_angles(
    cg_coords: np.ndarray, phi_shift: float = 180.0, psi_shift: float = 180.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phi/psi angles for an Ace-Ala-Nme 5-site CG trajectory.

    Expected bead order:
        0: ACE-C
        1: ALA-N
        2: ALA-CA
        3: ALA-C
        4: NME-N
    """
    if cg_coords.ndim != 3 or cg_coords.shape[1] != 5 or cg_coords.shape[2] != 3:
        raise ValueError(
            f"Expected cg_coords with shape (n_frames, 5, 3), got {cg_coords.shape}"
        )

    phi = np.empty(cg_coords.shape[0], dtype=float)
    psi = np.empty(cg_coords.shape[0], dtype=float)

    for i, frame in enumerate(cg_coords):
        phi[i] = wrap_degrees(
            compute_dihedral(frame[0], frame[1], frame[2], frame[3]) + phi_shift
        )
        psi[i] = wrap_degrees(
            compute_dihedral(frame[1], frame[2], frame[3], frame[4]) + psi_shift
        )

    return phi, psi



def save_dihedral_dat(output_path: Union[str, Path], phi: np.ndarray, psi: np.ndarray) -> None:
    """Save phi/psi angles in a simple whitespace-separated .dat file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([phi, psi])
    header = "phi_deg psi_deg"
    np.savetxt(output_path, data, fmt=["%.8f", "%.8f"], header=header)





### Plotting Tools


def plot_circular_arrow(ax, axis, origin=np.array([0,0,0]), radius=0.4, 
                         height_frac=0.5, color='red', arc_fraction=0.75):
    """
    Draw a circular arrow around an arbitrary axis vector.
    
    axis        : 3D vector defining the rotation axis
    origin      : base of the axis arrow
    height_frac : where along the axis to place the ring (0=base, 1=tip)
    arc_fraction: fraction of full circle to draw (e.g. 0.75 = 270°)
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    axis_hat = axis / norm

    # --- Build two vectors perpendicular to the axis ---
    # Pick an arbitrary vector not parallel to axis
    if abs(axis_hat[0]) < 0.9:
        tmp = np.array([1, 0, 0])
    else:
        tmp = np.array([0, 1, 0])

    u = np.cross(axis_hat, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis_hat, u)  # already unit length

    # --- Arc ---
    theta = np.linspace(0, arc_fraction * 2 * np.pi, 100)
    center = origin + height_frac * axis
    points = (center[:, None]
              + radius * np.cos(theta) * u[:, None]
              + radius * np.sin(theta) * v[:, None])

    ax.plot(points[0], points[1], points[2], color=color, linewidth=2)

    # --- Arrowhead tangent at end of arc ---
    end = theta[-1]
    tangent = -np.sin(end) * u + np.cos(end) * v
    ax.quiver(*points[:, -1], *tangent,
              color=color, length=radius*0.5, arrow_length_ratio=1.0, linewidth=2)

## Example usage
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# Arbitrary axis
# axis = np.array([1, 1, 2])
# origin = np.array([0, 0, 0])
#
# Draw the axis arrow
# ax.quiver(*origin, *axis, color='blue', arrow_length_ratio=0.1, linewidth=2)
#
# Draw circular arrow around it
# plot_circular_arrow(ax, axis, origin=origin, radius=0.4, height_frac=0.5)
# 
# ax.set_xlim(-1, 2); ax.set_ylim(-1, 2); ax.set_zlim(-1, 3)
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# ax.set_title('Torque around arbitrary axis')
# 
# plt.tight_layout()
# plt.show()





"""Historical helper functions retained for numerical compatibility."""
