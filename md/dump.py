"""Write LAMMPS dump files from cameo_cg trajectory NPZ arrays for OVITO."""

from pathlib import Path

import numpy as np


def write_lammps_dump(
    traj_npz: str,
    output_dump: str,
    padding: float = 20.0,
) -> None:
    """Read a trajectory NPZ and write a LAMMPS custom dump file.

    Format: ITEM TIMESTEP / NUMBER OF ATOMS / BOX BOUNDS pp pp pp /
            ATOMS id type xu yu zu fx fy fz

    Coordinates are centred on the trajectory mean so the protein sits at
    the origin — matching the style of LAMMPS dumps used elsewhere in the
    project.  Only real atoms (mask > 0) are written; padding beads are
    excluded.

    Args:
        traj_npz:    Path to trajectory NPZ produced by run_md.py.
        output_dump: Output .dump path.
        padding:     Extra Å added to each box half-width (default 20 Å).
    """
    traj = np.load(traj_npz, allow_pickle=False)

    R     = traj["R"]     # (n_frames, N, 3)
    F     = traj["F"]     # (n_frames, N, 3)
    steps = traj["step"]  # (n_frames,)

    species = traj["species"].astype(np.int32) if "species" in traj else np.zeros(R.shape[1], dtype=np.int32)
    valid   = traj["mask"] > 0 if "mask" in traj else np.ones(R.shape[1], dtype=bool)

    n_frames = R.shape[0]
    n_valid  = int(valid.sum())

    atom_ids   = np.where(valid)[0] + 1   # 1-indexed LAMMPS ids
    atom_types = species[valid] + 1        # 1-indexed LAMMPS types

    # Centre on the time-averaged centroid so the protein sits near the origin.
    centroid   = R[:, valid, :].mean(axis=(0, 1))
    R_centered = R - centroid[None, None, :]

    half_span = np.abs(R_centered[:, valid, :]).max(axis=(0, 1)) + padding
    bounds    = [(-h, h) for h in half_span]

    output_path = Path(output_dump)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        for fi in range(n_frames):
            R_f = R_centered[fi, valid, :]
            F_f = F[fi, valid, :]

            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{int(steps[fi])}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{n_valid}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            for lo, hi in bounds:
                fh.write(f"{lo:.10e} {hi:.10e}\n")
            fh.write("ITEM: ATOMS id type xu yu zu fx fy fz\n")
            for j in range(n_valid):
                fh.write(
                    f"{atom_ids[j]} {atom_types[j]} "
                    f"{R_f[j,0]:.8f} {R_f[j,1]:.8f} {R_f[j,2]:.8f} "
                    f"{F_f[j,0]:.8f} {F_f[j,1]:.8f} {F_f[j,2]:.8f}\n"
                )

    print(
        f"Wrote {n_frames} frames × {n_valid} atoms → {output_path}\n"
        f"  box: {['[{:.2f}, {:.2f}]'.format(lo, hi) for lo, hi in bounds]}\n"
        f"  atom types: {sorted(set(atom_types.tolist()))}"
    )
