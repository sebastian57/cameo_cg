"""Convert a cameo_cg trajectory NPZ to LAMMPS dump format for OVITO.

Usage:
    python scripts/traj_to_dump.py <traj.npz>
    python scripts/traj_to_dump.py <traj.npz> output.dump
    python scripts/traj_to_dump.py <traj.npz> output.dump --padding 20.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from md.dump import write_lammps_dump  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traj_npz", help="Trajectory NPZ from run_md.py")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .dump path (default: <traj_npz>.dump)")
    parser.add_argument("--padding", type=float, default=20.0,
                        help="Box padding in Å (default: 20)")
    args = parser.parse_args()

    output = args.output or str(Path(args.traj_npz).with_suffix(".dump"))
    write_lammps_dump(args.traj_npz, output, padding=args.padding)


if __name__ == "__main__":
    main()
