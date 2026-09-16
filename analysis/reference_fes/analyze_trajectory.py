#!/usr/bin/env python3
"""Compute phi/psi data and a Ramachandran FES for one CG trajectory or dataset."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np



def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", type=Path, required=True,
                    help="NPZ containing R with shape (frames,5,3)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--label", default="trajectory")
    ap.add_argument("--temperature-K", type=float, default=300.0)
    ap.add_argument("--bins", type=int, default=100)
    a = ap.parse_args(argv)
    from analysis.reference_fes.functions import compute_ramachandran_angles, save_dihedral_dat
    data = np.load(a.trajectory, allow_pickle=False)
    R = np.asarray(data["R"])
    phi, psi = compute_ramachandran_angles(R)
    a.outdir.mkdir(parents=True, exist_ok=True)
    dat = a.outdir / f"rama_{a.label}.dat"
    png = a.outdir / f"fes_{a.label}.png"
    save_dihedral_dat(dat, phi, psi)
    subprocess.run([
        sys.executable, "-m", "analysis.reference_fes.generate_fes",
        "--input", str(dat), "--output", str(png),
        "--temperature-K", str(a.temperature_K),
        "--bins-x", str(a.bins), "--bins-y", str(a.bins),
        "--xlabel", "Phi [deg]", "--ylabel", "Psi [deg]",
    ], check=True)
    print(f"loaded {len(R):,} frames; wrote {dat} and {png}")


if __name__ == "__main__":
    main()
