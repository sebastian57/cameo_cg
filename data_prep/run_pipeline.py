#!/usr/bin/env python3
"""
Full mdCATH pipeline: H5 directory  →  prior fit.

Steps executed:
  1. h5_dataset_npz_transform  – extract frames from each H5  →  per-protein NPZ
  2. cg_1bead                  – coarse-grain each NPZ to CA beads
  3. pad_and_combine_datasets  – merge all CG NPZs into one padded dataset
  4. prior_fitting_script      – fit bond/angle/dihedral priors  →  YAML + plots

Output layout (inside --out_dir):
  01_raw_npz/            per-protein extracted NPZs          (step 1)
  02_cg_npz/             per-protein CG NPZs                 (step 2)
  combined_dataset.npz   merged + padded dataset             (step 3)
  fitted_priors.yaml     prior-fit results                   (step 4)
  plots/                 prior-fit diagnostic figures        (step 4)

Usage:
    python run_pipeline.py --h5_dir /path/to/h5s --out_dir /path/to/output --nframes 100
"""

import os                                          # env vars must come before JAX

os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("Pipeline")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Locate sibling scripts  →  make them importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from h5_dataset_npz_transform import build_dataset       # noqa: E402
from cg_1bead import build_cg_dataset                    # noqa: E402
from pad_and_combine_datasets import combine_and_pad_npz, pad_individual_npz # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_protein_id(h5_path: Path) -> str:
    """mdcath_dataset_12asA00.h5  →  12asA00"""
    return h5_path.stem.replace("mdcath_dataset_", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full mdCATH pipeline: H5 directory → prior fit."
    )

    # --- required ---------------------------------------------------------
    parser.add_argument("--h5_dir", required=True,
                        help="Directory containing mdcath_dataset_*.h5 files.")
    parser.add_argument("--out_dir", required=True,
                        help="Root output directory (intermediates + final).")
    parser.add_argument("--nframes", type=int, required=True,
                        help="Frames per run to extract from each H5.")

    # --- step 1 options ---------------------------------------------------
    parser.add_argument("--convert_to_ev", action="store_true", default=False,
                        help="Convert forces kcal/mol → eV during extraction.")
    parser.add_argument("--temp", default=None,
                        help="Single temperature group to extract (e.g. 320). "
                             "Omit to use all available temperatures.")

    # --- step 2 options ---------------------------------------------------
    parser.add_argument("--no_aggforce", action="store_true", default=False,
                        help="Disable aggforce; use plain CA-sliced forces.")
    parser.add_argument("--normalize_forces", action="store_true", default=False,
                        help="Per-type force normalization after CG mapping.")
    parser.add_argument("--use_4way_grouping", action="store_true", default=False,
                        help="4-way charge-based species grouping instead of per-AA.")

    # --- step 4 pass-throughs ----------------------------------------------
    parser.add_argument("--T", type=float, default=320.0,
                        help="Temperature for prior fitting (K).")
    parser.add_argument("--angle_terms", type=int, default=10,
                        help="Fourier terms for angle PMF fit.")
    parser.add_argument("--skip_plots", action="store_true", default=False,
                        help="Skip plot generation in prior fitting.")

    # --- step 3 options ----------------------------------------------------
    parser.add_argument("--no_combine", action="store_true", default=False,
                        help="Pad each CG dataset individually to global N_max "
                             "and keep as separate files instead of merging into "
                             "one combined_dataset.npz.")

    # --- misc --------------------------------------------------------------
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # ---------------------------------------------------------------------------
    # Directories
    # ---------------------------------------------------------------------------
    out_dir   = Path(args.out_dir)
    npz_dir   = out_dir / "01_raw_npz"
    cg_dir    = out_dir / "02_cg_npz"
    plots_dir = out_dir / "plots"
    for d in (npz_dir, cg_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Collect H5 files
    # ---------------------------------------------------------------------------
    h5_files = sorted(Path(args.h5_dir).glob("*.h5"))
    if not h5_files:
        logger.error(f"No .h5 files found in {args.h5_dir}")
        sys.exit(1)
    logger.info(f"Found {len(h5_files)} H5 files")

    # ---------------------------------------------------------------------------
    # Steps 1 + 2  –  per-protein loop
    # ---------------------------------------------------------------------------
    temps = "all" if args.temp is None else [args.temp]
    cg_paths: list[str] = []
    n_failed = 0

    for i, h5_path in enumerate(h5_files):
        protein_id = extract_protein_id(h5_path)
        raw_npz    = npz_dir / f"{protein_id}.npz"
        cg_npz     = cg_dir  / f"{protein_id}_cg.npz"

        logger.info(f"[{i + 1}/{len(h5_files)}] {h5_path.name}  (protein={protein_id})")

        # -- Step 1: H5 → raw NPZ ----------------------------------------
        if raw_npz.exists():
            logger.debug(f"  step 1 skip – {raw_npz.name} exists")
        else:
            try:
                build_dataset(
                    h5_path=str(h5_path),
                    protein_id=protein_id,
                    out_file=str(raw_npz),
                    nframes=args.nframes,
                    temp_groups=temps,
                    convert_to_ev=args.convert_to_ev,
                )
            except Exception as exc:
                logger.warning(f"  step 1 FAILED – {exc}")
                n_failed += 1
                continue

        # -- Step 2: raw NPZ → CG NPZ -------------------------------------
        if cg_npz.exists():
            logger.debug(f"  step 2 skip – {cg_npz.name} exists")
        else:
            try:
                build_cg_dataset(
                    npz_in=str(raw_npz),
                    npz_out=str(cg_npz),
                    use_aggforce=not args.no_aggforce,
                    normalize_forces=args.normalize_forces,
                    use_4way_grouping=args.use_4way_grouping,
                )
            except Exception as exc:
                logger.warning(f"  step 2 FAILED – {exc}")
                n_failed += 1
                continue

        cg_paths.append(str(cg_npz))

    logger.info(f"Per-protein steps done: {len(cg_paths)} succeeded, {n_failed} failed")
    if not cg_paths:
        logger.error("No CG datasets produced – nothing to combine.")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Step 3: pad (+ combine) all CG NPZs
    # ---------------------------------------------------------------------------
    if args.no_combine:
        padded_dir = out_dir / "03_padded_npz"
        padded_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Step 3: padding {len(cg_paths)} CG datasets individually …")
        data_paths = pad_individual_npz(cg_paths, str(padded_dir))
    else:
        combined_path = out_dir / "combined_dataset.npz"
        logger.info(f"Step 3: combining {len(cg_paths)} CG datasets …")
        combine_and_pad_npz(cg_paths, str(combined_path))
        data_paths = [str(combined_path)]

    # ---------------------------------------------------------------------------
    # Step 4: prior fitting  (subprocess – isolates JAX monkey-patching)
    # ---------------------------------------------------------------------------
    logger.info("Step 4: prior fitting …")
    yaml_path = out_dir / "fitted_priors.yaml"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prior_fitting_script.py"),
        "--out_yaml",    str(yaml_path),
        "--plots_dir",   str(plots_dir),
        "--T",           str(args.T),
        "--angle_terms", str(args.angle_terms),
    ]
    for dp in data_paths:
        cmd.extend(["--data", dp])
    if args.skip_plots:
        cmd.append("--skip_plots")
    if args.verbose:
        cmd.append("--verbose")

    subprocess.run(cmd, check=True)

    # ---------------------------------------------------------------------------
    logger.info("=== Pipeline complete ===")
    if args.no_combine:
        logger.info(f"  Padded datasets  : {padded_dir}  ({len(data_paths)} files)")
    else:
        logger.info(f"  Combined dataset : {combined_path}")
    logger.info(f"  Fitted priors    : {yaml_path}")
    logger.info(f"  Plots            : {plots_dir}")


if __name__ == "__main__":
    main()
