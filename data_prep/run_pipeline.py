#!/usr/bin/env python3
"""
Full mdCATH Data Preparation Pipeline

Converts raw mdCATH H5 files to coarse-grained datasets with fitted priors.
Integrates with the cameo_cg framework for consistent data processing.

Steps executed:
  1. h5_dataset_npz_transform  – Extract frames from each H5  →  per-protein NPZ
  2. cg_1bead                  – Coarse-grain each NPZ to CA beads
  3. pad_and_combine_datasets  – Merge all CG NPZs into one padded dataset
  4. prior_fitting_script      – Fit bond/angle/dihedral priors  →  YAML + plots

Output layout (inside --out_dir):
  01_raw_npz/            per-protein extracted NPZs          (step 1)
  02_cg_npz/             per-protein CG NPZs                 (step 2)
  combined_dataset.npz   merged + padded dataset             (step 3)
  fitted_priors.yaml     prior-fit results                   (step 4)
  plots/                 prior-fit diagnostic figures        (step 4)

Usage:
    python data_prep/run_pipeline.py --h5_dir /path/to/h5s --out_dir /path/to/output --nframes 100

Example:
    >>> # Process mdCATH dataset with 100 frames per protein, fit spline priors
    >>> python data_prep/run_pipeline.py \\
    ...     --h5_dir /data/mdcath/h5_files \\
    ...     --out_dir /data/processed \\
    ...     --nframes 100 \\
    ...     --spline \\
    ...     --residue_specific_angles

Consolidated from:
- Original run_pipeline.py in data_prep/
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# =============================================================================
# Environment Setup
# =============================================================================
# These environment variables suppress verbose warnings from JAX/TensorFlow
# that occur in subprocess calls to prior_fitting_script.py.
# Must be set before importing any JAX-dependent modules.
# =============================================================================

def setup_environment() -> None:
    """
    Configure environment variables for clean pipeline execution.

    Sets:
        - JAX_TRACEBACK_FILTERING: Disable for full error traces
        - PYTHONWARNINGS: Suppress FutureWarning and UserWarning
        - TF_CPP_MIN_LOG_LEVEL: Reduce TensorFlow verbosity

    Called at module import time to ensure settings apply to subprocess calls.
    """
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Initialize environment before any other imports
setup_environment()


# =============================================================================
# Path Setup
# =============================================================================
# Add parent directory (cameo_cg root) to path for framework imports
# This allows data_prep scripts to use framework modules (data/, utils/, etc.)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CAMEO_CG_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CAMEO_CG_ROOT))


# =============================================================================
# Imports (after path setup)
# =============================================================================

from utils.logging import pipeline_logger  # noqa: E402

# Local data_prep modules
from h5_dataset_npz_transform import build_dataset  # noqa: E402
from cg_1bead import build_cg_dataset  # noqa: E402
from pad_and_combine_datasets import (  # noqa: E402
    combine_and_pad_npz,
    pad_individual_npz,
    combine_and_pad_npz_bucketed,
)


# =============================================================================
# Helper Functions
# =============================================================================

def extract_protein_id(h5_path: Path) -> str:
    """
    Extract protein ID from mdCATH H5 filename.

    Args:
        h5_path: Path to H5 file with pattern mdcath_dataset_<id>.h5

    Returns:
        Protein ID string (e.g., "12asA00")

    Example:
        >>> extract_protein_id(Path("mdcath_dataset_12asA00.h5"))
        '12asA00'
    """
    return h5_path.stem.replace("mdcath_dataset_", "")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """
    Execute full mdCATH preprocessing pipeline.

    Raises:
        SystemExit: If no H5 files found or all processing steps fail
    """
    parser = argparse.ArgumentParser(
        description="Full mdCATH pipeline: H5 directory → prior fit."
    )

    # ===== Required Arguments =====
    parser.add_argument(
        "--h5_dir",
        required=True,
        help="Directory containing mdcath_dataset_*.h5 files."
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Root output directory (intermediates + final)."
    )
    parser.add_argument(
        "--nframes",
        type=int,
        required=True,
        help="Frames per run to extract from each H5."
    )

    # ===== Step 1 Options (H5 extraction) =====
    parser.add_argument(
        "--convert_to_ev",
        action="store_true",
        default=False,
        help="Convert forces kcal/mol → eV during extraction."
    )
    parser.add_argument(
        "--temp",
        default=None,
        help="Single temperature group to extract (e.g. 320). Omit to use all available temperatures."
    )

    # ===== Step 2 Options (Coarse-graining) =====
    parser.add_argument(
        "--no_aggforce",
        action="store_true",
        default=False,
        help="Disable aggforce; use plain CA-sliced forces."
    )
    parser.add_argument(
        "--normalize_forces",
        action="store_true",
        default=False,
        help="Per-type force normalization after CG mapping."
    )
    parser.add_argument(
        "--use_4way_grouping",
        action="store_true",
        default=False,
        help="4-way charge-based species grouping instead of per-AA."
    )

    # ===== Step 3 Options (Dataset combination) =====
    parser.add_argument(
        "--no_combine",
        action="store_true",
        default=False,
        help="Pad each CG dataset individually to global N_max and keep as separate files "
             "instead of merging into one combined_dataset.npz."
    )
    parser.add_argument(
        "--bucket_boundaries",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Explicit bucket boundaries for protein-aware batching "
             "(e.g. --bucket_boundaries 100 200 → 3 buckets: ≤100, 101-200, >200). "
             "Mutually exclusive with --no_combine."
    )
    parser.add_argument(
        "--n_buckets",
        type=int,
        default=None,
        metavar="N",
        help="Enable protein-aware bucketing with N auto-computed equal-count buckets "
             "(e.g. --n_buckets 3). Ignored if --bucket_boundaries is also set. "
             "Mutually exclusive with --no_combine."
    )

    # ===== Step 4 Options (Prior fitting) =====
    parser.add_argument(
        "--T",
        type=float,
        default=320.0,
        help="Temperature for prior fitting (K)."
    )
    parser.add_argument(
        "--angle_terms",
        type=int,
        default=10,
        help="Fourier terms for angle PMF fit."
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        default=False,
        help="Skip plot generation in prior fitting."
    )
    parser.add_argument(
        "--spline",
        action="store_true",
        default=False,
        help="Also fit spline priors (KDE -> BI -> CubicSpline) and write NPZ."
    )
    parser.add_argument(
        "--spline_out",
        default=None,
        help="Optional spline NPZ output path. Default: <out_dir>/fitted_priors_spline.npz"
    )
    parser.add_argument(
        "--residue_specific_angles",
        action="store_true",
        default=False,
        help="Enable residue-specific angle splines when --spline is set."
    )
    parser.add_argument(
        "--angle_min_samples",
        type=int,
        default=500,
        help="Minimum samples per residue type for dedicated angle spline."
    )
    parser.add_argument(
        "--kde_bandwidth_factor",
        type=float,
        default=1.0,
        help="Bandwidth factor multiplier for spline KDE fits."
    )
    parser.add_argument(
        "--spline_grid_points",
        type=int,
        default=500,
        help="Grid points used in spline fitting."
    )

    # ===== Misc Options =====
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging."
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        pipeline_logger.setLevel(logging.DEBUG)

    # =============================================================================
    # Setup Directories
    # =============================================================================
    out_dir: Path = Path(args.out_dir)
    npz_dir: Path = out_dir / "01_raw_npz"
    cg_dir: Path = out_dir / "02_cg_npz"
    plots_dir: Path = out_dir / "plots"

    for d in (npz_dir, cg_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Collect H5 Files
    # =============================================================================
    h5_files: List[Path] = sorted(Path(args.h5_dir).glob("*.h5"))
    if not h5_files:
        pipeline_logger.error(f"No .h5 files found in {args.h5_dir}")
        sys.exit(1)

    pipeline_logger.info(f"Found {len(h5_files)} H5 files")

    # =============================================================================
    # Steps 1 + 2: Per-Protein Processing Loop
    # =============================================================================
    temps: List[str] = "all" if args.temp is None else [args.temp]
    cg_paths: List[str] = []
    n_failed: int = 0

    for i, h5_path in enumerate(h5_files):
        protein_id: str = extract_protein_id(h5_path)
        raw_npz: Path = npz_dir / f"{protein_id}.npz"
        cg_npz: Path = cg_dir / f"{protein_id}_cg.npz"

        pipeline_logger.info(
            f"[{i + 1}/{len(h5_files)}] {h5_path.name}  (protein={protein_id})"
        )

        # ===== Step 1: H5 → raw NPZ =====
        if raw_npz.exists():
            pipeline_logger.debug(f"  step 1 skip – {raw_npz.name} exists")
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
                pipeline_logger.warning(f"  step 1 FAILED – {exc}")
                n_failed += 1
                continue

        # ===== Step 2: raw NPZ → CG NPZ =====
        if cg_npz.exists():
            pipeline_logger.debug(f"  step 2 skip – {cg_npz.name} exists")
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
                pipeline_logger.warning(f"  step 2 FAILED – {exc}")
                n_failed += 1
                continue

        cg_paths.append(str(cg_npz))

    pipeline_logger.info(
        f"Per-protein steps done: {len(cg_paths)} succeeded, {n_failed} failed"
    )
    if not cg_paths:
        pipeline_logger.error("No CG datasets produced – nothing to combine.")
        sys.exit(1)

    # =============================================================================
    # Step 3: Pad (+ Combine) CG Datasets
    # =============================================================================
    data_paths: List[str]

    want_buckets = (args.bucket_boundaries is not None) or (args.n_buckets is not None)

    if want_buckets and args.no_combine:
        pipeline_logger.error("--bucket_boundaries / --n_buckets and --no_combine are mutually exclusive.")
        sys.exit(1)

    if want_buckets:
        bucketed_dir: Path = out_dir / "03_bucketed_npz"
        pipeline_logger.info(
            f"Step 3: bucketed combine of {len(cg_paths)} CG datasets …"
        )
        bucket_out = combine_and_pad_npz_bucketed(
            cg_paths,
            str(bucketed_dir),
            bucket_boundaries=args.bucket_boundaries,
            n_buckets=args.n_buckets if args.n_buckets is not None else 3,
        )
        data_paths = list(bucket_out.values())
    elif args.no_combine:
        padded_dir: Path = out_dir / "03_padded_npz"
        padded_dir.mkdir(parents=True, exist_ok=True)
        pipeline_logger.info(
            f"Step 3: padding {len(cg_paths)} CG datasets individually …"
        )
        data_paths = pad_individual_npz(cg_paths, str(padded_dir))
    else:
        combined_path: Path = out_dir / "combined_dataset.npz"
        pipeline_logger.info(f"Step 3: combining {len(cg_paths)} CG datasets …")
        combine_and_pad_npz(cg_paths, str(combined_path))
        data_paths = [str(combined_path)]

    # =============================================================================
    # Step 4: Prior Fitting (subprocess to isolate JAX monkey-patching)
    # =============================================================================
    pipeline_logger.info("Step 4: prior fitting …")
    yaml_path: Path = out_dir / "fitted_priors.yaml"

    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_DIR / "prior_fitting_script.py"),
        "--out_yaml", str(yaml_path),
        "--plots_dir", str(plots_dir),
        "--T", str(args.T),
        "--angle_terms", str(args.angle_terms),
    ]

    # Add all dataset paths
    for dp in data_paths:
        cmd.extend(["--data", dp])

    # Add spline fitting options if requested
    if args.spline:
        spline_out: str = (
            args.spline_out
            if args.spline_out is not None
            else str(out_dir / "fitted_priors_spline.npz")
        )
        cmd.extend([
            "--spline",
            "--spline_out", spline_out,
            "--angle_min_samples", str(args.angle_min_samples),
            "--kde_bandwidth_factor", str(args.kde_bandwidth_factor),
            "--spline_grid_points", str(args.spline_grid_points),
        ])
        if args.residue_specific_angles:
            cmd.append("--residue_specific_angles")

    # Add optional flags
    if args.skip_plots:
        cmd.append("--skip_plots")
    if args.verbose:
        cmd.append("--verbose")

    # Run prior fitting in subprocess
    subprocess.run(cmd, check=True)

    # =============================================================================
    # Pipeline Complete
    # =============================================================================
    pipeline_logger.info("=" * 60)
    pipeline_logger.info("Pipeline complete")
    pipeline_logger.info("=" * 60)

    if want_buckets:
        pipeline_logger.info(f"  Bucketed datasets: {bucketed_dir}  ({len(data_paths)} buckets)")
    elif args.no_combine:
        pipeline_logger.info(f"  Padded datasets  : {padded_dir}  ({len(data_paths)} files)")
    else:
        pipeline_logger.info(f"  Combined dataset : {data_paths[0]}")

    pipeline_logger.info(f"  Fitted priors    : {yaml_path}")
    pipeline_logger.info(f"  Plots            : {plots_dir}")


if __name__ == "__main__":
    main()
