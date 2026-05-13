"""MD simulation entry point for CAMEO CG force-field models.

Usage:
    python scripts/run_md.py <md_config.yaml> [job_id]

The MD config YAML contains only an `md:` section that points to:
  - training_config_path: the _config.yaml saved alongside the trained model
  - params_path:          the _params.pkl file from training
  - dataset_path:         NPZ dataset to draw initial coordinates from

All model architecture is loaded from the training config — no duplication.
"""

import csv
import os
import sys
import pickle
import logging
from pathlib import Path

# ── GPU / platform detection ────────────────────────────────────────────────
# jax_md/rigid_body.py runs jnp.linalg.eigh at *module import time*, which
# triggers cuSolver and crashes on login nodes without a GPU allocation.
# On this cluster GPU is only available inside a SLURM job (SLURM_JOB_ID set).
# Force CPU when running interactively — must happen before any JAX import.
# Override manually: JAX_PLATFORMS=cuda python scripts/run_md.py ...
if "JAX_PLATFORMS" not in os.environ:
    if not os.environ.get("SLURM_JOB_ID"):
        os.environ["JAX_PLATFORMS"] = "cpu"
# ────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must happen before any jax_md imports.
from utils.jax_setup import apply_jax_compat_shims
apply_jax_compat_shims()

import yaml
import jax
import jax.numpy as jnp
import numpy as np

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from md.runner import MDRunner
from md.units import to_akma
from md.dump import write_lammps_dump
from utils.logging import md_logger


def _project_root(config_file: str) -> Path:
    """Resolve the project root: CAMEO_CG_PROJECT_ROOT env var, else two levels up from config."""
    env = os.environ.get("CAMEO_CG_PROJECT_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(config_file).resolve().parent.parent


def _resolve(path_str: str, root: Path) -> Path:
    """Resolve a path: absolute paths pass through; relative paths anchor to root."""
    p = Path(path_str)
    return p if p.is_absolute() else (root / p)


def main(config_file: str, job_id: str = None) -> None:
    """Run an MD simulation from a YAML config file.

    Args:
        config_file: Path to MD YAML (must contain an `md:` section).
        job_id:      Optional identifier appended to output filenames.
    """
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

    root = _project_root(config_file)
    md_logger.info(f"Project root: {root}  (CAMEO_CG_PROJECT_ROOT={os.environ.get('CAMEO_CG_PROJECT_ROOT', 'not set')})")

    with open(config_file) as fh:
        md_cfg_raw = yaml.safe_load(fh)
    md_cfg = md_cfg_raw.get("md")
    if md_cfg is None:
        raise ValueError(f"Config file {config_file!r} has no 'md:' section.")

    # ------------------------------------------------------------------
    # 1. Load model architecture from the companion training config.
    # ------------------------------------------------------------------
    training_config_path = _resolve(md_cfg["training_config_path"], root)
    md_logger.info(f"Training config: {training_config_path}")
    training_config = ConfigManager(str(training_config_path))

    # ------------------------------------------------------------------
    # 2. Load initial conditions from dataset.
    # ------------------------------------------------------------------
    dataset_path = _resolve(md_cfg["dataset_path"], root)

    md_logger.info(f"Dataset: {dataset_path}")
    loader    = DatasetLoader(str(dataset_path))
    frame_idx = int(md_cfg.get("frame_idx", 0))

    R0      = jnp.asarray(loader.R[frame_idx], dtype=jnp.float32)
    mask    = jnp.asarray(loader.mask[frame_idx], dtype=jnp.float32)
    species = jnp.asarray(loader.species[frame_idx], dtype=jnp.int32)

    md_logger.info(
        f"Initial frame: idx={frame_idx} N_atoms={loader.N_max} "
        f"n_valid={int(mask.sum())} n_species={int(loader.species.max())+1}"
    )

    # Build box from coordinate extents (free-space mode).
    cutoff = training_config.get_cutoff()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=training_config.get_buffer_multiplier(),
        park_multiplier=training_config.get_park_multiplier(),
    )
    box, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    # Center initial frame in this box.
    R0 = jnp.asarray(preprocessor.center_and_park(
        np.asarray(R0)[None], np.asarray(mask)[None], box, R_shift
    )[0], dtype=jnp.float32)
    md_logger.info(f"Box: {jax.device_get(box)}")

    # ------------------------------------------------------------------
    # 3. Build CombinedModel (same init path as train.py).
    # ------------------------------------------------------------------
    n_species = int(np.max(loader.species)) + 1
    model = CombinedModel(
        config=training_config,
        R0=R0,
        box=box,
        species=species,
        N_max=loader.N_max,
        n_species_override=n_species,
    )
    md_logger.info(f"Model: {model}")

    # ------------------------------------------------------------------
    # 4. Load trained params.
    # ------------------------------------------------------------------
    params_path = _resolve(md_cfg["params_path"], root)
    md_logger.info(f"Params: {params_path}")
    with open(params_path, "rb") as fh:
        params = pickle.load(fh)
    # CombinedModel wraps params under {"ml": ..., "prior": ...}; unwrap if needed.
    if not isinstance(params, dict) or "ml" not in params:
        params = {"ml": params}

    # ------------------------------------------------------------------
    # 5. Convert physical units → AKMA, then run MD.
    # ------------------------------------------------------------------
    md_cfg = to_akma(md_cfg)
    output_dir = _resolve(md_cfg.get("output_dir", "local_work/md_runs"), root)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = MDRunner(model, params, md_cfg)
    rng    = jax.random.PRNGKey(int(md_cfg.get("seed", 0)))
    traj   = runner.run(R0, mask, species, rng)

    # ------------------------------------------------------------------
    # 6. Save trajectory.
    # ------------------------------------------------------------------
    filename = md_cfg.get("output_filename", f"traj_{job_id}.npz")
    output_path = output_dir / filename
    np.savez(str(output_path), **{k: np.asarray(v) for k, v in traj.items()})
    md_logger.info(f"Trajectory saved: {output_path}")
    md_logger.info(
        f"Frames: {traj['R'].shape[0]}  "
        f"T_mean={float(np.mean(traj['T'])):.1f} K  "
        f"PE_mean={float(np.mean(traj['PE'])):.3f} kcal/mol"
    )

    # ------------------------------------------------------------------
    # 7. Write observables CSV.
    # ------------------------------------------------------------------
    obs_keys = [k for k in md_cfg.get("observables", []) if f"obs_{k}" in traj]
    if obs_keys:
        obs_filename = md_cfg.get(
            "observables_filename",
            Path(filename).stem + "_observables.csv",
        )
        obs_path_raw = md_cfg.get("observables_path", None)
        if obs_path_raw:
            obs_path = Path(obs_path_raw)
            if not obs_path.is_absolute():
                obs_path = output_dir / obs_path
        else:
            obs_path = output_dir / obs_filename

        obs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(obs_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(obs_keys)
            n_rows = len(traj[f"obs_{obs_keys[0]}"])
            for i in range(n_rows):
                writer.writerow([float(traj[f"obs_{k}"][i]) for k in obs_keys])
        md_logger.info(f"Observables CSV: {obs_path}  ({n_rows} rows, cols: {obs_keys})")

    # ------------------------------------------------------------------
    # 8. Optionally convert to LAMMPS dump for OVITO.
    # ------------------------------------------------------------------
    if md_cfg.get("dump_for_ovito", False):
        dump_path = output_path.with_suffix(".dump")
        padding   = float(md_cfg.get("dump_padding", 20.0))
        write_lammps_dump(str(output_path), str(dump_path), padding=padding)
        md_logger.info(f"OVITO dump: {dump_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        logging.error("Usage: python scripts/run_md.py <md_config.yaml> [job_id]")
        sys.exit(1)
    config_file = sys.argv[1]
    job_id = sys.argv[2] if len(sys.argv) > 2 else None
    main(config_file, job_id)
