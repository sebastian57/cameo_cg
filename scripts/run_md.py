"""MD simulation entry point for CAMEO CG force-field models.

Usage (single run):
    python scripts/run_md.py <md_config.yaml> [job_id]

Usage (one specific replica, e.g. from a SLURM array):
    python scripts/run_md.py <md_config.yaml> [job_id] --replica 2

The MD config YAML contains only an `md:` section that points to:
  - training_config_path: the _config.yaml saved alongside the trained model
  - params_path:          the _params.pkl file from training
  - dataset_path:         NPZ dataset to draw initial coordinates from

All model architecture is loaded from the training config — no duplication.
"""

import argparse
import csv
import os
import sys
import pickle
import logging
import signal
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
from training.path_utils import repo_root_from_file, resolve_from_config_or_repo
from training.support_gate import build_support_gate_bank, support_gate_config, support_gate_enabled
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


def _replica_output_path(base_path: Path, replica_idx: int, n_replicas: int) -> Path:
    """Append _repXX to the stem when running more than one replica."""
    if n_replicas <= 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_rep{replica_idx:02d}{base_path.suffix}")


def _build_support_gate_bank_for_md(training_config: ConfigManager):
    """Rebuild the training-data support gate bank for direct Python/JAX MD."""
    if not support_gate_enabled(training_config):
        return None

    cfg = support_gate_config(training_config)
    descriptor = str(cfg.get("descriptor", "pairwise_distances")).strip().lower()
    if descriptor != "pairwise_distances":
        raise ValueError(
            "training.support_gate.descriptor currently supports only 'pairwise_distances'."
        )

    data_path = resolve_from_config_or_repo(
        training_config.get_data_path(),
        training_config.config_path,
        repo_root_from_file(__file__),
    )
    gate_loader = DatasetLoader(
        str(data_path),
        max_frames=training_config.get_max_frames(),
        seed=training_config.get_seed(),
    )
    bank = build_support_gate_bank(
        R=np.asarray(gate_loader.R, dtype=np.float32),
        mask=np.asarray(gate_loader.mask, dtype=np.float32),
        max_centers=int(cfg.get("max_centers", 512)),
        sigma_multiplier=float(cfg.get("sigma_multiplier", 1.0)),
        seed=int(cfg.get("seed", training_config.get_seed())),
        floor=float(cfg.get("floor", 0.0)),
        stop_gradient=bool(cfg.get("stop_gradient", False)),
    )
    md_logger.info(
        "[SupportGate] Runtime bank rebuilt from %s: centers=%d sigma=%.6g floor=%.3g stop_gradient=%s",
        data_path,
        int(bank.centers.shape[0]),
        float(bank.sigma),
        float(bank.floor),
        bool(bank.stop_gradient),
    )
    return bank


def _save_outputs(
    traj: dict,
    output_path: Path,
    md_cfg: dict,
    output_dir: Path,
) -> None:
    """Save trajectory NPZ, observables CSV, and optional OVITO dump."""
    np.savez(str(output_path), **{k: np.asarray(v) for k, v in traj.items()})
    md_logger.info(f"Trajectory saved: {output_path}")
    md_logger.info(
        f"Frames: {traj['R'].shape[0]}  "
        f"T_mean={float(np.mean(traj['T'])):.1f} K  "
        f"PE_mean={float(np.mean(traj['PE'])):.3f} kcal/mol"
    )

    # Observables CSV
    obs_keys = [k for k in md_cfg.get("observables", []) if f"obs_{k}" in traj]
    if obs_keys:
        obs_filename = md_cfg.get(
            "observables_filename",
            Path(output_path.name).stem + "_observables.csv",
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

    if "decomp_step" in traj:
        decomp_path = output_path.with_name(output_path.stem + "_decomp.csv")
        mask = np.asarray(traj.get("mask", np.ones(traj["R"].shape[1], dtype=np.float32)), dtype=bool)
        f_ml = np.asarray(traj.get("decomp_F_ml", np.zeros((0, traj["R"].shape[1], 3))), dtype=np.float32)
        f_prior = np.asarray(traj.get("decomp_F_prior", np.zeros_like(f_ml)), dtype=np.float32)
        with open(decomp_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "step",
                "E_ml",
                "E_prior",
                "F_ml_rms",
                "F_prior_rms",
                "gate_combined_alpha",
                "gate_torsion_alpha",
                "gate_distance_matrix_alpha",
                "gate_angular_alpha",
            ])
            n_rows = int(np.asarray(traj["decomp_step"]).shape[0])
            for i in range(n_rows):
                fml_i = f_ml[i][mask] if f_ml.shape[0] else np.zeros((0, 3), dtype=np.float32)
                fprior_i = f_prior[i][mask] if f_prior.shape[0] else np.zeros((0, 3), dtype=np.float32)
                writer.writerow([
                    int(np.asarray(traj["decomp_step"])[i]),
                    float(np.asarray(traj.get("decomp_E_ml", np.zeros(n_rows)))[i]),
                    float(np.asarray(traj.get("decomp_E_prior", np.zeros(n_rows)))[i]),
                    float(np.sqrt(np.mean(fml_i ** 2))) if fml_i.size else 0.0,
                    float(np.sqrt(np.mean(fprior_i ** 2))) if fprior_i.size else 0.0,
                    float(np.asarray(traj.get("decomp_gate_combined_alpha", np.ones(n_rows)))[i]),
                    float(np.asarray(traj.get("decomp_gate_torsion_alpha", np.ones(n_rows)))[i]),
                    float(np.asarray(traj.get("decomp_gate_distance_matrix_alpha", np.ones(n_rows)))[i]),
                    float(np.asarray(traj.get("decomp_gate_angular_alpha", np.ones(n_rows)))[i]),
                ])
        md_logger.info(f"Decomposition CSV: {decomp_path}  ({n_rows} rows)")

    # OVITO dump
    if md_cfg.get("dump_for_ovito", False):
        dump_path = output_path.with_suffix(".dump")
        padding = float(md_cfg.get("dump_padding", 20.0))
        write_lammps_dump(str(output_path), str(dump_path), padding=padding)
        md_logger.info(f"OVITO dump: {dump_path}")


def main(config_file: str, job_id: str = None, replica_idx: int = None) -> None:
    """Run MD simulation(s) from a YAML config file.

    Args:
        config_file: Path to MD YAML (must contain an `md:` section).
        job_id:      Optional identifier used in default output filenames.
        replica_idx: If set, run only this replica index (0-based).
                     If None, run all replicas 0..n_replicas-1 sequentially.
                     Use with SLURM array jobs (--replica $SLURM_ARRAY_TASK_ID).
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

    if md_cfg.get("override_use_priors", False):
        training_config.set("model", "use_priors", True)
        training_config.set("model", "train_priors", False)
        md_logger.info(
            "[Config] override_use_priors=true: prior energy added to ML energy. "
            "Use for prior-residual models (F_total = F_ML + F_prior)."
        )

    if md_cfg.get("prior_only", False):
        training_config.set("model", "use_priors", True)
        training_config.set("model", "train_priors", False)
        training_config.set("model", "prior_only", True)
        md_logger.info(
            "[Config] prior_only=true: MD evaluates only the configured fixed priors."
        )

    if "ml_energy_scale" in md_cfg:
        training_config.set("model", "ml_energy_scale", float(md_cfg["ml_energy_scale"]))
        md_logger.info(
            "[Config] md.ml_energy_scale=%.4g: scaling ML energy/forces during MD.",
            float(md_cfg["ml_energy_scale"]),
        )
    if "prior_energy_scale" in md_cfg:
        training_config.set("model", "prior_energy_scale", float(md_cfg["prior_energy_scale"]))
        md_logger.info(
            "[Config] md.prior_energy_scale=%.4g: scaling prior energy/forces during MD.",
            float(md_cfg["prior_energy_scale"]),
        )

    if "aa_integrated_baseline_component_scales" in md_cfg:
        priors_cfg = dict(training_config.get("model", "priors", default={}) or {})
        baseline_cfg = dict(priors_cfg.get("aa_integrated_baseline", {}) or {})
        component_scales = dict(md_cfg["aa_integrated_baseline_component_scales"] or {})
        baseline_cfg["component_scales"] = component_scales
        priors_cfg["aa_integrated_baseline"] = baseline_cfg
        training_config.set("model", "priors", priors_cfg)
        md_logger.info(
            "[Config] aa_integrated_baseline component scales=%s", component_scales
        )

    if "prior_overrides" in md_cfg:
        priors_cfg = dict(training_config.get("model", "priors", default={}) or {})
        overrides = dict(md_cfg["prior_overrides"] or {})
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(priors_cfg.get(key), dict):
                merged = dict(priors_cfg[key])
                merged.update(value)
                priors_cfg[key] = merged
            else:
                priors_cfg[key] = value
        training_config.set("model", "priors", priors_cfg)
        md_logger.info("[Config] applied md.prior_overrides keys=%s", sorted(overrides))

    if "robustness_gate" in md_cfg:
        gate_cfg = dict(md_cfg.get("robustness_gate") or {})
        training_config.set("model", "robustness_gate", gate_cfg)
        md_logger.info("[Config] md.robustness_gate=%s", gate_cfg)

    if "local_extrapolation_gate" in md_cfg:
        local_gate_cfg = dict(md_cfg.get("local_extrapolation_gate") or {})
        if local_gate_cfg.get("artifact_path"):
            local_gate_cfg["artifact_path"] = str(_resolve(local_gate_cfg["artifact_path"], root))
        training_config.set("model", "local_extrapolation_gate", local_gate_cfg)
        md_logger.info("[Config] md.local_extrapolation_gate=%s", local_gate_cfg)

    if "edge_distance_gate" in md_cfg:
        edge_gate_cfg = dict(md_cfg.get("edge_distance_gate") or {})
        if edge_gate_cfg.get("artifact_path"):
            edge_gate_cfg["artifact_path"] = str(_resolve(edge_gate_cfg["artifact_path"], root))
        training_config.set("model", "edge_distance_gate", edge_gate_cfg)
        md_logger.info("[Config] md.edge_distance_gate=%s", edge_gate_cfg)

    if md_cfg.get("cell_list", False):
        training_config.set("model", "neighbor_disable_cell_list", False)
        md_logger.info(
            "[Config] cell_list=true: cell list enabled for neighbor search "
            "(overrides training config neighbor_disable_cell_list=True)."
        )

    # ------------------------------------------------------------------
    # 2. Load initial conditions from dataset.
    # ------------------------------------------------------------------
    dataset_path = _resolve(md_cfg["dataset_path"], root)

    md_logger.info(f"Dataset: {dataset_path}")
    loader    = DatasetLoader(str(dataset_path))
    frame_indices_raw = md_cfg.get("frame_indices", None)
    if frame_indices_raw is None:
        frame_indices = [int(md_cfg.get("frame_idx", 0))]
    else:
        frame_indices = [int(idx) for idx in frame_indices_raw]
        if not frame_indices:
            raise ValueError("md.frame_indices was provided but is empty.")
    for idx in frame_indices:
        if idx < 0 or idx >= loader.R.shape[0]:
            raise IndexError(
                f"Initial frame index {idx} is out of range for dataset with "
                f"{loader.R.shape[0]} frames."
            )
    frame_idx = frame_indices[0]

    R0      = jnp.asarray(loader.R[frame_idx], dtype=jnp.float32)
    mask    = jnp.asarray(loader.mask[frame_idx], dtype=jnp.float32)
    species = jnp.asarray(loader.species[frame_idx], dtype=jnp.int32)

    md_logger.info(
        f"Initial frame: idx={frame_idx} N_atoms={loader.N_max} "
        f"n_valid={int(mask.sum())} n_species={int(loader.species.max())+1}"
    )
    if len(frame_indices) > 1:
        md_logger.info(f"Replica initial frames: {frame_indices}")

    # Build box from coordinate extents (free-space mode).
    cutoff = training_config.get_cutoff()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=training_config.get_buffer_multiplier(),
        park_multiplier=training_config.get_park_multiplier(),
    )
    box, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    R0_frames = []
    mask_frames = []
    species_frames = []
    for idx in frame_indices:
        mask_i = np.asarray(loader.mask[idx], dtype=np.float32)
        R_i = preprocessor.center_and_park(
            np.asarray(loader.R[idx], dtype=np.float32)[None],
            mask_i[None],
            box,
            R_shift,
        )[0]
        R0_frames.append(jnp.asarray(R_i, dtype=jnp.float32))
        mask_frames.append(jnp.asarray(mask_i, dtype=jnp.float32))
        species_frames.append(jnp.asarray(loader.species[idx], dtype=jnp.int32))
    R0 = R0_frames[0]
    mask = mask_frames[0]
    species = species_frames[0]
    md_logger.info(f"Box: {jax.device_get(box)}")

    # ------------------------------------------------------------------
    # 3. Build CombinedModel (same init path as train.py).
    # ------------------------------------------------------------------
    data_n_species = int(np.max(loader.species)) + 1
    config_n_species = training_config.get("model", "allegro", "num_types", default=None)
    n_species = max(data_n_species, int(config_n_species or 0))
    if n_species != data_n_species:
        md_logger.info(
            f"Using n_species={n_species} from training config "
            f"(dataset contains species ids up to {data_n_species - 1})."
        )
    support_gate_bank = _build_support_gate_bank_for_md(training_config)
    model = CombinedModel(
        config=training_config,
        R0=R0,
        box=box,
        species=species,
        N_max=loader.N_max,
        prior_only=training_config.prior_only_enabled(),
        n_species_override=n_species,
        support_gate_bank=support_gate_bank,
    )
    md_logger.info(f"Model: {model}")

    # ------------------------------------------------------------------
    # 4. Load trained params.
    # ------------------------------------------------------------------
    params_path = _resolve(md_cfg["params_path"], root)
    md_logger.info(f"Params: {params_path}")
    with open(params_path, "rb") as fh:
        params = pickle.load(fh)
    if isinstance(params, dict):
        if isinstance(params.get("params"), dict):
            md_logger.info("Detected trainer checkpoint format; extracting params.")
            params = params["params"]
        elif isinstance(params.get("best_params"), dict):
            md_logger.info("Detected trainer checkpoint format; extracting best_params (no params key).")
            params = params["best_params"]
    if not isinstance(params, dict) or "ml" not in params:
        params = {"ml": params}

    # ------------------------------------------------------------------
    # 5. Convert physical units → AKMA.
    # ------------------------------------------------------------------
    md_cfg = to_akma(md_cfg)

    output_dir = _resolve(md_cfg.get("output_dir", "local_work/md_runs"), root)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = md_cfg.get("output_filename", f"traj_{job_id}.npz")
    base_output_path = output_dir / filename

    # ------------------------------------------------------------------
    # 6. Determine replicas to run.
    # ------------------------------------------------------------------
    n_replicas = int(md_cfg.get("n_replicas", 1))
    base_seed  = int(md_cfg.get("seed", 0))
    if len(frame_indices) not in (1, n_replicas):
        raise ValueError(
            "md.frame_indices must contain either one frame index or exactly "
            f"n_replicas entries. Got {len(frame_indices)} frame index/indices "
            f"for n_replicas={n_replicas}."
        )

    if replica_idx is not None:
        if replica_idx < 0 or replica_idx >= n_replicas:
            raise ValueError(
                f"--replica {replica_idx} is out of range for n_replicas={n_replicas}."
            )
        replicas_to_run = [replica_idx]
    else:
        replicas_to_run = list(range(n_replicas))

    md_logger.info(
        f"Replicas: running {len(replicas_to_run)} of {n_replicas}  "
        f"(base_seed={base_seed}, indices={replicas_to_run})"
    )

    # ------------------------------------------------------------------
    # 7. Build MDRunner once — JIT is compiled on the first replica and
    #    reused for all subsequent ones (same model, same shapes).
    # ------------------------------------------------------------------
    # partial_output_path is updated per-replica before each run() call.
    md_cfg_runner = dict(md_cfg)
    first_output_path = _replica_output_path(base_output_path, replicas_to_run[0], n_replicas)
    first_partial_path = first_output_path.with_name(
        first_output_path.stem + ".partial" + first_output_path.suffix
    )
    md_cfg_runner["_partial_output_path"] = (
        str(first_partial_path) if md_cfg.get("continuous_output") else None
    )
    runner = MDRunner(model, params, md_cfg_runner)

    # ------------------------------------------------------------------
    # 8. Run replicas sequentially.
    # ------------------------------------------------------------------
    for i in replicas_to_run:
        replica_seed = base_seed + i
        start_slot = i if len(frame_indices) > 1 else 0
        start_frame_idx = frame_indices[start_slot]
        R0_i = R0_frames[start_slot]
        mask_i = mask_frames[start_slot]
        species_i = species_frames[start_slot]
        output_path  = _replica_output_path(base_output_path, i, n_replicas)
        partial_path = output_path.with_name(
            output_path.stem + ".partial" + output_path.suffix
        )

        runner.partial_output_path = partial_path if md_cfg.get("continuous_output") else None

        if n_replicas > 1:
            md_logger.info(
                f"\n{'='*60}\n"
                f"Replica {i} / {n_replicas - 1}  seed={replica_seed}  "
                f"frame_idx={start_frame_idx}  "
                f"→ {output_path.name}\n"
                f"{'='*60}"
            )

        rng  = jax.random.PRNGKey(replica_seed)
        traj = runner.run(R0_i, mask_i, species_i, rng)
        traj["initial_frame_idx"] = np.asarray(start_frame_idx, dtype=np.int32)
        if hasattr(loader, "raw_data") and isinstance(getattr(loader, "raw_data"), dict):
            source_indices = loader.raw_data.get("source_indices")
            if source_indices is not None:
                traj["initial_source_index"] = np.asarray(
                    int(np.asarray(source_indices)[start_frame_idx]), dtype=np.int32
                )

        _save_outputs(traj, output_path, md_cfg, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _handle_termination(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)

    parser = argparse.ArgumentParser(
        description="Run CAMEO CG MD simulation(s) from a YAML config."
    )
    parser.add_argument("config_file", help="Path to MD config YAML.")
    parser.add_argument("job_id", nargs="?", default=None,
                        help="Optional job identifier (default: SLURM_JOB_ID or 'local').")
    parser.add_argument("--replica", type=int, default=None, metavar="IDX",
                        help="Run only replica IDX (0-based). "
                             "Omit to run all replicas sequentially. "
                             "Intended for SLURM array jobs.")
    args = parser.parse_args()

    try:
        main(args.config_file, args.job_id, replica_idx=args.replica)
    except KeyboardInterrupt as exc:
        logging.info(f"MD interrupted cleanly ({exc}).")
        sys.exit(130)
