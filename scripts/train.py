"""
Unified Training Script for Allegro Coarse-Grained Protein Force Fields

Uses JAX distributed + chemtrain's shard_map for true multi-node training.

ARCHITECTURE (1 process per NODE, not per GPU!):
    - 1 node (4 GPUs) = 1 process with 4 local GPUs
    - 2 nodes (8 GPUs) = 2 processes, each with 4 local GPUs = 8 total GPUs

    After jax.distributed.initialize():
    - jax.devices() returns ALL 8 GPUs across both nodes
    - chemtrain creates: Mesh(jax.devices(), axis_names=('batch'))
    - This mesh spans ALL 8 GPUs for unified training
    - lax.pmean(grad, 'batch') aggregates gradients across ALL 8 GPUs
    - Result: ONE training run using ALL 8 GPUs together!

    global_batch_size = batch_per_device * total_gpus_across_all_nodes

Memory model:
    - Data is loaded ONCE per node (not per GPU)
    - shard_map splits batches across ALL GPUs in the mesh
    - Gradients are synchronized across ALL nodes via lax.pmean

Usage:
    Single-node (1 process, 4 GPUs):
        sbatch scripts/run_training.sh config.yaml

    Multi-node (2 processes, 8 total GPUs):
        sbatch --nodes=2 scripts/run_training.sh config.yaml

IMPORTANT: JAX distributed initialization must happen before any other JAX operations.
           This is why the initialization code is at the top of this file.
"""

# =============================================================================
# CRITICAL: JAX DISTRIBUTED INITIALIZATION (must be first!)
# =============================================================================
# JAX distributed must be initialized BEFORE any other JAX operations.
# This includes before importing modules that use jax.numpy, etc.
# =============================================================================

import os
import jax

def _initialize_jax_distributed():
    """
    Initialize JAX distributed training.

    MUST be called before any other JAX operations!

    Architecture: 1 process per NODE (not per GPU!)
        - Each process sees 4 local GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)
        - After initialization, jax.devices() returns ALL GPUs across ALL nodes
        - chemtrain's shard_map creates a mesh spanning ALL devices
        - lax.pmean synchronizes gradients across ALL devices in the mesh
        - Result: TRUE multi-node training with unified gradient updates

    Returns:
        Tuple of (is_distributed, rank, world_size)
        - rank: which NODE this process is (0, 1, 2, ...)
        - world_size: total number of NODES (each with 4 GPUs)
    """
    # Check if running under SLURM with multiple tasks
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")

    if slurm_job_id and slurm_ntasks and int(slurm_ntasks) > 1:
        # Multi-node SLURM job
        # IMPORTANT: JAX automatic detection assumes 1 GPU per task, which doesn't work
        # for our "1 process per node with 4 GPUs" architecture. Use manual initialization.
        num_processes = int(slurm_ntasks)
        process_id = int(slurm_procid) if slurm_procid else 0

        # Get coordinator address from SLURM nodelist
        # Use subprocess to parse nodelist since scontrol may not be available
        import subprocess
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostname", slurm_nodelist],
                capture_output=True, text=True, check=True
            )
            coordinator_host = result.stdout.strip().split('\n')[0]
            # Use FQDN for cross-node resolution on JUWELS Booster
            coordinator_address = f"{coordinator_host}.juwels"
        except Exception as e:
            print(f"[SLURM] Warning: Could not parse nodelist, using first node from {slurm_nodelist}: {e}")
            # Fallback: extract first node manually (handles simple cases like "node[001-002]")
            import re
            match = re.match(r'([a-zA-Z]+)(\d+)', slurm_nodelist.replace('[', '').replace(']', ''))
            if match:
                coordinator_address = f"{match.group(1)}{match.group(2)}.juwels"
            else:
                coordinator_address = f"{slurm_nodelist.split(',')[0].split('[')[0]}.juwels"

        # Use job-specific port to avoid conflicts
        coordinator_port = 29400 + (int(slurm_job_id) % 1000)

        # Derive local device count from CUDA_VISIBLE_DEVICES
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        n_local_gpus = len(cuda_vis.split(","))
        local_ids = list(range(n_local_gpus))

        print(f"[SLURM] Detected multi-node job: {num_processes} tasks")
        print(f"[SLURM] Process {process_id}/{num_processes}")
        print(f"[SLURM] Coordinator: {coordinator_address}:{coordinator_port}")
        print(f"[SLURM] CUDA_VISIBLE_DEVICES={cuda_vis} -> {n_local_gpus} local GPUs")

        jax.distributed.initialize(
            coordinator_address=f"{coordinator_address}:{coordinator_port}",
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=local_ids,
            initialization_timeout=300,
        )

        rank = jax.process_index()
        world_size = jax.process_count()

        print(f"[Rank {rank}/{world_size}] JAX distributed initialized")

        # Verify device counts - with 1 process per NODE, n_local should be 4 (GPUs per node)
        n_local = jax.local_device_count()
        n_global = jax.device_count()
        expected_local = 4  # GPUs per node on JUWELS Booster

        print(f"[Rank {rank}] Local GPUs: {n_local}, Total GPUs: {n_global}")
        print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        # Verify: with 1 process per NODE, we expect n_local=4 and n_global=num_processes*4
        if n_local != expected_local:
            print(f"WARNING: Expected {expected_local} local GPUs per process (1 process per node), got {n_local}")
        expected_global = world_size * expected_local
        if n_global != expected_global:
            print(f"WARNING: Expected {expected_global} total GPUs ({world_size} nodes × {expected_local} GPUs), got {n_global}")

        is_distributed = True
    else:
        # Single-node or not under SLURM - single process mode
        rank = 0
        world_size = 1
        is_distributed = False

        n_local = jax.local_device_count()
        print(f"[Single-process mode] Local devices: {n_local}")

    # Print all devices
    print(f"[Rank {rank}] Local devices: {jax.local_devices()}")
    if is_distributed:
        print(f"[Rank {rank}] All devices: {jax.devices()}")

    # Disable 64-bit precision for performance
    jax.config.update("jax_enable_x64", False)

    return is_distributed, rank, world_size


# Initialize JAX distributed FIRST, before any other imports that use JAX
_IS_DISTRIBUTED, _RANK, _WORLD_SIZE = _initialize_jax_distributed()

# =============================================================================
# Now safe to import modules that use JAX
# =============================================================================

import sys
import pickle
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from chemtrain.data.data_loaders import DataLoaders

# Add parent directory to path for clean_code_base imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from training.trainer import Trainer
from export.exporter import AllegroExporter
from evaluation.visualizer import LossPlotter
from utils.logging import data_logger, model_logger, training_logger, export_logger
import logging


def apply_numpy_dataloader_patch():
    """
    Apply necessary patch to NumpyDataLoader for cache_size.

    This ensures cache_size is never 0, which causes errors in chemtrain.
    """
    from jax_sgmc.data.numpy_loader import NumpyDataLoader as _NDL

    _orig_get_indices = _NDL._get_indices

    def _patched_get_indices(self, chain_id: int):
        chain = self._chains[chain_id]
        if chain.get("cache_size", 0) <= 0:
            chain["cache_size"] = 1
        return _orig_get_indices(self, chain_id)

    _NDL._get_indices = _patched_get_indices
    logging.info("[Patch] Applied NumpyDataLoader cache_size fix")


def find_latest_checkpoint(checkpoint_dir: Path):
    """
    Find the most recent checkpoint file in a directory.

    Looks for chemtrain checkpoint files (epoch*.pkl or stage_*.pkl)
    and returns the most recently modified one.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    # Look for chemtrain checkpoints (epoch*.pkl) and stage checkpoints (stage_*.pkl)
    checkpoints = list(checkpoint_dir.glob("epoch*.pkl")) + list(checkpoint_dir.glob("stage_*.pkl"))

    # Filter out metadata files
    checkpoints = [p for p in checkpoints if not p.name.endswith(".meta.pkl")]

    if not checkpoints:
        return None

    # Return most recently modified checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def main(config_file: str, job_id: str = None, resume_checkpoint: str = None):
    """
    Main training pipeline.

    Args:
        config_file: Path to YAML configuration file
        job_id: SLURM job ID (optional, for logging)
        resume_checkpoint: Path to checkpoint file to resume from (optional)
    """
    # Use global distributed state (initialized at module load)
    is_distributed = _IS_DISTRIBUTED
    rank = _RANK
    world_size = _WORLD_SIZE

    apply_numpy_dataloader_patch()

    config = ConfigManager(config_file)

    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Log distributed info
    logging.info("=" * 60)
    logging.info("JAX DISTRIBUTED STATUS")
    logging.info("=" * 60)
    logging.info(f"Distributed: {is_distributed}")
    logging.info(f"Rank: {rank}/{world_size}")
    logging.info(f"Local devices: {jax.local_device_count()}")
    logging.info(f"Global devices: {jax.device_count()}")
    logging.info("=" * 60)

    # ===== Output directories =====
    export_dir = Path(config.get_export_path())
    export_dir.mkdir(parents=True, exist_ok=True)

    model_context = config.get_model_context()
    model_id = config.get_model_id()
    model_name = f"{model_context}_{model_id}"

    # Save config copy
    config_path = export_dir / f"{model_name}_config.yaml"
    config.save(config_path)
    logging.info(f"[Config] Saved to: {config_path}")

    # ===== Load and preprocess data =====
    logging.info("\n" + "=" * 60)
    logging.info("LOADING DATA")
    logging.info("=" * 60)

    # Resolve data path relative to clean_code_base directory if it's relative
    data_path = config.get_data_path()
    data_path_obj = Path(data_path)

    if not data_path_obj.is_absolute():
        # Get the directory where this script is located (clean_code_base/scripts)
        script_dir = Path(__file__).parent
        clean_code_base_dir = script_dir.parent
        data_path_obj = clean_code_base_dir / data_path
        data_logger.info(f"Resolved relative path: {data_path} -> {data_path_obj}")

    # Load dataset (shuffling and limiting frames happens in constructor)
    max_frames = config.get_max_frames()
    seed = config.get_seed()
    loader = DatasetLoader(str(data_path_obj), max_frames=max_frames, seed=seed)

    # Get dataset info
    N_max = loader.N_max
    species0 = loader.species[0]

    data_logger.info(f"N_max: {N_max}")
    data_logger.info(f"Species: {species0}")
    data_logger.info(f"Total frames: {len(loader)}")

    # ===== Compute box from data =====
    cutoff = config.get_cutoff()
    buffer_mult = config.get_buffer_multiplier()
    park_mult = config.get_park_multiplier()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=buffer_mult,
        park_multiplier=park_mult
    )

    # Compute box extent from all data
    extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)

    # Preprocess all coordinates
    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, R_shift)

    box = extent
    data_logger.info(f"[Preprocessing] Computed box: {np.asarray(box)}")
    data_logger.info(f"[Preprocessing] R_shift: {np.asarray(R_shift)}")

    # ===== Initialize model =====
    logging.info("\n" + "=" * 60)
    logging.info("INITIALIZING MODEL")
    logging.info("=" * 60)

    R0 = dataset["R"][0]
    mask0 = dataset["mask"][0]

    model = CombinedModel(
        config=config,
        R0=R0,
        box=box,
        species=species0,
        N_max=N_max
    )

    model_logger.info(f"Initialized: {model}")

    # ===== Setup data loaders =====
    logging.info("\n" + "=" * 60)
    logging.info("PREPARING TRAINING")
    logging.info("=" * 60)

    val_fraction = config.get_val_fraction()
    N_train = int(np.round(len(dataset["R"]) * (1 - val_fraction)))
    N_val = len(dataset["R"]) - N_train

    # Check if validation set is too small for batching
    batch_per_device = config.get_batch_per_device()
    n_devices = jax.local_device_count()
    min_val_samples = batch_per_device * n_devices

    if N_val < min_val_samples:
        data_logger.warning(f"[Split] Validation set ({N_val} samples) is too small for batch size "
                          f"({batch_per_device} per device * {n_devices} devices = {min_val_samples})")
        data_logger.warning(f"[Split] Using training data for validation")
        N_train = len(dataset["R"])
        N_val = 0

    data_logger.info(f"[Split] Training: {N_train}, Validation: {N_val if N_val > 0 else 'using train'}")

    train_loader = NumpyDataLoader(
        R=dataset["R"][:N_train],
        F=dataset["F"][:N_train],
        mask=dataset["mask"][:N_train],
        species=dataset["species"][:N_train],
        copy=False
    )

    # Use training data for validation if validation set is too small or disabled
    if N_val == 0 or val_fraction == 0.0:
        val_loader = train_loader
    else:
        val_loader = NumpyDataLoader(
            R=dataset["R"][N_train:N_train + N_val],
            F=dataset["F"][N_train:N_train + N_val],
            mask=dataset["mask"][N_train:N_train + N_val],
            species=dataset["species"][N_train:N_train + N_val],
            copy=False
        )

    loaders = DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None
    )

    # ===== Training =====
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING")
    logging.info("=" * 60)

    # Prepare training data dict for prior pre-training (avoid _chains access)
    train_data = {
        "R": jnp.asarray(dataset["R"][:N_train]),
        "F": jnp.asarray(dataset["F"][:N_train]),
        "mask": jnp.asarray(dataset["mask"][:N_train]),
    }

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_data=train_data
    )

    # Handle resume from checkpoint
    resolved_checkpoint = None
    if resume_checkpoint:
        if resume_checkpoint == "auto":
            # Auto-find latest checkpoint
            checkpoint_dir = Path(config.get_checkpoint_path())
            resolved_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if resolved_checkpoint:
                training_logger.info(f"\nAuto-resume: found checkpoint {resolved_checkpoint}")
            else:
                training_logger.info("\nAuto-resume: no checkpoints found, starting fresh")
        else:
            # Use specified checkpoint
            resolved_checkpoint = resume_checkpoint
            training_logger.info(f"\nResuming from checkpoint: {resolved_checkpoint}")

    # Run full pipeline (prior pretrain + stage1 + stage2)
    results = trainer.train_full_pipeline(resume_from=resolved_checkpoint)

    # Save checkpoint after training
    checkpoint_path = export_dir / f"{model_name}_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path, metadata={"job_id": job_id, "results": results})

    training_logger.info("\nComplete!")
    for stage, result in results.items():
        training_logger.info(f"  {stage}: {result}")

    # ===== Diagnostics =====
    logging.info("\n" + "=" * 60)
    logging.info("DIAGNOSTICS")
    logging.info("=" * 60)

    best_params = trainer.get_best_params()

    # Energy components
    components = model.compute_components(
        best_params, R0, mask0, species0
    )
    model_logger.info("[Energy components]")
    for key, val in components.items():
        model_logger.info(f"  {key}: {val:.6f}")

    # Force components
    force_components = model.compute_force_components(
        best_params, R0, mask0, species0
    )
    model_logger.info("\n[Force component norms]")
    for key, val in force_components.items():
        norm = jnp.linalg.norm(val)
        model_logger.info(f"  {key}: {norm:.6f}")

    # ===== Export =====
    logging.info("\n" + "=" * 60)
    logging.info("EXPORTING MODEL")
    logging.info("=" * 60)

    mlir_path = export_dir / f"{model_name}.mlir"
    exporter = AllegroExporter.from_combined_model(
        model=model,
        params=best_params,
        box=box,
        species=species0
    )
    exporter.export_to_file(mlir_path)
    export_logger.info(f"MLIR: {mlir_path}")

    # Save parameters as pickle
    params_path = export_dir / f"{model_name}_params.pkl"
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    export_logger.info(f"Parameters: {params_path}")

    # ===== Plotting =====
    logging.info("\n" + "=" * 60)
    logging.info("GENERATING PLOTS")
    logging.info("=" * 60)

    log_file = f"train_allegro_{job_id}.log"
    if Path(log_file).exists():
        plotter = LossPlotter(log_file, config=config)
        plotter.parse_log()

        plot_path = export_dir / f"loss_curve_{job_id}_{model_id}.png"
        plotter.plot(plot_path)
        logging.info(f"[Plot] Loss curve: {plot_path}")

        data_path = export_dir / f"loss_data_{job_id}_{model_id}.txt"
        plotter.save_loss_data(data_path)
        logging.info(f"[Plot] Loss data: {data_path}")
    else:
        logging.warning(f"[Plot] Log file not found: {log_file}")

    logging.info("\n" + "=" * 60)
    logging.info("TRAINING PIPELINE COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python train.py <config.yaml> [job_id] [--resume checkpoint.pkl]")
        sys.exit(1)

    config_file = sys.argv[1]
    job_id = None
    resume_checkpoint = None

    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--resume":
            if i + 1 < len(sys.argv):
                resume_checkpoint = sys.argv[i + 1]
                i += 2
            else:
                logging.error("--resume requires a checkpoint path")
                sys.exit(1)
        else:
            # Assume positional job_id if not a flag
            if job_id is None:
                job_id = arg
            i += 1

    main(config_file, job_id, resume_checkpoint)
