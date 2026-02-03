"""
Ensemble Training Script for Allegro Coarse-Grained Protein Force Fields

Trains multiple models with different random seeds, computes variance
across models, and selects the best performing one (lowest validation loss).

Uses chemtrain's sequential training approach - models are trained one after
another. For parallel GPU training, see future Phase 2 implementation.

ARCHITECTURE (1 process per NODE, not per GPU!):
    Same as train.py - uses JAX distributed + chemtrain's shard_map.

Usage:
    # Enable in config:
    ensemble:
      enabled: true
      n_models: 5
      base_seed: 42
      save_all_models: false

    # Run:
    sbatch scripts/run_training.sh config.yaml  # Uses this script if ensemble enabled

IMPORTANT: JAX distributed initialization must happen before any other JAX operations.
"""

# =============================================================================
# CRITICAL: JAX DISTRIBUTED INITIALIZATION (must be first!)
# =============================================================================

import os
import jax

def _initialize_jax_distributed():
    """
    Initialize JAX distributed training.

    MUST be called before any other JAX operations!
    See train.py for detailed documentation.
    """
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_procid = os.environ.get("SLURM_PROCID")
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    coordinator_port = os.environ.get("JAX_COORDINATOR_PORT", "12345")

    if slurm_ntasks and coordinator_address:
        num_processes = int(slurm_ntasks)
        process_id = int(slurm_procid) if slurm_procid else 0

        jax.distributed.initialize(
            coordinator_address=f"{coordinator_address}:{coordinator_port}",
            num_processes=num_processes,
            process_id=process_id,
        )

        rank = jax.process_index()
        world_size = jax.process_count()

        print(f"[Rank {rank}/{world_size}] JAX distributed initialized")

        n_local = jax.local_device_count()
        n_global = jax.device_count()
        expected_local = 4

        print(f"[Rank {rank}] Local GPUs: {n_local}, Total GPUs: {n_global}")

        if n_local != expected_local:
            print(f"WARNING: Expected {expected_local} local GPUs per process, got {n_local}")

        is_distributed = True
    else:
        rank = 0
        world_size = 1
        is_distributed = False

        n_local = jax.local_device_count()
        print(f"[Single-process mode] Local devices: {n_local}")

    print(f"[Rank {rank}] Local devices: {jax.local_devices()}")
    if is_distributed:
        print(f"[Rank {rank}] All devices: {jax.devices()}")

    jax.config.update("jax_enable_x64", False)

    return is_distributed, rank, world_size


# Initialize JAX distributed FIRST
_IS_DISTRIBUTED, _RANK, _WORLD_SIZE = _initialize_jax_distributed()

# =============================================================================
# Now safe to import modules that use JAX
# =============================================================================

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import jax.numpy as jnp
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from chemtrain.data.data_loaders import DataLoaders

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
    """Apply necessary patch to NumpyDataLoader for cache_size."""
    from jax_sgmc.data.numpy_loader import NumpyDataLoader as _NDL

    _orig_get_indices = _NDL._get_indices

    def _patched_get_indices(self, chain_id: int):
        chain = self._chains[chain_id]
        if chain.get("cache_size", 0) <= 0:
            chain["cache_size"] = 1
        return _orig_get_indices(self, chain_id)

    _NDL._get_indices = _patched_get_indices
    logging.info("[Patch] Applied NumpyDataLoader cache_size fix")


def train_single_model(
    config,
    dataset,
    loaders,
    species0,
    box,
    N_max,
    seed: int,
    model_index: int,
    export_dir: Path,
    job_id: str,
):
    """
    Train a single model with a specific seed.

    Args:
        config: ConfigManager instance
        dataset: Preprocessed dataset dict
        loaders: DataLoaders tuple
        species0: Species array for first frame
        box: Box extent
        N_max: Maximum number of atoms
        seed: Random seed for this model
        model_index: Index of this model in the ensemble
        export_dir: Export directory
        job_id: SLURM job ID

    Returns:
        Dictionary with training results
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"TRAINING MODEL {model_index} (seed={seed})")
    logging.info(f"{'='*60}\n")

    # Get data shapes
    R0 = dataset["R"][0]
    mask0 = dataset["mask"][0]
    N_train = loaders.train_loader._chains[0]["R"].shape[0]

    # Initialize model (seed is passed to Trainer for param initialization)
    model = CombinedModel(
        config=config,
        R0=R0,
        box=box,
        species=species0,
        N_max=N_max,
    )

    model_logger.info(f"[Model {model_index}] Created model, will initialize with seed={seed}")

    # Prepare training data for prior pre-training
    train_data = {
        "R": jnp.asarray(dataset["R"][:N_train]),
        "F": jnp.asarray(dataset["F"][:N_train]),
        "mask": jnp.asarray(dataset["mask"][:N_train]),
    }

    # Create trainer with specific seed for this ensemble member
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders.train_loader,
        val_loader=loaders.val_loader,
        train_data=train_data,
        seed=seed,  # Use specific seed for this ensemble member
    )

    # Run training
    results = trainer.train_full_pipeline()

    # Get best params and final validation loss
    best_params = trainer.get_best_params()

    # Get final validation loss
    if hasattr(trainer, '_chemtrain_trainer') and trainer._chemtrain_trainer is not None:
        val_losses = trainer._chemtrain_trainer.val_losses
        final_val_loss = val_losses[-1] if val_losses else float('inf')
    else:
        final_val_loss = float('inf')

    training_logger.info(f"[Model {model_index}] Training complete")
    training_logger.info(f"[Model {model_index}] Final validation loss: {final_val_loss:.6f}")

    return {
        "seed": seed,
        "model_index": model_index,
        "params": best_params,
        "val_loss": float(final_val_loss),
        "results": results,
        "model": model,
    }


def save_ensemble_metadata(
    export_dir: Path,
    model_name: str,
    ensemble_results: list,
    ensemble_config: dict,
    best_idx: int,
):
    """Save ensemble training metadata to JSON."""
    seeds = [r["seed"] for r in ensemble_results]
    val_losses = [r["val_loss"] for r in ensemble_results]

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(ensemble_results),
        "base_seed": ensemble_config["base_seed"],
        "seeds": seeds,
        "validation_losses": val_losses,
        "mean_loss": float(np.mean(val_losses)),
        "std_loss": float(np.std(val_losses)),
        "variance_loss": float(np.var(val_losses)),
        "min_loss": float(np.min(val_losses)),
        "max_loss": float(np.max(val_losses)),
        "best_model_index": int(best_idx),
        "best_model_seed": int(seeds[best_idx]),
        "best_model_loss": float(val_losses[best_idx]),
        "save_all_models": ensemble_config["save_all_models"],
    }

    metadata_path = export_dir / f"{model_name}_ensemble_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"[Ensemble] Saved metadata to: {metadata_path}")

    return metadata


def main(config_file: str, job_id: str = None):
    """
    Main ensemble training pipeline.

    Args:
        config_file: Path to YAML configuration file
        job_id: SLURM job ID (optional, for logging)
    """
    is_distributed = _IS_DISTRIBUTED
    rank = _RANK
    world_size = _WORLD_SIZE

    apply_numpy_dataloader_patch()

    config = ConfigManager(config_file)

    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Check if ensemble is enabled
    ensemble_config = config.get_ensemble_config()

    if not ensemble_config["enabled"]:
        logging.info("Ensemble training not enabled. Running single model training.")
        logging.info("To enable, set ensemble.enabled: true in config.")
        # Fall back to single model training
        from train import main as single_main
        return single_main(config_file, job_id)

    n_models = ensemble_config["n_models"]
    base_seed = ensemble_config["base_seed"]
    save_all = ensemble_config["save_all_models"]

    logging.info("=" * 60)
    logging.info("ENSEMBLE TRAINING")
    logging.info("=" * 60)
    logging.info(f"Number of models: {n_models}")
    logging.info(f"Base seed: {base_seed}")
    logging.info(f"Seeds: {[base_seed + i for i in range(n_models)]}")
    logging.info(f"Save all models: {save_all}")
    logging.info("=" * 60)

    # ===== Output directories =====
    export_dir = Path(config.get_export_path())
    export_dir.mkdir(parents=True, exist_ok=True)

    model_context = config.get_model_context()
    model_id = config.get_model_id()
    model_name = f"{model_context}_{model_id}"

    # Save config copy
    config_path = export_dir / f"{model_name}_ensemble_config.yaml"
    config.save(config_path)
    logging.info(f"[Config] Saved to: {config_path}")

    # ===== Load and preprocess data (once for all models) =====
    logging.info("\n" + "=" * 60)
    logging.info("LOADING DATA")
    logging.info("=" * 60)

    data_path = config.get_data_path()
    data_path_obj = Path(data_path)

    if not data_path_obj.is_absolute():
        script_dir = Path(__file__).parent
        clean_code_base_dir = script_dir.parent
        data_path_obj = clean_code_base_dir / data_path
        data_logger.info(f"Resolved relative path: {data_path} -> {data_path_obj}")

    # Use base_seed for data loading (consistent across ensemble)
    max_frames = config.get_max_frames()
    loader = DatasetLoader(str(data_path_obj), max_frames=max_frames, seed=base_seed)

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

    extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)

    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, R_shift)

    box = extent
    data_logger.info(f"[Preprocessing] Computed box: {np.asarray(box)}")

    # ===== Setup data loaders (shared across ensemble) =====
    val_fraction = config.get_val_fraction()
    N_train = int(np.round(len(dataset["R"]) * (1 - val_fraction)))
    N_val = len(dataset["R"]) - N_train

    batch_per_device = config.get_batch_per_device()
    n_devices = jax.local_device_count()
    min_val_samples = batch_per_device * n_devices

    if N_val < min_val_samples:
        data_logger.warning(f"[Split] Validation set too small, using training data for validation")
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

    # ===== Train ensemble =====
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING ENSEMBLE")
    logging.info("=" * 60)

    ensemble_results = []

    for i in range(n_models):
        seed = base_seed + i

        result = train_single_model(
            config=config,
            dataset=dataset,
            loaders=loaders,
            species0=species0,
            box=box,
            N_max=N_max,
            seed=seed,
            model_index=i,
            export_dir=export_dir,
            job_id=job_id,
        )

        ensemble_results.append(result)

        # Log progress
        val_losses_so_far = [r["val_loss"] for r in ensemble_results]
        logging.info(f"\n[Ensemble] Progress: {i+1}/{n_models} models trained")
        logging.info(f"[Ensemble] Validation losses so far: {val_losses_so_far}")

    # ===== Compute ensemble statistics =====
    logging.info("\n" + "=" * 60)
    logging.info("ENSEMBLE RESULTS")
    logging.info("=" * 60)

    val_losses = np.array([r["val_loss"] for r in ensemble_results])
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    variance_loss = np.var(val_losses)
    best_idx = int(np.argmin(val_losses))

    logging.info(f"Validation losses: {val_losses}")
    logging.info(f"Mean:     {mean_loss:.6f}")
    logging.info(f"Std:      {std_loss:.6f}")
    logging.info(f"Variance: {variance_loss:.6f}")
    logging.info(f"Min:      {np.min(val_losses):.6f}")
    logging.info(f"Max:      {np.max(val_losses):.6f}")
    logging.info(f"Best model: index={best_idx}, seed={base_seed + best_idx}, loss={val_losses[best_idx]:.6f}")

    # ===== Export models =====
    logging.info("\n" + "=" * 60)
    logging.info("EXPORTING MODELS")
    logging.info("=" * 60)

    # Always export the best model
    best_result = ensemble_results[best_idx]
    best_params = best_result["params"]
    best_model = best_result["model"]

    # Export best model MLIR
    mlir_path = export_dir / f"{model_name}_best.mlir"
    exporter = AllegroExporter.from_combined_model(
        model=best_model,
        params=best_params,
        box=box,
        species=species0
    )
    exporter.export_to_file(mlir_path)
    export_logger.info(f"Best model MLIR: {mlir_path}")

    # Save best params
    params_path = export_dir / f"{model_name}_best_params.pkl"
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    export_logger.info(f"Best model params: {params_path}")

    # Optionally save all models
    if save_all:
        logging.info("\n[Ensemble] Saving all models...")
        for i, result in enumerate(ensemble_results):
            if i == best_idx:
                continue  # Already saved as best

            model_mlir_path = export_dir / f"{model_name}_ensemble_{i}.mlir"
            model_params_path = export_dir / f"{model_name}_ensemble_{i}_params.pkl"

            # Export MLIR
            exporter = AllegroExporter.from_combined_model(
                model=result["model"],
                params=result["params"],
                box=box,
                species=species0
            )
            exporter.export_to_file(model_mlir_path)

            # Save params
            with open(model_params_path, 'wb') as f:
                pickle.dump(result["params"], f)

            export_logger.info(f"Model {i}: {model_mlir_path}")

    # Save ensemble metadata
    save_ensemble_metadata(
        export_dir=export_dir,
        model_name=model_name,
        ensemble_results=ensemble_results,
        ensemble_config=ensemble_config,
        best_idx=best_idx,
    )

    # ===== Summary =====
    logging.info("\n" + "=" * 60)
    logging.info("ENSEMBLE TRAINING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Models trained: {n_models}")
    logging.info(f"Best model: index={best_idx}, seed={base_seed + best_idx}")
    logging.info(f"Best validation loss: {val_losses[best_idx]:.6f}")
    logging.info(f"Loss variance: {variance_loss:.6f}")
    logging.info(f"Exported to: {export_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python train_ensemble.py <config.yaml> [job_id]")
        sys.exit(1)

    config_file = sys.argv[1]
    job_id = sys.argv[2] if len(sys.argv) > 2 else None

    main(config_file, job_id)
