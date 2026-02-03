"""
Evaluation Script for Trained Allegro Models

Evaluates trained models on single frames or full datasets.
Generates force analysis plots and error metrics.

Usage:
    # Evaluate single frame
    python evaluate.py config.yaml params.pkl --frame 0

    # Evaluate full dataset
    python evaluate.py config.yaml params.pkl --full

    # Generate plots only
    python evaluate.py config.yaml params.pkl --frame 0 --plots-only
"""

import sys
import pickle
from pathlib import Path
import argparse
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from evaluation.evaluator import Evaluator
from evaluation.visualizer import ForceAnalyzer
from utils.logging import data_logger, model_logger, eval_logger
import logging


def load_params(params_path: str):
    """Load parameters from pickle file."""
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    logging.info(f"[Params] Loaded from: {params_path}")
    return params


def evaluate_single_frame(
    evaluator: Evaluator,
    dataset: dict,
    frame_idx: int,
    output_dir: Path,
    model_name: str
):
    """
    Evaluate model on a single frame and generate plots.

    Args:
        evaluator: Evaluator instance
        dataset: Dataset dictionary
        frame_idx: Frame index to evaluate
        output_dir: Output directory for plots
        model_name: Model name for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    R = dataset["R"][frame_idx]
    F_ref = dataset["F"][frame_idx]
    mask = dataset["mask"][frame_idx]
    species = dataset["species"][frame_idx]

    logging.info(f"\n[Evaluating frame {frame_idx}]")

    # Compute metrics
    metrics = evaluator.evaluate_frame(R, F_ref, mask, species)

    # Print results
    logging.info("\nMetrics:")
    logging.info(f"  Energy: {metrics['energy']:.6f}")
    logging.info(f"  Force RMSE: {metrics['force_rmse']:.6f}")
    logging.info(f"  Force MAE: {metrics['force_mae']:.6f}")
    logging.info(f"  Max Force Error: {metrics['force_max_error']:.6f}")

    if "energy_components" in metrics:
        logging.info("\nEnergy Components:")
        for key, val in metrics["energy_components"].items():
            print(f"  {key}: {val:.6f}")

    if "force_components" in metrics:
        logging.info("\nForce Component Norms:")
        for key, val in metrics["force_components"].items():
            norm = jnp.linalg.norm(val)
            print(f"  {key}: {norm:.6f}")

    # Generate plots
    F_pred = metrics["forces"]

    logging.info("\n[Generating plots]")

    # Force component scatter plots
    comp_path = output_dir / f"{model_name}_force_components_frame{frame_idx}.png"
    ForceAnalyzer.plot_force_components(
        F_pred=np.asarray(F_pred),
        F_ref=np.asarray(F_ref),
        output_path=str(comp_path),
        mask=np.asarray(mask)
    )

    # Force magnitude plots
    mag_path = output_dir / f"{model_name}_force_magnitude_frame{frame_idx}.png"
    ForceAnalyzer.plot_force_magnitude(
        F_pred=np.asarray(F_pred),
        F_ref=np.asarray(F_ref),
        R=np.asarray(R),
        output_path=str(mag_path),
        mask=np.asarray(mask)
    )

    # Force distribution plots
    dist_path = output_dir / f"{model_name}_force_distribution_frame{frame_idx}.png"
    ForceAnalyzer.plot_force_distribution(
        F_pred=np.asarray(F_pred),
        F_ref=np.asarray(F_ref),
        output_path=str(dist_path),
        mask=np.asarray(mask)
    )

    logging.info(f"\nPlots saved to: {output_dir}")

    return metrics


def evaluate_dataset(
    evaluator: Evaluator,
    dataset: dict,
    output_dir: Path,
    model_name: str
):
    """
    Evaluate model on full dataset.

    Args:
        evaluator: Evaluator instance
        dataset: Dataset dictionary
        output_dir: Output directory for results
        model_name: Model name for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(dataset["R"])
    logging.info(f"\n[Evaluating {n_frames} frames]")

    all_metrics = []

    for i in range(n_frames):
        R = dataset["R"][i]
        F_ref = dataset["F"][i]
        mask = dataset["mask"][i]
        species = dataset["species"][i]

        metrics = evaluator.evaluate_frame(R, F_ref, mask, species)
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_frames}")

    # Aggregate statistics
    energies = np.array([m["energy"] for m in all_metrics])
    force_rmse = np.array([m["force_rmse"] for m in all_metrics])
    force_mae = np.array([m["force_mae"] for m in all_metrics])
    force_max = np.array([m["force_max_error"] for m in all_metrics])

    logging.info("\n" + "=" * 60)
    logging.info("DATASET STATISTICS")
    logging.info("=" * 60)

    logging.info(f"\nEnergy:")
    logging.info(f"  Mean: {np.mean(energies):.6f}")
    logging.info(f"  Std: {np.std(energies):.6f}")
    logging.info(f"  Min: {np.min(energies):.6f}")
    logging.info(f"  Max: {np.max(energies):.6f}")

    logging.info(f"\nForce RMSE:")
    logging.info(f"  Mean: {np.mean(force_rmse):.6f}")
    logging.info(f"  Std: {np.std(force_rmse):.6f}")
    logging.info(f"  Min: {np.min(force_rmse):.6f}")
    logging.info(f"  Max: {np.max(force_rmse):.6f}")

    logging.info(f"\nForce MAE:")
    logging.info(f"  Mean: {np.mean(force_mae):.6f}")
    logging.info(f"  Std: {np.std(force_mae):.6f}")

    logging.info(f"\nMax Force Error:")
    logging.info(f"  Mean: {np.mean(force_max):.6f}")
    logging.info(f"  Max: {np.max(force_max):.6f}")

    # Save statistics to file
    stats_path = output_dir / f"{model_name}_eval_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("Dataset Evaluation Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"N_frames: {n_frames}\n\n")
        f.write(f"Energy: mean={np.mean(energies):.6f}, std={np.std(energies):.6f}\n")
        f.write(f"Force RMSE: mean={np.mean(force_rmse):.6f}, std={np.std(force_rmse):.6f}\n")
        f.write(f"Force MAE: mean={np.mean(force_mae):.6f}, std={np.std(force_mae):.6f}\n")
        f.write(f"Max Force Error: mean={np.mean(force_max):.6f}, max={np.max(force_max):.6f}\n")

    logging.info(f"\nStatistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Allegro model"
    )
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("params", type=str, help="Path to parameters pickle file")
    parser.add_argument("--frame", type=int, default=None, help="Evaluate single frame (default: 0)")
    parser.add_argument("--full", action="store_true", help="Evaluate full dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: ./evaluation)")

    args = parser.parse_args()

    # Disable 64-bit for consistency
    jax.config.update("jax_enable_x64", False)

    # ===== Load configuration =====
    config = ConfigManager(args.config)
    params = load_params(args.params)

    model_context = config.get_model_context()
    model_id = config.get_model_id()
    model_name = f"{model_context}_{model_id}"

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("./evaluation") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load data =====
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

    # Load dataset (same preprocessing as training)
    max_frames = config.get_max_frames()
    seed = config.get_seed()
    loader = DatasetLoader(str(data_path_obj), max_frames=max_frames, seed=seed)

    N_max = loader.N_max
    species0 = loader.species[0]

    # Compute box (same as training)
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
    data_logger.info(f"Loaded {len(dataset['R'])} frames")
    data_logger.info(f"Box: {np.asarray(box)}")

    # ===== Initialize model =====
    logging.info("\n" + "=" * 60)
    logging.info("INITIALIZING MODEL")
    logging.info("=" * 60)

    R0 = dataset["R"][0]

    model = CombinedModel(
        config=config,
        R0=R0,
        box=box,
        species=species0,
        N_max=N_max
    )

    model_logger.info(f"{model}")

    # ===== Create evaluator =====
    evaluator = Evaluator(model, params, config)

    # ===== Evaluate =====
    if args.full:
        evaluate_dataset(evaluator, dataset, output_dir, model_name)
    else:
        frame_idx = args.frame if args.frame is not None else 0
        evaluate_single_frame(evaluator, dataset, frame_idx, output_dir, model_name)

    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
