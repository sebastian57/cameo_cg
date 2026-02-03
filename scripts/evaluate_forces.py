#!/usr/bin/env python3
"""
Force Evaluation Script

Evaluates trained model forces against reference data and creates diagnostic plots.
Output files are named using the model_id from the config file.

Usage:
    python scripts/evaluate_forces.py <params.pkl> <config.yaml> [--frames N] [--output DIR]

Examples:
    # Evaluate on first 10 frames
    python scripts/evaluate_forces.py exported_models/model_params.pkl config.yaml

    # Evaluate on 50 random frames
    python scripts/evaluate_forces.py exported_models/model_params.pkl config.yaml --frames 50

    # Custom output directory
    python scripts/evaluate_forces.py exported_models/model_params.pkl config.yaml --output eval_results/
"""

import sys
import warnings
import os
from pathlib import Path

# =============================================================================
# Suppress ALL warnings (JAX, numpy, etc.) for clean console output
# =============================================================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs if present
# =============================================================================

# Add clean_code_base to path for imports
script_dir = Path(__file__).parent
clean_code_base = script_dir.parent
if str(clean_code_base) not in sys.path:
    sys.path.insert(0, str(clean_code_base))

# =============================================================================
# JAX/jax_md compatibility patch (must be before any jax_md imports)
# =============================================================================
# jax_md uses jax.random.KeyArray which was removed in newer JAX versions
import jax
jax.random.KeyArray = jax.Array

# Clear cached modules to ensure patch takes effect
_to_uncache = [mod for mod in sys.modules if mod.startswith('jax.random')]
for mod in _to_uncache:
    del sys.modules[mod]
# =============================================================================

import argparse
import pickle
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from contextlib import redirect_stdout
import io

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from evaluation.evaluator import Evaluator
from evaluation.visualizer import ForceAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model forces")
    parser.add_argument("params", help="Path to trained parameters pickle file")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to evaluate (default: 10)")
    parser.add_argument("--output", default="./force_eval", help="Output directory for plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for frame selection")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Force Evaluation")
    print("=" * 60)
    print(f"Parameters: {args.params}")
    print(f"Config: {args.config}")
    print(f"Frames: {args.frames}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load config
    config = ConfigManager(args.config)

    # Get model_id for file naming
    model_id = config.get("model_id", default="model")
    print(f"Model ID: {model_id}")

    # Load parameters
    print("\nLoading parameters...")
    with open(args.params, 'rb') as f:
        params = pickle.load(f)
    print(f"  Loaded parameters with keys: {list(params.keys()) if isinstance(params, dict) else 'single array'}")

    # Load dataset
    print("\nLoading dataset...")
    data_path = config.get_data_path()
    data_path_obj = Path(data_path)
    if not data_path_obj.is_absolute():
        data_path_obj = clean_code_base / data_path

    loader = DatasetLoader(str(data_path_obj), max_frames=None, seed=config.get_seed())
    print(f"  Total frames: {len(loader)}")
    print(f"  N_max: {loader.N_max}")

    # Preprocess coordinates
    cutoff = config.get_cutoff()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier()
    )

    extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, R_shift)
    box = extent

    print(f"  Box: {np.asarray(box)}")

    # Initialize model (suppress Allegro's verbose output)
    print("\nInitializing model...")
    R0 = dataset["R"][0]
    species0 = dataset["species"][0]

    with redirect_stdout(io.StringIO()):
        model = CombinedModel(
            config=config,
            R0=R0,
            box=box,
            species=species0,
            N_max=loader.N_max
        )
    print(f"  Model: {model}")
    print(f"  Use priors: {model.use_priors}")

    # Create evaluator
    evaluator = Evaluator(model, params, config)

    # Select frames to evaluate
    np.random.seed(args.seed)
    n_frames = min(args.frames, len(loader))
    frame_indices = np.random.choice(len(loader), size=n_frames, replace=False)
    frame_indices = np.sort(frame_indices)

    print(f"\nEvaluating {n_frames} frames...")

    # Collect predictions and references
    all_F_pred = []
    all_F_ref = []
    all_masks = []
    all_results = []

    for idx in tqdm(frame_indices, desc="Evaluating frames", unit="frame"):
        R = jnp.asarray(dataset["R"][idx])
        F_ref = jnp.asarray(dataset["F"][idx])
        mask = jnp.asarray(dataset["mask"][idx])
        species = jnp.asarray(dataset["species"][idx])

        # Suppress Allegro's verbose print statements during evaluation
        with redirect_stdout(io.StringIO()):
            result = evaluator.evaluate_frame(R, F_ref, mask, species)
        all_results.append(result)

        all_F_pred.append(np.asarray(result["forces"]))
        all_F_ref.append(np.asarray(F_ref))
        all_masks.append(np.asarray(mask))

    # Aggregate forces (flatten across all frames)
    F_pred_all = np.concatenate(all_F_pred, axis=0)
    F_ref_all = np.concatenate(all_F_ref, axis=0)
    mask_all = np.concatenate(all_masks, axis=0)

    # Apply mask to get only real atoms
    real_mask = mask_all > 0
    F_pred_real = F_pred_all[real_mask]
    F_ref_real = F_ref_all[real_mask]

    print(f"\nTotal atoms evaluated: {len(F_pred_real)}")

    # Compute aggregate metrics
    force_errors = F_pred_real - F_ref_real
    rmse = np.sqrt(np.mean(force_errors ** 2))
    mae = np.mean(np.abs(force_errors))

    # Per-component metrics
    print("\n" + "=" * 60)
    print("Force Error Metrics")
    print("=" * 60)
    print(f"Overall RMSE: {rmse:.4f} kcal/mol/Å")
    print(f"Overall MAE:  {mae:.4f} kcal/mol/Å")
    print()

    for i, comp in enumerate(['X', 'Y', 'Z']):
        comp_rmse = np.sqrt(np.mean(force_errors[:, i] ** 2))
        comp_mae = np.mean(np.abs(force_errors[:, i]))
        print(f"  {comp}-component RMSE: {comp_rmse:.4f}, MAE: {comp_mae:.4f}")

    print("=" * 60)

    # Create plots (named with model_id to avoid overwriting)
    print("\nGenerating plots...")

    # 1. Force component scatter plots (pred vs ref for x, y, z)
    component_plot = output_dir / f"{model_id}_force_components.png"
    ForceAnalyzer.plot_force_components(F_pred_real, F_ref_real, component_plot)
    print(f"  Saved: {component_plot}")

    # 2. Force distribution plots
    distribution_plot = output_dir / f"{model_id}_force_distribution.png"
    ForceAnalyzer.plot_force_distribution(F_pred_real, F_ref_real, distribution_plot)
    print(f"  Saved: {distribution_plot}")

    # 3. Force magnitude plot (using first frame's coordinates for position info)
    R_all = np.concatenate([dataset["R"][i] for i in frame_indices], axis=0)
    R_real = R_all[real_mask]
    magnitude_plot = output_dir / f"{model_id}_force_magnitude.png"
    ForceAnalyzer.plot_force_magnitude(F_pred_real, F_ref_real, R_real, magnitude_plot)
    print(f"  Saved: {magnitude_plot}")

    # Save numerical results
    results_file = output_dir / f"{model_id}_force_metrics.txt"
    with open(results_file, 'w') as f:
        f.write("Force Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Parameters: {args.params}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Frames evaluated: {n_frames}\n")
        f.write(f"Total atoms: {len(F_pred_real)}\n")
        f.write("\n")
        f.write(f"Overall RMSE: {rmse:.6f}\n")
        f.write(f"Overall MAE: {mae:.6f}\n")
        f.write("\n")
        f.write("Per-component metrics:\n")
        for i, comp in enumerate(['X', 'Y', 'Z']):
            comp_rmse = np.sqrt(np.mean(force_errors[:, i] ** 2))
            comp_mae = np.mean(np.abs(force_errors[:, i]))
            f.write(f"  {comp}: RMSE={comp_rmse:.6f}, MAE={comp_mae:.6f}\n")
    print(f"  Saved: {results_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
