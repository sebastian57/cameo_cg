#!/usr/bin/env python3
"""
Force Evaluation Script

Evaluates trained model forces against reference data and creates diagnostic plots.
Supports three evaluation modes:
  - full: Evaluate complete model (ML + priors if configured)
  - prior-only: Evaluate ONLY prior terms (parametric, spline, or trained)
  - ml-only: Evaluate ONLY ML model (force disable priors)

Usage:
    python scripts/evaluate_forces.py [params.pkl] <config.yaml> [options]

Examples:
    # Full model evaluation (default)
    python scripts/evaluate_forces.py exported_models/model_params.pkl config.yaml

    # Evaluate parametric priors only (from config)
    python scripts/evaluate_forces.py config_preprior.yaml --mode prior-only

    # Evaluate spline priors only (from config)
    python scripts/evaluate_forces.py config_template.yaml --mode prior-only --frames 50

    # Evaluate trained priors (from params.pkl)
    python scripts/evaluate_forces.py exported_models/params.pkl config.yaml --mode prior-only

    # ML-only (disable priors even if config enables them)
    python scripts/evaluate_forces.py exported_models/params.pkl config.yaml --mode ml-only
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
from contextlib import redirect_stdout, redirect_stderr
import io

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from evaluation.evaluator import Evaluator
from evaluation.visualizer import ForceAnalyzer
from evaluation.per_residue import (
    compute_per_residue_errors,
    plot_per_residue_rmse,
    save_per_residue_txt,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model forces")
    parser.add_argument("params", nargs='?', default=None,
                        help="Path to trained parameters pickle file (optional for prior-only mode)")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to evaluate (default: 10)")
    parser.add_argument("--output", default="./force_eval",
                        help="Output directory for plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for frame selection")
    parser.add_argument("--mode", choices=['full', 'prior-only', 'ml-only', 'per-residue'],
                        default='full',
                        help="Evaluation mode: full (ML+priors), prior-only, ml-only, "
                             "or per-residue (RMSE broken down by AA type) (default: full)")
    args = parser.parse_args()

    # Validate mode and params combination
    if args.mode not in ('prior-only', 'per-residue') and args.params is None:
        parser.error(f"params file is required for --mode {args.mode}")

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Force Evaluation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Parameters: {args.params if args.params else 'None (using config priors)'}")
    print(f"Config: {args.config}")
    print(f"Frames: {args.frames}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load config
    config = ConfigManager(args.config)

    # Validate prior-only mode requirements
    if args.mode == 'prior-only' and not config.use_priors():
        print("\nERROR: --mode prior-only requires model.use_priors=true in config")
        print("       Please enable priors in your config file")
        sys.exit(1)

    # Validate spline file exists and resolve to an absolute path.
    # PriorEnergy resolves relative spline paths from the config file's
    # directory, which breaks when evaluate_forces.py is called with a
    # config copy saved inside exported_models/.  We resolve the path here
    # and write the absolute path back into the config so that CombinedModel
    # always sees an absolute path regardless of which config copy is used.
    if config.use_spline_priors_enabled():
        spline_path_str = config.get_spline_file_path()
        spline_path = Path(spline_path_str)

        if not spline_path.is_absolute():
            # Try in order: CWD, clean_code_base, config file's directory
            candidates = [
                Path.cwd() / spline_path_str,
                clean_code_base / spline_path_str,
                Path(args.config).parent / spline_path_str,
            ]
            for candidate in candidates:
                if candidate.exists():
                    spline_path = candidate
                    break

        if not spline_path.exists():
            print(f"\nERROR: Spline file not found: {spline_path_str}")
            print(f"  Searched in: CWD, {clean_code_base}, config dir")
            sys.exit(1)

        # Patch config with the resolved absolute path so PriorEnergy
        # does not attempt a second (wrong) relative resolution.
        abs_spline = str(spline_path.resolve())
        config._config.setdefault('model', {}).setdefault('priors', {})['spline_file'] = abs_spline
        print(f"  Spline file: {abs_spline}")

    # Get model_id for file naming
    model_id = config.get("model_id", default="model")
    print(f"Model ID: {model_id}")

    # Load parameters (optional for prior-only mode)
    params = None
    prior_type = None  # Track which prior type we're using

    if args.mode == 'prior-only':
        # Prior-only mode: params file is optional
        if args.params is not None:
            print("\nLoading trained prior parameters...")
            with open(args.params, 'rb') as f:
                loaded = pickle.load(f)

                # Handle different checkpoint formats
                full_params = None
                # First check if it's a dict (most common case)
                if isinstance(loaded, dict):
                    # Check for chemtrain checkpoint dict with 'trainer_state' key
                    if 'trainer_state' in loaded:
                        print(f"  Detected chemtrain checkpoint dict")
                        trainer_state = loaded['trainer_state']
                        if isinstance(trainer_state, dict) and 'params' in trainer_state:
                            full_params = trainer_state['params']
                        elif hasattr(trainer_state, 'params'):
                            full_params = trainer_state.params
                    # Check for exported checkpoint dict with 'params' key
                    elif 'params' in loaded and isinstance(loaded['params'], dict):
                        full_params = loaded['params']
                    else:
                        # Direct params dict
                        full_params = loaded
                # Check if it's a trainer object
                elif hasattr(loaded, 'trainer_state'):
                    print(f"  Detected trainer object")
                    if hasattr(loaded.trainer_state, 'params'):
                        full_params = loaded.trainer_state.params
                    elif isinstance(loaded.trainer_state, dict) and 'params' in loaded.trainer_state:
                        full_params = loaded.trainer_state['params']
                elif hasattr(loaded, 'params'):
                    print(f"  Detected legacy trainer object")
                    full_params = loaded.params
                else:
                    print(f"  Warning: Unknown checkpoint format, using config priors")
                    full_params = None

                # Extract prior params if available
                if full_params is not None and isinstance(full_params, dict) and 'prior' in full_params:
                    params = {'prior': full_params['prior']}
                    prior_type = "trained"
                    print(f"  Loaded trained prior params")
                else:
                    print("  Warning: No 'prior' key in params file, using config priors")
                    params = None

        # Determine prior type from config if not trained
        if params is None:
            if config.use_spline_priors_enabled():
                prior_type = "spline"
                spline_path = config.get_spline_file_path()
                print(f"\nUsing spline priors from: {spline_path}")
            else:
                prior_type = "parametric"
                print("\nUsing parametric priors from config (histogram-fitted)")
    else:
        # Full or ml-only mode: params file is required
        print("\nLoading parameters...")
        with open(args.params, 'rb') as f:
            loaded = pickle.load(f)

            # Handle different checkpoint formats
            # First check if it's a dict (most common case)
            if isinstance(loaded, dict):
                # Check for chemtrain checkpoint dict with 'trainer_state' key
                if 'trainer_state' in loaded:
                    print(f"  Detected chemtrain checkpoint dict")
                    trainer_state = loaded['trainer_state']
                    # trainer_state can be dict or object
                    if isinstance(trainer_state, dict) and 'params' in trainer_state:
                        params = trainer_state['params']
                    elif hasattr(trainer_state, 'params'):
                        params = trainer_state.params
                    else:
                        print(f"  ERROR: Could not extract params from trainer_state")
                        print(f"  trainer_state type: {type(trainer_state)}")
                        if isinstance(trainer_state, dict):
                            print(f"  trainer_state keys: {list(trainer_state.keys())}")
                        else:
                            print(f"  trainer_state attrs: {dir(trainer_state)}")
                        sys.exit(1)
                # Check for exported checkpoint dict with 'params' key
                elif 'params' in loaded:
                    # Could be exported checkpoint or nested dict
                    if isinstance(loaded['params'], dict) and ('allegro' in loaded['params'] or 'prior' in loaded['params']):
                        print(f"  Detected exported checkpoint dict")
                        params = loaded['params']
                    else:
                        # Nested structure, might need to go deeper
                        print(f"  Detected nested params dict")
                        params = loaded['params']
                else:
                    # Direct params dict
                    print(f"  Detected direct params dict")
                    params = loaded
            # Check if it's a trainer object (less common)
            elif hasattr(loaded, 'trainer_state'):
                print(f"  Detected trainer object")
                if hasattr(loaded.trainer_state, 'params'):
                    params = loaded.trainer_state.params
                elif isinstance(loaded.trainer_state, dict) and 'params' in loaded.trainer_state:
                    params = loaded.trainer_state['params']
                else:
                    print(f"  ERROR: Could not extract params from trainer object")
                    sys.exit(1)
            elif hasattr(loaded, 'params'):
                # Legacy trainer format - params is an attribute
                print(f"  Detected legacy trainer object")
                params = loaded.params
            else:
                print(f"  ERROR: Unknown checkpoint format")
                print(f"  Type: {type(loaded)}")
                if isinstance(loaded, dict):
                    print(f"  Keys: {list(loaded.keys())}")
                else:
                    print(f"  Attributes: {dir(loaded)}")
                sys.exit(1)

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

    # Override config based on mode
    if args.mode == 'prior-only':
        # Force enable priors for prior-only evaluation
        if not config._config.get('model'):
            config._config['model'] = {}
        config._config['model']['use_priors'] = True
        print("  Mode: prior-only (priors enabled, ML will not be used)")
    elif args.mode == 'ml-only':
        # Force disable priors for ML-only evaluation
        if not config._config.get('model'):
            config._config['model'] = {}
        config._config['model']['use_priors'] = False
        print("  Mode: ml-only (priors disabled)")
    else:
        print(f"  Mode: full (use_priors={config.use_priors()})")

    # Suppress both stdout and stderr during model initialization
    # (Allegro prints to both, and we don't need to see it for prior-only mode)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        model = CombinedModel(
            config=config,
            R0=R0,
            box=box,
            species=species0,
            N_max=loader.N_max,
            prior_only=(args.mode == 'prior-only')  # Skip ML computation for prior-only mode
        )
    print(f"  Model: {model}")
    print(f"  Use priors: {model.use_priors}")

    # -------------------------------------------------------------------------
    # Per-residue mode: early exit with dedicated evaluation
    # -------------------------------------------------------------------------
    if args.mode == 'per-residue':
        if args.params is None:
            print("\nERROR: --mode per-residue requires a params file")
            sys.exit(1)

        # Build id_to_aa from loader
        id_to_aa = {v: k for k, v in loader.aa_to_id.items()} if loader.aa_to_id else {}

        print(f"\nRunning per-residue RMSE evaluation ({args.frames} frames)…")
        results = compute_per_residue_errors(
            model=model,
            params=params,
            dataset=dataset,
            n_frames=args.frames,
            seed=args.seed,
        )

        filename_base = f"{model_id}_per_residue"
        plot_path = output_dir / f"{filename_base}_rmse.png"
        txt_path = output_dir / f"{filename_base}_rmse.txt"

        plot_per_residue_rmse(results, id_to_aa, str(plot_path), title=f"Per-Residue RMSE — {model_id}")
        save_per_residue_txt(results, id_to_aa, str(txt_path), model_id=model_id)

        # Print summary
        real_mask = results["mask"] > 0
        rmse_real = results["rmse_per_atom"][real_mask]
        print(f"\nOverall per-atom RMSE: {float(np.mean(rmse_real)):.4f} kcal/mol/Å")
        print("Done!")
        return

    # For prior-only mode, ensure params dict structure
    if args.mode == 'prior-only':
        if params is None:
            # Create empty params dict - PriorEnergy will use its own params from config
            params = {}

        # Prior-only mode now truly skips ML computation, so we need a dummy ML param
        # for Evaluator compatibility (it expects params['allegro'] to exist)
        if 'allegro' not in params:
            # Use a minimal dummy dict instead of full initialization
            params['allegro'] = {'dummy': jnp.array(0.0)}

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

    # Create plots (named with model_id and mode to avoid overwriting)
    print("\nGenerating plots...")

    # Build filename base with mode suffix
    mode_suffix = ""
    if args.mode == 'prior-only':
        mode_suffix = f"_prior_only_{prior_type}" if prior_type else "_prior_only"
    elif args.mode == 'ml-only':
        mode_suffix = "_ml_only"

    filename_base = f"{model_id}{mode_suffix}"

    # 1. Force component scatter plots (pred vs ref for x, y, z)
    component_plot = output_dir / f"{filename_base}_force_components.png"
    ForceAnalyzer.plot_force_components(F_pred_real, F_ref_real, component_plot)
    print(f"  Saved: {component_plot}")

    # 2. Force distribution plots
    distribution_plot = output_dir / f"{filename_base}_force_distribution.png"
    ForceAnalyzer.plot_force_distribution(F_pred_real, F_ref_real, distribution_plot)
    print(f"  Saved: {distribution_plot}")

    # 3. Force magnitude plot (using first frame's coordinates for position info)
    R_all = np.concatenate([dataset["R"][i] for i in frame_indices], axis=0)
    R_real = R_all[real_mask]
    magnitude_plot = output_dir / f"{filename_base}_force_magnitude.png"
    ForceAnalyzer.plot_force_magnitude(F_pred_real, F_ref_real, R_real, magnitude_plot)
    print(f"  Saved: {magnitude_plot}")

    # Save numerical results
    results_file = output_dir / f"{filename_base}_force_metrics.txt"
    with open(results_file, 'w') as f:
        f.write("Force Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mode: {args.mode}\n")
        if args.mode == 'prior-only':
            f.write(f"Prior type: {prior_type}\n")
        f.write(f"Parameters: {args.params if args.params else 'None (config priors)'}\n")
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
