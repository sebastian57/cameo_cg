"""
Model Evaluator for Single-Point Analysis

Evaluates trained models on individual frames or batches.
Computes energy components, forces, and error metrics.

Extracted from:
- compute_single_multi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from config.types import SingleFrameMetrics, BatchMetrics
from utils.logging import eval_logger


class Evaluator:
    """
    Evaluator for trained CG protein force field models.

    Computes energy, forces, and error metrics for model analysis.

    Example:
        >>> evaluator = Evaluator(model, params, config)
        >>> results = evaluator.evaluate_frame(R, F_ref, mask, species)
        >>> print(f"Energy: {results['energy']:.4f}")
        >>> print(f"Force RMSE: {results['force_rmse']:.4f}")
    """

    def __init__(self, model, params: Dict[str, Any], config):
        """
        Initialize evaluator.

        Args:
            model: CombinedModel instance
            params: Trained model parameters
            config: ConfigManager instance
        """
        self.model = model
        self.params = params
        self.config = config

    def evaluate_frame(
        self,
        R: jax.Array,
        F_ref: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None
    ) -> SingleFrameMetrics:
        """
        Evaluate model on a single frame.

        Args:
            R: Coordinates, shape (n_atoms, 3)
            F_ref: Reference forces, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)

        Returns:
            Dictionary with:
                - energy: Total energy
                - energy_components: Dict of energy breakdown
                - forces: Predicted forces
                - force_components: Dict of force breakdown (if use_priors)
                - force_rmse: RMSE on real atoms
                - force_mae: MAE on real atoms
                - max_force_error: Maximum force error magnitude
                - n_real_atoms: Number of real (non-padded) atoms
        """
        # Compute energy components
        energy_components = self.model.compute_components(
            self.params, R, mask, species, neighbor
        )

        # Compute forces via autodiff
        def energy_fn(R_):
            return self.model.compute_energy(
                self.params, R_, mask, species, neighbor
            )

        F_pred = -jax.grad(energy_fn)(R)

        # Compute force components if using priors
        if self.model.use_priors:
            force_components = self.model.compute_force_components(
                self.params, R, mask, species
            )
        else:
            force_components = {
                "F_total": F_pred,
                "F_allegro": F_pred,
            }

        # Compute errors on real atoms only
        real_mask = mask > 0
        F_pred_real = F_pred[real_mask]
        F_ref_real = F_ref[real_mask]

        force_diff = F_pred_real - F_ref_real
        force_error_magnitude = jnp.linalg.norm(force_diff, axis=-1)

        rmse = float(jnp.sqrt(jnp.mean(force_diff ** 2)))
        mae = float(jnp.mean(jnp.abs(force_diff)))
        max_error = float(jnp.max(force_error_magnitude))

        return {
            "energy": float(energy_components["E_total"]),
            "energy_components": {k: float(v) for k, v in energy_components.items()},
            "forces": F_pred,
            "force_components": {k: v for k, v in force_components.items()},
            "force_rmse": rmse,
            "force_mae": mae,
            "max_force_error": max_error,
            "n_real_atoms": int(jnp.sum(real_mask)),
        }

    def evaluate_batch(
        self,
        R_batch: jax.Array,
        F_batch: jax.Array,
        mask_batch: jax.Array,
        species_batch: jax.Array
    ) -> BatchMetrics:
        """
        Evaluate model on a batch of frames.

        Args:
            R_batch: Coordinates, shape (n_frames, n_atoms, 3)
            F_batch: Reference forces, shape (n_frames, n_atoms, 3)
            mask_batch: Validity masks, shape (n_frames, n_atoms)
            species_batch: Species IDs, shape (n_frames, n_atoms)

        Returns:
            Dictionary with aggregated statistics
        """
        n_frames = R_batch.shape[0]

        energies = []
        force_rmses = []
        force_maes = []
        max_errors = []

        for i in range(n_frames):
            result = self.evaluate_frame(
                R_batch[i], F_batch[i], mask_batch[i], species_batch[i]
            )
            energies.append(result["energy"])
            force_rmses.append(result["force_rmse"])
            force_maes.append(result["force_mae"])
            max_errors.append(result["max_force_error"])

        return {
            "n_frames": n_frames,
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "mean_force_rmse": float(np.mean(force_rmses)),
            "std_force_rmse": float(np.std(force_rmses)),
            "mean_force_mae": float(np.mean(force_maes)),
            "std_force_mae": float(np.std(force_maes)),
            "max_force_error": float(np.max(max_errors)),
            "energies": energies,
            "force_rmses": force_rmses,
            "force_maes": force_maes,
        }

    def print_summary(self, results: Dict[str, Any], frame_idx: Optional[int] = None):
        """
        Print evaluation results summary.

        Args:
            results: Results from evaluate_frame() or evaluate_batch()
            frame_idx: Frame index (optional, for single frame)
        """
        eval_logger.info("\n" + "="*60)
        if frame_idx is not None:
            eval_logger.info(f"Evaluation Results - Frame {frame_idx}")
        else:
            eval_logger.info("Evaluation Results")
        eval_logger.info("="*60)

        if "n_frames" in results:
            # Batch results
            eval_logger.info(f"\nBatch Statistics ({results['n_frames']} frames):")
            eval_logger.info(f"  Energy: {results['mean_energy']:.4f} ± {results['std_energy']:.4f}")
            eval_logger.info(f"  Force RMSE: {results['mean_force_rmse']:.4f} ± {results['std_force_rmse']:.4f}")
            eval_logger.info(f"  Force MAE: {results['mean_force_mae']:.4f} ± {results['std_force_mae']:.4f}")
            eval_logger.info(f"  Max Force Error: {results['max_force_error']:.4f}")
        else:
            # Single frame results
            eval_logger.info(f"\nEnergy: {results['energy']:.6f}")

            if "energy_components" in results:
                eval_logger.info("\nEnergy Components:")
                for key, val in results["energy_components"].items():
                    eval_logger.info(f"  {key}: {val:.6f}")

            eval_logger.info(f"\nForce Metrics (n_atoms={results['n_real_atoms']}):")
            eval_logger.info(f"  RMSE: {results['force_rmse']:.6f}")
            eval_logger.info(f"  MAE: {results['force_mae']:.6f}")
            eval_logger.info(f"  Max Error: {results['max_force_error']:.6f}")

        eval_logger.info("="*60 + "\n")

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to file.

        Args:
            results: Results dictionary
            output_path: Path to save results (NPZ or pickle)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npz":
            # Save as NPZ (good for arrays)
            np.savez(output_path, **results)
        else:
            # Save as pickle (handles mixed types)
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)

        eval_logger.info(f"Saved results to: {output_path}")

    def __repr__(self) -> str:
        return f"Evaluator(model={self.model})"
