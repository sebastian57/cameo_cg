"""
Per-Residue Force Error Decomposition

Computes per-atom RMSE grouped by amino acid type and chain position.
Useful for identifying which residue types or chain regions are hardest to model.

Usage (via evaluate_forces.py):
    python scripts/evaluate_forces.py params.pkl config.yaml --mode per-residue --frames 100

Or directly:
    from evaluation.per_residue import compute_per_residue_errors, plot_per_residue_rmse
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, List, Optional


def compute_per_residue_errors(
    model,
    params: Dict[str, Any],
    dataset: Dict[str, Any],
    n_frames: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute per-atom force RMSE over n_frames randomly selected frames.

    For each atom position i:
        RMSE[i] = sqrt( mean_over_frames( ||F_pred[frame,i] - F_ref[frame,i]||^2 ) )

    Args:
        model:    CombinedModel instance (already initialized)
        params:   Model parameters dict
        dataset:  Dict with 'R', 'F', 'mask', 'species' arrays (all frames)
        n_frames: Number of frames to evaluate (randomly sampled)
        seed:     Random seed for frame selection

    Returns:
        dict with:
            'rmse_per_atom'  float32[N_max]   per-position RMSE (padded positions = 0)
            'mae_per_atom'   float32[N_max]   per-position MAE
            'species'        int32[N_max]     AA type per position (from frame 0)
            'mask'           float32[N_max]   1=real atom, 0=padding (from frame 0)
            'aa_names'       list[str]        id_to_aa mapping as a list
    """
    R_all = np.asarray(dataset["R"])
    F_all = np.asarray(dataset["F"])
    mask_all = np.asarray(dataset["mask"])
    species_all = np.asarray(dataset["species"])

    n_total = R_all.shape[0]
    N_max = R_all.shape[1]
    n_frames = min(n_frames, n_total)

    rng = np.random.RandomState(seed)
    frame_indices = rng.choice(n_total, size=n_frames, replace=False)
    frame_indices = np.sort(frame_indices)

    # Build force prediction function for a single frame
    def predict_forces(R, mask, species):
        species = jnp.where(mask > 0, species, 0).astype(jnp.int32)

        def energy_fn(R_):
            return model.compute_energy(params, R_, mask, species)

        return -jax.grad(energy_fn)(R)

    # Accumulate squared errors and absolute errors over frames
    sq_errors = np.zeros((N_max, 3), dtype=np.float64)
    abs_errors = np.zeros((N_max, 3), dtype=np.float64)
    n_counted = np.zeros(N_max, dtype=np.int32)

    for frame_idx in frame_indices:
        R = jnp.asarray(R_all[frame_idx])
        F_ref = jnp.asarray(F_all[frame_idx])
        mask = jnp.asarray(mask_all[frame_idx])
        species = jnp.asarray(species_all[frame_idx])

        F_pred = predict_forces(R, mask, species)
        err = np.asarray(F_pred) - np.asarray(F_ref)  # (N_max, 3)

        sq_errors += err ** 2
        abs_errors += np.abs(err)
        n_counted += 1

    # Per-atom RMSE and MAE (mean over 3 components and frames)
    rmse_per_atom = np.sqrt(np.mean(sq_errors / n_counted[:, None], axis=-1)).astype(np.float32)
    mae_per_atom = np.mean(abs_errors / n_counted[:, None], axis=-1).astype(np.float32)

    # Mask out padded positions
    mask_0 = mask_all[frame_indices[0]]  # use first selected frame
    rmse_per_atom = np.where(mask_0 > 0, rmse_per_atom, 0.0)
    mae_per_atom = np.where(mask_0 > 0, mae_per_atom, 0.0)

    # Species and aa_names from frame 0 (constant across frames for single protein)
    species_0 = species_all[frame_indices[0]]

    return {
        "rmse_per_atom": rmse_per_atom,
        "mae_per_atom": mae_per_atom,
        "species": species_0.astype(np.int32),
        "mask": mask_0.astype(np.float32),
        "n_frames_evaluated": n_frames,
    }


def plot_per_residue_rmse(
    results: Dict[str, Any],
    id_to_aa: Dict[int, str],
    output_path: str,
    title: str = "Per-Residue Force RMSE",
):
    """
    Generate two plots:
      1. Bar chart: mean RMSE per amino acid type (sorted descending)
      2. Line plot: RMSE vs chain position index

    Args:
        results:     Output of compute_per_residue_errors
        id_to_aa:    Mapping from species ID to AA name
        output_path: Path to save the PNG figure
        title:       Figure title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmse = results["rmse_per_atom"]
    species = results["species"]
    mask = results["mask"]

    real_idx = np.where(mask > 0)[0]
    rmse_real = rmse[real_idx]
    species_real = species[real_idx]

    # ---- Per-AA bar chart ----
    aa_ids = sorted(set(species_real.tolist()))
    aa_labels = [id_to_aa.get(i, str(i)) for i in aa_ids]
    aa_rmse = [float(np.mean(rmse_real[species_real == i])) for i in aa_ids]

    # Sort by RMSE descending
    order = np.argsort(aa_rmse)[::-1]
    aa_labels_sorted = [aa_labels[i] for i in order]
    aa_rmse_sorted = [aa_rmse[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    bars = ax.bar(range(len(aa_labels_sorted)), aa_rmse_sorted, color="steelblue")
    ax.set_xticks(range(len(aa_labels_sorted)))
    ax.set_xticklabels(aa_labels_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean RMSE (kcal/mol/Å)")
    ax.set_title("RMSE by Amino Acid Type")
    ax.bar_label(bars, fmt="%.3f", fontsize=7)

    # ---- Chain position line plot ----
    ax2 = axes[1]
    ax2.plot(np.arange(len(real_idx)), rmse_real, color="darkorange", linewidth=1.0)
    ax2.set_xlabel("Chain position index")
    ax2.set_ylabel("RMSE (kcal/mol/Å)")
    ax2.set_title("RMSE vs Chain Position")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_per_residue_txt(
    results: Dict[str, Any],
    id_to_aa: Dict[int, str],
    output_path: str,
    model_id: str = "",
):
    """Save per-AA RMSE summary to a text file."""
    rmse = results["rmse_per_atom"]
    species = results["species"]
    mask = results["mask"]

    real_idx = np.where(mask > 0)[0]
    rmse_real = rmse[real_idx]
    species_real = species[real_idx]

    aa_ids = sorted(set(species_real.tolist()))
    with open(output_path, "w") as f:
        f.write(f"Per-Residue Force RMSE — {model_id}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Frames evaluated: {results['n_frames_evaluated']}\n")
        f.write(f"Real atoms: {len(real_idx)}\n\n")
        f.write(f"{'AA':>6}  {'Mean RMSE':>12}  {'n_atoms':>8}\n")
        f.write("-" * 30 + "\n")

        rows = []
        for i in aa_ids:
            mask_aa = species_real == i
            rmse_aa = float(np.mean(rmse_real[mask_aa]))
            n_aa = int(np.sum(mask_aa))
            rows.append((rmse_aa, id_to_aa.get(i, str(i)), n_aa))

        for rmse_aa, aa_name, n_aa in sorted(rows, reverse=True):
            f.write(f"{aa_name:>6}  {rmse_aa:>12.4f}  {n_aa:>8}\n")

        f.write("-" * 30 + "\n")
        f.write(f"{'Overall':>6}  {float(np.mean(rmse_real)):>12.4f}  {len(real_idx):>8}\n")

    print(f"  Saved: {output_path}")
