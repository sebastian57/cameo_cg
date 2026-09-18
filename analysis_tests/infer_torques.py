"""
Inference wrapper for the trained torque-matching model.

The model predicts forces (-dE/dR) and torques (-dE/dO projected onto SO(3))
for a system of CG beads with positions R and orientation matrices O.

Coordinate convention (from training preprocessing):
  - The training data was shifted by R_shift = [22., 22., 22.] Å into a box of
    [47., 44., 44.] Å. Pass R in the same shifted frame, or use
    `preprocess_positions` to apply the shift to raw data-frame coordinates.

Usage example:
    import sys
    sys.path.insert(0, "/e/project1/cameo/edelkoetter2/work/cameo_cg")

    from analysis_tests.infer_torques import load_model, predict, preprocess_positions
    import numpy as np

    EXPORT_DIR = (
        "/e/project1/cameo/edelkoetter2/work/cameo_cg"
        "/local_work/outputs/torque_matching_toy_model_1_fixed_oO/exports"
    )

    model, params = load_model(EXPORT_DIR)

    # R: positions in *raw* data coordinates, shape (2, 3)
    # O: orientation matrices, shape (2, 3, 3)
    R_raw = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    O     = np.eye(3, dtype=np.float32)[None].repeat(2, axis=0)  # (2,3,3) identities

    R = preprocess_positions(R_raw)   # shift into model box
    result = predict(model, params, R, O)
    print("Energy :", result["energy"])
    print("Forces :", result["forces"])   # shape (2, 3)
    print("Torques:", result["torques"])  # shape (2, 3)
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Training-time preprocessing constants (from dataset + config)
# ---------------------------------------------------------------------------
# Box extent computed from the training dataset (gb_data_only_orientation_1_fixed.npz)
BOX   = np.array([47.0, 44.0, 44.0], dtype=np.float32)
# Shift that maps raw dataset coordinates into [0, BOX] frame
R_SHIFT = np.array([22.0, 22.0, 22.0], dtype=np.float32)
# Both beads are species-1; N_max = 2
DEFAULT_SPECIES = np.array([1, 1], dtype=np.int32)
DEFAULT_MASK    = np.array([1, 1], dtype=np.float32)


def preprocess_positions(R_raw: np.ndarray) -> np.ndarray:
    """
    Apply the same coordinate shift used during training.

    Args:
        R_raw: Positions in the original data-file frame, shape (N, 3).

    Returns:
        R shifted into the model box, shape (N, 3).
    """
    return (np.asarray(R_raw, dtype=np.float32) + R_SHIFT)


def load_model(export_dir: str):
    """
    Build the CombinedModel and load trained parameters from the export directory.

    Args:
        export_dir: Path to the exports folder that contains
                    ``*_config.yaml`` and ``*_params.pkl``.

    Returns:
        (model, params) ready for use with `predict`.
    """
    export_dir = Path(export_dir)

    # --- find config and params files ---
    config_files = sorted(export_dir.glob("*_config.yaml"))
    params_files = sorted(export_dir.glob("*_params.pkl"))
    if not config_files:
        raise FileNotFoundError(f"No *_config.yaml found in {export_dir}")
    if not params_files:
        raise FileNotFoundError(f"No *_params.pkl found in {export_dir}")
    config_path = config_files[-1]
    params_path = params_files[-1]

    # --- project root on sys.path ---
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.jax_setup import apply_jax_compat_shims
    apply_jax_compat_shims()

    from config.manager import ConfigManager
    from models.combined_model import CombinedModel

    jax.config.update("jax_enable_x64", False)

    config = ConfigManager(str(config_path))

    # R0: two beads within cutoff inside the box, used only to allocate neighbor list
    R0 = np.array([[22.0, 22.0, 22.0],
                   [22.0, 22.0, 25.5]], dtype=np.float32)
    O0 = np.stack([np.eye(3, dtype=np.float32)] * 2)  # (2, 3, 3)

    model = CombinedModel(
        config=config,
        R0=R0,
        O0=O0,
        box=BOX,
        species=DEFAULT_SPECIES,
        N_max=2,
    )

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    return model, params


def predict(
    model,
    params: Dict[str, Any],
    R: np.ndarray,
    O: np.ndarray,
    box: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    species: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run the trained model on a single frame and return energy, forces, and torques.

    Args:
        model:   CombinedModel instance returned by `load_model`.
        params:  Trained parameters returned by `load_model`.
        R:       Bead positions in the model box frame, shape (N, 3).
                 Use `preprocess_positions` to convert raw dataset coordinates.
        O:       Orientation matrices, shape (N, 3, 3). Each O[i] is a
                 rotation matrix whose columns are the body-frame axes of bead i.
        box:     Box dimensions (3,). Defaults to training box [47, 44, 44].
        mask:    Validity mask (N,). 1 for real beads, 0 for padding.
                 Defaults to all-ones (both beads are real).
        species: Species IDs (N,). Defaults to [1, 1] (from training data).

    Returns:
        dict with:
            "energy"  : float — total potential energy
            "forces"  : ndarray (N, 3) — forces on each bead (-dE/dR)
            "torques" : ndarray (N, 3) — torques on each bead (-dE/dO projected)
    """
    R       = jnp.asarray(R,       dtype=jnp.float32)
    O       = jnp.asarray(O,       dtype=jnp.float32)
    box_arr = jnp.asarray(box if box is not None else BOX,             dtype=jnp.float32)
    mask_arr= jnp.asarray(mask if mask is not None else DEFAULT_MASK,  dtype=jnp.float32)
    sp_arr  = jnp.asarray(species if species is not None else DEFAULT_SPECIES, dtype=jnp.int32)

    def energy_R(R_):
        return model.compute_energy(params, R_, mask_arr, sp_arr, box=box_arr, O=O)

    def energy_O(O_):
        return model.compute_energy(params, R,  mask_arr, sp_arr, box=box_arr, O=O_)

    energy  = float(energy_R(R))
    forces  = -jax.grad(energy_R)(R)

    # Torque: τ_i = -∑_col  O[:,col,:] × (dE/dO)[:,col,:]
    dU_dO   = jax.grad(energy_O)(O)
    torques = -jnp.cross(
        O.transpose(0, 2, 1),      # (N, 3_col, 3_spatial)
        dU_dO.transpose(0, 2, 1),  # (N, 3_col, 3_spatial)
        axis=-1,
    ).sum(axis=1)                  # sum over columns → (N, 3)

    return {
        "energy":  energy,
        "forces":  np.asarray(forces),
        "torques": np.asarray(torques),
    }
