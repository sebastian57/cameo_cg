"""
Type definitions for chemtrain clean code base.

Provides type aliases and TypedDict classes for better type safety.
"""

from pathlib import Path
from typing import Union, TypedDict, Dict, Optional
import jax
import jax.numpy as jnp


# =============================================================================
# Path Type Aliases
# =============================================================================

PathLike = Union[str, Path]
"""Type alias for paths - accepts both str and Path objects."""


def as_path(p: PathLike) -> Path:
    """
    Convert str or Path to Path object.

    Args:
        p: Path-like object (str or Path)

    Returns:
        Path object

    Example:
        >>> path = as_path("data/file.npz")
        >>> path = as_path(Path("data/file.npz"))
    """
    return Path(p) if not isinstance(p, Path) else p


# =============================================================================
# Training Result TypedDicts
# =============================================================================

class PretrainResult(TypedDict):
    """Result from prior pre-training (LBFGS)."""
    train_loss: float
    val_loss: float
    steps: int
    converged: bool
    grad_norm: float
    loss_history: jnp.ndarray
    fitted_params: Dict[str, jnp.ndarray]


class StageResult(TypedDict):
    """Result from a training stage (AdaBelief, Yogi, etc.)."""
    train_loss: float
    val_loss: float


class TrainingResults(TypedDict, total=False):
    """Results from full training pipeline."""
    prior_pretrain: Optional[PretrainResult]
    stage1: StageResult
    stage2: Optional[StageResult]


# =============================================================================
# Evaluation Result TypedDicts
# =============================================================================

class EnergyComponents(TypedDict, total=False):
    """Energy component breakdown."""
    E_total: float
    E_prior: Optional[float]
    E_allegro: float
    # Prior energy components
    E_bond: Optional[float]
    E_angle: Optional[float]
    E_rep: Optional[float]
    E_dih: Optional[float]


class ForceComponents(TypedDict):
    """Force component breakdown."""
    F_total: jax.Array
    F_prior: Optional[jax.Array]
    F_allegro: jax.Array


class SingleFrameMetrics(TypedDict):
    """Evaluation metrics for a single frame."""
    energy: float
    force_rmse: float
    force_mae: float
    max_force_error: float
    n_real_atoms: int
    energy_components: Optional[EnergyComponents]


class BatchMetrics(TypedDict):
    """Evaluation metrics for a batch of frames."""
    n_frames: int
    mean_energy: float
    std_energy: float
    mean_force_rmse: float
    std_force_rmse: float
    mean_force_mae: float
    std_force_mae: float
    max_force_error: float


# =============================================================================
# Data TypedDicts
# =============================================================================

class DatasetDict(TypedDict):
    """Dataset structure returned by DatasetLoader."""
    R: jax.Array  # Coordinates (n_frames, n_atoms, 3)
    F: jax.Array  # Forces (n_frames, n_atoms, 3)
    mask: jax.Array  # Mask (n_frames, n_atoms)
    species: jax.Array  # Species IDs (n_frames, n_atoms)


class TopologyDict(TypedDict):
    """Topology structure with bond/angle/dihedral indices."""
    bonds: jax.Array
    angles: jax.Array
    dihedrals: jax.Array
    rep_pairs: jax.Array


# =============================================================================
# Model Parameter TypedDicts
# =============================================================================

class PriorParams(TypedDict, total=False):
    """Prior energy parameters."""
    r0: float  # Bond equilibrium distance
    kr: float  # Bond force constant
    theta0: float  # Angle equilibrium (unused)
    k_theta: float  # Angle force constant (unused)
    epsilon: float  # LJ epsilon
    sigma: float  # LJ sigma
    a: jax.Array  # Angle Fourier coefficients
    b: jax.Array  # Angle Fourier coefficients
    k_dih: jax.Array  # Dihedral force constants
    gamma_dih: jax.Array  # Dihedral phase angles


class ModelParams(TypedDict, total=False):
    """Combined model parameters."""
    prior: Optional[PriorParams]
    allegro: Dict  # Allegro params (complex nested structure)


# =============================================================================
# Config TypedDicts (optional - for future use)
# =============================================================================

class AllegroConfig(TypedDict, total=False):
    """Allegro model configuration."""
    max_ell: int
    num_layers: int
    n_radial_basis: int
    envelope_p: int
    embed_n_hidden: list
    species_embed: int
    mlp_n_hidden: int
    mlp_n_layers: int
    avg_num_neighbors: int


class OptimizerConfig(TypedDict, total=False):
    """Optimizer configuration."""
    lr: float
    peak_lr: float
    end_lr: float
    warmup_epochs: int
    decay_steps: int
    beta1: float
    beta2: float
    eps: float
    grad_clip: float
    weight_decay: float
