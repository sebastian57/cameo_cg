"""
Combined Prior + ML Model

Composes physics-based prior energy with an ML model (Allegro, MACE, or PaiNN).
Supports pure ML, pure prior, or combined training via config.
"""

import jax
import jax.numpy as jnp
import inspect
from typing import Dict, Any, Optional

from config.types import EnergyComponents, ForceComponents
from .base_model import get_ml_model_class
from .prior_energy import PriorEnergy
from .topology import TopologyBuilder
from utils.logging import model_logger

# Eagerly import standard backends so their @register_ml_model fires.
# cuEq variants are registered on import of allegro_cueq_model (lazy).
from . import allegro_model as _am  # noqa: F401
from . import mace_model as _mm  # noqa: F401
from . import painn_model as _pm  # noqa: F401


class CombinedModel:
    """
    Combined model with prior energy and ML (Allegro, MACE, or PaiNN) terms.

    Can operate in two modes (controlled by config):
    1. use_priors=True: Prior + ML (default)
    2. use_priors=False: Pure ML only

    The ML backbone is selected via config `model.ml_model`:
    - "allegro" (default): Allegro equivariant neural network
    - "allegro_cuEq" / "allegro_cueq": Allegro with cuEquivariance backend
    - "allegro_cueq_fast": Allegro with cuEquivariance fast backend
    - "mace": MACE equivariant neural network
    - "painn": PaiNN polarizable interaction neural network

    Example:
        >>> config = ConfigManager("config.yaml")
        >>> model = CombinedModel(config, R0, box, species0, N_max)
        >>> params = model.initialize_params(jax.random.PRNGKey(0))
        >>> energy = model.compute_energy(params, R, mask, species)
        >>> components = model.compute_components(params, R, mask, species)
    """

    def __init__(self, config, R0: jax.Array, box: jax.Array, species: jax.Array, N_max: int,
                 init_mask: Optional[jax.Array] = None,
                 prior_only: bool = False, n_species_override: Optional[int] = None,
                 id_to_aa: Optional[Dict[int, str]] = None):
        """
        Initialize combined model.

        Args:
            config: ConfigManager instance
            R0: Initial coordinates, shape (n_atoms, 3)
            box: Simulation box dimensions, shape (3,)
            species: Species IDs, shape (n_atoms,)
            N_max: Maximum number of atoms
            prior_only: If True, skip ML computation entirely (only compute priors)
            n_species_override: Optional global species cardinality used to
                force a consistent embedding size across datasets/buckets.
            id_to_aa: Optional species->resname mapping from dataset metadata,
                used by typed prior terms (DH/stickiness/salt_bridge).
        """
        self.config = config
        self.N_max = N_max
        self.prior_only = prior_only
        self.use_priors = config.use_priors()
        self.train_priors = config.train_priors_enabled()
        self.topology = TopologyBuilder(
            N_max=N_max,
            min_repulsive_sep=config.get_min_repulsive_sep(),
        )
        self.ml_model_type = config.get_ml_model_type()

        # cuEq variants need lazy import to trigger registration
        if self.ml_model_type in ("allegro_cueq", "allegro_cueq_fast"):
            from . import allegro_cueq_model as _cueq  # noqa: F401

        ModelClass = get_ml_model_class(self.ml_model_type)
        ml_kwargs = {
            "n_species_override": n_species_override,
        }
        if "init_mask" in inspect.signature(ModelClass.__init__).parameters:
            ml_kwargs["init_mask"] = init_mask
        self.ml_model = ModelClass(
            config, R0, box, species, N_max,
            **ml_kwargs,
        )
        model_logger.info(f"ML backbone: {self.ml_model_type}")

        if self.use_priors:
            self.prior = PriorEnergy(
                config, self.topology, self.ml_model.displacement, id_to_aa=id_to_aa
            )
            model_logger.info(f"Mode: Prior + {self.ml_model_type.upper()}")
            model_logger.info(f"Prior weights: {self.prior.weights}")
        else:
            self.prior = None
            model_logger.info(f"Mode: Pure {self.ml_model_type.upper()} (no priors)")

    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Initialize model parameters.

        Returns:
            Dictionary with 'ml' (ML backbone params) and optionally 'prior'.
        """
        params = {
            'ml': self.ml_model.initialize_params(rng_key),
        }

        if self.use_priors:
            params['prior'] = self.prior.params

        return params

    def compute_energy(
        self,
        params: Dict[str, Any],
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute total energy (ML + Prior if enabled, or prior-only).

        Args:
            params: Model parameters dict
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)
            segment_id: Optional segment IDs used to preserve disconnected
                packed structures in tiled mode.

        Returns:
            Total energy (scalar)
        """
        if self.prior_only:
            if not self.use_priors:
                raise ValueError("prior_only=True requires use_priors=True in config")
            # Block gradient flow through padded atom coordinates.
            # Start from a fully detached copy, then re-attach gradients only for
            # valid atoms.  This avoids allocating a second full-size array.
            R_detached = jax.lax.stop_gradient(R)
            mask_3d = mask[:, None]
            R_masked = jnp.where(mask_3d > 0, R, R_detached)
            if self.train_priors and "prior" in params:
                return self.prior.compute_total_energy(
                    R_masked, mask, species=species, params=params["prior"]
                )
            else:
                return self.prior.compute_total_energy(R_masked, mask, species=species)

        E_ml = self.ml_model.compute_energy(
            params['ml'], R, mask, species, neighbor, segment_id=segment_id
        )

        if self.use_priors:
            # Stop gradient flow through padded atoms, re-attach only for valid ones

            R_detached = jax.lax.stop_gradient(R)
            mask_3d = mask[:, None]
            R_masked = jnp.where(mask_3d > 0, R, R_detached)
            if self.train_priors and "prior" in params:
                E_prior = self.prior.compute_total_energy(
                    R_masked, mask, species=species, params=params["prior"]
                )
            else:
                E_prior = self.prior.compute_total_energy(R_masked, mask, species=species)
            return E_ml + E_prior
        else:
            return E_ml

    def compute_total_energy(
        self,
        params: Dict[str, Any],
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute total energy (alias for compute_energy for compatibility).

        This method exists for backward compatibility with the exporter,
        which expects compute_total_energy() method.

        Args:
        """
        return self.compute_energy(
            params, R, mask, species, neighbor, segment_id=segment_id
        )

    def compute_components(
        self,
        params: Dict[str, Any],
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None
    ) -> EnergyComponents:
        """
        Compute energy breakdown for analysis.

        Args:
            params: Model parameters
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)

        Returns:
            Dictionary with energy components:
                - E_total: Total energy
                - E_ml: ML energy (0.0 if prior_only)
                - E_bond: Bond energy (if use_priors)
                - E_angle: Angle energy (if use_priors)
                - E_repulsive: Repulsive energy (if use_priors)
                - E_dihedral: Dihedral energy (if use_priors)
                - E_prior_total: Total prior energy (if use_priors)
        """
        if self.prior_only:
            E_ml = 0.0
        else:
            E_ml = self.ml_model.compute_energy(
                params['ml'], R, mask, species, neighbor
            )

        components = {
            "E_ml": E_ml,
        }

        if self.use_priors:
            R_detached = jax.lax.stop_gradient(R)
            mask_3d = mask[:, None]
            R_masked = jnp.where(mask_3d > 0, R, R_detached)
            if self.train_priors and "prior" in params:
                prior_components = self.prior.compute_energy(
                    R_masked, mask, species=species, params=params["prior"]
                )
            else:
                prior_components = self.prior.compute_energy(R_masked, mask, species=species)
            components.update({
                "E_bond": prior_components["E_bond"],
                "E_angle": prior_components["E_angle"],
                "E_repulsive": prior_components["E_repulsive"],
                "E_dihedral": prior_components["E_dihedral"],
                "E_excluded_volume": prior_components["E_excluded_volume"],
                "E_dh": prior_components.get("E_dh", 0.0),
                "E_stickiness": prior_components.get("E_stickiness", 0.0),
                "E_salt_bridge": prior_components.get("E_salt_bridge", 0.0),
                "E_prior_total": prior_components["E_total"],
            })
            components["E_total"] = E_ml + prior_components["E_total"]
        else:
            components["E_total"] = E_ml

        return components

    def compute_force_components(
        self,
        params: Dict[str, Any],
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array
    ) -> ForceComponents:
        """
        Compute force breakdown via autodiff.

        Uses jax.vjp to perform ONE forward pass through the model, then runs a
        separate backward pass per component.  This replaces the previous approach
        of calling jax.grad N times (each of which triggered a full forward pass),
        reducing forward-pass cost from O(N) to O(1).

        Args:
            params: Model parameters
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)

        Returns:
            Dictionary with force components:
                - F_total: Total forces
                - F_ml: ML forces
                - F_bond, F_angle, F_repulsive, F_dihedral, F_excluded_volume (if use_priors)
        """
        if self.use_priors:
            def all_energies(R_):
                comps = self.compute_components(params, R_, mask, species)
                return (
                    comps["E_total"],
                    comps["E_ml"],
                    comps["E_bond"],
                    comps["E_angle"],
                    comps["E_repulsive"],
                    comps["E_dihedral"],
                    comps["E_excluded_volume"],
                )

            # Single forward pass; vjp_fn holds stored residuals for backward.
            _, vjp_fn = jax.vjp(all_energies, R)

            # Each vjp_fn call is a backward-only pass (no re-forward).
            def _force(idx, n=7):
                ct = tuple(1.0 if i == idx else 0.0 for i in range(n))
                return -vjp_fn(ct)[0]

            return {
                "F_total":           _force(0),
                "F_ml":              _force(1),
                "F_bond":            _force(2),
                "F_angle":           _force(3),
                "F_repulsive":       _force(4),
                "F_dihedral":        _force(5),
                "F_excluded_volume": _force(6),
            }
        else:
            def all_energies(R_):
                comps = self.compute_components(params, R_, mask, species)
                return comps["E_total"], comps["E_ml"]

            _, vjp_fn = jax.vjp(all_energies, R)

            return {
                "F_total": -vjp_fn((1.0, 0.0))[0],
                "F_ml":    -vjp_fn((0.0, 1.0))[0],
            }

    def energy_fn_template(self, params: Dict[str, Any]):
        """
        Create energy function template for chemtrain ForceMatching.

        This returns a function that can be used with chemtrain's trainer.

        Args:
            params: Model parameters

        Returns:
            Energy function: (R, neighbor, **kwargs) -> scalar energy
        """
        def energy_fn(R: jax.Array, neighbor: Any, **kwargs) -> jax.Array:
            mask = kwargs["mask"]
            species = kwargs["species"]
            segment_id = kwargs.get("segment_id")

            species = jnp.where(mask > 0, species, 0).astype(jnp.int32)

            E = self.compute_energy(
                params, R, mask, species, neighbor=neighbor, segment_id=segment_id
            )
            return E

        return energy_fn

    @property
    def initial_neighbors(self) -> Any:
        """Get initial neighbor list for training."""
        return self.ml_model.initial_neighbors

    @property
    def displacement(self):
        """Get displacement function (from ML model)."""
        return self.ml_model.displacement

    @property
    def nneigh_fn(self):
        """Get neighbor list function (from ML model)."""
        return self.ml_model.nneigh_fn

    def __repr__(self) -> str:
        ml = self.ml_model_type.upper()
        mode = f"Prior+{ml}" if self.use_priors else f"Pure{ml}"
        return f"CombinedModel(mode={mode}, N_max={self.N_max})"
