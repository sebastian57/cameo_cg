"""
Combined Prior + ML Model

Composes physics-based prior energy with an ML model (Allegro, MACE, or PaiNN).
Supports pure ML, pure prior, or combined training via config.

Extracted from:
- allegro_energyfn_multiple_proteins.py
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple

from config.types import EnergyComponents, ForceComponents
from .prior_energy import PriorEnergy
from .allegro_model import AllegroModel
from .mace_model import MACEModel
from .painn_model import PaiNNModel
from .topology import TopologyBuilder
from utils.logging import model_logger


class CombinedModel:
    """
    Combined model with prior energy and ML (Allegro, MACE, or PaiNN) terms.

    Can operate in two modes (controlled by config):
    1. use_priors=True: Prior + ML (default)
    2. use_priors=False: Pure ML only

    The ML backbone is selected via config `model.ml_model`:
    - "allegro" (default): Allegro equivariant neural network
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
                 prior_only: bool = False):
        """
        Initialize combined model.

        Args:
            config: ConfigManager instance
            R0: Initial coordinates, shape (n_atoms, 3)
            box: Simulation box dimensions, shape (3,)
            species: Species IDs, shape (n_atoms,)
            N_max: Maximum number of atoms
            prior_only: If True, skip ML computation entirely (only compute priors)
        """
        self.config = config
        self.N_max = N_max
        self.prior_only = prior_only

        # Check if priors are enabled
        self.use_priors = config.use_priors()
        self.train_priors = config.train_priors_enabled()

        # Create topology builder (always needed for proper initialization)
        self.topology = TopologyBuilder(N_max=N_max, min_repulsive_sep=6)

        # Determine which ML backbone to use
        self.ml_model_type = config.get_ml_model_type()

        if self.ml_model_type == "mace":
            self.ml_model = MACEModel(config, R0, box, species, N_max)
            model_logger.info("ML backbone: MACE")
        elif self.ml_model_type == "painn":
            self.ml_model = PaiNNModel(config, R0, box, species, N_max)
            model_logger.info("ML backbone: PaiNN")
        else:
            self.ml_model = AllegroModel(config, R0, box, species, N_max)
            model_logger.info("ML backbone: Allegro")

        # Backward-compatible alias: existing code references self.allegro
        self.allegro = self.ml_model

        # Create prior model (if enabled)
        if self.use_priors:
            self.prior = PriorEnergy(config, self.topology, self.ml_model.displacement)
            model_logger.info(f"Mode: Prior + {self.ml_model_type.upper()}")
            model_logger.info(f"Prior weights: {self.prior.weights}")
        else:
            self.prior = None
            model_logger.info(f"Mode: Pure {self.ml_model_type.upper()} (no priors)")

    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Initialize model parameters.

        Returns:
            Dictionary with:
                - 'allegro': Allegro model parameters
                - 'prior': Prior parameters (if use_priors=True)
        """
        params = {
            'allegro': self.allegro.initialize_params(rng_key),
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
        neighbor: Optional[Any] = None
    ) -> jax.Array:
        """
        Compute total energy (Allegro + Prior if enabled, or prior-only).

        Args:
            params: Model parameters dict with 'allegro' and optionally 'prior'
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)

        Returns:
            Total energy (scalar)
        """
        # Prior-only mode: skip ML computation entirely
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

        # Normal mode: compute ML (and optionally add priors)
        # ML energy (has internal R_masked handling)
        E_ml = self.ml_model.compute_energy(
            params['allegro'], R, mask, species, neighbor
        )

        # Add prior energy if enabled
        if self.use_priors:
            # Block gradient flow through padded atom coordinates.
            # Start from a fully detached copy, then re-attach gradients only for
            # valid atoms.  This avoids allocating a second full-size array.
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
        neighbor: Optional[Any] = None
    ) -> jax.Array:
        """
        Compute total energy (alias for compute_energy for compatibility).

        This method exists for backward compatibility with the exporter,
        which expects compute_total_energy() method.

        Args:
            params: Model parameters dict with 'allegro' and optionally 'prior'
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)

        Returns:
            Total energy (scalar)
        """
        return self.compute_energy(params, R, mask, species, neighbor)

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
                - E_allegro: Allegro ML energy (0.0 if prior_only)
                - E_bond: Bond energy (if use_priors)
                - E_angle: Angle energy (if use_priors)
                - E_repulsive: Repulsive energy (if use_priors)
                - E_dihedral: Dihedral energy (if use_priors)
                - E_prior_total: Total prior energy (if use_priors)
        """
        # ML energy (skip if prior_only mode)
        if self.prior_only:
            E_ml = 0.0
        else:
            E_ml = self.ml_model.compute_energy(
                params['allegro'], R, mask, species, neighbor
            )

        components = {
            "E_allegro": E_ml,  # Key kept as "E_allegro" for backward compat
        }

        # Add prior components if enabled
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
                - F_allegro: Allegro forces
                - F_bond, F_angle, F_repulsive, F_dihedral, F_excluded_volume (if use_priors)
        """
        if self.use_priors:
            # Returns a 7-tuple of scalars: one per energy component.
            def all_energies(R_):
                comps = self.compute_components(params, R_, mask, species)
                return (
                    comps["E_total"],
                    comps["E_allegro"],
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
                "F_allegro":         _force(1),
                "F_bond":            _force(2),
                "F_angle":           _force(3),
                "F_repulsive":       _force(4),
                "F_dihedral":        _force(5),
                "F_excluded_volume": _force(6),
            }
        else:
            def all_energies(R_):
                comps = self.compute_components(params, R_, mask, species)
                return comps["E_total"], comps["E_allegro"]

            _, vjp_fn = jax.vjp(all_energies, R)

            return {
                "F_total":   -vjp_fn((1.0, 0.0))[0],
                "F_allegro": -vjp_fn((0.0, 1.0))[0],
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

            # Ensure species are valid for masked atoms
            species = jnp.where(mask > 0, species, 0).astype(jnp.int32)

            E = self.compute_energy(params, R, mask, species, neighbor=neighbor)
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
