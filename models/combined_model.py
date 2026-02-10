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

    def __init__(self, config, R0: jax.Array, box: jax.Array, species: jax.Array, N_max: int):
        """
        Initialize combined model.

        Args:
            config: ConfigManager instance
            R0: Initial coordinates, shape (n_atoms, 3)
            box: Simulation box dimensions, shape (3,)
            species: Species IDs, shape (n_atoms,)
            N_max: Maximum number of atoms
        """
        self.config = config
        self.N_max = N_max

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
        Compute total energy (Allegro + Prior if enabled).

        Args:
            params: Model parameters dict with 'allegro' and optionally 'prior'
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)

        Returns:
            Total energy (scalar)
        """
        # ML energy (has internal R_masked handling)
        E_ml = self.ml_model.compute_energy(
            params['allegro'], R, mask, species, neighbor
        )

        # Add prior energy if enabled
        if self.use_priors:
            # CRITICAL: Apply stop_gradient to padded atom coordinates!
            # This prevents NaN gradients from undefined geometry:
            # - For parked atoms at same location, d(norm)/dR = v/||v|| is undefined
            # - Without stop_gradient, this NaN propagates through the gradient
            # - With stop_gradient, padded atoms don't contribute to gradients
            mask_3d = mask[:, None]
            R_masked = jnp.where(
                mask_3d > 0,
                R,
                jax.lax.stop_gradient(R)  # Block gradient for padded atoms
            )
            if self.train_priors and "prior" in params:
                E_prior = self.prior.compute_total_energy(
                    R_masked, mask, params=params["prior"]
                )
            else:
                E_prior = self.prior.compute_total_energy(R_masked, mask)
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
                - E_allegro: Allegro ML energy
                - E_bond: Bond energy (if use_priors)
                - E_angle: Angle energy (if use_priors)
                - E_repulsive: Repulsive energy (if use_priors)
                - E_dihedral: Dihedral energy (if use_priors)
                - E_prior_total: Total prior energy (if use_priors)
        """
        # ML energy
        E_ml = self.ml_model.compute_energy(
            params['allegro'], R, mask, species, neighbor
        )

        components = {
            "E_allegro": E_ml,  # Key kept as "E_allegro" for backward compat
        }

        # Add prior components if enabled
        if self.use_priors:
            # Apply stop_gradient to padded atoms (same as in compute_energy)
            mask_3d = mask[:, None]
            R_masked = jnp.where(
                mask_3d > 0,
                R,
                jax.lax.stop_gradient(R)
            )
            if self.train_priors and "prior" in params:
                prior_components = self.prior.compute_energy(
                    R_masked, mask, params=params["prior"]
                )
            else:
                prior_components = self.prior.compute_energy(R_masked, mask)
            components.update({
                "E_bond": prior_components["E_bond"],
                "E_angle": prior_components["E_angle"],
                "E_repulsive": prior_components["E_repulsive"],
                "E_dihedral": prior_components["E_dihedral"],
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

        Args:
            params: Model parameters
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)

        Returns:
            Dictionary with force components:
                - F_total: Total forces
                - F_allegro: Allegro forces
                - F_bond, F_angle, F_repulsive, F_dihedral (if use_priors)
        """
        # Define energy functions for each component
        def energy_components_at_R(R_):
            return self.compute_components(params, R_, mask, species)

        def E_total_fn(R_):
            return energy_components_at_R(R_)["E_total"]

        def E_allegro_fn(R_):
            return energy_components_at_R(R_)["E_allegro"]

        # Compute forces
        F_total = -jax.grad(E_total_fn)(R)
        F_allegro = -jax.grad(E_allegro_fn)(R)

        force_components = {
            "F_total": F_total,
            "F_allegro": F_allegro,
        }

        # Add prior force components if enabled
        if self.use_priors:
            def E_bond_fn(R_):
                return energy_components_at_R(R_)["E_bond"]

            def E_angle_fn(R_):
                return energy_components_at_R(R_)["E_angle"]

            def E_rep_fn(R_):
                return energy_components_at_R(R_)["E_repulsive"]

            def E_dih_fn(R_):
                return energy_components_at_R(R_)["E_dihedral"]

            force_components.update({
                "F_bond": -jax.grad(E_bond_fn)(R),
                "F_angle": -jax.grad(E_angle_fn)(R),
                "F_repulsive": -jax.grad(E_rep_fn)(R),
                "F_dihedral": -jax.grad(E_dih_fn)(R),
            })

        return force_components

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
