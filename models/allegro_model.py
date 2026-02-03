"""
Allegro Equivariant Neural Network Model Wrapper

Wraps the Allegro model initialization and inference for force field training.
Handles neighbor lists, species types, and coordinate masking.

Extracted from:
- allegro_energyfn_multiple_proteins.py
"""

import jax
import jax.numpy as jnp
from jax_md import space, partition
from chemutils.models.allegro.model import allegro_neighborlist_pp
from typing import Optional, Tuple, Any

from utils.logging import model_logger


class AllegroModel:
    """
    Wrapper for Allegro equivariant graph neural network.

    Handles:
    - Allegro model initialization from config
    - Neighbor list management
    - Species handling
    - Coordinate masking for padded systems

    Example:
        >>> config = ConfigManager("config.yaml")
        >>> model = AllegroModel(config, R0, box, species0, N_max)
        >>> params = model.initialize_params(jax.random.PRNGKey(0))
        >>> energy = model.compute_energy(params, R, mask, species)
    """

    def __init__(self, config, R0: jax.Array, box: jax.Array, species: jax.Array, N_max: int):
        """
        Initialize Allegro model.

        Args:
            config: ConfigManager instance
            R0: Initial coordinates for setup, shape (n_atoms, 3)
            box: Simulation box dimensions, shape (3,)
            species: Species IDs for atoms, shape (n_atoms,)
            N_max: Maximum number of atoms
        """
        self.config = config
        self.N_max = N_max

        # Model parameters from config
        self.cutoff = config.get_cutoff()
        self.dr_threshold = config.get_dr_threshold()

        # Get Allegro hyperparameters
        # Support different model sizes: default, large, med
        allegro_size = config.get_allegro_size()
        self.allegro_config = config.get_allegro_config(size=allegro_size)

        model_logger.info(f"Using Allegro size: {allegro_size}")

        # Setup JAX-MD displacement and neighbor list
        self.displacement, self.shift = space.free()

        # Convert box to safe float32
        safe_box = jnp.asarray(box, dtype=jnp.float32)

        self.nneigh_fn = partition.neighbor_list(
            self.displacement,
            box=safe_box,
            r_cutoff=self.cutoff,
            dr_threshold=self.dr_threshold,
            fractional_coordinates=False
        )

        # Allocate initial neighbor list
        self.nbrs_init = self.nneigh_fn.allocate(R0, extra_capacity=64)

        # Determine number of species
        self.n_species = int(jnp.max(species)) + 1
        species_safe = jnp.asarray(species, dtype=jnp.int32)

        model_logger.info(f"Detected {self.n_species} unique species")
        model_logger.info(f"Using Allegro config size: {allegro_size}")

        # Initialize Allegro model
        self.init_allegro, self.apply_allegro = allegro_neighborlist_pp(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=R0,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=1.25,
            mode="energy",
            **self.allegro_config
        )

        # Store initialization parameters
        self._R0 = R0
        self._species0 = species_safe

    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Any:
        """
        Initialize Allegro model parameters.

        Args:
            rng_key: JAX random key for initialization

        Returns:
            Allegro model parameters (pytree)
        """
        params = self.init_allegro(rng_key, self._R0, self.nbrs_init, self._species0)
        return params

    def get_neighborlist(self, R: jax.Array, nbrs: Optional[Any] = None) -> Any:
        """
        Get or update neighbor list for coordinates.

        Args:
            R: Coordinates, shape (n_atoms, 3)
            nbrs: Existing neighbor list (optional, will allocate if None)

        Returns:
            Updated neighbor list
        """
        if nbrs is None:
            nbrs = self.nneigh_fn.allocate(R)

        nbrs = self.nneigh_fn.update(R, nbrs)
        return nbrs

    def compute_energy(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None
    ) -> jax.Array:
        """
        Compute Allegro energy for given coordinates.

        Args:
            params: Allegro model parameters
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional, will compute if None)

        Returns:
            Total energy (scalar)
        """
        # Apply mask to coordinates (stop gradient for padded atoms)
        mask_3d = mask[:, None]
        R_masked = jnp.where(
            mask_3d > 0,
            R,
            jax.lax.stop_gradient(R)
        )

        # Get or update neighbor list
        if neighbor is None:
            nbrs = self.nneigh_fn.allocate(R_masked)
            nbrs = self.nneigh_fn.update(R_masked, nbrs)
        else:
            nbrs = neighbor

        # Ensure species are valid (masked atoms -> species 0)
        species_masked = jnp.where(mask > 0, species, 0).astype(jnp.int32)

        # Compute energy
        E_allegro = self.apply_allegro(params, R_masked, nbrs, species_masked)

        return E_allegro

    def compute_energy_and_forces(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute energy and forces via automatic differentiation.

        Args:
            params: Allegro model parameters
            R: Coordinates, shape (n_atoms, 3)
            mask: Validity mask, shape (n_atoms,)
            species: Species IDs, shape (n_atoms,)
            neighbor: Neighbor list (optional)

        Returns:
            energy: Total energy (scalar)
            forces: Forces, shape (n_atoms, 3)
        """
        def energy_fn(R_):
            return self.compute_energy(params, R_, mask, species, neighbor)

        E = energy_fn(R)
        F = -jax.grad(energy_fn)(R)

        return E, F

    @property
    def initial_neighbors(self) -> Any:
        """Get initial neighbor list for training setup."""
        return self.nbrs_init

    def __repr__(self) -> str:
        return (
            f"AllegroModel(cutoff={self.cutoff}, n_species={self.n_species}, "
            f"N_max={self.N_max})"
        )
