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
        self._pad_spacing = jnp.asarray(self.cutoff + self.dr_threshold + 1.0, dtype=jnp.float32)

        # Precompute the final padded-atom parking positions (constant for given N_max).
        n = N_max
        idx = jnp.arange(n, dtype=jnp.float32)
        offsets = jnp.stack(
            [idx * self._pad_spacing, jnp.zeros(n), jnp.zeros(n)], axis=1
        )
        self._padded_positions = jnp.asarray([1e6, 1e6, 1e6], dtype=jnp.float32) + offsets

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

        # Sanitize padded entries for Allegro initialization.
        species_arr = jnp.asarray(species)
        init_padded_mask = species_arr < 0
        R0_safe = self._spread_padded_coordinates(jnp.asarray(R0, dtype=jnp.float32), init_padded_mask)

        # Allocate initial neighbor list
        self.nbrs_init = self.nneigh_fn.allocate(R0_safe, extra_capacity=64)

        # Compute actual average number of neighbors from the initial neighbor list
        # and use it instead of the hardcoded config value. The config value is often
        # wrong (copy-pasted from other models/cutoffs), which mis-scales Allegro's
        # many-body interaction output. Make a mutable copy of the dict first.
        self.allegro_config = dict(self.allegro_config)
        # Optional graph-cap controls from YAML:
        #   model.allegro.max_edge_multiplier: float (default 1.25)
        #   model.allegro.max_edges: int (default None -> inferred)
        self.max_edge_multiplier = float(self.allegro_config.pop("max_edge_multiplier", 1.25))
        max_edges_cfg = self.allegro_config.pop("max_edges", None)
        self.max_edges = None if max_edges_cfg is None else int(max_edges_cfg)
        n_atoms = int(R0_safe.shape[0])
        valid_neighbor_slots = (self.nbrs_init.idx >= 0) & (self.nbrs_init.idx < n_atoms)
        actual_avg_neighbors = float(jnp.mean(jnp.sum(valid_neighbor_slots, axis=-1).astype(jnp.float32)))
        config_avg = self.allegro_config.get("avg_num_neighbors", 12)
        if abs(actual_avg_neighbors - config_avg) > 2.0:
            model_logger.warning(
                f"avg_num_neighbors: config={config_avg}, "
                f"computed from data={actual_avg_neighbors:.1f}. Using computed value."
            )
        self.allegro_config["avg_num_neighbors"] = actual_avg_neighbors
        model_logger.info(f"avg_num_neighbors = {actual_avg_neighbors:.1f} (computed from initial neighbor list)")
        if self.max_edges is not None:
            model_logger.info(
                f"Using configured Allegro max_edges={self.max_edges} "
                f"(max_edge_multiplier={self.max_edge_multiplier:.3f})"
            )
        else:
            model_logger.info(
                f"Using inferred Allegro max_edges "
                f"(max_edge_multiplier={self.max_edge_multiplier:.3f})"
            )

        # Determine number of species
        species_safe = jnp.where(species_arr >= 0, species_arr, 0).astype(jnp.int32)
        self.n_species = int(jnp.max(species_safe)) + 1

        model_logger.info(f"Detected {self.n_species} unique species")
        model_logger.info(f"Using Allegro config size: {allegro_size}")

        # Initialize Allegro model
        self.init_allegro, self.apply_allegro = allegro_neighborlist_pp(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=R0_safe,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=self.max_edge_multiplier,
            max_edges=self.max_edges,
            mode="energy",
            **self.allegro_config
        )

        # Store initialization parameters
        self._R0 = R0_safe
        self._species0 = species_safe

    def _spread_padded_coordinates(self, R: jax.Array, padded_mask: jax.Array) -> jax.Array:
        """
        Place padded atoms far apart from all atoms so they cannot form spurious edges.

        Uses self._padded_positions (precomputed at init) — a single jnp.where per call.
        """
        return jnp.where(padded_mask[:, None], self._padded_positions, R)

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
            nbrs = self.nbrs_init

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
        # Apply mask to coordinates and spread padded atoms away from all real atoms.
        valid_mask = mask > 0
        padded_mask = jnp.logical_not(valid_mask)
        R_safe = self._spread_padded_coordinates(R, padded_mask)
        R_masked = jnp.where(valid_mask[:, None], R, jax.lax.stop_gradient(R_safe))

        # Reuse the pre-allocated neighbor list structure; only update positions.
        # Avoids expensive O(N^2) allocate() on every forward pass.
        nbrs = neighbor if neighbor is not None else self.nbrs_init
        nbrs = self.nneigh_fn.update(R_masked, nbrs)

        # Ensure species are valid (masked atoms -> species 0)
        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)

        # Compute energy
        E_allegro = self.apply_allegro(
            params, R_masked, nbrs, species_masked, mask=valid_mask.astype(jnp.bool_)
        )

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
    def model_apply_fn(self):
        """Get raw Haiku apply function (for exporter compatibility)."""
        return self.apply_allegro

    @property
    def initial_neighbors(self) -> Any:
        """Get initial neighbor list for training setup."""
        return self.nbrs_init

    def __repr__(self) -> str:
        return (
            f"AllegroModel(cutoff={self.cutoff}, n_species={self.n_species}, "
            f"N_max={self.N_max})"
        )
