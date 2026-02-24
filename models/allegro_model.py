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


def _resolve_compute_dtype(config) -> tuple[str, jnp.dtype]:
    """Resolve model compute dtype from config."""
    name = str(config.get_compute_dtype()).lower()
    if name == "bfloat16":
        return name, jnp.bfloat16
    return "float32", jnp.float32


def _resolve_mlp_activation(value, allow_linear: bool = False) -> tuple[str, Any]:
    """Resolve activation from config value (string/callable; optional linear)."""
    if callable(value):
        return getattr(value, "__name__", "callable"), value

    raw = str(value).strip().lower()
    aliases = {
        "swish": "silu",
        "none": "linear",
        "identity": "linear",
        "off": "linear",
        "false": "linear",
        "null": "linear",
    }
    raw = aliases.get(raw, raw)

    if allow_linear and raw == "linear":
        return "linear", None

    options = {
        "mish": jax.nn.mish,
        "silu": jax.nn.silu,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "elu": jax.nn.elu,
        "tanh": jnp.tanh,
    }
    if raw not in options:
        extra = ", linear" if allow_linear else ""
        supported = ", ".join(sorted(options.keys()))
        raise ValueError(
            f"Unsupported model.allegro.mlp_activation='{value}'. "
            f"Supported values: {supported}{extra}."
        )
    return raw, options[raw]


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
        self.compute_dtype_name, self.compute_dtype = _resolve_compute_dtype(config)
        self.remat_level = int(config.get_remat_level())
        self.remat_policy = str(config.get_remat_policy())

        # Model parameters from config
        self.cutoff = config.get_cutoff()
        self.dr_threshold = config.get_dr_threshold()

        # Get Allegro hyperparameters
        # Support different model sizes: default, large, med
        allegro_size = config.get_allegro_size()
        self.allegro_config = config.get_allegro_config(size=allegro_size)
        self._pad_spacing = jnp.asarray(
            self.cutoff + self.dr_threshold + 1.0, dtype=self.compute_dtype
        )

        # Note: padded positions are computed on-the-fly in _spread_padded_coordinates
        # so that the method is compatible with symbolic/abstract shapes during jax.export().

        model_logger.info(f"Using Allegro size: {allegro_size}")
        model_logger.info(f"Allegro compute dtype: {self.compute_dtype_name}")
        model_logger.info(
            f"Allegro remat policy: level={self.remat_level}, policy={self.remat_policy}"
        )

        # Setup JAX-MD displacement and neighbor list
        self.displacement, self.shift = space.free()

        # Match neighbor-list math to selected compute dtype.
        safe_box = jnp.asarray(box, dtype=self.compute_dtype)

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
        R0_safe = self._spread_padded_coordinates(
            jnp.asarray(R0, dtype=self.compute_dtype), init_padded_mask
        )

        # Allocate initial neighbor list
        self.nbrs_init = self.nneigh_fn.allocate(R0_safe, extra_capacity=64)

        # Compute actual average number of neighbors from the initial neighbor list
        # and use it instead of the hardcoded config value. The config value is often
        # wrong (copy-pasted from other models/cutoffs), which mis-scales Allegro's
        # many-body interaction output. Make a mutable copy of the dict first.
        self.allegro_config = dict(self.allegro_config)
        hidden_raw = self.allegro_config.get(
            "mlp_hidden_activation",
            self.allegro_config.get("mlp_activation", "mish"),
        )
        output_raw = self.allegro_config.get("mlp_output_activation", "linear")
        self.mlp_hidden_activation_name, self.mlp_hidden_activation = _resolve_mlp_activation(
            hidden_raw
        )
        self.mlp_output_activation_name, self.mlp_output_activation = _resolve_mlp_activation(
            output_raw, allow_linear=True
        )
        # Keep backward compatibility with chemutils kwargs naming.
        self.allegro_config["mlp_activation"] = self.mlp_hidden_activation
        self.allegro_config["mlp_output_activation"] = self.mlp_output_activation
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
        model_logger.info(
            f"Allegro MLP activations: hidden={self.mlp_hidden_activation_name}, "
            f"output={self.mlp_output_activation_name}"
        )

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
        self._apply_allegro_for_training = self.apply_allegro
        if self.remat_level > 0:
            # Coarse remat boundary around full model apply.
            self._apply_allegro_for_training = jax.checkpoint(self.apply_allegro)

        # Store initialization parameters
        self._R0 = R0_safe
        self._species0 = species_safe

    def _spread_padded_coordinates(self, R: jax.Array, padded_mask: jax.Array) -> jax.Array:
        """
        Place padded atoms far apart from all atoms so they cannot form spurious edges.

        Safe positions are computed from R.shape[0] rather than from a stored concrete array,
        so this method is compatible with abstract/symbolic tracing during jax.export().
        During normal JIT-compiled training R.shape[0] is concrete and JAX constant-folds this.
        """
        n = R.shape[0]
        dtype = getattr(R, "dtype", jnp.float32)
        idx = jnp.arange(n, dtype=dtype)
        pad_spacing = jnp.asarray(self._pad_spacing, dtype=dtype)
        base = jnp.asarray(1e6, dtype=dtype)
        safe_positions = jnp.stack(
            [
                base + idx * pad_spacing,
                jnp.full((n,), base, dtype=dtype),
                jnp.full((n,), base, dtype=dtype),
            ],
            axis=1,
        )
        return jnp.where(padded_mask[:, None], safe_positions, R)

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

        ref_position = getattr(nbrs, "reference_position", None)
        target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
        nbrs = self.nneigh_fn.update(jnp.asarray(R, dtype=target_dtype), nbrs)
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
        # Run coordinate and neighbor-list path in selected compute precision.
        R_base = jnp.asarray(R, dtype=self.compute_dtype)
        padded_mask = jnp.logical_not(valid_mask)
        R_safe = self._spread_padded_coordinates(R_base, padded_mask)
        R_masked = jnp.where(valid_mask[:, None], R_base, jax.lax.stop_gradient(R_safe))

        # Reuse/update a neighbor list for training. During MLIR export, the
        # exporter may pass a graph-derived sparse NeighborList shell where
        # `error` is None; calling jax_md update() on that object fails.
        if neighbor is None:
            base_nbrs = self.nbrs_init
            ref_position = getattr(base_nbrs, "reference_position", None)
            target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
            nbrs = self.nneigh_fn.update(jnp.asarray(R_masked, dtype=target_dtype), base_nbrs)
        else:
            nbr_error = getattr(neighbor, "error", None)
            if nbr_error is None:
                # Export path: use the provided graph connectivity as-is.
                nbrs = neighbor
            else:
                ref_position = getattr(neighbor, "reference_position", None)
                target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
                nbrs = self.nneigh_fn.update(jnp.asarray(R_masked, dtype=target_dtype), neighbor)

        # Ensure species are valid (masked atoms -> species 0)
        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)
        # Keep model input in compute precision while leaving neighbor indices untouched.
        R_model = jnp.asarray(R_masked, dtype=self.compute_dtype)

        # Compute energy
        E_allegro = self._apply_allegro_for_training(
            params, R_model, nbrs, species_masked, mask=valid_mask.astype(jnp.bool_)
        )
        # Keep scalar losses/reductions in float32 for numerical stability.
        return jnp.asarray(E_allegro, dtype=jnp.float32)

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
