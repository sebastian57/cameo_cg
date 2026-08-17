"""
PaiNN Equivariant Neural Network Model Wrapper

Wraps the PaiNN model initialization and inference for force field training.
Handles neighbor lists, species types, and coordinate masking.
"""

import jax
import jax.numpy as jnp
from jax_md import space, partition
from chemutils.models.painn.model import painn_neighborlist_pp
from typing import Optional, Any

from .base_model import BaseMLModel, register_ml_model, resolve_compute_dtype
from .neighborlist_utils import resolve_neighbor_list_format
from utils.logging import model_logger


@register_ml_model("painn")
class PaiNNModel(BaseMLModel):
    """Wrapper for PaiNN (Polarizable Interaction Neural Network).

    Same interface as AllegroModel/MACEModel — can be used as a drop-in
    replacement in CombinedModel.
    """

    def __init__(
        self,
        config,
        R0: jax.Array,
        box: jax.Array,
        species: jax.Array,
        N_max: int,
        n_species_override: Optional[int] = None,
    ):
        self.config = config
        self.N_max = N_max
        self.compute_dtype_name, self.compute_dtype = resolve_compute_dtype(config)

        self.cutoff = config.get_cutoff()
        self.dr_threshold = config.get_dr_threshold()
        self.neighbor_list_format_name, self.neighbor_list_format = resolve_neighbor_list_format(
            config.get_neighbor_list_format()
        )

        self.painn_config = config.get_painn_config()

        model_logger.info(f"PaiNN compute dtype: {self.compute_dtype_name}")

        self.displacement, self.shift = space.free()
        safe_box = jnp.asarray(box, dtype=jnp.float32)

        self.nneigh_fn = partition.neighbor_list(
            self.displacement,
            box=safe_box,
            r_cutoff=self.cutoff,
            dr_threshold=self.dr_threshold,
            fractional_coordinates=False,
            format=self.neighbor_list_format,
        )
        model_logger.info(f"Neighbor list format: {self.neighbor_list_format_name}")

        self.painn_config = dict(self.painn_config)
        self._neighbor_extra_capacity = int(
            self.painn_config.pop("neighbor_extra_capacity", 64)
        )
        self.max_edge_multiplier = float(
            self.painn_config.pop("max_edge_multiplier", 1.25)
        )
        self.nbrs_init = self.nneigh_fn.allocate(
            R0, extra_capacity=self._neighbor_extra_capacity
        )

        species_safe = jnp.where(jnp.asarray(species) >= 0, species, 0).astype(jnp.int32)
        n_species_data = int(jnp.max(species_safe)) + 1
        self.n_species = (
            max(n_species_data, int(n_species_override))
            if n_species_override is not None
            else n_species_data
        )

        model_logger.info(f"Detected {self.n_species} unique species")

        self.init_fn, self.apply_fn = painn_neighborlist_pp(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=R0,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=self.max_edge_multiplier,
            mode="energy",
            **self.painn_config,
        )

        self._R0 = R0
        self._species0 = species_safe

    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Any:
        return self.init_fn(rng_key, self._R0, self.nbrs_init, self._species0)

    def compute_energy(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
        box: Optional[jax.Array] = None,
    ) -> jax.Array:
        R_base = jnp.asarray(R, dtype=jnp.float32)
        mask_3d = mask[:, None]
        R_masked = jnp.where(mask_3d > 0, R_base, jax.lax.stop_gradient(R_base))

        if neighbor is None:
            nbrs = self.nneigh_fn.allocate(R_masked)
            nbrs = self.nneigh_fn.update(R_masked, nbrs)
        else:
            nbrs = neighbor

        species_masked = jnp.where(mask > 0, species, 0).astype(jnp.int32)
        R_model = jnp.asarray(R_masked, dtype=self.compute_dtype)

        E_painn = self.apply_fn(params, R_model, nbrs, species_masked)
        return jnp.asarray(E_painn, dtype=jnp.float32)

    @property
    def model_apply_fn(self):
        return self.apply_fn

    def __repr__(self) -> str:
        return f"PaiNNModel(cutoff={self.cutoff}, n_species={self.n_species}, N_max={self.N_max})"
