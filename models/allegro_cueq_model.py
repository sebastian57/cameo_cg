"""
Allegro + cuEquivariance Model Wrapper

Drop-in replacement for AllegroModel that uses the cuEquivariance-accelerated
backend (allegro_cueq_v2.py).  Spherical harmonics and tensor products are
computed by cuequivariance_jax; everything else (neighbor lists, masking,
parameter initialisation) is identical to AllegroModel.

Two new config keys are recognised under model.allegro (or model.allegro_cuEq / model.allegro_cueq):
  mlp_dtype:  "bfloat16" | "float32"  (default "float32")
              Use bfloat16 inside every MLP; inputs and outputs remain float32.
  logging:    true | false             (default true)
              Emit jax.debug.print messages inside compiled model blocks.
"""

import jax
import jax.numpy as jnp
from jax_md import space, partition
from typing import Optional, Any
from jax_md_mod import custom_partition

from .base_model import BaseMLModel, register_ml_model, resolve_compute_dtype
from .allegro_model import _resolve_mlp_activation
from .neighborlist_utils import resolve_neighbor_list_format, compute_avg_num_neighbors
from utils.logging import model_logger

# allegro_cueq_v2 is imported lazily inside __init__ to avoid a hard dependency
# on cuequivariance at import time (non-cueq runs would fail otherwise).


def _resolve_mlp_dtype(cfg_value) -> tuple[str, jnp.dtype]:
    """Resolve MLP compute dtype from a config string or jnp.dtype."""
    if isinstance(cfg_value, str) and cfg_value.strip().lower() == "bfloat16":
        return "bfloat16", jnp.bfloat16
    return "float32", jnp.float32


def _normalize_export_tp_token(value: Any, *, mode_token: bool = False) -> str:
    """Normalize tp_method / tp_mode aliases used by export logic."""
    token = str(value).strip().lower().replace("-", "_")
    aliases = {
        "linear_1d": "uniform_1d",
        "block_linear_1d": "block_uniform_1d" if mode_token else "uniform_1d",
    }
    return aliases.get(token, token)


@register_ml_model("allegro_cueq", "allegro_cueq_fast")
class AllegroModelCuEq(BaseMLModel):
    """
    Allegro equivariant neural network backed by cuEquivariance.

    Interface is identical to AllegroModel.  Pass ``ml_model: allegro_cuEq``
    (or ``allegro_cueq``) in your config to select this backend.

    New config keys under model.allegro (or model.allegro_cuEq / model.allegro_cueq):
      mlp_dtype: "bfloat16" | "float32"   - MLP layer precision (default float32)
      logging:   true | false              - debug prints inside JIT (default true)

    Example::

        model:
          ml_model: "allegro_cuEq"
          allegro:
            max_ell: 2
            num_layers: 3
            mlp_n_hidden: 96
            mlp_n_layers: 3
            mlp_dtype: "bfloat16"   # <-- new
            logging: false           # <-- new
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
        """
        Initialize AllegroModelCuEq.

        Args:
            config:  ConfigManager instance
            R0:      Initial coordinates for neighbor-list setup, shape (n_atoms, 3)
            box:     Simulation box dimensions, shape (3,)
            species: Species IDs, shape (n_atoms,)
            N_max:   Maximum number of atoms (for padding)
            n_species_override: Optional global species cardinality override.
        """
        self.config = config
        self.N_max = N_max
        self.compute_dtype_name, self.compute_dtype = resolve_compute_dtype(config)
        self.remat_level = int(config.get_remat_level())
        self.remat_policy = str(config.get_remat_policy())

        self.cutoff = config.get_cutoff()
        self.dr_threshold = config.get_dr_threshold()
        self.neighbor_list_format_name, self.neighbor_list_format = resolve_neighbor_list_format(
            config.get_neighbor_list_format()
        )

        self.allegro_config = dict(config.get_allegro_config())
        self._pad_spacing = jnp.asarray(
            self.cutoff + self.dr_threshold + 1.0, dtype=self.compute_dtype
        )

        # ------------------------------------------------------------------ #
        #  cuEq-specific parameters - extract before passing to factory       #
        # ------------------------------------------------------------------ #

        # MLP dtype: controls bfloat16 inside MLP layers only
        mlp_dtype_cfg = self.allegro_config.pop("mlp_dtype", "float32")
        self.mlp_dtype_name, self.mlp_dtype = _resolve_mlp_dtype(mlp_dtype_cfg)

        enable_logging = config.debug_model_logging()
        self.allegro_config.pop("logging", None)

        # ------------------------------------------------------------------ #
        #  Activation resolution (mirrors allegro_model.py)                  #
        # ------------------------------------------------------------------ #

        hidden_raw = self.allegro_config.get(
            "mlp_hidden_activation",
            self.allegro_config.get("mlp_activation", "mish"),
        )
        output_raw = self.allegro_config.get("mlp_output_activation", "linear")
        self.mlp_hidden_activation_name, self.mlp_hidden_activation = (
            _resolve_mlp_activation(hidden_raw)
        )
        self.mlp_output_activation_name, self.mlp_output_activation = _resolve_mlp_activation(
            output_raw, allow_linear=True
        )
        # Write resolved callables back so the factory receives functions.
        self.allegro_config["mlp_activation"] = self.mlp_hidden_activation
        self.allegro_config["mlp_output_activation"] = self.mlp_output_activation

        # Graph-cap controls
        self.max_edge_multiplier = float(
            self.allegro_config.pop("max_edge_multiplier", 1.1)
        )
        max_edges_cfg = self.allegro_config.pop("max_edges", None)
        self.max_edges = None if max_edges_cfg is None else int(max_edges_cfg)

        # ------------------------------------------------------------------ #
        #  Logging                                                            #
        # ------------------------------------------------------------------ #

        model_logger.info(f"Using AllegroModelCuEq (cuEquivariance backend)")
        model_logger.info(f"  compute_dtype    = {self.compute_dtype_name}")
        model_logger.info(f"  mlp_dtype        = {self.mlp_dtype_name}")
        model_logger.info(
            f"  remat            = level={self.remat_level}, policy={self.remat_policy}"
        )
        model_logger.info(
            f"  MLP activations  = hidden={self.mlp_hidden_activation_name}, "
            f"output={self.mlp_output_activation_name}"
        )
        model_logger.info(f"  debug logging    = {enable_logging}")

        # ------------------------------------------------------------------ #
        #  JAX-MD neighbor list                                               #
        # ------------------------------------------------------------------ #

        self.displacement, self.shift = space.free()
        safe_box = jnp.asarray(box, dtype=self.compute_dtype)

        self.nneigh_fn = partition.neighbor_list(
            self.displacement,
            box=safe_box,
            r_cutoff=self.cutoff,
            dr_threshold=self.dr_threshold,
            fractional_coordinates=False,
            format=self.neighbor_list_format,
        )
        model_logger.info(f"  neighbor format = {self.neighbor_list_format_name}")

        # Sanitize padded entries for initialization
        species_arr = jnp.asarray(species)
        init_padded_mask = species_arr < 0
        R0_safe = self._spread_padded_coordinates(
            jnp.asarray(R0, dtype=self.compute_dtype), init_padded_mask
        )

        self._neighbor_extra_capacity = int(
            self.allegro_config.pop("neighbor_extra_capacity", 10)
        )
        self.nbrs_init = self.nneigh_fn.allocate(
            R0_safe, extra_capacity=self._neighbor_extra_capacity
        )
        model_logger.info(
            f"  neighbor extra_capacity = {self._neighbor_extra_capacity}"
        )

        # Resolve avg_num_neighbors without letting a single initialization
        # tile override a dataset-sampled runtime calibration.
        n_atoms = int(R0_safe.shape[0])
        actual_avg = compute_avg_num_neighbors(self.nbrs_init, n_atoms)
        config_avg = float(self.allegro_config.get("avg_num_neighbors", 12))
        avg_source = str(self.allegro_config.pop("avg_num_neighbors_source", "auto")).strip().lower()
        if avg_source in ("dataset_sample", "config"):
            effective_avg = config_avg
            if abs(actual_avg - config_avg) > 2.0:
                model_logger.info(
                    "avg_num_neighbors: honoring configured %.1f from %s; "
                    "initial neighbor list estimate was %.1f.",
                    config_avg,
                    avg_source,
                    actual_avg,
                )
            else:
                model_logger.info(
                    "avg_num_neighbors = %.1f (configured from %s; initial estimate %.1f)",
                    config_avg,
                    avg_source,
                    actual_avg,
                )
        else:
            effective_avg = actual_avg
            if abs(actual_avg - config_avg) > 2.0:
                model_logger.warning(
                    "avg_num_neighbors: config=%s, computed from data=%.1f. Using computed value.",
                    config_avg,
                    actual_avg,
                )
            model_logger.info(
                "avg_num_neighbors = %.1f (computed from initial neighbor list)",
                actual_avg,
            )
        self.allegro_config["avg_num_neighbors"] = float(effective_avg)

        if self.max_edges is not None:
            model_logger.info(
                f"max_edges = {self.max_edges} (max_edge_multiplier={self.max_edge_multiplier:.3f})"
            )
        else:
            model_logger.info(
                f"max_edges = inferred (max_edge_multiplier={self.max_edge_multiplier:.3f})"
            )

        # Species
        species_safe = jnp.where(species_arr >= 0, species_arr, 0).astype(jnp.int32)
        n_species_data = int(jnp.max(species_safe)) + 1
        if n_species_override is not None:
            self.n_species = max(n_species_data, int(n_species_override))
        else:
            self.n_species = n_species_data
        model_logger.info(f"Detected {self.n_species} unique species")

        # ------------------------------------------------------------------ #
        #  Initialise cuEq Allegro factory                                    #
        # ------------------------------------------------------------------ #

        # mlp_dtype flows to Allegro.__init__ via **allegro_kwargs;
        # logging is an explicit param of the factory (not forwarded to Allegro).
        ml_model_type = config.get_ml_model_type()
        self.ml_model_type = ml_model_type
        self._enable_logging = bool(enable_logging)
        layer_methods = self.allegro_config.get("tp_method_by_layer")
        if isinstance(layer_methods, (list, tuple)):
            self._export_tp_methods = tuple(
                _normalize_export_tp_token(method) for method in layer_methods
            )
        else:
            self._export_tp_methods = (
                _normalize_export_tp_token(self.allegro_config.get("tp_method", "naive")),
            )
        self._export_tp_mode = _normalize_export_tp_token(
            self.allegro_config.get("tp_mode", "mixed_naive"),
            mode_token=True,
        )
        self._export_apply_cache: dict[str, Any] = {}
        if ml_model_type == "allegro_cueq_fast":
            from .allegro_cueq_fast_1103 import (
                allegro_neighborlist_pp,  # lazy cuequivariance import
            )
            model_logger.info("  backend         = allegro_cueq_fast_1103")
        else:
            from .allegro_cueq_v2 import (
                allegro_neighborlist_pp,  # lazy cuequivariance import
            )
            model_logger.info("  backend         = allegro_cueq_v2")
        self._export_factory = allegro_neighborlist_pp
        self.init_allegro, self.apply_allegro = allegro_neighborlist_pp(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=R0_safe,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=self.max_edge_multiplier,
            max_edges=self.max_edges,
            mode="energy",
            logging=enable_logging,
            mlp_dtype=self.mlp_dtype,
            **self.allegro_config,
        )

        self._apply_allegro_for_training = self.apply_allegro
        if self.remat_level > 0:
            self._apply_allegro_for_training = jax.checkpoint(self.apply_allegro)
        self._export_apply_cache["current"] = self.apply_allegro

        self._R0 = R0_safe
        self._species0 = species_safe

    # ---------------------------------------------------------------------- #
    #  Internal helpers                                                       #
    # ---------------------------------------------------------------------- #

    def _spread_padded_coordinates(
        self, R: jax.Array, padded_mask: jax.Array
    ) -> jax.Array:
        """Place padded atoms far from all real atoms to prevent spurious edges."""
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

    # ---------------------------------------------------------------------- #
    #  Public interface (identical to AllegroModel)                          #
    # ---------------------------------------------------------------------- #

    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Any:
        """Initialize cuEq Allegro parameters."""
        return self.init_allegro(rng_key, self._R0, self.nbrs_init, self._species0)

    def get_neighborlist(self, R: jax.Array, nbrs: Optional[Any] = None) -> Any:
        """Get or update neighbor list for coordinates."""
        if nbrs is None:
            nbrs = self.nbrs_init
        ref_position = getattr(nbrs, "reference_position", None)
        target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
        return self.nneigh_fn.update(jnp.asarray(R, dtype=target_dtype), nbrs)

    def _compute_energy_with_apply(
        self,
        apply_fn: Any,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute cuEq Allegro energy for given coordinates."""
        valid_mask = mask > 0
        R_base = jnp.asarray(R, dtype=self.compute_dtype)
        padded_mask = jnp.logical_not(valid_mask)
        R_safe = self._spread_padded_coordinates(R_base, padded_mask)
        R_masked = jnp.where(valid_mask[:, None], R_base, jax.lax.stop_gradient(R_safe))

        if neighbor is None:
            base_nbrs = self.nbrs_init
            ref_position = getattr(base_nbrs, "reference_position", None)
            target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
            nbrs = self.nneigh_fn.update(
                jnp.asarray(R_masked, dtype=target_dtype), base_nbrs
            )
        else:
            nbr_error = getattr(neighbor, "error", None)
            if nbr_error is None:
                nbrs = neighbor
            else:
                ref_position = getattr(neighbor, "reference_position", None)
                target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
                nbrs = self.nneigh_fn.update(
                    jnp.asarray(R_masked, dtype=target_dtype), neighbor
                )

        if segment_id is not None:
            nbrs = custom_partition.mask_neighbor_list(
                nbrs,
                mask=valid_mask.astype(jnp.bool_),
                segment_id=jnp.asarray(segment_id, dtype=jnp.int32),
            )

        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)
        R_model = jnp.asarray(R_masked, dtype=self.compute_dtype)

        E = apply_fn(
            params, R_model, nbrs, species_masked,
            mask=valid_mask.astype(jnp.bool_),
        )
        return jnp.asarray(E, dtype=jnp.float32)

    def compute_energy(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute cuEq Allegro energy for given coordinates."""
        return self._compute_energy_with_apply(
            self._apply_allegro_for_training,
            params,
            R,
            mask,
            species,
            neighbor,
            segment_id=segment_id,
        )

    @property
    def export_tp_methods(self):
        methods = list(self._export_tp_methods)
        if self._export_tp_mode == "block_uniform_1d":
            methods.append("uniform_1d")
        return tuple(methods)

    def build_export_apply_fn(self, *, tp_method_override: Optional[str] = None):
        """Rebuild the raw apply_fn for export-time backend overrides."""
        if tp_method_override is None:
            return self.apply_allegro

        normalized_override = _normalize_export_tp_token(tp_method_override)
        if normalized_override != "naive":
            raise ValueError(
                f"Unsupported export tp_method_override={tp_method_override!r}; expected 'naive'."
            )
        if self.ml_model_type != "allegro_cueq_fast":
            raise ValueError(
                "Cross-backend export overrides are only supported for allegro_cueq_fast."
            )
        cached = self._export_apply_cache.get(normalized_override)
        if cached is not None:
            return cached

        export_allegro_config = dict(self.allegro_config)
        layer_methods = export_allegro_config.get("tp_method_by_layer")
        if isinstance(layer_methods, (list, tuple)):
            mapped_methods = [
                "uniform_1d_naive"
                if _normalize_export_tp_token(method) == "uniform_1d"
                else "naive"
                for method in layer_methods
            ]
            export_allegro_config["tp_method_by_layer"] = mapped_methods
            export_allegro_config["tp_method"] = mapped_methods[0] if mapped_methods else "naive"
        else:
            current_method = _normalize_export_tp_token(
                export_allegro_config.get("tp_method", "naive")
            )
            export_allegro_config["tp_method"] = (
                "uniform_1d_naive" if current_method == "uniform_1d" else "naive"
            )

        if self._export_tp_mode == "block_uniform_1d":
            export_allegro_config["tp_mode"] = "block_naive"

        model_logger.info(
            "Export override for allegro_cueq_fast: rebuilding apply_fn with export-compatible "
            "naive TP execution while preserving the trained layout."
        )
        _, export_apply = self._export_factory(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=self._R0,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=self.max_edge_multiplier,
            max_edges=self.max_edges,
            mode="energy",
            logging=self._enable_logging,
            mlp_dtype=self.mlp_dtype,
            **export_allegro_config,
        )
        self._export_apply_cache[normalized_override] = export_apply
        return export_apply
    @property
    def model_apply_fn(self):
        return self.apply_allegro

    def __repr__(self) -> str:
        return (
            f"AllegroModelCuEq(cutoff={self.cutoff}, n_species={self.n_species}, "
            f"N_max={self.N_max}, mlp_dtype={self.mlp_dtype_name})"
        )
