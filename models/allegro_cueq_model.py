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
import numpy as np
from jax.sharding import PartitionSpec
try:
    from jax.sharding import get_abstract_mesh
except ImportError:
    class _EmptyAbstractMesh:
        empty = True

    def get_abstract_mesh():
        return _EmptyAbstractMesh()
from jax_md import space, partition
from pathlib import Path
from typing import Optional, Any
from jax_md_mod import custom_partition

from .base_model import BaseMLModel, register_ml_model, resolve_compute_dtype
from .allegro_model import _resolve_mlp_activation
from .neighborlist_utils import resolve_neighbor_list_format, compute_avg_num_neighbors
from training.edge_distance_gate import EdgeDistanceGateBank, edge_distance_gate_config
from utils.logging import model_logger

# allegro_cueq_v2 is imported lazily inside __init__ to avoid a hard dependency
# on cuequivariance at import time (non-cueq runs would fail otherwise).


def _replicate_params_when_mesh_active(params: Any) -> Any:
    """Make parameter shardings explicit inside active mesh contexts."""
    if get_abstract_mesh().empty:
        return params
    return jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, PartitionSpec())
                  if isinstance(x, jax.Array) else x,
        params,
    )


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
        init_mask: Optional[jax.Array] = None,
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
        self._neighbor_disable_cell_list = bool(config.neighbor_disable_cell_list_enabled())

        self.allegro_config = dict(config.get_allegro_config())
        self._neighbor_capacity_multiplier = float(
            self.allegro_config.pop("neighbor_capacity_multiplier", 1.25)
        )
        if self._neighbor_capacity_multiplier < 1.0:
            raise ValueError(
                "neighbor_capacity_multiplier must be >= 1.0, got "
                f"{self._neighbor_capacity_multiplier}."
            )
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

        safe_box = jnp.asarray(box, dtype=self.compute_dtype)
        if config.use_pbc_enabled():
            self.displacement, self.shift = space.periodic_general(
                safe_box, fractional_coordinates=False
            )
            self._pbc = True
            model_logger.info(
                f"  PBC mode       = space.periodic_general, box={jax.device_get(safe_box)}"
            )
        else:
            self.displacement, self.shift = space.free()
            self._pbc = False

        self.nneigh_fn = custom_partition.masked_neighbor_list(
            self.displacement,
            box=safe_box,
            r_cutoff=self.cutoff,
            dr_threshold=self.dr_threshold,
            capacity_multiplier=self._neighbor_capacity_multiplier,
            fractional_coordinates=False,
            disable_cell_list=self._neighbor_disable_cell_list,
            format=self.neighbor_list_format,
        )
        model_logger.info(
            f"  neighbor format = {self.neighbor_list_format_name} "
            f"(disable_cell_list={self._neighbor_disable_cell_list}, "
            f"capacity_multiplier={self._neighbor_capacity_multiplier:.3f})"
        )

        # Mask-aware neighbor init avoids coordinate teleporting and excludes
        # padded nodes from neighbor candidate generation.
        species_arr = jnp.asarray(species)
        if init_mask is not None:
            init_valid_mask = jnp.asarray(init_mask > 0, dtype=jnp.bool_)
        else:
            init_valid_mask = species_arr >= 0
        R0_safe = jnp.asarray(R0, dtype=self.compute_dtype)

        self._neighbor_extra_capacity = int(
            self.allegro_config.pop("neighbor_extra_capacity", 10)
        )
        self.nbrs_init = self.nneigh_fn.allocate(
            R0_safe,
            extra_capacity=self._neighbor_extra_capacity,
            mask=init_valid_mask,
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
        self.output_mode = config.get_model_output_mode()
        self.direct_force_config = config.get_direct_force_config()
        if (
            self.output_mode == "direct_force"
            and self.direct_force_config["require_bidirectional_edges"]
        ):
            idx = np.asarray(jax.device_get(self.nbrs_init.idx), dtype=np.int64)
            if self.neighbor_list_format_name == "sparse":
                receivers_np, senders_np = idx[0], idx[1]
            else:
                n_centers, n_slots = idx.shape
                senders_np = np.repeat(np.arange(n_centers, dtype=np.int64), n_slots)
                receivers_np = idx.reshape(-1)
            valid_nodes = np.asarray(jax.device_get(init_valid_mask), dtype=bool)
            valid = (
                (senders_np >= 0)
                & (senders_np < N_max)
                & (receivers_np >= 0)
                & (receivers_np < N_max)
            )
            senders_valid = senders_np[valid]
            receivers_valid = receivers_np[valid]
            valid = valid_nodes[senders_valid] & valid_nodes[receivers_valid]
            pairs = set(zip(senders_valid[valid].tolist(), receivers_valid[valid].tolist()))
            missing = [(i, j) for i, j in pairs if (j, i) not in pairs]
            if missing:
                raise ValueError(
                    "Direct-force central symmetrization requires bidirectional edges; "
                    f"initial graph is missing {len(missing)} reverse edges."
                )
            model_logger.info(
                "  direct-force graph validation = bidirectional (%d directed edges)",
                len(pairs),
            )
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

        edge_gate_cfg = edge_distance_gate_config(config)
        self.edge_distance_gate_enabled = bool(edge_gate_cfg.get("enabled", False))
        if self.output_mode == "direct_force" and self.edge_distance_gate_enabled:
            raise ValueError("model.edge_distance_gate is not supported in direct-force teacher mode.")
        self.edge_distance_gate_bank = None
        if self.edge_distance_gate_enabled:
            if ml_model_type != "allegro_cueq_fast":
                raise ValueError("model.edge_distance_gate is supported only for ml_model=allegro_cueq_fast")
            artifact_path = edge_gate_cfg.get("artifact_path")
            if not artifact_path:
                raise ValueError("model.edge_distance_gate.enabled=true requires artifact_path")
            gate_path = Path(artifact_path)
            if not gate_path.is_absolute():
                config_relative = (Path(config.config_path).parent / gate_path).resolve()
                cwd_relative = (Path.cwd() / gate_path).resolve()
                gate_path = config_relative if config_relative.exists() else cwd_relative
            fragment_torsion_gate_path = edge_gate_cfg.get(
                "fragment_torsion_gate_path",
                edge_gate_cfg.get("torsion_gate_path"),
            )
            ala2_combined_gate_path = edge_gate_cfg.get(
                "ala2_combined_gate_path",
                edge_gate_cfg.get("combined_gate_path"),
            )
            torsion_gate_path = None
            if fragment_torsion_gate_path:
                torsion_gate_path = Path(fragment_torsion_gate_path)
                if not torsion_gate_path.is_absolute():
                    config_relative = (
                        Path(config.config_path).parent / torsion_gate_path
                    ).resolve()
                    cwd_relative = (Path.cwd() / torsion_gate_path).resolve()
                    torsion_gate_path = (
                        config_relative if config_relative.exists() else cwd_relative
                    )
            combined_gate_path = None
            if ala2_combined_gate_path:
                combined_gate_path = Path(ala2_combined_gate_path)
                if not combined_gate_path.is_absolute():
                    config_relative = (
                        Path(config.config_path).parent / combined_gate_path
                    ).resolve()
                    cwd_relative = (Path.cwd() / combined_gate_path).resolve()
                    combined_gate_path = (
                        config_relative if config_relative.exists() else cwd_relative
                    )
            self.edge_distance_gate_bank = EdgeDistanceGateBank.from_file(
                gate_path,
                falloff_percent=float(edge_gate_cfg.get("falloff_percent", 0.05)),
                onset_percent=float(edge_gate_cfg.get("onset_percent", 0.0)),
                offset_percent=float(edge_gate_cfg.get("offset_percent", edge_gate_cfg.get("falloff_percent", 0.05))),
                floor=float(edge_gate_cfg.get("floor", 0.0)),
                alpha_power=float(edge_gate_cfg.get("alpha_power", 1.0)),
                stop_gradient=bool(edge_gate_cfg.get("stop_gradient", True)),
                fragment_torsion_gate_path=torsion_gate_path,
                ala2_combined_gate_path=combined_gate_path,
            )
            model_logger.info(
                "  edge distance gate = %s onset_percent=%.4g offset_percent=%.4g floor=%.4g alpha_power=%.4g stop_gradient=%s",
                gate_path,
                float(self.edge_distance_gate_bank.onset_percent),
                float(self.edge_distance_gate_bank.offset_percent),
                float(self.edge_distance_gate_bank.floor),
                float(self.edge_distance_gate_bank.alpha_power),
                bool(self.edge_distance_gate_bank.stop_gradient),
            )
            if self.edge_distance_gate_bank.has_fragment_torsion_gate:
                model_logger.info(
                    "  fragment torsion gate = %s k=%d onset_score=%.4gdeg offset_score=%.4gdeg",
                    torsion_gate_path,
                    int(self.edge_distance_gate_bank.fragment_torsion_k),
                    float(self.edge_distance_gate_bank.fragment_torsion_onset_score_deg),
                    float(self.edge_distance_gate_bank.fragment_torsion_offset_score_deg),
                )
            if self.edge_distance_gate_bank.has_ala2_combined_gate:
                model_logger.info(
                    "  ala2 combined gate = %s components=%s",
                    combined_gate_path,
                    ",".join(self.edge_distance_gate_bank.ala2_combined_components),
                )

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
        direct_force_kwargs = {
            "direct_force_hidden": int(self.direct_force_config["hidden"]),
            "direct_force_layers": int(self.direct_force_config["layers"]),
            "direct_force_envelope_p": int(self.direct_force_config["envelope_p"]),
            "direct_force_zero_init": bool(self.direct_force_config["zero_init"]),
        }
        factory_mode = "direct_forces" if self.output_mode == "direct_force" else "energy"
        self.init_allegro, self.apply_allegro = allegro_neighborlist_pp(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=R0_safe,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=self.max_edge_multiplier,
            max_edges=self.max_edges,
            mode=factory_mode,
            logging=enable_logging,
            mlp_dtype=self.mlp_dtype,
            **(direct_force_kwargs if self.output_mode == "direct_force" else {}),
            **({"edge_distance_gate": self.edge_distance_gate_bank} if ml_model_type == "allegro_cueq_fast" else {}),
            **self.allegro_config,
        )
        # Per-atom version for export: per_particle=True in the closure so the
        # function returns shape (n_atoms,) instead of a scalar total.  The
        # scalar version (_apply_allegro_for_training) is kept for training
        # where jax.grad needs a scalar output.
        if self.output_mode == "energy":
            _, self.apply_allegro_per_atom = allegro_neighborlist_pp(
                displacement=self.displacement,
                r_cutoff=self.cutoff,
                n_species=self.n_species,
                positions_test=R0_safe,
                neighbor_test=self.nbrs_init,
                max_edge_multiplier=self.max_edge_multiplier,
                max_edges=self.max_edges,
                mode="energy",
                per_particle=True,
                logging=enable_logging,
                mlp_dtype=self.mlp_dtype,
                **({"edge_distance_gate": self.edge_distance_gate_bank} if ml_model_type == "allegro_cueq_fast" else {}),
                **self.allegro_config,
            )
        else:
            self.apply_allegro_per_atom = None
        if ml_model_type == "allegro_cueq_fast" and self.output_mode == "energy":
            _, self.apply_allegro_al_features = allegro_neighborlist_pp(
                displacement=self.displacement,
                r_cutoff=self.cutoff,
                n_species=self.n_species,
                positions_test=R0_safe,
                neighbor_test=self.nbrs_init,
                max_edge_multiplier=self.max_edge_multiplier,
                max_edges=self.max_edges,
                mode="al_features",
                logging=enable_logging,
                mlp_dtype=self.mlp_dtype,
                **self.allegro_config,
            )
        else:
            self.apply_allegro_al_features = None

        self._apply_allegro_for_training = self.apply_allegro
        if self.remat_level > 0:
            _REMAT_POLICIES = {
                "none": None,
                "allegro_blocks_coarse": jax.checkpoint_policies.dots_saveable,
                "allegro_blocks_deep": jax.checkpoint_policies.nothing_saveable,
            }
            _policy = _REMAT_POLICIES.get(self.remat_policy)
            self._apply_allegro_for_training = jax.checkpoint(
                self.apply_allegro, policy=_policy
            )
        if self.apply_allegro_per_atom is not None:
            self._export_apply_cache["current"] = self.apply_allegro_per_atom

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
        valid_mask = jnp.ones((int(jnp.asarray(R).shape[0]),), dtype=jnp.bool_)
        return self.nneigh_fn.update(
            jnp.asarray(R, dtype=target_dtype),
            nbrs,
            mask=valid_mask,
        )

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
        R_masked = R_base

        if neighbor is None:
            base_nbrs = self.nbrs_init
            ref_position = getattr(base_nbrs, "reference_position", None)
            target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
            nbrs = self.nneigh_fn.update(
                jnp.asarray(R_masked, dtype=target_dtype),
                base_nbrs,
                mask=valid_mask.astype(jnp.bool_),
            )
        else:
            nbr_error = getattr(neighbor, "error", None)
            if nbr_error is None:
                nbrs = neighbor
            else:
                ref_position = getattr(neighbor, "reference_position", None)
                target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
                nbrs = self.nneigh_fn.update(
                    jnp.asarray(R_masked, dtype=target_dtype),
                    neighbor,
                    mask=valid_mask.astype(jnp.bool_),
                )

        nbrs = custom_partition.mask_neighbor_list(
            nbrs,
            mask=valid_mask.astype(jnp.bool_),
            segment_id=jnp.asarray(segment_id, dtype=jnp.int32) if segment_id is not None else None,
        )

        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)
        R_model = jnp.asarray(R_masked, dtype=self.compute_dtype)

        # JAX >= 0.10 strictly enforces that arrays used inside shard_map's
        # Manual mesh context must not carry Auto-mesh sharding. Haiku parameters
        # are created with Auto-mesh sharding, so we re-annotate them with
        # PartitionSpec() (replicated, no axis partitioning) which is valid in
        # both Auto and Manual contexts. Skip when no mesh is active (e.g.,
        # during export tracing) since with_sharding_constraint requires a mesh.
        params = _replicate_params_when_mesh_active(params)

        E = apply_fn(
            params, R_model, nbrs, species_masked,
            mask=valid_mask.astype(jnp.bool_),
        )
        return jnp.asarray(E, dtype=jnp.float32)

    def _compute_per_atom_energy_with_apply(
        self,
        apply_fn: Any,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute per-atom energies with a custom apply_fn for export."""
        valid_mask = mask > 0
        R_base = jnp.asarray(R, dtype=self.compute_dtype)
        R_masked = R_base

        if neighbor is None:
            base_nbrs = self.nbrs_init
            ref_position = getattr(base_nbrs, "reference_position", None)
            target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
            nbrs = self.nneigh_fn.update(
                jnp.asarray(R_masked, dtype=target_dtype),
                base_nbrs,
                mask=valid_mask.astype(jnp.bool_),
            )
        else:
            nbr_error = getattr(neighbor, "error", None)
            if nbr_error is None:
                nbrs = neighbor
            else:
                ref_position = getattr(neighbor, "reference_position", None)
                target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
                nbrs = self.nneigh_fn.update(
                    jnp.asarray(R_masked, dtype=target_dtype),
                    neighbor,
                    mask=valid_mask.astype(jnp.bool_),
                )

        nbrs = custom_partition.mask_neighbor_list(
            nbrs,
            mask=valid_mask.astype(jnp.bool_),
            segment_id=jnp.asarray(segment_id, dtype=jnp.int32) if segment_id is not None else None,
        )

        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)
        R_model = jnp.asarray(R_masked, dtype=self.compute_dtype)

        params = _replicate_params_when_mesh_active(params)

        E = apply_fn(
            params,
            R_model,
            nbrs,
            species_masked,
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
        if self.output_mode != "energy":
            raise RuntimeError("Direct-force teacher models do not define a scalar energy.")
        return self._compute_energy_with_apply(
            self._apply_allegro_for_training,
            params,
            R,
            mask,
            species,
            neighbor,
            segment_id=segment_id,
        )

    def compute_direct_force(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute central direct forces without differentiating an energy."""
        if self.output_mode != "direct_force":
            raise RuntimeError("compute_direct_force requires model.output_mode=direct_force.")

        valid_mask = jnp.asarray(mask > 0, dtype=jnp.bool_)
        R_model = jnp.asarray(R, dtype=self.compute_dtype)
        if neighbor is None or getattr(neighbor, "error", None) is not None:
            base_nbrs = self.nbrs_init if neighbor is None else neighbor
            ref_position = getattr(base_nbrs, "reference_position", None)
            target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
            nbrs = self.nneigh_fn.update(
                jnp.asarray(R_model, dtype=target_dtype),
                base_nbrs,
                mask=valid_mask,
            )
        else:
            nbrs = neighbor
        nbrs = custom_partition.mask_neighbor_list(
            nbrs,
            mask=valid_mask,
            segment_id=(
                jnp.asarray(segment_id, dtype=jnp.int32)
                if segment_id is not None
                else None
            ),
        )
        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)
        params = _replicate_params_when_mesh_active(params)
        forces = self._apply_allegro_for_training(
            params,
            R_model,
            nbrs,
            species_masked,
            mask=valid_mask,
        )
        return jnp.asarray(forces, dtype=jnp.float32) * valid_mask[:, None]

    def compute_per_atom_energy(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute per-atom cuEq Allegro energies for segment-level gating."""
        if self.output_mode != "energy":
            raise RuntimeError("Direct-force teacher models do not define per-atom energies.")
        return self._compute_per_atom_energy_with_apply(
            self.apply_allegro_per_atom,
            params,
            R,
            mask,
            species,
            neighbor,
            segment_id=segment_id,
        )

    def compute_al_features(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> dict[str, jax.Array]:
        """Return final invariant edge features for active-learning scoring."""
        if self.apply_allegro_al_features is None:
            raise NotImplementedError(
                "Active-learning feature extraction is currently implemented "
                "for ml_model='allegro_cueq_fast' only."
            )

        valid_mask = mask > 0
        R_base = jnp.asarray(R, dtype=self.compute_dtype)

        if neighbor is None:
            base_nbrs = self.nbrs_init
            ref_position = getattr(base_nbrs, "reference_position", None)
            target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
            nbrs = self.nneigh_fn.update(
                jnp.asarray(R_base, dtype=target_dtype),
                base_nbrs,
                mask=valid_mask.astype(jnp.bool_),
            )
        else:
            nbr_error = getattr(neighbor, "error", None)
            if nbr_error is None:
                nbrs = neighbor
            else:
                ref_position = getattr(neighbor, "reference_position", None)
                target_dtype = getattr(ref_position, "dtype", self.compute_dtype)
                nbrs = self.nneigh_fn.update(
                    jnp.asarray(R_base, dtype=target_dtype),
                    neighbor,
                    mask=valid_mask.astype(jnp.bool_),
                )

        nbrs = custom_partition.mask_neighbor_list(
            nbrs,
            mask=valid_mask.astype(jnp.bool_),
            segment_id=jnp.asarray(segment_id, dtype=jnp.int32) if segment_id is not None else None,
        )

        species_masked = jnp.where(valid_mask, species, 0).astype(jnp.int32)
        return self.apply_allegro_al_features(
            params,
            jnp.asarray(R_base, dtype=self.compute_dtype),
            nbrs,
            species_masked,
            mask=valid_mask.astype(jnp.bool_),
        )

    @property
    def export_tp_methods(self):
        methods = list(self._export_tp_methods)
        if self._export_tp_mode == "block_uniform_1d":
            methods.append("uniform_1d")
        return tuple(methods)

    @property
    def model_export_apply_fn(self):
        """Per-atom apply function for use in the MLIR export path."""
        if self.edge_distance_gate_enabled:
            raise NotImplementedError(
                "MLIR export does not yet support model.edge_distance_gate; "
                "disable the gate or use direct Python/JAX MD."
            )
        return self.apply_allegro_per_atom

    def build_export_apply_fn(self, *, tp_method_override: Optional[str] = None):
        """Rebuild the raw apply_fn for export-time backend overrides."""
        if self.edge_distance_gate_enabled:
            raise NotImplementedError(
                "MLIR export does not yet support model.edge_distance_gate; "
                "disable the gate or use direct Python/JAX MD."
            )
        if tp_method_override is None:
            return self.apply_allegro_per_atom

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
            per_particle=True,
            logging=self._enable_logging,
            mlp_dtype=self.mlp_dtype,
            edge_distance_gate=None,
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
