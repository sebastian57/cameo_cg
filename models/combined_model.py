"""
Combined Prior + ML Model

Composes physics-based prior energy with an ML model (Allegro, MACE, or PaiNN).
Supports pure ML, pure prior, or combined training via config.
"""

import jax
import jax.numpy as jnp
import inspect
import haiku as hk
from typing import Dict, Any, Optional

from config.types import EnergyComponents, ForceComponents
from .base_model import get_ml_model_class
from .prior_energy import PriorEnergy
from .topology import TopologyBuilder
from utils.logging import model_logger
from training.support_gate import SupportGateBank, rbf_segment_supports, rbf_structure_support, support_gate_scope
from training.edge_distance_gate import compute_ala2_combined_gate_diagnostics
from .local_extrapolation_gate import LocalExtrapolationGate

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
                 id_to_aa: Optional[Dict[int, str]] = None,
                 support_gate_bank: Optional[SupportGateBank] = None):
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
            support_gate_bank: Optional fitted support bank used to gate the ML
                residual energy by distance to training structures.
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

        self.ml_energy_scale = float(config.get("model", "ml_energy_scale", default=1.0))
        teacher_cfg = config.get("training", "teacher_distillation", default={}) or {}
        teacher_feature_cfg = teacher_cfg.get("feature", {}) if isinstance(teacher_cfg, dict) else {}
        self.teacher_feature_distillation_enabled = bool(teacher_feature_cfg.get("enabled", False))
        self.teacher_feature_key = str(teacher_feature_cfg.get("feature_key", "edge_features"))
        self.teacher_feature_source_dim = int(teacher_feature_cfg.get("source_dim", 32))
        self.teacher_feature_target_dim = int(teacher_feature_cfg.get("target_dim", 32))
        self.teacher_feature_projection_hidden_dim = int(teacher_feature_cfg.get("projection_hidden_dim", 64))
        self._teacher_projection_init = None
        self._teacher_projection_apply = None
        if self.teacher_feature_distillation_enabled:
            def _teacher_projection(x):
                x = hk.Linear(self.teacher_feature_projection_hidden_dim, name="hidden")(x)
                x = jax.nn.silu(x)
                return hk.Linear(self.teacher_feature_target_dim, name="output")(x)

            self._teacher_projection_init, self._teacher_projection_apply = hk.without_apply_rng(
                hk.transform(_teacher_projection)
            )
            model_logger.info(
                "Teacher feature distillation head: %s [%d -> %d -> %d]",
                self.teacher_feature_key,
                self.teacher_feature_source_dim,
                self.teacher_feature_projection_hidden_dim,
                self.teacher_feature_target_dim,
            )
        self.prior_energy_scale = float(config.get("model", "prior_energy_scale", default=1.0))
        gate_cfg = config.get("model", "robustness_gate", default={}) or {}
        self.robustness_gate_enabled = bool(gate_cfg.get("enabled", False))
        self.robustness_gate_threshold = float(gate_cfg.get("threshold", 250.0))
        self.robustness_gate_width = max(float(gate_cfg.get("width", 50.0)), 1.0e-6)
        self.robustness_gate_floor = float(gate_cfg.get("floor", 0.0))
        self.robustness_gate_stop_gradient = bool(gate_cfg.get("stop_gradient", True))
        self.local_extrapolation_gate = None
        self.local_extrapolation_ml_apply_fn = None
        local_gate_cfg = config.get("model", "local_extrapolation_gate", default={}) or {}
        self.local_extrapolation_gate_enabled = bool(local_gate_cfg.get("enabled", False))
        self.local_extrapolation_gate_stop_gradient = bool(local_gate_cfg.get("stop_gradient_gate", True))
        if self.local_extrapolation_gate_enabled:
            artifact_path = local_gate_cfg.get("artifact_path")
            if not artifact_path:
                raise ValueError("model.local_extrapolation_gate.enabled=true requires artifact_path")
            self.local_extrapolation_gate = LocalExtrapolationGate.from_file(artifact_path)
            if hasattr(self.ml_model, "build_export_apply_fn"):
                try:
                    self.local_extrapolation_ml_apply_fn = self.ml_model.build_export_apply_fn(
                        tp_method_override="naive"
                    )
                except NotImplementedError as exc:
                    model_logger.info(
                        "Runtime local extrapolation gate: export-compatible per-atom apply_fn "
                        "unavailable (%s); using direct compute_per_atom_energy().",
                        exc,
                    )
            model_logger.info(
                "Runtime local extrapolation gate: descriptor=%s artifact=%s onset=%.4g offset=%.4g stop_gradient=%s",
                self.local_extrapolation_gate.artifact.get("descriptor", "unknown"),
                artifact_path,
                self.local_extrapolation_gate.onset,
                self.local_extrapolation_gate.offset,
                self.local_extrapolation_gate_stop_gradient,
            )

        self.support_gate_bank = support_gate_bank
        support_cfg = config.get("training", "support_gate", default={}) or {}
        self.support_gate_scope = support_gate_scope(config)
        if self.support_gate_scope not in ("segment", "batch"):
            raise ValueError("training.support_gate.scope must be 'segment' or 'batch'")
        self.support_gate_enabled = bool(support_cfg.get("enabled", False)) and support_gate_bank is not None
        if bool(support_cfg.get("enabled", False)) and support_gate_bank is None:
            model_logger.warning(
                "training.support_gate.enabled=true but no support gate bank was provided; "
                "ML energy will not be support-gated."
            )
        if self.ml_energy_scale != 1.0 or self.prior_energy_scale != 1.0:
            model_logger.info(
                "Runtime energy scales: ml_energy_scale=%.4g prior_energy_scale=%.4g",
                self.ml_energy_scale,
                self.prior_energy_scale,
            )
        if self.robustness_gate_enabled:
            model_logger.info(
                "Runtime robustness gate: threshold=%.4g width=%.4g floor=%.4g stop_gradient=%s",
                self.robustness_gate_threshold,
                self.robustness_gate_width,
                self.robustness_gate_floor,
                self.robustness_gate_stop_gradient,
            )
        if self.support_gate_enabled:
            model_logger.info(
                "Runtime support gate: scope=%s centers=%d sigma=%.4g floor=%.4g stop_gradient=%s",
                self.support_gate_scope,
                int(self.support_gate_bank.centers.shape[0]),
                float(self.support_gate_bank.sigma),
                float(self.support_gate_bank.floor),
                bool(self.support_gate_bank.stop_gradient),
            )

        if config.use_pbc_enabled() and self.use_priors:
            model_logger.warning(
                "PBC mode is active with priors enabled. Prior bond/angle/dihedral/repulsive "
                "distance terms use free-space (Ri - Rj) convention, which is valid when all "
                "bonded pairs are well within box/2. Disable priors for very small periodic boxes."
            )

        if self.use_priors:
            self.prior = PriorEnergy(
                config, self.topology, self.ml_model.displacement, id_to_aa=id_to_aa
            )
            self._residual_dsm_prior = None
            model_logger.info(f"Mode: Prior + {self.ml_model_type.upper()}")
            model_logger.info(f"Prior weights: {self.prior.weights}")
        else:
            self.prior = None
            self._residual_dsm_prior = (
                PriorEnergy(config, self.topology, self.ml_model.displacement, id_to_aa=id_to_aa)
                if config.prior_residual_enabled()
                else None
            )
            model_logger.info(f"Mode: Pure {self.ml_model_type.upper()} (no priors)")
            if self._residual_dsm_prior is not None:
                model_logger.info(
                    "DSM/HVP residual-prior mode: auxiliary curvature/noise targets will use "
                    "ML + frozen prior energy."
                )

    def _robustness_gate_alpha(self, E_prior: jax.Array) -> jax.Array:
        if not self.robustness_gate_enabled:
            return jnp.asarray(1.0, dtype=jnp.asarray(E_prior).dtype)
        x = (jnp.asarray(E_prior) - self.robustness_gate_threshold) / self.robustness_gate_width
        alpha = self.robustness_gate_floor + (1.0 - self.robustness_gate_floor) * jax.nn.sigmoid(-x)
        alpha = jnp.clip(alpha, self.robustness_gate_floor, 1.0)
        if self.robustness_gate_stop_gradient:
            alpha = jax.lax.stop_gradient(alpha)
        return alpha

    def _support_gate_alpha(
        self,
        R: jax.Array,
        mask: jax.Array,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        if not self.support_gate_enabled or self.support_gate_bank is None:
            return jnp.asarray(1.0, dtype=jnp.asarray(R).dtype)
        if self.support_gate_scope == "segment" and segment_id is not None:
            num_segments = int(R.shape[0])
            alphas = rbf_segment_supports(
                R, mask, segment_id, self.support_gate_bank, num_segments=num_segments
            )
            active = ((mask > 0) & (segment_id >= 0)).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(active), 1.0)
            atom_alpha = alphas[jnp.clip(segment_id, 0, num_segments - 1)]
            return jnp.sum(jnp.where(active > 0, atom_alpha, 0.0)) / denom
        return rbf_structure_support(R, mask, self.support_gate_bank)

    def _support_gated_ml_energy(
        self,
        ml_params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any],
        segment_id: Optional[jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.local_extrapolation_gate_enabled and self.local_extrapolation_gate is not None:
            if not hasattr(self.ml_model, "compute_per_atom_energy"):
                raise NotImplementedError(
                    "model.local_extrapolation_gate requires the ML backend to expose compute_per_atom_energy()."
                )
            if (
                self.local_extrapolation_ml_apply_fn is not None
                and hasattr(self.ml_model, "_compute_per_atom_energy_with_apply")
            ):
                per_atom_raw = self.ml_model._compute_per_atom_energy_with_apply(
                    self.local_extrapolation_ml_apply_fn,
                    ml_params,
                    R,
                    mask,
                    species,
                    neighbor,
                    segment_id=segment_id,
                )
            else:
                per_atom_raw = self.ml_model.compute_per_atom_energy(
                    ml_params, R, mask, species, neighbor, segment_id=segment_id
                )
            per_atom = self.ml_energy_scale * per_atom_raw
            gates = self.local_extrapolation_gate.compute_gates(R, mask)
            if self.local_extrapolation_gate_stop_gradient:
                gates = jax.lax.stop_gradient(gates)
            valid = (mask > 0).astype(jnp.float32)
            E_ml_raw = jnp.sum(jnp.where(valid > 0, per_atom, 0.0))
            E_ml = jnp.sum(jnp.where(valid > 0, gates * per_atom, 0.0))
            mean_alpha = jnp.sum(jnp.where(valid > 0, gates, 0.0)) / jnp.maximum(jnp.sum(valid), 1.0)
            return E_ml, E_ml_raw, mean_alpha

        if (
            not self.support_gate_enabled
            or self.support_gate_bank is None
            or self.support_gate_scope == "batch"
            or segment_id is None
        ):
            E_ml_raw = self.ml_energy_scale * self.ml_model.compute_energy(
                ml_params, R, mask, species, neighbor, segment_id=segment_id
            )
            if not self.support_gate_enabled or self.support_gate_bank is None:
                one = jnp.asarray(1.0, dtype=jnp.float32)
                return E_ml_raw, E_ml_raw, one
            support_alpha = self._support_gate_alpha(R, mask)
            return support_alpha * E_ml_raw, E_ml_raw, support_alpha
        if not hasattr(self.ml_model, "compute_per_atom_energy"):
            raise NotImplementedError(
                "training.support_gate.scope='segment' requires the ML backend to expose "
                "compute_per_atom_energy(). Use scope='batch' for this backend."
            )
        per_atom = self.ml_energy_scale * self.ml_model.compute_per_atom_energy(
            ml_params, R, mask, species, neighbor, segment_id=segment_id
        )
        num_segments = int(R.shape[0])
        seg_safe = jnp.where((mask > 0) & (segment_id >= 0), segment_id, 0).astype(jnp.int32)
        seg_energy = jax.ops.segment_sum(per_atom, seg_safe, num_segments=num_segments)
        alphas = rbf_segment_supports(
            R, mask, segment_id, self.support_gate_bank, num_segments=num_segments
        )
        E_ml_raw = jnp.sum(per_atom)
        E_ml = jnp.sum(seg_energy * alphas)
        active = ((mask > 0) & (segment_id >= 0)).astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(active), 1.0)
        atom_alpha = alphas[jnp.clip(segment_id, 0, num_segments - 1)]
        mean_alpha = jnp.sum(jnp.where(active > 0, atom_alpha, 0.0)) / denom
        return E_ml, E_ml_raw, mean_alpha

    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Initialize model parameters.

        Returns:
            Dictionary with 'ml' (ML backbone params) and optionally 'prior'.
        """
        params = {'ml': self.ml_model.initialize_params(rng_key)}

        if self.teacher_feature_distillation_enabled:
            dummy = jnp.zeros((1, self.teacher_feature_source_dim), dtype=jnp.float32)
            params['teacher_projection'] = self._teacher_projection_init(
                jax.random.fold_in(rng_key, 173), dummy
            )

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
                    R_masked, mask, species=species, params=params["prior"], segment_id=segment_id
                )
            else:
                return self.prior.compute_total_energy(
                    R_masked, mask, species=species, segment_id=segment_id
                )

        E_ml, _, _ = self._support_gated_ml_energy(
            params['ml'], R, mask, species, neighbor, segment_id
        )

        if self.use_priors:
            # Stop gradient flow through padded atoms, re-attach only for valid ones

            R_detached = jax.lax.stop_gradient(R)
            mask_3d = mask[:, None]
            R_masked = jnp.where(mask_3d > 0, R, R_detached)
            if self.train_priors and "prior" in params:
                E_prior = self.prior.compute_total_energy(
                    R_masked, mask, species=species, params=params["prior"], segment_id=segment_id
                )
            else:
                E_prior = self.prior.compute_total_energy(
                    R_masked, mask, species=species, segment_id=segment_id
                )
            alpha = self._robustness_gate_alpha(E_prior)
            return alpha * E_ml + self.prior_energy_scale * E_prior
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

    def compute_al_features(
        self,
        params: Dict[str, Any],
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        """Return ML-backbone edge features used by active-learning scoring."""
        if self.prior_only:
            raise ValueError("Active-learning features require an ML model.")
        if not hasattr(self.ml_model, "compute_al_features"):
            raise NotImplementedError(
                f"ML model {type(self.ml_model).__name__} does not expose active-learning features."
            )
        return self.ml_model.compute_al_features(
            params["ml"],
            R,
            mask,
            species,
            neighbor,
            segment_id=segment_id,
        )

    def compute_teacher_projected_features(
        self,
        params: Dict[str, Any],
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Return learned AA-teacher-space node scalars for auxiliary training only."""
        if not self.teacher_feature_distillation_enabled or self._teacher_projection_apply is None:
            raise ValueError("Teacher feature projection is not enabled in this model config.")
        if "teacher_projection" not in params:
            raise KeyError("Model params do not contain teacher_projection.")
        aux = self.compute_al_features(params, R, mask, species, neighbor, segment_id=segment_id)
        if self.teacher_feature_key not in aux:
            raise KeyError(f"Requested teacher feature key {self.teacher_feature_key!r} is unavailable.")
        features = jnp.asarray(aux[self.teacher_feature_key])
        valid = jnp.asarray(aux["valid_edges"], dtype=features.dtype)
        senders = jnp.where(jnp.asarray(aux["valid_edges"], dtype=bool), aux["senders"], 0)
        sums = jax.ops.segment_sum(features * valid[:, None], senders, num_segments=R.shape[0])
        counts = jax.ops.segment_sum(valid, senders, num_segments=R.shape[0])
        node_features = sums / jnp.maximum(counts[:, None], 1.0)
        if node_features.shape[-1] != self.teacher_feature_source_dim:
            raise ValueError(
                "Teacher projection source dimension mismatch: "
                f"expected {self.teacher_feature_source_dim}, got {node_features.shape[-1]}."
            )
        return self._teacher_projection_apply(params["teacher_projection"], node_features)

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
            support_alpha = jnp.asarray(1.0)
        else:
            E_ml, E_ml_raw_scaled, support_alpha = self._support_gated_ml_energy(
                params['ml'], R, mask, species, neighbor, None
            )

        components = {
            "E_ml": E_ml,
            "E_ml_raw_scaled": E_ml_raw_scaled,
            "E_ml_support_alpha": support_alpha,
        }
        if (
            hasattr(self.ml_model, "edge_distance_gate_bank")
            and self.ml_model.edge_distance_gate_bank is not None
            and self.ml_model.edge_distance_gate_bank.has_ala2_combined_gate
        ):
            gate_diag = compute_ala2_combined_gate_diagnostics(
                R,
                mask,
                self.ml_model.edge_distance_gate_bank,
            )
            components.update({
                "E_ml_edge_gate_torsion_alpha": gate_diag["torsion_alpha"],
                "E_ml_edge_gate_distance_matrix_alpha": gate_diag["distance_matrix_alpha"],
                "E_ml_edge_gate_angular_alpha": gate_diag["angular_alpha"],
                "E_ml_edge_gate_combined_structure_alpha": gate_diag["combined_structure_alpha"],
            })

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
                "E_lj": prior_components.get("E_lj", 0.0),
                "E_wca": prior_components.get("E_wca", 0.0),
                "E_fene": prior_components.get("E_fene", 0.0),
                "E_leash": prior_components.get("E_leash", 0.0),
                "E_dh": prior_components.get("E_dh", 0.0),
                "E_stickiness": prior_components.get("E_stickiness", 0.0),
                "E_salt_bridge": prior_components.get("E_salt_bridge", 0.0),
                "E_local_in": prior_components.get("E_local_in", 0.0),
                "E_local_bond_in": prior_components.get("E_local_bond_in", 0.0),
                "E_crowding_wall": prior_components.get("E_crowding_wall", 0.0),
                "E_five_particle_flat_bottom": prior_components.get("E_five_particle_flat_bottom", 0.0),
                "E_aa_integrated_baseline": prior_components.get("E_aa_integrated_baseline", 0.0),
                "E_ala2_feature_recovery": prior_components.get("E_ala2_feature_recovery", 0.0),
                "E_ala2_rama_recovery": prior_components.get("E_ala2_rama_recovery", 0.0),
                "E_ala2_geometry_support_recovery": prior_components.get("E_ala2_geometry_support_recovery", 0.0),
                "E_prior_total": prior_components["E_total"],
            })
            gate_alpha = self._robustness_gate_alpha(prior_components["E_total"])
            components["E_ml_gate_alpha"] = gate_alpha
            components["E_ml"] = gate_alpha * components["E_ml"]
            components["E_prior_total"] = self.prior_energy_scale * components["E_prior_total"]
            components["E_total"] = components["E_ml"] + self.prior_energy_scale * prior_components["E_total"]
        else:
            components["E_total"] = components["E_ml"]

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
                    comps.get("E_lj", 0.0),
                    comps.get("E_wca", 0.0),
                    comps.get("E_fene", 0.0),
                    comps.get("E_leash", 0.0),
                    comps.get("E_local_in", 0.0),
                    comps.get("E_local_bond_in", 0.0),
                    comps.get("E_crowding_wall", 0.0),
                )

            # Single forward pass; vjp_fn holds stored residuals for backward.
            _, vjp_fn = jax.vjp(all_energies, R)

            # Each vjp_fn call is a backward-only pass (no re-forward).
            def _force(idx, n=14):
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
                "F_lj":              _force(7),
                "F_wca":             _force(8),
                "F_fene":            _force(9),
                "F_leash":           _force(10),
                "F_local_in":        _force(11),
                "F_local_bond_in":   _force(12),
                "F_crowding_wall":   _force(13),
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

    def dsm_energy_fn_template(self, params: Dict[str, Any]):
        """
        Energy template for DSM targets.

        In ordinary ML-only or direct ML+prior training this is identical to
        ``energy_fn_template``.  In prior-residual mode the force-matching
        target is the residual force, but DSM should regularize the exported
        total potential, so this template adds the frozen prior energy back in.
        """
        if self._residual_dsm_prior is None:
            return self.energy_fn_template(params)

        def energy_fn(R: jax.Array, neighbor: Any, **kwargs) -> jax.Array:
            mask = kwargs["mask"]
            species = kwargs["species"]
            segment_id = kwargs.get("segment_id")

            species = jnp.where(mask > 0, species, 0).astype(jnp.int32)
            E_ml_raw = self.ml_model.compute_energy(
                params["ml"],
                R,
                mask,
                species,
                neighbor,
                segment_id=segment_id,
            )
            E_ml = self.ml_energy_scale * E_ml_raw

            R_detached = jax.lax.stop_gradient(R)
            mask_3d = mask[:, None]
            R_masked = jnp.where(mask_3d > 0, R, R_detached)
            E_prior = self._residual_dsm_prior.compute_total_energy(
                R_masked,
                mask,
                species=species,
                segment_id=segment_id,
            )
            alpha = self._robustness_gate_alpha(E_prior)
            return alpha * E_ml + self.prior_energy_scale * E_prior

        return energy_fn

    def hvp_energy_fn_template(self, params: Dict[str, Any]):
        """
        Energy template for HVP targets.

        In residual-prior training, force matching fits ML to ``F_ref - F_prior``.
        HVP targets, however, are full reference curvature targets and should be
        compared against the exported potential that MD samples: ML + frozen
        prior. Reuse the DSM total-potential template for that case.
        """
        return self.dsm_energy_fn_template(params)

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
