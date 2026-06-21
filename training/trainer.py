"""
Trainer for Force Matching

Orchestrates training of combined Prior + ML models using chemtrain.
Supports multi-stage training, prior pre-training, and checkpointing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple, Callable
import pickle
import time

from chemtrain.trainers.trainers import ForceMatching
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from chemtrain.data.data_loaders import DataLoaders

from config.types import PretrainResult, TrainingResults, StageResult
from .optimizers import create_optimizer_from_config
from .dsm import add_dsm_noise_fields, dsm_config, dsm_enabled, dsm_error, make_dsm_quantity
from .hvp_matching import hvp_config, hvp_error, make_hvp_quantity
from .safety_regularization import (
    SAFETY_FIELD_KEYS,
    make_safety_quantities,
    safety_config,
    safety_error_fns,
    safety_gammas,
    safety_weights_keys,
)
from .noised_residual import (
    attach_noised_residual_fields,
    noised_residual_config_parsed,
)
from .diagnostics import log_neighbor_debug_once
from utils.logging import training_logger
from data.loader import build_tiled_dataset, attach_batch_metadata


HVP_FIELD_KEYS = ("hvp_probe", "HVP", "hvp_loss_mask")


def valid_component_mse(predictions, targets, weights=None):
    """Average squared force error over valid force components only."""
    squared_differences = jnp.square(targets - predictions)
    if weights is None:
        return jnp.mean(squared_differences)

    weights = jnp.asarray(weights, dtype=squared_differences.dtype)
    if weights.ndim == squared_differences.ndim - 1:
        weights = weights[..., None]
    try:
        weights = jnp.broadcast_to(weights, squared_differences.shape)
    except ValueError as exc:
        raise ValueError(
            "force_loss_mask must match force target shape after broadcasting. "
            f"Got weights shape {weights.shape} and force shape {squared_differences.shape}."
        ) from exc
    numerator = jnp.sum(squared_differences * weights)
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    return numerator / denominator


class Trainer:
    """
    Trainer for force matching with Prior + ML models.

    Supports:
    - Multi-stage training with different optimizers
    - Optional prior pre-training (LBFGS or gradient-based)
    - Checkpointing and model export
    - Multi-GPU training
    - Single-node and multi-node distributed training

    Example:
        >>> trainer = Trainer(model, config, train_loader, val_loader)
        >>> # Optional prior pre-training
        >>> if config.get("training", "pretrain_prior"):
        ...     trainer.pretrain_prior(epochs=50)
        >>> # Main training
        >>> trainer.train_stage("adabelief", epochs=100)
        >>> trainer.train_stage("yogi", epochs=50)
        >>> # Export
        >>> trainer.export_model("model.mlir")
    """

    def __init__(
        self,
        model,  # CombinedModel instance
        config,  # ConfigManager instance
        train_loader,  # DatasetLoader or NumpyDataLoader
        val_loader: Optional[Any] = None,
        train_data: Optional[Dict[str, jax.Array]] = None,
        tiled_train_source: Optional[Dict[str, np.ndarray]] = None,
        noised_id_to_aa: Optional[Dict[int, str]] = None,
        noised_fitted_params: Optional[Dict[str, np.ndarray]] = None,
        seed: Optional[int] = None,  # Optional seed override for ensemble training
    ):
        """
        Initialize trainer.

        Args:
            model: CombinedModel instance
            config: ConfigManager instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            train_data: Optional dict with R, F, mask for prior pre-training
            tiled_train_source: Untiled training structures used to rebuild tiled batches
            noised_id_to_aa: Optional species metadata for noised residual prior forces
            noised_fitted_params: Optional fitted prior parameters for noised residual prior forces
            seed: Optional seed override (for ensemble training). If None, uses config seed.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training parameters
        self.batch_per_device = config.get_batch_per_device()
        self.batch_cache = config.get_batch_cache()
        self.gammas = config.get_gammas()
        self._dsm_cfg = dsm_config(config)
        if self._dsm_cfg["enabled"]:
            self.gammas = dict(self.gammas)
            self.gammas["DSM"] = float(self._dsm_cfg["lambda"])
        self._hvp_cfg = hvp_config(config)
        if self._hvp_cfg["enabled"]:
            self.gammas = dict(self.gammas)
            self.gammas.setdefault("HVP", float(self._hvp_cfg["lambda"]))
        self._safety_cfg = safety_config(config)
        if self._safety_cfg["enabled"]:
            self.gammas = dict(self.gammas)
            self.gammas.update(safety_gammas(config))
            training_logger.info(
                "[Safety] Regularization enabled: gammas=%s",
                {k: self.gammas[k] for k in safety_gammas(config)},
            )
        self._dsm_refresh_interval_steps = int(self._dsm_cfg.get("refresh_interval_steps", 0))
        self._noised_residual_cfg = noised_residual_config_parsed(config)
        self._noised_residual_enabled = bool(self._noised_residual_cfg.get("enabled", False))
        self._noised_refresh_interval_epochs = int(
            self._noised_residual_cfg.get("refresh_interval_epochs", 1)
        )
        self._noised_id_to_aa = noised_id_to_aa
        self._noised_fitted_params = noised_fitted_params
        self._dsm_refresh_count = 0
        self._dsm_optimizer_steps = 0
        self._seed = seed if seed is not None else config.get_seed()
        self.checkpoint_path = Path(config.get_checkpoint_path())
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._rank = jax.process_index()
        self._world_size = jax.process_count()
        self._batch_mode = config.get_batch_mode()
        self._global_device_count = jax.device_count()
        self._global_batch_size = self.batch_per_device * self._global_device_count
        self._grad_accum_steps = max(1, int(os.environ.get("CHEMTRAIN_GRAD_ACCUM_STEPS", "1")))
        self._grad_accum_mode = str(
            os.environ.get("CHEMTRAIN_GRAD_ACCUM_MODE", "stack_scan")
        ).strip().lower()
        self._force_loss_normalization = config.get_force_loss_normalization()
        self._config_tile_rebuild_each_epoch = (
            self._batch_mode == "tiled" and config.tile_rebuild_each_epoch_enabled()
        )
        self._tile_rebuild_each_epoch = (
            self._config_tile_rebuild_each_epoch
            or (
                self._batch_mode == "tiled"
                and self._noised_residual_enabled
                and self._noised_refresh_interval_epochs > 0
            )
        )
        self._tile_shuffle_structures = config.tile_shuffle_structures_enabled()
        self._tile_sort_by_size = config.tile_sort_by_size_enabled()
        self._tile_drop_incomplete = config.tile_drop_incomplete_enabled()
        self._tile_target_beads = (
            config.get_tile_target_beads() if self._batch_mode == "tiled" else None
        )
        self._tile_bucket_beads = (
            config.get_tile_bucket_beads() if self._batch_mode == "tiled" else None
        )
        self._tile_target_edges = (
            config.get_tile_target_edges() if self._batch_mode == "tiled" else None
        )
        self._tile_bucket_edges = (
            config.get_tile_bucket_edges() if self._batch_mode == "tiled" else None
        )
        self._tile_edge_estimate_scale = (
            config.get_tile_edge_estimate_scale() if self._batch_mode == "tiled" else 15.0
        )
        self._tile_edge_estimate_mode = (
            config.get_tile_edge_estimate_mode() if self._batch_mode == "tiled" else "valid_scaled"
        )
        self._tile_edge_estimate_cutoff = (
            config.get_tile_edge_estimate_cutoff() if self._batch_mode == "tiled" else None
        )
        self._tile_sort_by_estimated_edges = (
            self._batch_mode == "tiled" and config.tile_sort_by_estimated_edges_enabled()
        )
        self._tile_isolate_large_structures = (
            self._batch_mode == "tiled" and config.tile_isolate_large_structures_enabled()
        )
        self._tile_large_structure_threshold = (
            config.get_tile_large_structure_threshold() if self._batch_mode == "tiled" else None
        )
        self._tile_large_structure_edge_threshold = (
            config.get_tile_large_structure_edge_threshold() if self._batch_mode == "tiled" else None
        )
        self._tile_spatial_separation = (
            self._batch_mode == "tiled" and config.tile_spatial_separation_enabled()
        )
        self._tile_structure_gap = (
            config.get_tile_structure_gap() if self._batch_mode == "tiled" else 25.0
        )
        self._tiled_train_source = None
        if tiled_train_source is not None:
            self._tiled_train_source = {
                key: np.asarray(value) for key, value in tiled_train_source.items()
            }
        self._dsm_standard_train_source = None

        # Runtime precision + JIT buffer donation policy.
        self._mixed_precision_enabled = config.mixed_precision_enabled()
        self._compute_dtype = config.get_compute_dtype()
        self._param_dtype = config.get_param_dtype()
        self._reduce_dtype = config.get_reduce_dtype()
        self._enable_buffer_donation = config.buffer_donation_enabled()
        self._donate_mode = config.get_donate_mode()

        # Conservative default when mixed precision is disabled.
        if not self._mixed_precision_enabled:
            self._compute_dtype = "float32"
            self._reduce_dtype = "float32"

        # Make policy visible to the vendored chemtrain update path.
        os.environ["CHEMTRAIN_COMPUTE_DTYPE"] = self._compute_dtype
        os.environ["CHEMTRAIN_PARAM_DTYPE"] = self._param_dtype
        os.environ["CHEMTRAIN_REDUCE_DTYPE"] = self._reduce_dtype
        os.environ["CHEMTRAIN_ENABLE_BUFFER_DONATION"] = (
            "1" if self._enable_buffer_donation else "0"
        )
        os.environ["CHEMTRAIN_DONATE_MODE"] = self._donate_mode

        training_logger.info(
            "[RuntimePolicy] mixed_precision=%s compute_dtype=%s param_dtype=%s "
            "reduce_dtype=%s buffer_donation=%s donate_mode=%s",
            self._mixed_precision_enabled,
            self._compute_dtype,
            self._param_dtype,
            self._reduce_dtype,
            self._enable_buffer_donation,
            self._donate_mode,
        )

        # Optional JAX profiler configuration (controlled by YAML)
        profiling_cfg = config.get_profiling_config()
        self._profiling_enabled = bool(profiling_cfg.get("enabled", False))
        self._profiling_jax_trace_enabled = bool(
            profiling_cfg.get("jax_trace_enabled", True)
        )
        self._profiling_trace_dir = Path(str(profiling_cfg.get("trace_dir", "./profiles")))
        self._profiling_trace_rank0_only = bool(profiling_cfg.get("trace_rank0_only", True))
        self._profiling_log_compiles = bool(profiling_cfg.get("log_compiles", False))
        self._batch_profiler_enabled = bool(profiling_cfg.get("batch_profiler_enabled", False))
        self._batch_profiler_warmup = int(profiling_cfg.get("batch_profiler_warmup", 5))
        self._batch_profiler_samples = int(profiling_cfg.get("batch_profiler_samples", 50))
        self._batch_stats_enabled = bool(profiling_cfg.get("batch_stats_enabled", False))
        self._batch_stats_rank0_only = bool(profiling_cfg.get("batch_stats_rank0_only", True))
        self._batch_stats_log_every = max(1, int(profiling_cfg.get("batch_stats_log_every", 1)))
        self._loss_profile_enabled = bool(profiling_cfg.get("loss_profile_enabled", False))
        self._loss_profile_steps = max(0, int(profiling_cfg.get("loss_profile_steps", 4)))
        self._epoch_summary_enabled = bool(profiling_cfg.get("epoch_summary_enabled", False))
        self._loader_timing_enabled = bool(
            self._profiling_enabled
            and (self._batch_profiler_enabled or self._batch_stats_enabled)
        )
        self._loader_timing_limit = max(
            self._batch_profiler_samples,
            self._loss_profile_steps,
            self._batch_stats_log_every,
            8,
        )
        self._host_breakdown_limit = min(self._loader_timing_limit, 8)
        self._profile_step_records = []
        self._dataset_profile = None
        self._loader_setup_records = []
        self._tile_build_records = []
        self._batch_fetch_records = []
        self._pending_batch_fetch_profiles = deque()
        env_true = ("1", "true", "yes", "on")
        self._edge_profiler_enabled = (
            self._batch_profiler_enabled
            and str(os.getenv("CHEMTRAIN_PROFILE_EDGE_COUNTS", "1")).strip().lower() in env_true
        )
        self._edge_profiler_structures = max(
            1, int(os.getenv("CHEMTRAIN_PROFILE_EDGE_COUNT_SAMPLES", "1"))
        )
        self._edge_profiler_stride = max(
            1, int(os.getenv("CHEMTRAIN_PROFILE_EDGE_COUNT_STRIDE", "1"))
        )
        self._edge_profiler_rank0_only = (
            str(os.getenv("CHEMTRAIN_PROFILE_EDGE_COUNT_RANK0_ONLY", "1")).strip().lower()
            in env_true
        )
        self._edge_profiler_warned_missing_batch = False
        self._edge_profiler_prev_mean = None
        self._neighbor_debug_enabled = config.debug_neighbor_logging()
        self._neighbor_debug_rank0_only = config.debug_neighbor_rank0_only()
        self._neighbor_debug_logged = False
        # Env override for emergency/no-code toggles in SLURM scripts.
        # 1/true/on -> enable traces, 0/false/off -> disable traces.
        env_trace_toggle = os.getenv("CHEMTRAIN_PROFILE_JAX_TRACE")
        if env_trace_toggle is not None:
            self._profiling_jax_trace_enabled = env_trace_toggle.strip().lower() in (
                "1", "true", "yes", "on"
            )

        if self._profiling_log_compiles:
            try:
                jax.config.update("jax_log_compiles", True)
                training_logger.info("[Profiling] Enabled jax_log_compiles=True")
            except Exception as e:
                training_logger.warning(f"[Profiling] Could not enable jax_log_compiles: {e}")

        if self._loader_timing_enabled:
            os.environ["CHEMTRAIN_PROFILE_BATCH_BREAKDOWN"] = "1"
            os.environ["CHEMTRAIN_PROFILE_UPDATE_BREAKDOWN"] = "1"
            os.environ["CHEMTRAIN_PROFILE_TASK_TIMING"] = "1"
            os.environ["CHEMTRAIN_PROFILE_RANK0_ONLY"] = "1"
            os.environ[
                "CHEMTRAIN_PROFILE_BATCH_BREAKDOWN_LIMIT"
            ] = str(self._host_breakdown_limit)

        if self._profiling_enabled:
            self._profiling_trace_dir.mkdir(parents=True, exist_ok=True)
            if self._loader_timing_enabled and self._should_batch_stats_this_rank():
                training_logger.info(
                    "[Profiling] Loader timing enabled (batch_breakdown_limit=%d, update_breakdown=%s, task_timing=%s)",
                    self._host_breakdown_limit,
                    True,
                    True,
                )
            if self._edge_profiler_enabled:
                training_logger.info(
                    "[Profiling] EdgeProfiler enabled "
                    "(sample_structures=%d, stride=%d, rank0_only=%s)",
                    self._edge_profiler_structures,
                    self._edge_profiler_stride,
                    self._edge_profiler_rank0_only,
                )
            if not self._profiling_jax_trace_enabled:
                training_logger.info(
                    f"[Profiling] JAX trace export disabled on rank {self._rank} "
                    "(jax_trace_enabled=false)"
                )
            elif self._profiling_trace_rank0_only and self._rank != 0:
                training_logger.info(
                    f"[Profiling] Enabled in config, but rank {self._rank} tracing is disabled "
                    "(trace_rank0_only=true)"
                )
            else:
                training_logger.info(
                    f"[Profiling] JAX tracing enabled (rank={self._rank}, "
                    f"output={self._profiling_trace_dir})"
                )

        # Store training data for prior pre-training
        if train_data is not None:
            self._train_data = train_data
            if "species" not in self._train_data:
                self._train_data["species"] = jnp.zeros_like(
                    self._train_data["mask"], dtype=jnp.int32
                )
        else:
            # Try to extract from loader (for backwards compatibility)
            # NumpyDataLoader stores data in _chains internally
            try:
                if hasattr(train_loader, '_chains') and len(train_loader._chains) > 0:
                    chain_data = train_loader._chains[0]
                    self._train_data = {
                        "R": jnp.asarray(chain_data["R"]),
                        "F": jnp.asarray(chain_data["F"]),
                        "mask": jnp.asarray(chain_data["mask"]),
                        "species": jnp.asarray(
                            chain_data["species"]
                            if "species" in chain_data
                            else np.zeros_like(chain_data["mask"], dtype=np.int32)
                        ),
                    }
                else:
                    training_logger.warning("Could not extract training data from loader. Prior pre-training may not work.")
                    self._train_data = None
            except Exception as e:
                training_logger.warning(f"Could not extract training data: {e}. Prior pre-training may not work.")
                self._train_data = None

        if self._dsm_cfg["enabled"] and self._batch_mode != "tiled":
            dsm_keys = {"DSM", "dsm_eps", "dsm_sigma", "dsm_loss_mask"}
            self._dsm_standard_train_source = {
                key: np.asarray(value)
                for key, value in self._loader_reference_data(self.train_loader).items()
                if key not in dsm_keys
            }

        training_logger.info(
            "[Training] Force loss normalization: %s",
            self._force_loss_normalization,
        )

        if self._batch_stats_enabled and self._should_batch_stats_this_rank():
            training_logger.info(
                "[Profiling] Batch stats enabled (log_every=%d, loss_profile=%s, epoch_summary=%s)",
                self._batch_stats_log_every,
                self._loss_profile_enabled,
                self._epoch_summary_enabled,
            )

        self._initialize_dataset_profile()

        # Initialize model parameters
        self.params = model.initialize_params(jax.random.PRNGKey(self._seed))
        training_logger.info(f"Initialized model with seed={self._seed}")
        if self._dsm_refresh_interval_steps > 0:
            training_logger.info(
                "[DSM] Training noise refresh enabled every %d optimizer steps.",
                self._dsm_refresh_interval_steps,
            )
        if self._tile_rebuild_each_epoch:
            training_logger.info(
                "[Tiling] Epoch-wise tile rebuild enabled (shuffle=%s, sort_by_size=%s, target_beads=%s, bucket_beads=%s)",
                self._tile_shuffle_structures,
                self._tile_sort_by_size,
                self._tile_target_beads,
                self._tile_bucket_beads,
            )
        self.best_params = None

        # Current trainer instance (will be set during training)
        self._chemtrain_trainer = None

        # Optimizer state to restore on next train_stage call (set by load_chemtrain_checkpoint)
        self._resume_opt_state = None

        # Apply NumpyDataLoader patch if needed
        self._apply_dataloader_patch()

    def _apply_dataloader_patch(self):
        """
        Apply patch to NumpyDataLoader to fix cache_size issue.

        This is needed for chemtrain compatibility.
        """
        from jax_sgmc.data.numpy_loader import NumpyDataLoader as _NDL

        if not hasattr(_NDL, '_original_get_indices'):
            _orig_get_indices = _NDL._get_indices

            def _patched_get_indices(self, chain_id: int):
                chain = self._chains[chain_id]
                if chain.get("cache_size", 0) <= 0:
                    chain["cache_size"] = 1
                return _orig_get_indices(self, chain_id)

            _NDL._get_indices = _patched_get_indices
            _NDL._original_get_indices = _orig_get_indices
            training_logger.info("Applied NumpyDataLoader patch")

    def _should_trace_this_rank(self) -> bool:
        """Return True if JAX tracing should run on this rank."""
        if not self._profiling_enabled:
            return False
        if not self._profiling_jax_trace_enabled:
            return False
        if self._profiling_trace_rank0_only and self._rank != 0:
            return False
        return True

    def _should_batch_profile_this_rank(self) -> bool:
        """Return True if batch-profiler wrappers should run on this rank."""
        if not self._batch_profiler_enabled:
            return False
        if self._profiling_trace_rank0_only and self._rank != 0:
            return False
        return True


    def _should_batch_stats_this_rank(self) -> bool:
        """Return True if batch/accounting diagnostics should run on this rank."""
        if not self._batch_stats_enabled:
            return False
        if self._batch_stats_rank0_only and self._rank != 0:
            return False
        return True

    @staticmethod
    def _tree_l2_norm(tree: Any) -> float:
        """Compute an L2 norm for numeric leaves of a pytree on the host."""
        total = 0.0
        for leaf in jax.tree_util.tree_leaves(tree):
            try:
                arr = np.asarray(jax.device_get(leaf))
            except Exception:
                continue
            if not np.issubdtype(arr.dtype, np.number):
                continue
            total += float(np.sum(np.square(arr, dtype=np.float64), dtype=np.float64))
        return float(np.sqrt(total))

    def _batch_array_to_host(self, value: Any) -> Optional[np.ndarray]:
        """Move a batch field to host and flatten stack-scan microbatches."""
        if value is None:
            return None
        arr = np.asarray(jax.device_get(value))
        if (
            self._grad_accum_steps > 1
            and self._grad_accum_mode == "stack_scan"
            and arr.ndim >= 2
            and arr.shape[0] <= self._grad_accum_steps
            and arr.shape[1] == self._global_batch_size
        ):
            return arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:])
        return arr

    def _profile_batch_to_host(self, batch: Any) -> Dict[str, np.ndarray]:
        """Extract relevant batch fields to host arrays for diagnostics."""
        keys = (
            "R",
            "F",
            "mask",
            "species",
            "segment_id",
            "n_valid",
            "n_segments",
            "meta_batch_item_id",
            "meta_capacity",
            "meta_fill_ratio",
            "meta_n_force_components",
            "meta_source_structure_ids",
            "meta_source_structure_n_valid",
            "meta_structure_size_min",
            "meta_structure_size_mean",
            "meta_structure_size_max",
            "meta_structure_size_std",
        )
        out: Dict[str, np.ndarray] = {}
        for key in keys:
            value = self._get_batch_field(batch, key)
            if value is not None:
                out[key] = self._batch_array_to_host(value)
        return out

    def _set_dataset_profile(self, data: Optional[Dict[str, Any]], log: bool = False) -> None:
        """Cache dataset-level counts used to contextualize per-step logs."""
        if data is None:
            return
        item_ids = self._batch_array_to_host(data.get("meta_batch_item_id"))
        n_valid = self._batch_array_to_host(data.get("n_valid"))
        n_segments = self._batch_array_to_host(data.get("n_segments"))
        fill_ratio = self._batch_array_to_host(data.get("meta_fill_ratio"))
        n_force = self._batch_array_to_host(data.get("meta_n_force_components"))
        source_ids = self._batch_array_to_host(data.get("meta_source_structure_ids"))
        if item_ids is None or n_valid is None:
            return

        valid_source_ids = np.asarray([], dtype=np.int32)
        if source_ids is not None:
            valid_source_ids = np.asarray(source_ids, dtype=np.int32)
            valid_source_ids = valid_source_ids[valid_source_ids >= 0]

        self._dataset_profile = {
            "n_items": int(item_ids.shape[0]),
            "total_valid_beads": int(np.sum(n_valid, dtype=np.int64)),
            "total_force_components": int(np.sum(n_force, dtype=np.int64)) if n_force is not None else int(np.sum(n_valid, dtype=np.int64) * 3),
            "total_structures": int(np.unique(valid_source_ids).size) if valid_source_ids.size else int(item_ids.shape[0]),
            "mean_fill_ratio": float(np.mean(fill_ratio)) if fill_ratio is not None else float("nan"),
            "mean_segments": float(np.mean(n_segments)) if n_segments is not None else 1.0,
        }
        if log and self._should_batch_stats_this_rank():
            training_logger.info(
                "[Profiling][Dataset] mode=%s items=%d total_structures=%d total_valid_beads=%d total_force_components=%d mean_fill_ratio=%.3f mean_segments=%.2f",
                self._batch_mode,
                self._dataset_profile["n_items"],
                self._dataset_profile["total_structures"],
                self._dataset_profile["total_valid_beads"],
                self._dataset_profile["total_force_components"],
                self._dataset_profile["mean_fill_ratio"],
                self._dataset_profile["mean_segments"],
            )

    def _initialize_dataset_profile(self) -> None:
        self._set_dataset_profile(self._train_data, log=True)

    def _device_slot_means(self, values: np.ndarray) -> Dict[int, float]:
        """Summarize a per-sample metric by device slot within a global batch."""
        if values.size == 0:
            return {}
        local_positions = np.arange(values.shape[0], dtype=np.int32) % max(
            self._global_batch_size, 1
        )
        device_slots = local_positions // max(self.batch_per_device, 1)
        slot_means: Dict[int, float] = {}
        for slot in range(self._global_device_count):
            slot_mask = device_slots == slot
            if np.any(slot_mask):
                slot_means[int(slot)] = float(np.mean(values[slot_mask]))
        return slot_means

    def _compute_manual_loss_views(
        self, params: Dict[str, Any], batch_host: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, float]]:
        """Recompute sampled force-loss views to expose weighting differences."""
        required = ("R", "F", "mask", "species")
        if any(key not in batch_host for key in required):
            return None

        if not hasattr(self, "_manual_force_profile_fn"):
            def _single_force(params_, R_, mask_, species_, segment_id_):
                species_safe = jnp.where(mask_ > 0, species_, 0).astype(jnp.int32)
                def _energy_fn(R_eval):
                    return self.model.compute_energy(
                        params_, R_eval, mask_, species_safe, segment_id=segment_id_
                    )
                return -jax.grad(_energy_fn)(R_)

            self._manual_force_profile_fn = jax.jit(
                jax.vmap(_single_force, in_axes=(None, 0, 0, 0, 0))
            )

        R = jnp.asarray(batch_host["R"])
        F_ref = jnp.asarray(batch_host["F"])
        mask = jnp.asarray(batch_host["mask"])
        species = jnp.asarray(batch_host["species"])
        segment_id = batch_host.get("segment_id")
        if segment_id is None:
            segment_id = np.zeros_like(batch_host["mask"], dtype=np.int32)
        segment_id = jnp.asarray(segment_id, dtype=jnp.int32)

        F_pred = self._manual_force_profile_fn(params, R, mask, species, segment_id)
        sq = np.asarray(jax.device_get(jnp.square(F_ref - F_pred)), dtype=np.float64)
        mask_np = np.asarray(batch_host["mask"], dtype=np.float64)
        mask3 = np.broadcast_to(mask_np[..., None], sq.shape)
        legacy_component_mse = float(np.mean(sq))
        valid_component_mse = float(
            np.sum(sq * mask3, dtype=np.float64)
            / max(np.sum(mask3, dtype=np.float64), 1.0)
        )
        if self._force_loss_normalization == "valid_components":
            current_component_mse = valid_component_mse
        elif self._force_loss_normalization == "per_structure_components":
            current_component_mse = float("nan")
        else:
            current_component_mse = legacy_component_mse
        valid_bead_vector_mse = float(
            np.sum(sq * mask3, dtype=np.float64)
            / max(np.sum(mask_np, dtype=np.float64), 1.0)
        )
        per_tile_component = np.mean(sq.reshape((sq.shape[0], -1)), axis=1)
        per_tile_component_mse = float(np.mean(per_tile_component))
        per_tile_valid_component = np.sum(sq * mask3, axis=(1, 2)) / np.maximum(
            np.sum(mask3, axis=(1, 2)), 1.0
        )
        per_tile_valid_component_mse = float(np.mean(per_tile_valid_component))

        n_segments = batch_host.get("n_segments")
        if n_segments is None:
            n_segments = np.ones((sq.shape[0],), dtype=np.int32)
        n_segments = np.asarray(n_segments, dtype=np.int32)
        per_structure_component = []
        for batch_idx in range(sq.shape[0]):
            if "segment_id" in batch_host:
                seg = np.asarray(batch_host["segment_id"][batch_idx], dtype=np.int32)
                for seg_id in range(int(n_segments[batch_idx])):
                    seg_mask = seg == seg_id
                    if np.any(seg_mask):
                        per_structure_component.append(float(np.mean(sq[batch_idx][seg_mask])))
            else:
                valid = mask_np[batch_idx] > 0
                if np.any(valid):
                    per_structure_component.append(float(np.mean(sq[batch_idx][valid])))

        per_structure_component_mse = (
            float(np.mean(per_structure_component)) if per_structure_component else float("nan")
        )
        if self._force_loss_normalization == "per_structure_components":
            current_component_mse = per_structure_component_mse

        loss_views = {
            "current_component_mse": current_component_mse,
            "legacy_component_mse": legacy_component_mse,
            "valid_component_mse": valid_component_mse,
            "valid_bead_vector_mse": valid_bead_vector_mse,
            "per_tile_component_mse": per_tile_component_mse,
            "per_tile_valid_component_mse": per_tile_valid_component_mse,
            "per_structure_component_mse": per_structure_component_mse,
            "padding_dilution_ratio": legacy_component_mse / max(valid_component_mse, 1e-12),
        }
        fill_ratio = batch_host.get("meta_fill_ratio")
        if fill_ratio is not None and fill_ratio.size > 1:
            corr = np.corrcoef(
                np.asarray(fill_ratio, dtype=np.float64),
                np.asarray(per_tile_valid_component, dtype=np.float64),
            )[0, 1]
            if np.isfinite(corr):
                loss_views["fill_ratio_vs_tile_loss_corr"] = float(corr)
        return loss_views

    def _record_profiled_step(
        self,
        trainer: Any,
        batch: Any,
        step_idx: int,
        params_before: Dict[str, Any],
        params_after: Dict[str, Any],
        train_loss: Any,
        curr_grad: Any,
        dispatch_ms: Optional[float] = None,
        barrier_ms: Optional[float] = None,
    ) -> None:
        """Capture optimizer-step accounting and optional loss re-evaluations."""
        if not self._should_batch_stats_this_rank():
            return

        batch_host = self._profile_batch_to_host(batch)
        batch_ids = batch_host.get("meta_batch_item_id")
        if batch_ids is None:
            return

        epoch_idx = int(getattr(trainer, "_epoch", 0))
        n_valid = np.asarray(batch_host.get("n_valid"), dtype=np.int32)
        n_segments = np.asarray(batch_host.get("n_segments"), dtype=np.int32)
        n_force = np.asarray(batch_host.get("meta_n_force_components"), dtype=np.int32)
        fill_ratio = np.asarray(batch_host.get("meta_fill_ratio"), dtype=np.float64)
        source_ids = np.asarray(batch_host.get("meta_source_structure_ids"), dtype=np.int32)
        valid_source_ids = source_ids[source_ids >= 0] if source_ids.size else np.asarray([], dtype=np.int32)

        param_norm = self._tree_l2_norm(params_before)
        update_norm = self._tree_l2_norm(
            jax.tree_util.tree_map(lambda new, old: new - old, params_after, params_before)
        )
        grad_norm = self._tree_l2_norm(curr_grad)
        train_loss_value = float(np.asarray(jax.device_get(train_loss)))

        fetch_record = self._pending_batch_fetch_profiles.popleft() if self._pending_batch_fetch_profiles else None

        record = {
            "epoch": epoch_idx,
            "step": int(step_idx),
            "item_ids": [int(x) for x in np.asarray(batch_ids).reshape(-1)],
            "n_items": int(np.asarray(batch_ids).reshape(-1).shape[0]),
            "n_structures": int(np.sum(n_segments, dtype=np.int64)),
            "n_valid_beads": int(np.sum(n_valid, dtype=np.int64)),
            "n_force_components": int(np.sum(n_force, dtype=np.int64)),
            "unique_structures": int(np.unique(valid_source_ids).size) if valid_source_ids.size else int(np.sum(n_segments, dtype=np.int64)),
            "mean_fill_ratio": float(np.mean(fill_ratio)) if fill_ratio.size else float("nan"),
            "train_loss": train_loss_value,
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "update_norm": update_norm,
            "update_to_param_ratio": update_norm / max(param_norm, 1e-12),
            "dispatch_ms": float(dispatch_ms) if dispatch_ms is not None else float("nan"),
            "barrier_ms": float(barrier_ms) if barrier_ms is not None else float("nan"),
            "device_valid_means": self._device_slot_means(n_valid.astype(np.float64)),
            "device_fill_means": self._device_slot_means(fill_ratio.astype(np.float64)),
            "device_segment_means": self._device_slot_means(n_segments.astype(np.float64)),
        }

        if fetch_record is not None:
            record["fetch_ms"] = float(fetch_record["fetch_ms"])
            record["fetch_refresh_before"] = bool(fetch_record["refresh_before"])
            record["fetch_cache_count"] = int(fetch_record["cache_count"])
            record["fetch_line_before"] = int(fetch_record["line_before"])
            record["fetch_line_after"] = int(fetch_record["line_after"])

        if self._dataset_profile is not None:
            record["structure_fraction"] = (
                record["unique_structures"]
                / max(self._dataset_profile["total_structures"], 1)
            )
            record["bead_fraction"] = (
                record["n_valid_beads"]
                / max(self._dataset_profile["total_valid_beads"], 1)
            )
            record["force_component_fraction"] = (
                record["n_force_components"]
                / max(self._dataset_profile["total_force_components"], 1)
            )

        if self._loss_profile_enabled and step_idx < self._loss_profile_steps:
            loss_views = self._compute_manual_loss_views(params_before, batch_host)
            if loss_views is not None:
                record.update(loss_views)

        self._profile_step_records.append(record)

        if step_idx % self._batch_stats_log_every == 0:
            log_msg = (
                "[BatchStats] epoch=%d step=%d mode=%s items=%d structures=%d unique_structures=%d "
                "valid_beads=%d force_components=%d fill_mean=%.3f train_loss=%.6e grad_norm=%.3e "
                "update_ratio=%.3e fetch_ms=%s refresh=%s item_ids=%s"
            )
            training_logger.info(
                log_msg,
                record["epoch"],
                record["step"],
                self._batch_mode,
                record["n_items"],
                record["n_structures"],
                record["unique_structures"],
                record["n_valid_beads"],
                record["n_force_components"],
                record["mean_fill_ratio"],
                record["train_loss"],
                record["grad_norm"],
                record["update_to_param_ratio"],
                f"{record['fetch_ms']:.3f}" if "fetch_ms" in record else "n/a",
                record.get("fetch_refresh_before", False),
                record["item_ids"][: min(8, len(record["item_ids"]))],
            )
            if "structure_fraction" in record:
                training_logger.info(
                    "[BatchStats] epoch=%d step=%d dataset_fraction structures=%.3f beads=%.3f force_components=%.3f device_valid=%s device_fill=%s",
                    record["epoch"],
                    record["step"],
                    record["structure_fraction"],
                    record["bead_fraction"],
                    record["force_component_fraction"],
                    record["device_valid_means"],
                    record["device_fill_means"],
                )
            if "current_component_mse" in record:
                training_logger.info(
                    "[LossViews] epoch=%d step=%d current_component=%.6e valid_component=%.6e per_structure=%.6e valid_bead_vector=%.6e padding_ratio=%.3f",
                    record["epoch"],
                    record["step"],
                    record["current_component_mse"],
                    record["valid_component_mse"],
                    record["per_structure_component_mse"],
                    record["valid_bead_vector_mse"],
                    record["padding_dilution_ratio"],
                )

    def _report_epoch_profiles(self) -> None:
        """Emit aggregated epoch summaries from the collected step records."""
        if (
            not self._epoch_summary_enabled
            or not self._should_batch_stats_this_rank()
            or not self._profile_step_records
        ):
            return

        by_epoch = defaultdict(list)
        for record in self._profile_step_records:
            by_epoch[int(record["epoch"])].append(record)

        sorted_epochs = sorted(by_epoch)
        for idx, epoch in enumerate(sorted_epochs):
            records = by_epoch[epoch]
            item_ids = []
            for record in records:
                item_ids.extend(record["item_ids"])
            unique_items = len(set(item_ids))
            mean_loss = float(np.mean([r["train_loss"] for r in records]))
            mean_grad = float(np.mean([r["grad_norm"] for r in records]))
            mean_update_ratio = float(np.mean([r["update_to_param_ratio"] for r in records]))
            mean_fill = float(np.mean([r["mean_fill_ratio"] for r in records]))
            total_structures = int(np.sum([r["n_structures"] for r in records], dtype=np.int64))
            total_valid_beads = int(np.sum([r["n_valid_beads"] for r in records], dtype=np.int64))
            total_force_components = int(np.sum([r["n_force_components"] for r in records], dtype=np.int64))
            unique_structures_seen = int(max(r["unique_structures"] for r in records)) if records else 0
            order_preview = item_ids[: min(24, len(item_ids))]
            training_logger.info(
                "[EpochProfile] epoch=%d steps=%d unique_items=%d total_structures=%d total_valid_beads=%d total_force_components=%d mean_fill=%.3f mean_train_loss=%.6e mean_grad_norm=%.3e mean_update_ratio=%.3e item_order_prefix=%s",
                epoch,
                len(records),
                unique_items,
                total_structures,
                total_valid_beads,
                total_force_components,
                mean_fill,
                mean_loss,
                mean_grad,
                mean_update_ratio,
                order_preview,
            )
            if self._dataset_profile is not None:
                training_logger.info(
                    "[EpochProfile] epoch=%d dataset_coverage items=%.3f structures<=%.3f beads=%.3f force_components=%.3f",
                    epoch,
                    unique_items / max(self._dataset_profile["n_items"], 1),
                    unique_structures_seen / max(self._dataset_profile["total_structures"], 1),
                    total_valid_beads / max(self._dataset_profile["total_valid_beads"], 1),
                    total_force_components / max(self._dataset_profile["total_force_components"], 1),
                )

            fetch_records = [r for r in records if "fetch_ms" in r]
            if fetch_records:
                fetch_values = np.asarray([r["fetch_ms"] for r in fetch_records], dtype=np.float64)
                refresh_values = np.asarray([r["fetch_ms"] for r in fetch_records if r.get("fetch_refresh_before")], dtype=np.float64)
                steady_values = np.asarray([r["fetch_ms"] for r in fetch_records if not r.get("fetch_refresh_before")], dtype=np.float64)
                training_logger.info(
                    "[EpochProfile] epoch=%d fetch_ms mean=%.3f p50=%.3f p95=%.3f refreshes=%d refresh_mean=%s steady_mean=%s",
                    epoch,
                    float(np.mean(fetch_values)),
                    float(np.percentile(fetch_values, 50)),
                    float(np.percentile(fetch_values, 95)),
                    int(refresh_values.size),
                    f"{float(np.mean(refresh_values)):.3f}" if refresh_values.size else "n/a",
                    f"{float(np.mean(steady_values)):.3f}" if steady_values.size else "n/a",
                )

            device_valid = defaultdict(list)
            device_fill = defaultdict(list)
            for record in records:
                for slot, value in record["device_valid_means"].items():
                    device_valid[slot].append(value)
                for slot, value in record["device_fill_means"].items():
                    device_fill[slot].append(value)
            if device_valid:
                valid_summary = {
                    slot: round(float(np.mean(values)), 2)
                    for slot, values in sorted(device_valid.items())
                }
                fill_summary = {
                    slot: round(float(np.mean(device_fill.get(slot, [float("nan")]))), 3)
                    for slot in sorted(device_valid)
                }
                training_logger.info(
                    "[EpochProfile] epoch=%d per_device_valid_beads=%s per_device_fill=%s",
                    epoch,
                    valid_summary,
                    fill_summary,
                )

            sampled_loss_records = [r for r in records if "current_component_mse" in r]
            if sampled_loss_records:
                training_logger.info(
                    "[EpochProfile] epoch=%d sampled_loss_views current_component=%.6e valid_component=%.6e per_structure=%.6e valid_bead_vector=%.6e padding_ratio=%.3f",
                    epoch,
                    float(np.mean([r["current_component_mse"] for r in sampled_loss_records])),
                    float(np.mean([r["valid_component_mse"] for r in sampled_loss_records])),
                    float(np.mean([r["per_structure_component_mse"] for r in sampled_loss_records])),
                    float(np.mean([r["valid_bead_vector_mse"] for r in sampled_loss_records])),
                    float(np.mean([r["padding_dilution_ratio"] for r in sampled_loss_records])),
                )

            if idx > 0:
                prev_records = by_epoch[sorted_epochs[idx - 1]]
                prev_items = set(item for record in prev_records for item in record["item_ids"])
                curr_items = set(item_ids)
                union = prev_items | curr_items
                jaccard = len(prev_items & curr_items) / max(len(union), 1)
                training_logger.info(
                    "[EpochProfile] epoch=%d item_set_jaccard_vs_prev=%.3f",
                    epoch,
                    jaccard,
                )

    def _build_trace_dir(
        self, optimizer_name: str, start_epoch: int, remaining_epochs: int
    ) -> Path:
        """Build a unique trace output directory for one stage."""
        stage_end = start_epoch + remaining_epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = (
            f"stage_{optimizer_name}_rank{self._rank}_"
            f"epoch{start_epoch:04d}_to_{stage_end:04d}_{timestamp}"
        )
        return self._profiling_trace_dir / run_name

    def _start_jax_trace(
        self, optimizer_name: str, start_epoch: int, remaining_epochs: int
    ) -> Optional[Path]:
        """Start JAX profiler tracing for a stage."""
        if not self._should_trace_this_rank():
            return None

        trace_dir = self._build_trace_dir(optimizer_name, start_epoch, remaining_epochs)
        trace_dir.mkdir(parents=True, exist_ok=True)
        try:
            jax.profiler.start_trace(str(trace_dir))
            training_logger.info(f"[Profiling] Started JAX trace: {trace_dir}")
            return trace_dir
        except Exception as e:
            training_logger.warning(f"[Profiling] Failed to start JAX trace: {e}")
            return None

    def _stop_jax_trace(self, trace_dir: Optional[Path]) -> None:
        """Stop JAX profiler tracing if active."""
        if trace_dir is None:
            return
        try:
            jax.profiler.stop_trace()
            training_logger.info(f"[Profiling] Saved JAX trace to: {trace_dir}")
        except Exception as e:
            training_logger.warning(f"[Profiling] Failed to stop JAX trace cleanly: {e}")

    def _attach_batch_profiler(
        self, trainer, n_warmup: int = 5, n_samples: int = 50
    ) -> None:
        """
        Monkey-patch trainer._update_fn to record per-batch timing.

        Three timestamps are captured around each profiled _update_fn call:
          t0  — entry into _update_fn (dispatch start)
          t1  — return from _update_fn (dispatch end; GPU work is async at this point)
          t2  — return from jax.effects_barrier() (GPU compute complete)

        Derived metrics
        ---------------
        dispatch_ms  = t1 - t0   ~ time to queue the JIT work (should be ~0 if async)
        barrier_ms   = t2 - t1   ~ pure GPU compute time per batch
        gap_ms       = t0[i+1] - t0[i]  ~ wall time between batch starts

        Key diagnostic — gap / barrier ratio:
          ≈ 1.0  CPU blocks every step (current code: onp.asarray forces a sync per batch)
          ≈ 0.0  GPU is fully pipelined with data loading (target after async-sync fix)
        """
        call_ts: list = []
        dispatch_ts: list = []
        barrier_ts: list = []
        step = [0]
        original_fn = trainer._update_fn

        # Prefer effects_barrier (precise); fall back to block_until_ready on a dummy op.
        if hasattr(jax, "effects_barrier"):
            _barrier = jax.effects_barrier
        else:
            def _barrier():
                jax.block_until_ready(jnp.zeros(()))

        # Keep wrapper fully pass-through so profiler remains compatible when
        # _update_fn gains new keyword arguments (e.g., microbatch_count).
        def _timed_update_fn(*args, **kwargs):
            idx = step[0]
            step[0] += 1
            batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
            params_before = args[0] if len(args) > 0 else None
            timed_window = n_warmup <= idx < n_warmup + n_samples

            if not timed_window:
                result = original_fn(*args, **kwargs)
                if self._should_batch_stats_this_rank() and params_before is not None:
                    self._record_profiled_step(
                        trainer,
                        batch,
                        idx,
                        params_before,
                        result[0],
                        result[2],
                        result[3],
                    )
                return result

            t0 = time.perf_counter()
            result = original_fn(*args, **kwargs)
            t1 = time.perf_counter()
            _barrier()
            t2 = time.perf_counter()

            call_ts.append(t0)
            dispatch_ts.append(t1)
            barrier_ts.append(t2)
            self._log_edge_count_stats_for_batch(batch, idx)
            if self._should_batch_stats_this_rank() and params_before is not None:
                self._record_profiled_step(
                    trainer,
                    batch,
                    idx,
                    params_before,
                    result[0],
                    result[2],
                    result[3],
                    dispatch_ms=(t1 - t0) * 1e3,
                    barrier_ms=(t2 - t1) * 1e3,
                )
            return result

        trainer._update_fn = _timed_update_fn
        # Store on self so _report_batch_profiler can access after train() returns.
        self._batch_profiler_data = (call_ts, dispatch_ts, barrier_ts, n_warmup, n_samples)

    @staticmethod
    def _get_batch_field(batch: Any, key: str) -> Optional[Any]:
        """Best-effort accessor for batch containers used by chemtrain."""
        if batch is None:
            return None
        if isinstance(batch, dict):
            return batch.get(key)
        try:
            return batch[key]
        except Exception:
            pass
        return getattr(batch, key, None)

    @staticmethod
    def _coerce_neighbor_meta(value: Any) -> str:
        """Convert neighbor-list metadata (possibly device arrays) into text."""
        if value is None:
            return "None"
        try:
            host = jax.device_get(value)
            arr = np.asarray(host)
            if arr.shape == ():
                return str(arr.item())
            return str(arr)
        except Exception:
            return str(value)

    def _log_neighbor_debug_once(self) -> None:
        if not self._neighbor_debug_enabled or self._neighbor_debug_logged:
            return
        if self._neighbor_debug_rank0_only and self._rank != 0:
            return
        self._neighbor_debug_logged = log_neighbor_debug_once(self)

    def _edge_count_for_structure(self, R_sample: Any, mask_sample: Optional[Any]) -> Optional[Tuple[int, int, int]]:
        """
        Compute valid edge count for one structure using the model neighborlist update.

        Returns:
            Tuple (edge_count, valid_atom_count, edge_slots)
        """
        ml_model = getattr(self.model, "ml_model", None)
        if ml_model is None:
            return None
        if not hasattr(ml_model, "nneigh_fn") or not hasattr(ml_model, "nbrs_init"):
            return None

        compute_dtype = getattr(ml_model, "compute_dtype", jnp.float32)
        R_base = jnp.asarray(R_sample, dtype=compute_dtype)
        if mask_sample is None:
            valid_mask = jnp.ones((R_base.shape[0],), dtype=jnp.bool_)
        else:
            valid_mask = jnp.asarray(mask_sample) > 0

        if hasattr(ml_model, "_spread_padded_coordinates") and not getattr(ml_model, "_pbc", False):
            padded_mask = jnp.logical_not(valid_mask)
            R_safe = ml_model._spread_padded_coordinates(R_base, padded_mask)
            R_eval = jnp.where(valid_mask[:, None], R_base, jax.lax.stop_gradient(R_safe))
        else:
            R_eval = R_base

        nbrs = ml_model.nneigh_fn.update(R_eval, ml_model.nbrs_init)
        idx = np.asarray(jax.device_get(nbrs.idx))
        valid_mask_np = np.asarray(jax.device_get(valid_mask), dtype=bool)
        n_atoms = int(valid_mask_np.shape[0])
        if n_atoms <= 0:
            return 0, 0, int(idx.size)

        if idx.ndim == 2 and idx.shape[0] == 2:
            senders, receivers = idx[0], idx[1]
            in_range = (
                (senders >= 0)
                & (senders < n_atoms)
                & (receivers >= 0)
                & (receivers < n_atoms)
            )
            senders_safe = np.where(in_range, senders, 0)
            receivers_safe = np.where(in_range, receivers, 0)
            edge_valid = (
                in_range
                & valid_mask_np[senders_safe]
                & valid_mask_np[receivers_safe]
            )
            edge_slots = int(idx.shape[1])
        else:
            in_range = (idx >= 0) & (idx < n_atoms)
            idx_safe = np.where(in_range, idx, 0)
            neighbor_valid = valid_mask_np[idx_safe]
            center_valid = valid_mask_np[:, None]
            edge_valid = in_range & neighbor_valid & center_valid
            edge_slots = int(idx.size)

        edge_count = int(np.sum(edge_valid, dtype=np.int64))
        valid_atoms = int(np.sum(valid_mask_np, dtype=np.int64))
        return edge_count, valid_atoms, edge_slots

    def _log_edge_count_stats_for_batch(self, batch: Any, step_idx: int) -> None:
        """Log sampled per-batch neighbor edge statistics."""
        if not self._edge_profiler_enabled:
            return
        if self._edge_profiler_rank0_only and self._rank != 0:
            return
        if step_idx % self._edge_profiler_stride != 0:
            return

        R_batch = self._get_batch_field(batch, "R")
        mask_batch = self._get_batch_field(batch, "mask")
        if R_batch is None:
            if not self._edge_profiler_warned_missing_batch:
                training_logger.warning(
                    "[EdgeProfiler] Could not access batch['R']; disabling edge logging."
                )
                self._edge_profiler_warned_missing_batch = True
            return

        try:
            n_struct_total = int(R_batch.shape[0])
        except Exception:
            if not self._edge_profiler_warned_missing_batch:
                training_logger.warning(
                    "[EdgeProfiler] Unexpected batch shape for 'R'; disabling edge logging."
                )
                self._edge_profiler_warned_missing_batch = True
            return

        n_struct = min(self._edge_profiler_structures, n_struct_total)
        edge_counts = []
        valid_atoms = []
        edge_slots = []
        for i in range(n_struct):
            mask_i = None if mask_batch is None else mask_batch[i]
            try:
                stats = self._edge_count_for_structure(R_batch[i], mask_i)
            except Exception as e:
                training_logger.warning(
                    "[EdgeProfiler] Failed to compute edge stats (%s). Disabling edge logging.",
                    e,
                )
                self._edge_profiler_enabled = False
                return
            if stats is None:
                return
            e_count, v_atoms, e_slots = stats
            edge_counts.append(float(e_count))
            valid_atoms.append(float(v_atoms))
            edge_slots.append(float(e_slots))

        if not edge_counts:
            return

        edges = np.asarray(edge_counts, dtype=np.float64)
        atoms = np.asarray(valid_atoms, dtype=np.float64)
        slots = np.asarray(edge_slots, dtype=np.float64)
        occupancy = edges / np.maximum(slots, 1.0)
        mean_edges = float(np.mean(edges))
        delta = (
            mean_edges - self._edge_profiler_prev_mean
            if self._edge_profiler_prev_mean is not None
            else None
        )
        self._edge_profiler_prev_mean = mean_edges

        if delta is None:
            training_logger.info(
                "[EdgeProfiler] step=%d sample_n=%d edge_count mean=%.1f min=%.1f max=%.1f "
                "valid_atoms mean=%.1f occ=%.4f",
                step_idx,
                n_struct,
                mean_edges,
                float(np.min(edges)),
                float(np.max(edges)),
                float(np.mean(atoms)),
                float(np.mean(occupancy)),
            )
        else:
            training_logger.info(
                "[EdgeProfiler] step=%d sample_n=%d edge_count mean=%.1f min=%.1f max=%.1f "
                "valid_atoms mean=%.1f occ=%.4f delta_mean=%+.1f",
                step_idx,
                n_struct,
                mean_edges,
                float(np.min(edges)),
                float(np.max(edges)),
                float(np.mean(atoms)),
                float(np.mean(occupancy)),
                float(delta),
            )

    def _report_batch_profiler(self) -> None:
        """Log batch-profiler statistics collected by _attach_batch_profiler."""
        if not hasattr(self, "_batch_profiler_data"):
            return

        call_ts, dispatch_ts, barrier_ts, n_warmup, n_samples = self._batch_profiler_data
        del self._batch_profiler_data  # clean up

        n = len(call_ts)
        if n < 2:
            training_logger.warning("[BatchProfiler] Too few samples collected (n=%d).", n)
            return

        dispatch_ms = np.array([(dispatch_ts[i] - call_ts[i]) * 1e3 for i in range(n)])
        barrier_ms  = np.array([(barrier_ts[i]  - dispatch_ts[i]) * 1e3 for i in range(n)])
        gap_ms      = np.array([(call_ts[i + 1] - call_ts[i]) * 1e3 for i in range(n - 1)])

        def _fmt(arr):
            return (
                f"mean={arr.mean():.2f} ± {arr.std():.2f} ms  "
                f"p50={np.median(arr):.2f}  p95={np.percentile(arr, 95):.2f}"
            )

        mean_dispatch = float(np.mean(dispatch_ms))
        mean_barrier = float(np.mean(barrier_ms))
        ratio = float(np.mean(gap_ms) / max(mean_barrier, 1e-6))

        training_logger.info(
            "\n[BatchProfiler] Per-batch timing (%d samples, %d warmup skipped):",
            n, n_warmup,
        )
        training_logger.info("  dispatch_fn  : %s", _fmt(dispatch_ms))
        training_logger.info("  gpu_barrier  : %s", _fmt(barrier_ms))
        training_logger.info("  inter-batch gap: %s", _fmt(gap_ms))
        training_logger.info(
            "  gap / barrier ratio: %.3f  "
            "(1.0 = CPU blocks each step; 0.0 = GPU fully pipelined)",
            ratio,
        )

        if mean_dispatch > max(5.0 * mean_barrier, 5.0):
            training_logger.warning(
                "  [!!] _update_fn appears synchronous (dispatch includes compute). "
                "The large inter-batch gap is outside _update_fn (likely dataloader/Python loop overhead)."
            )
        elif ratio > 0.8:
            training_logger.warning(
                "  [!!] CPU is BLOCKING on every batch step. "
                "The onp.asarray() syncs in chemtrain._update are the dominant overhead. "
                "Deferred batch-sync fix will reduce this gap."
            )
        elif ratio < 0.3:
            training_logger.info(
                "  [OK] GPU is well-pipelined. Async dispatch is working."
            )
        else:
            training_logger.info(
                "  [~] Partial pipelining — some async benefit but host overhead visible."
            )

    @staticmethod
    def _block_until_ready(tree: Any) -> None:
        """Block host until device work for a pytree is complete."""
        try:
            jax.block_until_ready(tree)
            return
        except Exception:
            pass

        # Fallback: block every leaf individually
        leaves = jax.tree_util.tree_leaves(tree)
        for leaf in leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    @staticmethod
    def _to_int_scalar(value: Any, default: int = -1) -> int:
        """Safely convert scalar-like arrays to Python ints for logging."""
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return default
            return int(arr.reshape(-1)[0])
        except Exception:
            return default

    def _record_loader_setup(self, stage: str, label: str, elapsed_ms: float, **extra: Any) -> None:
        """Store and optionally log loader/tile setup timings."""
        record = {"stage": stage, "label": label, "elapsed_ms": float(elapsed_ms)}
        record.update(extra)
        self._loader_setup_records.append(record)
        if self._loader_timing_enabled and self._should_batch_stats_this_rank():
            details = " ".join(f"{key}={value}" for key, value in extra.items())
            training_logger.info(
                "[LoaderSetup] stage=%s label=%s elapsed_ms=%.3f%s%s",
                stage,
                label,
                float(elapsed_ms),
                " " if details else "",
                details,
            )

    def _install_batch_fetch_profiler(self, trainer: Any, stage: str = "training") -> None:
        """Wrap chemtrain's host-side batch supplier to time batch fetches directly."""
        if not self._loader_timing_enabled:
            return
        original = getattr(trainer, "_get_batch_fns", {}).get(stage)
        if original is None or getattr(original, "_cameo_loader_profile_wrapped", False):
            return

        def _wrapped_batch_fn(state, information: bool = False):
            line_before = self._to_int_scalar(getattr(state, "current_line", -1))
            cache_count = self._to_int_scalar(getattr(state, "cached_batches_count", -1))
            refresh_before = cache_count >= 0 and line_before == cache_count
            t_start = time.perf_counter()
            new_state, train_batch = original(state, information=information)
            t_end = time.perf_counter()
            fetch_record = {
                "stage": stage,
                "fetch_ms": (t_end - t_start) * 1e3,
                "refresh_before": bool(refresh_before),
                "line_before": int(line_before),
                "cache_count": int(cache_count),
                "line_after": self._to_int_scalar(getattr(new_state, "current_line", -1)),
                "information": bool(information),
            }
            self._batch_fetch_records.append(fetch_record)
            if stage == "training" and not information:
                self._pending_batch_fetch_profiles.append(fetch_record)
            return new_state, train_batch

        _wrapped_batch_fn._cameo_loader_profile_wrapped = True
        trainer._get_batch_fns[stage] = _wrapped_batch_fn
        if self._should_batch_stats_this_rank():
            training_logger.info("[Profiling] Wrapped host batch fetch for stage=%s", stage)

    def _split_loader_kwargs(self, split: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Build loader kwargs while preserving auxiliary metadata arrays."""
        loader_kwargs = {"copy": False}
        for key, value in split.items():
            if key in (
                "R",
                "F",
                "mask",
                "species",
                "segment_id",
                "force_loss_mask",
                "force_loss_weights",
                "DSM",
                "dsm_eps",
                "dsm_sigma",
                "dsm_loss_mask",
                *HVP_FIELD_KEYS,
                *SAFETY_FIELD_KEYS,
            ):
                loader_kwargs[key] = value
            elif key.startswith("meta_") or key in ("n_valid", "n_segments"):
                loader_kwargs[key] = value
        return loader_kwargs

    def _build_epoch_tiled_split(self, epoch_idx: int) -> Dict[str, np.ndarray]:
        """Rebuild tiled training data for a specific epoch using the untiled source."""
        if self._tiled_train_source is None:
            raise ValueError("Missing tiled_train_source for epoch-wise tile rebuilding.")

        epoch_seed = int(self._seed + epoch_idx)
        t_build_start = time.perf_counter()
        train_source = self._tiled_train_source
        if self._noised_residual_enabled:
            train_source = attach_noised_residual_fields(
                train_source,
                self.config,
                id_to_aa=self._noised_id_to_aa,
                seed=epoch_seed,
                split_seed=epoch_seed,
                fitted_params=self._noised_fitted_params,
            )
        tiled = build_tiled_dataset(
            R=train_source["R"],
            F=train_source["F"],
            mask=train_source["mask"],
            species=train_source["species"],
            structure_ids=train_source.get("structure_ids"),
            target_beads=int(self._tile_target_beads),
            bucket_beads=self._tile_bucket_beads,
            target_edges=self._tile_target_edges,
            bucket_edges=self._tile_bucket_edges,
            edge_estimate_scale=self._tile_edge_estimate_scale,
            edge_estimate_mode=self._tile_edge_estimate_mode,
            edge_estimate_cutoff=self._tile_edge_estimate_cutoff,
            shuffle_structures=self._tile_shuffle_structures,
            sort_by_size=self._tile_sort_by_size,
            sort_by_estimated_edges=self._tile_sort_by_estimated_edges,
            drop_incomplete=self._tile_drop_incomplete,
            isolate_large_structures=self._tile_isolate_large_structures,
            large_structure_threshold=self._tile_large_structure_threshold,
            large_structure_edge_threshold=self._tile_large_structure_edge_threshold,
            spatial_separation=self._tile_spatial_separation,
            structure_gap=self._tile_structure_gap,
            seed=epoch_seed,
            extra_per_atom_fields={
                key: np.asarray(train_source[key], dtype=np.float32)
                for key in HVP_FIELD_KEYS
                if key in train_source
            },
        )
        t_build_end = time.perf_counter()
        tiled = attach_batch_metadata(
            tiled, np.arange(tiled["R"].shape[0], dtype=np.int32)
        )
        if dsm_enabled(self.config):
            tiled = add_dsm_noise_fields(tiled, self.config, seed=epoch_seed)
        t_meta_end = time.perf_counter()
        build_ms = (t_build_end - t_build_start) * 1e3
        metadata_ms = (t_meta_end - t_build_end) * 1e3
        total_ms = (t_meta_end - t_build_start) * 1e3
        self._record_loader_setup(
            "training",
            "epoch_tile_rebuild",
            total_ms,
            epoch=epoch_idx,
            build_ms=f"{build_ms:.3f}",
            metadata_ms=f"{metadata_ms:.3f}",
            tiles=int(tiled["R"].shape[0]),
            sort_by_size=self._tile_sort_by_size,
        )
        training_logger.info(
            "[Tiling][EpochBuild] epoch=%d seed=%d tiles=%d mean_structures_per_tile=%.2f "
            "mean_valid_beads=%.1f fill_ratio=%.3f mean_est_edges=%.1f max_est_edges=%.1f "
            "sort_by_size=%s sort_by_estimated_edges=%s isolate_large=%s "
            "large_bead_threshold=%s large_edge_threshold=%s "
            "spatial_separation=%s structure_gap=%.2f build_ms=%.3f metadata_ms=%.3f",
            epoch_idx,
            epoch_seed,
            int(tiled["R"].shape[0]),
            float(np.mean(tiled["n_segments"])),
            float(np.mean(tiled["n_valid"])),
            float(np.mean(tiled["meta_fill_ratio"])),
            float(np.mean(tiled.get("meta_estimated_edges", np.zeros((1,), dtype=np.float32)))),
            float(np.max(tiled.get("meta_estimated_edges", np.zeros((1,), dtype=np.float32)))),
            self._tile_sort_by_size,
            self._tile_sort_by_estimated_edges,
            self._tile_isolate_large_structures,
            self._tile_large_structure_threshold,
            self._tile_large_structure_edge_threshold,
            self._tile_spatial_separation,
            self._tile_structure_gap,
            build_ms,
            metadata_ms,
        )
        return tiled

    def _install_epochwise_tile_rebuild(self, trainer: Any, stage_start_epoch: int) -> None:
        """Refresh tiled loader composition at each epoch boundary."""
        if not self._tile_rebuild_each_epoch:
            return
        if self._tiled_train_source is None:
            training_logger.warning(
                "[Tiling] tile_rebuild_each_epoch=true but no untiled source split was provided; keeping static tiles."
            )
            return

        def _refresh_tiles(chemtrain_trainer, *args, **kwargs):
            epoch_idx = int(stage_start_epoch + getattr(chemtrain_trainer, "_epoch", 0))
            if (
                not self._config_tile_rebuild_each_epoch
                and self._noised_residual_enabled
                and self._noised_refresh_interval_epochs > 1
            ):
                if epoch_idx % self._noised_refresh_interval_epochs != 0:
                    return
            train_split = self._build_epoch_tiled_split(epoch_idx)
            train_loader = NumpyDataLoader(**self._split_loader_kwargs(train_split))
            chemtrain_trainer.set_loader(train_loader, stage="training")
            chemtrain_trainer.set_loader(train_loader, stage="validation")
            self.train_loader = train_loader
            self.val_loader = train_loader
            self._set_dataset_profile(train_split, log=False)

        trainer.add_task("pre_epoch", _refresh_tiles)

    def _build_refreshed_dsm_split(self, refresh_idx: int) -> Dict[str, np.ndarray]:
        """Rebuild the training split used for DSM noise refresh."""
        if self._batch_mode == "tiled":
            if self._tiled_train_source is None:
                raise ValueError("Missing tiled_train_source for DSM tile refresh.")
            return self._build_epoch_tiled_split(refresh_idx)

        if self._dsm_standard_train_source is None:
            raise ValueError("Missing standard training source for DSM noise refresh.")
        split = {
            key: np.asarray(value)
            for key, value in self._dsm_standard_train_source.items()
        }
        return add_dsm_noise_fields(split, self.config, seed=int(self._seed + refresh_idx))

    def _install_dsm_step_refresh(self, trainer: Any) -> None:
        """Regenerate DSM noise, and tiled packing when needed, every N optimizer steps."""
        if not self._dsm_cfg["enabled"] or self._dsm_refresh_interval_steps <= 0:
            return
        if self._batch_mode == "tiled" and self._tiled_train_source is None:
            training_logger.warning(
                "[DSM] refresh_interval_steps=%d requested, but tiled_train_source is missing; "
                "keeping fixed DSM noise.",
                self._dsm_refresh_interval_steps,
            )
            return
        if self._batch_mode != "tiled" and self._dsm_standard_train_source is None:
            training_logger.warning(
                "[DSM] refresh_interval_steps=%d requested, but standard train source is missing; "
                "keeping fixed DSM noise.",
                self._dsm_refresh_interval_steps,
            )
            return

        def _refresh_after_batch(chemtrain_trainer, *args, **kwargs):
            self._dsm_optimizer_steps += 1
            if self._dsm_optimizer_steps % self._dsm_refresh_interval_steps != 0:
                return

            self._dsm_refresh_count += 1
            refresh_idx = int(self._dsm_refresh_count)
            train_split = self._build_refreshed_dsm_split(refresh_idx)
            train_loader = NumpyDataLoader(**self._split_loader_kwargs(train_split))
            chemtrain_trainer.set_loader(train_loader, stage="training")
            self.train_loader = train_loader
            self._set_dataset_profile(train_split, log=False)
            training_logger.info(
                "[DSM] Refreshed training noise%s after optimizer_step=%d (refresh=%d).",
                " and rebuilt tiles" if self._batch_mode == "tiled" else "",
                self._dsm_optimizer_steps,
                refresh_idx,
            )

        trainer.add_task("post_batch", _refresh_after_batch)

    def _force_matching_error_fns(self) -> Optional[Dict[str, Callable]]:
        """Return custom per-target error functions for chemtrain."""
        fns = {}
        if self._force_loss_normalization in ("valid_components", "per_structure_components"):
            fns["F"] = valid_component_mse
        if self._dsm_cfg["enabled"]:
            fns["DSM"] = dsm_error
        if self._hvp_cfg["enabled"]:
            fns["HVP"] = hvp_error
        if self._safety_cfg["enabled"]:
            fns.update(safety_error_fns(self.config))
        return fns or None

    def _force_matching_weights_keys(self) -> Optional[Dict[str, str]]:
        """Return dataset weight-key mapping for custom chemtrain losses."""
        keys = {}
        if self._force_loss_normalization == "valid_components":
            keys["F"] = "force_loss_mask"
        elif self._force_loss_normalization == "per_structure_components":
            keys["F"] = "force_loss_weights"
        if self._dsm_cfg["enabled"]:
            keys["DSM"] = "dsm_loss_mask"
        if self._hvp_cfg["enabled"]:
            keys["HVP"] = str(self._hvp_cfg["loss_mask_key"])
        if self._safety_cfg["enabled"]:
            keys.update(safety_weights_keys(self.config))
        return keys or None

    def _force_matching_additional_targets(self) -> Optional[Dict[str, Callable]]:
        """Return Chemtrain additional target quantities."""
        targets: Dict[str, Callable] = {}
        if self._dsm_cfg["enabled"]:
            targets["DSM"] = make_dsm_quantity(
                self.model.dsm_energy_fn_template,
                kT=float(self._dsm_cfg["kT"]),
            )
        if self._hvp_cfg["enabled"]:
            hvp_energy_template = getattr(self.model, "hvp_energy_fn_template", None)
            if hvp_energy_template is None:
                hvp_energy_template = self.model.energy_fn_template
            targets["HVP"] = make_hvp_quantity(
                hvp_energy_template,
                probe_key=str(self._hvp_cfg["probe_key"]),
            )
        if self._safety_cfg["enabled"]:
            targets.update(make_safety_quantities(self.model, self.config))
        return targets or None

    def _loader_reference_data(self, loader: Any) -> Dict[str, Any]:
        """Extract loader arrays while preserving auxiliary batch metadata."""
        if hasattr(loader, "reference_data"):
            reference_data = getattr(loader, "reference_data")
            if isinstance(reference_data, dict):
                return {key: value for key, value in reference_data.items()}

        n_samples = None
        if hasattr(loader, "R"):
            n_samples = int(np.asarray(loader.R).shape[0])
        elif hasattr(loader, "n_frames"):
            n_samples = int(loader.n_frames)
        if n_samples is None:
            raise ValueError("Could not infer sample count when converting loader to NumpyDataLoader.")

        reference_data = {}
        for key, value in vars(loader).items():
            if key.startswith("_"):
                continue
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) == 0 or int(shape[0]) != n_samples:
                continue
            reference_data[key] = value
        return reference_data

    def _numpy_loader_reference_data(self, loader: NumpyDataLoader) -> Dict[str, Any]:
        if hasattr(loader, "reference_data"):
            reference_data = getattr(loader, "reference_data")
            if isinstance(reference_data, dict):
                return reference_data
        if hasattr(loader, "_reference_data"):
            reference_data = getattr(loader, "_reference_data")
            if isinstance(reference_data, dict):
                return reference_data
        return self._loader_reference_data(loader)

    def _validate_hvp_reference_data(self, reference_data: Dict[str, Any], label: str) -> None:
        if not self._hvp_cfg["enabled"] or not self._hvp_cfg.get("require_targets", True):
            return
        missing = [
            key
            for key in (str(self._hvp_cfg["probe_key"]), str(self._hvp_cfg["target_key"]))
            if key not in reference_data
        ]
        if missing:
            raise ValueError(
                f"training.hvp.enabled=true requires {label} batch data to contain "
                f"{missing}; available keys: {sorted(reference_data.keys())}"
            )

    def _create_chemtrain_loaders(self) -> DataLoaders:
        """
        Create chemtrain DataLoaders from our loaders.

        Returns:
            chemtrain.data.data_loaders.DataLoaders instance
        """
        # Convert our DatasetLoader to NumpyDataLoader if needed.
        # DatasetLoader stores NumPy arrays, so no device transfer is required.
        if not isinstance(self.train_loader, NumpyDataLoader):
            train_reference_data = self._loader_reference_data(self.train_loader)
            self._validate_hvp_reference_data(train_reference_data, "training")
            train_np_loader = NumpyDataLoader(
                copy=False,
                **train_reference_data,
            )
        else:
            self._validate_hvp_reference_data(self._numpy_loader_reference_data(self.train_loader), "training")
            train_np_loader = self.train_loader

        if self.val_loader is not None:
            if not isinstance(self.val_loader, NumpyDataLoader):
                val_reference_data = self._loader_reference_data(self.val_loader)
                self._validate_hvp_reference_data(val_reference_data, "validation")
                val_np_loader = NumpyDataLoader(
                    copy=False,
                    **val_reference_data,
                )
            else:
                self._validate_hvp_reference_data(self._numpy_loader_reference_data(self.val_loader), "validation")
                val_np_loader = self.val_loader
        else:
            val_np_loader = train_np_loader  # Use training data for validation

        return DataLoaders(
            train_loader=train_np_loader,
            val_loader=val_np_loader,
            test_loader=None
        )

    def train_stage(
        self,
        optimizer_name: str,
        epochs: int,
        start_epoch: int = 0,
        checkpoint_freq: int = 0
    ) -> StageResult:
        """
        Train for a single stage with a specific optimizer.

        Args:
            optimizer_name: Name of optimizer (e.g., "adabelief", "yogi")
            epochs: Total number of epochs for this stage
            start_epoch: Epoch to start from (for resume, default 0)
            checkpoint_freq: Save checkpoint every N epochs (0 = only at end)

        Returns:
            Dictionary with final losses
        """
        remaining_epochs = epochs - start_epoch
        if remaining_epochs <= 0:
            training_logger.info(f"Stage {optimizer_name} already complete (epoch {start_epoch}/{epochs})")
            return {"train_loss": 0.0, "val_loss": 0.0, "skipped": True}

        training_logger.info(f"\n{'='*60}")
        training_logger.info(f"Training Stage: {optimizer_name.upper()} ({remaining_epochs} epochs, starting from {start_epoch})")
        training_logger.info(f"{'='*60}")

        # Create optimizer
        optimizer = create_optimizer_from_config(self.config, optimizer_name)

        # Create chemtrain loaders
        t_loader_start = time.perf_counter()
        loaders = self._create_chemtrain_loaders()
        t_loader_end = time.perf_counter()
        self._record_loader_setup(
            "training",
            "create_chemtrain_loaders",
            (t_loader_end - t_loader_start) * 1e3,
            batch_mode=self._batch_mode,
            batch_per_device=self.batch_per_device,
        )

        # Create energy function template
        energy_fn_template = self.model.energy_fn_template

        # Create ForceMatching trainer
        t_trainer_init_start = time.perf_counter()
        trainer = ForceMatching(
            init_params=self.params,
            optimizer=optimizer,
            energy_fn_template=energy_fn_template,
            nbrs_init=self.model.initial_neighbors,
            gammas=self.gammas,
            error_fns=self._force_matching_error_fns(),
            weights_keys=self._force_matching_weights_keys(),
            additional_targets=self._force_matching_additional_targets(),
            checkpoint_path=str(self.checkpoint_path),
            batch_per_device=self.batch_per_device,
            batch_cache=self.batch_cache,
            disable_shmap=False,
        )
        t_trainer_init_end = time.perf_counter()
        self._record_loader_setup(
            "training",
            "force_matching_init",
            (t_trainer_init_end - t_trainer_init_start) * 1e3,
            global_batch_size=self._global_batch_size,
            batch_cache=self.batch_cache,
        )

        # Set loaders
        t_set_train_loader_start = time.perf_counter()
        trainer.set_loader(loaders.train_loader, stage="training")
        t_set_train_loader_end = time.perf_counter()
        self._record_loader_setup(
            "training",
            "set_loader_training",
            (t_set_train_loader_end - t_set_train_loader_start) * 1e3,
            observations=int(loaders.train_loader.static_information["observation_count"]),
        )
        t_set_val_loader_start = time.perf_counter()
        trainer.set_loader(loaders.val_loader, stage="validation")
        t_set_val_loader_end = time.perf_counter()
        self._record_loader_setup(
            "validation",
            "set_loader_validation",
            (t_set_val_loader_end - t_set_val_loader_start) * 1e3,
            observations=int(loaders.val_loader.static_information["observation_count"]),
        )
        self._install_batch_fetch_profiler(trainer, stage="training")
        self._install_epochwise_tile_rebuild(trainer, stage_start_epoch=start_epoch)
        self._install_dsm_step_refresh(trainer)
        self._log_neighbor_debug_once()

        # Restore optimizer state from checkpoint if available.
        # This ensures the LR schedule continues from where it left off instead of
        # restarting from step 0, which would cause the LR to jump to its initial value.
        if self._resume_opt_state is not None:
            try:
                restored_opt_state = jax.tree_util.tree_map(jnp.asarray, self._resume_opt_state)
                # TrainerState is a NamedTuple; use _replace to create an updated copy
                trainer.state = trainer.state._replace(opt_state=restored_opt_state)
                training_logger.info("Restored optimizer state from checkpoint (LR schedule continues)")
            except Exception as e:
                training_logger.warning(
                    f"Could not restore optimizer state: {e}. "
                    "LR schedule will restart from step 0."
                )
            finally:
                self._resume_opt_state = None  # Only restore once per resume

        # Attach per-batch timing profiler if requested (non-invasive monkey-patch).
        # Must be done BEFORE trainer.train() so it intercepts from step 0.
        self._profile_step_records = []
        if self._should_batch_profile_this_rank() or self._should_batch_stats_this_rank():
            training_logger.info(
                "[BatchProfiler] Attaching to _update_fn "
                f"(warmup={self._batch_profiler_warmup}, "
                f"samples={self._batch_profiler_samples})"
            )
            self._attach_batch_profiler(
                trainer,
                n_warmup=self._batch_profiler_warmup,
                n_samples=self._batch_profiler_samples,
            )

        # Train with periodic checkpointing
        stage_start_time = time.perf_counter()
        trace_dir = self._start_jax_trace(optimizer_name, start_epoch, remaining_epochs)
        trace_annotation = getattr(jax.profiler, "TraceAnnotation", None)
        try:
            if trace_annotation is not None:
                with trace_annotation(f"train_stage_{optimizer_name}"):
                    trainer.train(
                        remaining_epochs,
                        checkpoint_freq=checkpoint_freq if checkpoint_freq > 0 else None
                    )
            else:
                trainer.train(
                    remaining_epochs,
                    checkpoint_freq=checkpoint_freq if checkpoint_freq > 0 else None
                )
            self._block_until_ready(trainer.params)
        finally:
            self._stop_jax_trace(trace_dir)
            # Report batch profiler results even if training is interrupted mid-stage.
            if self._batch_profiler_enabled:
                self._report_batch_profiler()
            if self._batch_stats_enabled:
                self._report_epoch_profiles()
        stage_wall_seconds = time.perf_counter() - stage_start_time

        # Update parameters
        self.params = trainer.params
        self.best_params = trainer.best_inference_params
        self._chemtrain_trainer = trainer
        if self.model.use_priors and getattr(self.model, "train_priors", False) and "prior" in self.params:
            self.model.prior.params = self.params["prior"]

        # Save stage checkpoint with metadata for resume capability
        if checkpoint_freq > 0:
            self._save_stage_checkpoint(optimizer_name, epochs)

        # Extract gradient norm history (per-step, logged by chemtrain internally)
        grad_norms_raw = list(getattr(trainer, 'gradient_norm_history', []))
        grad_norms = [float(np.asarray(v)) for v in grad_norms_raw]
        if grad_norms:
            training_logger.info(
                f"Gradient norms — mean: {np.mean(grad_norms):.4e}, "
                f"max: {max(grad_norms):.4e}, "
                f"final: {grad_norms[-1]:.4e}"
            )
        # Store on self so _save_stage_checkpoint can include it in metadata
        self._last_gradient_norms = grad_norms

        # Compute total parameter L2 norm on-device (single scalar transfer)
        total_param_norm = float(jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda acc, v: acc + jnp.sum(v * v),
                self.params,
                initializer=jnp.float32(0.0),
            )
        ))
        training_logger.info(f"Total parameter L2 norm: {total_param_norm:.4e}")

        # Get final losses
        final_losses = {
            "train_loss": float(trainer.train_losses[-1]) if trainer.train_losses else 0.0,
            "val_loss": float(trainer.val_losses[-1]) if trainer.val_losses else 0.0,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_final": float(grad_norms[-1]) if grad_norms else 0.0,
            "param_norm": total_param_norm,
            "stage_wall_seconds": stage_wall_seconds,
            "stage_wall_minutes": stage_wall_seconds / 60.0,
            "epoch_wall_seconds_est": stage_wall_seconds / max(remaining_epochs, 1),
        }
        if trace_dir is not None:
            final_losses["jax_trace_dir"] = str(trace_dir)

        training_logger.info(
            f"Stage wall time: {stage_wall_seconds:.2f} s "
            f"({stage_wall_seconds / 60.0:.2f} min), "
            f"~{final_losses['epoch_wall_seconds_est']:.2f} s/epoch"
        )

        training_logger.info(f"\nStage complete: train_loss={final_losses['train_loss']:.6f}, "
                           f"val_loss={final_losses['val_loss']:.6f}")

        return final_losses

    def _save_stage_checkpoint(self, stage_name: str, completed_epochs: int):
        """
        Save checkpoint with stage metadata for resume capability.

        Args:
            stage_name: Name of the completed stage (e.g., "adabelief", "yogi")
            completed_epochs: Total epochs completed in this stage
        """
        import time

        checkpoint_file = self.checkpoint_path / f"stage_{stage_name}_epoch{completed_epochs}.pkl"
        meta_file = checkpoint_file.with_suffix(".meta.pkl")

        # Use chemtrain's save_trainer which includes optimizer state
        if self._chemtrain_trainer is not None:
            self._chemtrain_trainer.save_trainer(checkpoint_file)
            training_logger.info(f"Saved stage checkpoint: {checkpoint_file}")

            # Save metadata separately for resume logic
            metadata = {
                "stage": stage_name,
                "completed_epochs": completed_epochs,
                "timestamp": time.time(),
                "train_losses": list(self._chemtrain_trainer.train_losses),
                "val_losses": list(self._chemtrain_trainer.val_losses),
                "gradient_norm_history": getattr(self, '_last_gradient_norms', []),
            }
            with open(meta_file, 'wb') as f:
                pickle.dump(metadata, f)
            training_logger.info(f"Saved stage metadata: {meta_file}")

    def pretrain_prior(
        self,
        max_steps: int = None,
        tol_grad: float = None
    ) -> PretrainResult:
        """
        Pre-train prior energy parameters using LBFGS force matching.

        This optimizes ONLY the prior parameters to match reference forces,
        using the LBFGS optimizer as in the original implementation.

        Args:
            max_steps: Maximum LBFGS iterations (default from config)
            tol_grad: Gradient tolerance for convergence (default from config)

        Returns:
            Dictionary with keys: train_loss, val_loss, steps, converged,
            grad_norm, loss_history, fitted_params

        Note:
            Only works if model.use_priors is True.
            Uses LBFGS optimization with jax.lax.while_loop for convergence.
            Always uses LBFGS optimizer (not configurable).
            In multi-node mode, runs LBFGS on rank 0 only and broadcasts results.
        """
        # Read defaults from config
        if max_steps is None:
            max_steps = self.config.get_pretrain_prior_max_steps()
        if tol_grad is None:
            tol_grad = self.config.get_pretrain_prior_tol_grad()

        # Minimum steps before convergence check (from config)
        min_steps = self.config.get_pretrain_prior_min_steps()
        if not self.model.use_priors:
            training_logger.info("Skipping prior pre-training (use_priors=False)")
            return {"train_loss": 0.0, "val_loss": 0.0, "converged": True}

        prior = self.model.prior
        if prior is not None and getattr(prior, "uses_splines", False):
            training_logger.info(
                "Spline priors detected - skipping LBFGS prior pre-training "
                "(no parametric prior parameters to optimize)."
            )
            return {"train_loss": 0.0, "val_loss": 0.0, "converged": True}

        # Check if we're in multi-node distributed mode
        is_distributed = jax.process_count() > 1
        rank = jax.process_index()

        training_logger.info(f"\n{'='*60}")
        training_logger.info(f"Prior Pre-Training (LBFGS, max_steps={max_steps})")
        if is_distributed:
            training_logger.info(f"[Distributed] Running on rank 0, broadcasting to {jax.process_count()} processes")
        training_logger.info(f"{'='*60}")

        from typing import NamedTuple

        # Get training data (stored in __init__ to avoid _chains[0] access)
        if self._train_data is None:
            training_logger.error("Training data not available. Cannot perform prior pre-training.")
            training_logger.error("Ensure train_data parameter is passed to Trainer.__init__")
            raise ValueError("Training data required for prior pre-training")

        train_data = self._train_data

        # Get prior components
        displacement = prior.displacement
        bonds = prior.bonds
        angles = prior.angles
        rep_pairs = prior.rep_pairs

        # Initial prior parameters
        params0 = prior.params

        # Define prior force computation
        def prior_forces(params, R, mask, species):
            """Compute forces from prior energy only."""
            def energy_of_R(R_):
                return prior.compute_total_energy_from_params(
                    params, R_, mask, species=species
                )
            return -jax.grad(energy_of_R)(R)

        # Define force matching loss
        def force_matching_loss(params):
            """Compute L2 loss between predicted and reference forces."""
            R = train_data["R"]
            F_ref = train_data["F"]
            mask = train_data["mask"]
            species = train_data["species"]

            # Vectorized force prediction over batch
            F_pred = jax.vmap(
                lambda R_f, m_f, s_f: prior_forces(params, R_f, m_f, s_f)
            )(R, mask, species)

            # Masked squared error
            m3 = mask[..., None]  # Broadcast mask to (batch, atoms, 3)
            diff = (F_pred - F_ref) * m3

            # Normalize by number of real atoms
            denom = jnp.maximum(jnp.sum(m3), 1.0)
            return jnp.sum(diff * diff) / denom

        # Create LBFGS optimizer
        opt = optax.lbfgs(learning_rate=1.0)
        value_and_grad = optax.value_and_grad_from_state(force_matching_loss)

        # LBFGS state
        class FitState(NamedTuple):
            params: Dict[str, jax.Array]
            opt_state: optax.OptState
            step: jax.Array
            loss: jax.Array
            loss_hist: jax.Array

        # Initialize state
        def init_state(p0):
            opt_state = opt.init(p0)
            value0, grad0 = value_and_grad(p0, state=opt_state)
            loss_hist = jnp.full((max_steps,), jnp.nan, dtype=jnp.float32)
            loss_hist = loss_hist.at[0].set(value0.astype(jnp.float32))
            return FitState(
                params=p0,
                opt_state=opt_state,
                step=jnp.array(0, dtype=jnp.int32),
                loss=value0,
                loss_hist=loss_hist,
            )

        # Convergence condition
        def cond_fn(st: FitState):
            not_done = st.step < max_steps

            # Check gradient norm
            grad = optax.tree.get(st.opt_state, "grad")
            grad_norm = optax.tree.norm(grad)
            not_converged_grad = jnp.logical_or(st.step < min_steps, grad_norm >= tol_grad)

            return jnp.logical_and(not_done, not_converged_grad)

        # LBFGS update step
        def body_fn(st: FitState):
            p, s, k = st.params, st.opt_state, st.step

            # Compute value and gradient
            value, grad = value_and_grad(p, state=s)

            # LBFGS update
            updates, s_new = opt.update(
                grad, s, p,
                value=value,
                grad=grad,
                value_fn=force_matching_loss,
            )
            p_new = optax.apply_updates(p, updates)

            # Compute new loss
            value_new = force_matching_loss(p_new)

            # Record loss
            loss_hist = st.loss_hist.at[k].set(value.astype(jnp.float32))

            return FitState(
                params=p_new,
                opt_state=s_new,
                step=k + 1,
                loss=value_new,
                loss_hist=loss_hist,
            )

        # Run LBFGS optimization (only on rank 0 in distributed mode)
        if is_distributed:
            if rank == 0:
                training_logger.info("[LBFGS] Starting optimization on rank 0...")
                st0 = init_state(params0)
                stF = jax.lax.while_loop(cond_fn, body_fn, st0)

                # Extract results on rank 0
                fitted_params = stF.params
                loss_hist = stF.loss_hist
                final_step = int(stF.step)
                final_loss = float(stF.loss)

                # Get gradient norm for convergence check
                grad_final = optax.tree.get(stF.opt_state, "grad")
                grad_norm_final = float(optax.tree.norm(grad_final))
                converged = grad_norm_final < tol_grad

                training_logger.info(f"[LBFGS] Completed: {final_step} steps")
                training_logger.info(f"[LBFGS] Final loss: {final_loss:.6e}")
                training_logger.info(f"[LBFGS] Grad norm: {grad_norm_final:.6e} (tol={tol_grad:.6e})")
                training_logger.info(f"[LBFGS] Converged: {converged}")
            else:
                # Other ranks wait and will receive broadcasted params
                training_logger.info(f"[LBFGS] Rank {rank} waiting for broadcast from rank 0...")
                fitted_params = None
                final_step = 0
                final_loss = 0.0
                grad_norm_final = 0.0
                converged = False
                loss_hist = None

            # Broadcast fitted parameters from rank 0 to all other ranks
            # Use jax.experimental.multihost_utils for multi-process broadcast
            from jax.experimental import multihost_utils

            # Broadcast each parameter array individually
            if rank == 0:
                broadcast_params = fitted_params
            else:
                # Create placeholder with same structure as params0
                broadcast_params = jax.tree.map(lambda x: jnp.zeros_like(x), params0)

            # Synchronize parameters across all processes
            fitted_params = multihost_utils.broadcast_one_to_all(broadcast_params, is_source=(rank == 0))

            # Also broadcast scalar results
            final_step = int(multihost_utils.broadcast_one_to_all(
                jnp.array(final_step, dtype=jnp.int32), is_source=(rank == 0)
            ))
            final_loss = float(multihost_utils.broadcast_one_to_all(
                jnp.array(final_loss, dtype=jnp.float32), is_source=(rank == 0)
            ))
            grad_norm_final = float(multihost_utils.broadcast_one_to_all(
                jnp.array(grad_norm_final, dtype=jnp.float32), is_source=(rank == 0)
            ))
            converged = bool(multihost_utils.broadcast_one_to_all(
                jnp.array(converged, dtype=jnp.bool_), is_source=(rank == 0)
            ))

            training_logger.info(f"[LBFGS] Rank {rank} received broadcasted parameters")
        else:
            # Single-node mode: run LBFGS directly
            training_logger.info("[LBFGS] Starting optimization...")
            st0 = init_state(params0)
            stF = jax.lax.while_loop(cond_fn, body_fn, st0)

            # Extract results
            fitted_params = stF.params
            loss_hist = stF.loss_hist
            final_step = int(stF.step)
            final_loss = float(stF.loss)

            # Get gradient norm for convergence check
            grad_final = optax.tree.get(stF.opt_state, "grad")
            grad_norm_final = float(optax.tree.norm(grad_final))
            converged = grad_norm_final < tol_grad

            training_logger.info(f"[LBFGS] Completed: {final_step} steps")
            training_logger.info(f"[LBFGS] Final loss: {final_loss:.6e}")
            training_logger.info(f"[LBFGS] Grad norm: {grad_norm_final:.6e} (tol={tol_grad:.6e})")
            training_logger.info(f"[LBFGS] Converged: {converged}")

        # Update model parameters (all ranks now have the same fitted_params)
        self.model.prior.params = fitted_params
        if 'prior' in self.params:
            self.params['prior'] = fitted_params

        # Print fitted parameters (transfer only scalars/small arrays)
        training_logger.info("\n[LBFGS] Fitted parameters:")
        for key, val in fitted_params.items():
            if jnp.ndim(val) == 0:
                training_logger.info(f"  {key}: {float(val):.6f}")
            else:
                val_np = np.asarray(val)
                if val_np.size <= 10:
                    training_logger.info(f"  {key}: {val_np}")
                else:
                    training_logger.info(f"  {key}: shape={val_np.shape}, norm={np.linalg.norm(val_np):.6f}")

        # Prepare loss history (may be None for non-rank-0 in distributed mode)
        if loss_hist is not None:
            loss_history = np.array(loss_hist[:final_step])
        else:
            loss_history = np.array([])

        return {
            "train_loss": final_loss,
            "val_loss": final_loss,  # No separate validation in LBFGS
            "steps": final_step,
            "converged": converged,
            "grad_norm": grad_norm_final,
            "loss_history": loss_history,
            "fitted_params": {k: np.array(v) for k, v in fitted_params.items()},
        }

    def train_full_pipeline(
        self,
        resume_from: Optional[str] = None,
        checkpoint_freq: Optional[int] = None
    ) -> TrainingResults:
        """
        Run full training pipeline as configured in YAML.

        Reads training configuration and runs:
        1. Optional prior pre-training
        2. Stage 1 optimizer (e.g., AdaBelief)
        3. Stage 2 optimizer (e.g., Yogi)

        Args:
            resume_from: Path to checkpoint to resume from (optional)
            checkpoint_freq: Override checkpoint frequency from config (optional)

        Returns:
            Dictionary with training results
        """
        results = {}

        # Get checkpoint frequency from config if not overridden
        if checkpoint_freq is None:
            checkpoint_freq = self.config.get_checkpoint_freq()

        # Resume state tracking
        resume_stage = None
        resume_epoch = 0

        if resume_from:
            metadata = self.load_chemtrain_checkpoint(resume_from)
            resume_stage = metadata.get("stage", "unknown")
            resume_epoch = metadata.get("completed_epochs", 0)
            training_logger.info(f"Resuming from stage '{resume_stage}' at epoch {resume_epoch}")

        stages = self.config.get_training_stages()
        stage_names = [s["optimizer"] for s in stages]

        # Check if prior pre-training is enabled (skip if resuming from a training stage)
        pretrain_prior = self.config.pretrain_prior_enabled()
        skip_pretrain = resume_stage in stage_names or resume_stage in ["stage1", "stage2"]

        if pretrain_prior and self.model.use_priors and not skip_pretrain:
            max_steps = self.config.get_pretrain_prior_max_steps()
            tol_grad = self.config.get_pretrain_prior_tol_grad()
            results["prior_pretrain"] = self.pretrain_prior(
                max_steps=max_steps,
                tol_grad=tol_grad
            )

        # Iterate over configured training stages
        found_resume_stage = resume_stage is None
        for idx, stage in enumerate(stages):
            opt_name = stage["optimizer"]
            n_epochs = stage["epochs"]

            if n_epochs <= 0:
                continue

            # Skip stages that precede the resume point
            if not found_resume_stage:
                if resume_stage == opt_name or resume_stage == f"stage{idx+1}":
                    found_resume_stage = True
                else:
                    continue

            start_epoch = resume_epoch if resume_stage == opt_name else 0
            results[f"stage{idx+1}"] = self.train_stage(
                opt_name,
                n_epochs,
                start_epoch=start_epoch,
                checkpoint_freq=checkpoint_freq
            )
            # Only use resume_epoch for the stage we're resuming into
            resume_stage = None
            resume_epoch = 0

        return results

    def evaluate_frame(self, frame_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate model on a single frame.

        Args:
            frame_idx: Frame index to evaluate

        Returns:
            Dictionary with energy components and force errors
        """
        R = self.train_loader.R[frame_idx]
        F_ref = self.train_loader.F[frame_idx]
        mask = self.train_loader.mask[frame_idx]
        species = self.train_loader.species[frame_idx]

        # Compute energy components
        components = self.model.compute_components(
            self.best_params or self.params,
            R, mask, species
        )

        # Compute forces
        def energy_fn(R_):
            return self.model.compute_energy(
                self.best_params or self.params,
                R_, mask, species
            )

        F_pred = -jax.grad(energy_fn)(R)

        # Compute errors (only for real atoms)
        real_mask = mask > 0
        F_pred_real = F_pred[real_mask]
        F_ref_real = F_ref[real_mask]

        rmse = float(jnp.sqrt(jnp.mean((F_pred_real - F_ref_real) ** 2)))
        mae = float(jnp.mean(jnp.abs(F_pred_real - F_ref_real)))

        return {
            "energy_components": {k: float(v) for k, v in components.items()},
            "force_rmse": rmse,
            "force_mae": mae,
        }

    def save_params(self, output_path: str):
        """
        Save model parameters to pickle file.

        Args:
            output_path: Path to save parameters
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params_to_save = self.best_params if self.best_params is not None else self.params

        with open(output_path, 'wb') as f:
            pickle.dump(params_to_save, f)

        training_logger.info(f"Saved parameters to: {output_path}")

    def load_params(self, input_path: str):
        """
        Load model parameters from pickle file.

        Args:
            input_path: Path to load parameters from
        """
        input_path = Path(input_path)

        with open(input_path, 'rb') as f:
            self.params = pickle.load(f)

        self.best_params = self.params
        training_logger.info(f"Loaded parameters from: {input_path}")

    def initialize_params_from_checkpoint(
        self,
        input_path: str,
        source_key: str = "best_params",
    ) -> None:
        """Initialize model params from a checkpoint without resuming optimizer state."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {input_path}")

        with input_path.open("rb") as f:
            payload = pickle.load(f)

        params = payload
        key = str(source_key or "best_params")
        if isinstance(payload, dict):
            if key == "trainer_state.params":
                trainer_state = payload.get("trainer_state", {})
                params = trainer_state.get("params")
            elif key in payload:
                params = payload[key]
            elif key == "params" and isinstance(payload.get("trainer_state"), dict):
                params = payload["trainer_state"].get("params")
            else:
                raise KeyError(
                    f"Checkpoint {input_path} does not contain source_key={key!r}. "
                    f"Available top-level keys: {sorted(payload.keys())}"
                )

        if isinstance(params, dict) and "ml" not in params:
            params = {"ml": params}
        if not isinstance(params, dict) or "ml" not in params:
            raise TypeError(
                f"Unsupported initial params payload from {input_path}: {type(params)}"
            )

        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        self.best_params = self.params
        self._resume_opt_state = None
        training_logger.info(
            "Initialized model params from checkpoint %s (source_key=%s); "
            "optimizer state was not restored.",
            input_path,
            key,
        )

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from training."""
        return self.best_params if self.best_params is not None else self.params

    def save_checkpoint(self, output_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save full training checkpoint for resume capability.

        Args:
            output_path: Path to save checkpoint
            metadata: Optional metadata dict (e.g., current epoch, stage info)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "params": self.params,
            "best_params": self.best_params,
            "metadata": metadata or {},
        }

        with open(output_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        training_logger.info(f"Saved checkpoint to: {output_path}")

    def load_checkpoint(self, input_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint to resume training.

        Args:
            input_path: Path to checkpoint file

        Returns:
            Checkpoint metadata (e.g., epoch info)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {input_path}")

        with open(input_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]
        self.best_params = checkpoint.get("best_params", checkpoint["params"])
        metadata = checkpoint.get("metadata", {})

        training_logger.info(f"Loaded checkpoint from: {input_path}")
        if metadata:
            training_logger.info(f"Checkpoint metadata: {metadata}")

        return metadata

    def load_chemtrain_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a chemtrain trainer checkpoint for resumption.

        Supports two formats saved by chemtrain's save_trainer():
        1. Dict format (full_checkpoint=False, default): keys are 'trainer_state',
           'best_params', 'train_losses', 'val_losses', etc.
        2. Full trainer object (full_checkpoint=True): has .params, .opt_state, etc.

        Also restores the optimizer state so the LR schedule continues seamlessly.

        Args:
            checkpoint_path: Path to chemtrain checkpoint file (.pkl)

        Returns:
            Dictionary with resume metadata:
                - stage: Stage name (e.g., "adabelief", "yogi")
                - completed_epochs: Number of epochs completed
                - train_losses: Training loss history
                - val_losses: Validation loss history
        """
        import re
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            saved = pickle.load(f)

        # chemtrain's save_trainer() (full_checkpoint=False, the default) saves a plain
        # dict with keys: 'trainer_state' (containing 'params' and 'opt_state'),
        # 'best_params', 'train_losses', 'val_losses', etc.
        if isinstance(saved, dict):
            trainer_state = saved.get('trainer_state', {})

            if 'params' in trainer_state:
                self.params = jax.tree_util.tree_map(jnp.asarray, trainer_state['params'])
                training_logger.info("Restored model params from trainer_state['params']")
            else:
                training_logger.warning("No 'params' key in trainer_state — params not restored!")

            if 'opt_state' in trainer_state:
                # Store for restoration in the next train_stage call
                self._resume_opt_state = trainer_state['opt_state']
                training_logger.info("Saved optimizer state for restoration in train_stage")

            if 'best_params' in saved:
                self.best_params = jax.tree_util.tree_map(jnp.asarray, saved['best_params'])
                training_logger.info("Restored best_params from checkpoint")
            else:
                self.best_params = self.params

            train_losses = list(saved.get('train_losses', []))
            val_losses = list(saved.get('val_losses', []))

        else:
            # Full trainer object (full_checkpoint=True) — less common
            if hasattr(saved, 'params'):
                self.params = saved.params
            if hasattr(saved, 'best_inference_params'):
                self.best_params = saved.best_inference_params
            elif hasattr(saved, 'best_params'):
                self.best_params = saved.best_params
            else:
                self.best_params = self.params
            # Try to extract opt_state from the full trainer object
            if hasattr(saved, 'state') and hasattr(saved.state, 'opt_state'):
                self._resume_opt_state = saved.state.opt_state
            train_losses = list(getattr(saved, 'train_losses', []))
            val_losses = list(getattr(saved, 'val_losses', []))

        training_logger.info(f"Loaded chemtrain checkpoint from: {checkpoint_path}")

        # Try to load metadata from companion .meta.pkl file
        meta_path = checkpoint_path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            training_logger.info(f"Loaded metadata: stage={metadata.get('stage')}, "
                               f"epochs={metadata.get('completed_epochs')}")
        else:
            # Infer epoch count from filename (e.g. epoch00040.pkl → 40)
            epoch_match = re.match(r'epoch0*(\d+)', checkpoint_path.stem)
            inferred_epoch = int(epoch_match.group(1)) if epoch_match else 0

            # Default stage to stage1 optimizer since epoch*.pkl files are only written
            # during stage 1 (stage checkpoints have a different naming convention)
            inferred_stage = self.config.get_stage1_optimizer()

            metadata = {
                "stage": inferred_stage,
                "completed_epochs": inferred_epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            training_logger.info(
                f"No metadata file found — inferred from filename: "
                f"stage='{inferred_stage}', completed_epochs={inferred_epoch}"
            )

        return metadata

    def __repr__(self) -> str:
        return (
            f"Trainer(model={self.model}, batch_per_device={self.batch_per_device}, "
            f"checkpoint_path={self.checkpoint_path})"
        )
