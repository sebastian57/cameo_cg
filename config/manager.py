"""
Configuration Manager for Chemtrain Pipeline

Handles loading, validation, and access to YAML configuration files.
"""

import os
import warnings
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigManager:
    """
    Manages configuration for training, models, and system parameters.

    Loads YAML configuration files and provides convenient accessor methods
    with default values and type checking.

    Example:
        >>> config = ConfigManager("config.yaml")
        >>> cutoff = config.get_model_param("cutoff", default=10.0)
        >>> batch_size = config.get_training_param("batch_per_device", default=4)
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is not valid YAML
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        """
        Validate that required configuration sections exist.

        Raises:
            ValueError: If required sections are missing
        """
        required_sections = ['data', 'model', 'training', 'optimizer']
        missing = [s for s in required_sections if s not in self._config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested config value by keys.

        Args:
            *keys: Sequence of keys to traverse (e.g., "model", "cutoff")
            default: Default value if key path doesn't exist

        Returns:
            Configuration value or default

        Example:
            >>> config.get("model", "cutoff", default=10.0)
            12.0
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    # ===== Convenience Methods =====

    def get_seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.get("seed", default=42)

    def get_model_context(self) -> str:
        """Get model context identifier."""
        return self.get("model_context", default="allegro_cg_protein")

    def get_model_id(self) -> str:
        """Get model ID."""
        return self.get("model_id", default="default")

    def get_protein_name(self) -> str:
        """Get protein name."""
        return self.get("protein_name", default="unknown")

    # ----- Data Section -----

    def get_data_path(self) -> str:
        """Get path to NPZ dataset."""
        return self.get("data", "path", default=None)

    def get_max_frames(self) -> Optional[int]:
        """Get maximum number of frames to use from dataset."""
        return self.get("data", "max_frames", default=None)

    def get_batch_mode(self) -> str:
        """
        Get dataset batch construction mode.

        Returns:
            "standard" for legacy per-structure batching or
            "tiled" for disconnected packed tile batching.
        """
        raw = str(self.get("data", "batch_mode", default="standard")).strip().lower()
        if raw not in ("standard", "tiled"):
            raise ValueError(
                f"Unsupported data.batch_mode='{raw}'. "
                "Expected one of: standard, tiled."
            )
        return raw

    def get_tile_target_beads(self) -> int:
        """Target number of valid beads per tile in tiled batch mode."""
        value = int(self.get("data", "tile_target_beads", default=1000))
        if value <= 0:
            raise ValueError(
                f"data.tile_target_beads must be > 0, got {value}."
            )
        return value

    def get_tile_bucket_beads(self) -> Optional[list[int]]:
        """
        Optional fixed tile-size buckets (in bead count) for tiled mode.

        Returns:
            Sorted unique positive bucket sizes, or None if not configured.
        """
        values = self.get("data", "tile_bucket_beads", default=None)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                "data.tile_bucket_beads must be a list of positive integers."
            )
        parsed = sorted({int(v) for v in values})
        if not parsed or parsed[0] <= 0:
            raise ValueError(
                f"data.tile_bucket_beads must contain positive integers, got {values}."
            )
        return parsed

    def tile_shuffle_structures_enabled(self) -> bool:
        """Whether to shuffle structures before greedy tile packing."""
        return bool(self.get("data", "tile_shuffle_structures", default=False))

    def tile_sort_by_size_enabled(self) -> bool:
        """Whether tiled packing should sort structures by valid bead count."""
        return bool(self.get("data", "tile_sort_by_size", default=True))

    def tile_rebuild_each_epoch_enabled(self) -> bool:
        """Whether tiled training should rebuild tile composition at each epoch."""
        return bool(self.get("data", "tile_rebuild_each_epoch", default=False))

    def tile_drop_incomplete_enabled(self) -> bool:
        """Whether to drop the final under-filled tile."""
        return bool(self.get("data", "tile_drop_incomplete", default=False))

    def tile_train_only_enabled(self) -> bool:
        """Whether tiled mode should only be applied to training batches."""
        return bool(self.get("data", "tile_train_only", default=True))

    # ----- Preprocessing Section -----

    def get_buffer_multiplier(self) -> float:
        """Get buffer multiplier for box extent computation (default: 2.0)."""
        return self.get("preprocessing", "buffer_multiplier", default=2.0)

    def get_park_multiplier(self) -> float:
        """Get parking location multiplier for padded atoms (default: 0.95)."""
        return self.get("preprocessing", "park_multiplier", default=0.95)

    # ----- Model Section -----

    def get_cutoff(self) -> float:
        """Get neighbor list cutoff distance."""
        return self.get("model", "cutoff", default=10.0)

    def get_dr_threshold(self) -> float:
        """Get neighbor list rebuild threshold."""
        return self.get("model", "dr_threshold", default=0.5)

    def get_neighbor_list_format(self) -> str:
        """
        Get neighbor list storage format.

        Returns:
            Format name: "dense" or "sparse"
        """
        env_override = os.environ.get("CHEMTRAIN_NEIGHBOR_LIST_FORMAT")
        if env_override is not None and str(env_override).strip() != "":
            raw = str(env_override)
        else:
            raw_from_cfg = self.get("model", "neighbor_list_format", default=None)
            if raw_from_cfg is None:
                warnings.warn(
                    "model.neighbor_list_format is not set; defaulting to 'dense'. "
                    "Set model.neighbor_list_format: sparse (or export "
                    "CHEMTRAIN_NEIGHBOR_LIST_FORMAT=sparse) to enable sparse "
                    "neighbor lists."
                )
                raw = "dense"
            else:
                raw = str(raw_from_cfg)
        normalized = raw.strip().lower().replace("-", "_")
        if normalized not in ("dense", "sparse"):
            raise ValueError(
                f"Unsupported model.neighbor_list_format='{raw}'. "
                "Expected one of: dense, sparse."
            )
        return normalized

    def get_allegro_config(self, size: str = "default") -> Dict[str, Any]:
        """
        Get Allegro model configuration.

        Args:
            size: Model size variant ("default", "large", "med")

        Returns:
            Dictionary of Allegro hyperparameters
        """
        use_cueq_cfg = self.get_ml_model_type() in ("allegro_cueq", "allegro_cueq_fast")
        cfg: Dict[str, Any] = {}

        def _merge_if_dict(key: str):
            value = self.get("model", key, default=None)
            if isinstance(value, dict):
                cfg.update(value)

        # Start from the generic Allegro settings, then layer backend-specific
        # overrides on top. This keeps cuEq and e3nn runs aligned by default.
        _merge_if_dict("allegro")
        if size != "default":
            _merge_if_dict(f"allegro_{size}")

        if use_cueq_cfg:
            _merge_if_dict("allegro_cuEq")
            _merge_if_dict("allegro_cueq")
            if size != "default":
                _merge_if_dict(f"allegro_cuEq_{size}")
                _merge_if_dict(f"allegro_cueq_{size}")

        # Keep activation explicit/configurable while preserving current behavior.
        cfg = dict(cfg)
        cfg.setdefault("mlp_activation", "mish")
        cfg.setdefault("mlp_hidden_activation", cfg.get("mlp_activation", "mish"))
        cfg.setdefault("mlp_output_activation", "linear")
        return cfg

    def get_prior_params(self) -> Dict[str, Any]:
        """Get prior energy parameters (r0, kr, a, b, etc.)."""
        return self.get("model", "priors", default={})

    def use_spline_priors_enabled(self) -> bool:
        """
        Check if spline priors are enabled.

        Backward compatibility:
        - If explicit boolean `model.priors.use_spline_priors` is set, use it.
        - Otherwise, enable spline priors if `model.priors.spline_file` exists.
        """
        explicit = self.get("model", "priors", "use_spline_priors", default=None)
        if explicit is not None:
            return bool(explicit)
        return self.get("model", "priors", "spline_file", default=None) is not None

    def get_spline_file_path(self) -> Optional[str]:
        """Get spline prior file path (if configured)."""
        return self.get("model", "priors", "spline_file", default=None)

    def get_residue_specific_angles(self) -> bool:
        """Check if residue-specific angle splines are requested."""
        return self.get("model", "priors", "residue_specific_angles", default=False)

    # ----- Optimizer Section -----

    def get_optimizer_config(self, name: str) -> Dict[str, Any]:
        """
        Get optimizer configuration by name.

        Args:
            name: Optimizer name (e.g., "adabelief", "yogi", "adam")

        Returns:
            Dictionary of optimizer hyperparameters
        """
        return self.get("optimizer", name, default={})

    def get_grad_clip(self) -> float:
        """Get global gradient clipping value."""
        return self.get("optimizer", "grad_clip", default=1.0)

    # ----- Training Section -----

    def get_epochs(self, optimizer: str) -> int:
        """
        Get number of epochs for a specific optimizer stage.

        Args:
            optimizer: Optimizer name (e.g., "adabelief", "yogi")

        Returns:
            Number of epochs
        """
        key = f"epochs_{optimizer}"
        return self.get("training", key, default=100)

    def get_val_fraction(self) -> float:
        """Get validation set fraction."""
        return self.get("training", "val_fraction", default=0.1)

    def get_batch_per_device(self) -> int:
        """Get batch size per GPU device."""
        return self.get("training", "batch_per_device", default=4)

    def get_batch_cache(self) -> int:
        """Get number of batches to cache."""
        return self.get("training", "batch_cache", default=10)

    def mixed_precision_enabled(self) -> bool:
        """
        Check if mixed precision should be enabled.

        Backward compatibility:
        - If explicit boolean `training.enable_mixed_precision` is set, use it.
        - Otherwise infer from `training.compute_dtype != "float32"`.
        """
        explicit = self.get("training", "enable_mixed_precision", default=None)
        if explicit is not None:
            return bool(explicit)
        return self.get_compute_dtype() != "float32"

    def get_compute_dtype(self) -> str:
        """Get compute dtype for model forward/backward."""
        raw = str(self.get("training", "compute_dtype", default="float32")).lower()
        if raw not in ("float32", "bfloat16"):
            raise ValueError(
                f"Unsupported training.compute_dtype='{raw}'. "
                "Expected one of: float32, bfloat16."
            )
        return raw

    def get_param_dtype(self) -> str:
        """Get master parameter dtype."""
        raw = str(self.get("training", "param_dtype", default="float32")).lower()
        if raw not in ("float32",):
            raise ValueError(
                f"Unsupported training.param_dtype='{raw}'. "
                "Currently only float32 is supported."
            )
        return raw

    def get_reduce_dtype(self) -> str:
        """Get collective reduction / optimizer math dtype."""
        raw = str(self.get("training", "reduce_dtype", default="float32")).lower()
        if raw not in ("float32", "bfloat16"):
            raise ValueError(
                f"Unsupported training.reduce_dtype='{raw}'. "
                "Expected one of: float32, bfloat16."
            )
        return raw

    def buffer_donation_enabled(self) -> bool:
        """Check if update-step buffer donation is enabled."""
        return bool(self.get("training", "enable_buffer_donation", default=False))

    def get_donate_mode(self) -> str:
        """Get donation mode for JIT update functions."""
        raw = str(self.get("training", "donate_mode", default="state_only")).lower()
        if raw not in ("state_only", "state_and_batch"):
            raise ValueError(
                f"Unsupported training.donate_mode='{raw}'. "
                "Expected one of: state_only, state_and_batch."
            )
        return raw

    def get_remat_level(self) -> int:
        """Get activation rematerialization level (0=off, 1=coarse, 2=deeper)."""
        raw = int(self.get("training", "remat_level", default=0))
        if raw not in (0, 1, 2):
            raise ValueError(
                f"Unsupported training.remat_level='{raw}'. "
                "Expected one of: 0, 1, 2."
            )
        return raw

    def get_remat_policy(self) -> str:
        """Get remat policy name for model wrappers."""
        raw = str(self.get("training", "remat_policy", default="none")).lower()
        if raw not in ("none", "allegro_blocks_coarse", "allegro_blocks_deep"):
            raise ValueError(
                f"Unsupported training.remat_policy='{raw}'. "
                "Expected one of: none, allegro_blocks_coarse, allegro_blocks_deep."
            )
        return raw

    def get_gammas(self) -> Dict[str, float]:
        """
        Get force matching weights (gammas).

        Returns:
            Dictionary with 'F' (force) and 'U' (energy) weights
        """
        return self.get("training", "gammas", default={"F": 1.0, "U": 0.0})

    def get_force_loss_normalization(self) -> str:
        """Get force-loss normalization mode for force matching."""
        raw = str(
            self.get("training", "force_loss_normalization", default="legacy_mean")
        ).strip().lower()
        if raw not in ("legacy_mean", "valid_components", "per_structure_components"):
            raise ValueError(
                f"Unsupported training.force_loss_normalization='{raw}'. "
                "Expected one of: legacy_mean, valid_components, per_structure_components."
            )
        return raw

    def prior_residual_enabled(self) -> bool:
        """
        Check if prior-force residual training mode is enabled.

        In this mode, prior forces are precomputed on untiled data and force
        targets are transformed to residual targets:
            F_residual = F_ref - F_prior.
        """
        return bool(self.get("training", "prior_residual", "enabled", default=False))

    def prior_residual_cache_enabled(self) -> bool:
        """Check if residual-prior precompute cache is enabled."""
        return bool(
            self.get("training", "prior_residual", "cache_enabled", default=True)
        )

    def get_prior_residual_cache_path(self) -> Optional[str]:
        """
        Get optional cache path for residual-prior precompute data.

        If None, callers should use a default path under checkpoint_path.
        """
        path = self.get("training", "prior_residual", "cache_path", default=None)
        if path is None:
            return None
        path = str(path).strip()
        if path == "":
            return None
        return path

    def prior_residual_force_recompute(self) -> bool:
        """Whether to bypass cache and recompute prior forces."""
        return bool(
            self.get("training", "prior_residual", "force_recompute", default=False)
        )

    def get_prior_residual_compute_batch_size(self) -> int:
        """Batch size used for chunked prior-force precompute."""
        value = int(
            self.get("training", "prior_residual", "compute_batch_size", default=128)
        )
        if value <= 0:
            raise ValueError(
                "training.prior_residual.compute_batch_size must be > 0, "
                f"got {value}."
            )
        return value

    def get_checkpoint_path(self) -> str:
        """Get checkpoint directory path."""
        return self.get("training", "checkpoint_path", default="./checkpoints_allegro")

    def get_checkpoint_freq(self) -> int:
        """Get checkpoint frequency in epochs (0 = only at end)."""
        return self.get("training", "checkpoint_freq", default=0)

    def get_export_path(self) -> str:
        """Get model export directory path."""
        return self.get("training", "export_path", default="./exported_models")

    def export_combined_ml_priors_enabled(self) -> bool:
        """Whether eval/export should reconstruct full forces as ML + priors."""
        return bool(self.get("training", "export_combined_ml_priors", default=True))

    def get_profiling_config(self) -> Dict[str, Any]:
        """
        Get JAX profiling configuration.

        Returns:
            Dictionary with profiling settings:
                - enabled: enable profiling features
                - jax_trace_enabled: enable JAX trace collection/export
                - trace_dir: output directory for trace files
                - trace_rank0_only: only trace rank 0 in distributed runs
                - log_compiles: enable JAX/XLA compile logging
                - batch_profiler_enabled: enable per-batch dispatch/barrier timing
                - batch_profiler_warmup: batches to skip before sampling (JIT warmup)
                - batch_profiler_samples: number of batches to profile per stage
                - batch_stats_enabled: log optimizer-step accounting and batch composition
                - batch_stats_rank0_only: only emit batch stats on rank 0
                - batch_stats_log_every: log every N profiled steps
                - loss_profile_enabled: recompute sampled force-loss views for diagnosis
                - loss_profile_steps: number of profiled steps to run manual loss views on
                - epoch_summary_enabled: emit epoch-level summaries from the profiled steps
        """
        return {
            "enabled": self.get("training", "profiling", "enabled", default=False),
            "jax_trace_enabled": self.get(
                "training", "profiling", "jax_trace_enabled", default=True
            ),
            "trace_dir": self.get("training", "profiling", "trace_dir", default="./profiles"),
            "trace_rank0_only": self.get(
                "training", "profiling", "trace_rank0_only", default=True
            ),
            "log_compiles": self.get("training", "profiling", "log_compiles", default=False),
            "batch_profiler_enabled": self.get(
                "training", "profiling", "batch_profiler_enabled", default=False
            ),
            "batch_profiler_warmup": int(self.get(
                "training", "profiling", "batch_profiler_warmup", default=5
            )),
            "batch_profiler_samples": int(self.get(
                "training", "profiling", "batch_profiler_samples", default=50
            )),
            "batch_stats_enabled": self.get(
                "training", "profiling", "batch_stats_enabled", default=False
            ),
            "batch_stats_rank0_only": self.get(
                "training", "profiling", "batch_stats_rank0_only", default=True
            ),
            "batch_stats_log_every": int(self.get(
                "training", "profiling", "batch_stats_log_every", default=1
            )),
            "loss_profile_enabled": self.get(
                "training", "profiling", "loss_profile_enabled", default=False
            ),
            "loss_profile_steps": int(self.get(
                "training", "profiling", "loss_profile_steps", default=4
            )),
            "epoch_summary_enabled": self.get(
                "training", "profiling", "epoch_summary_enabled", default=False
            ),
        }

    # ----- Model Configuration (New) -----

    def use_priors(self) -> bool:
        """Check if prior energy terms should be used."""
        return self.get("model", "use_priors", default=True)

    def train_priors_enabled(self) -> bool:
        """Check if prior parameters should be trained during force matching."""
        return self.get("model", "train_priors", default=False)

    def prior_only_enabled(self) -> bool:
        """Check if model should run in prior-only mode (no ML computation)."""
        return self.get("model", "prior_only", default=False)

    def get_ml_model_type(self) -> str:
        """
        Get which ML model backbone to use.

        Returns:
            Canonical model type:
            - "allegro"
            - "allegro_cueq"
            - "allegro_cueq_fast"
            - "mace"
            - "painn"
        """
        raw = str(self.get("model", "ml_model", default="allegro"))
        normalized = raw.strip().lower().replace("-", "_")

        aliases = {
            "allegro": "allegro",
            "allegro_cueq": "allegro_cueq",
            "allegro_cueq_opt": "allegro_cueq",
            "allegro_cueq_b1": "allegro_cueq",
            "allegro_cueq_fast": "allegro_cueq_fast",
            "allegro_cueq_fast_1103": "allegro_cueq_fast",
            "mace": "mace",
            "painn": "painn",
        }
        canonical = aliases.get(normalized)
        if canonical is None:
            allowed = ", ".join(sorted(aliases.keys()))
            raise ValueError(
                f"Unsupported model.ml_model='{raw}'. "
                f"Expected one of: {allowed}"
            )
        return canonical

    def get_allegro_size(self) -> str:
        """
        Get Allegro model size variant.

        Returns:
            Size name: "default", "large", or "med"
        """
        return self.get("model", "allegro_size", default="default")

    def get_mace_size(self) -> str:
        """
        Get MACE model size variant.

        Returns:
            Size name: "default", "large", or "small"
        """
        return self.get("model", "mace_size", default="default")

    def get_mace_config(self, size: str = "default") -> Dict[str, Any]:
        """
        Get MACE model configuration.

        Args:
            size: Model size variant ("default", "large", "small")

        Returns:
            Dictionary of MACE hyperparameters passed to mace_neighborlist_pp
        """
        if size == "default":
            return self.get("model", "mace", default={})
        else:
            key = f"mace_{size}"
            return self.get("model", key, default=self.get("model", "mace", default={}))

    def get_painn_size(self) -> str:
        """
        Get PaiNN model size variant.

        Returns:
            Size name: "default", "large", or "small"
        """
        return self.get("model", "painn_size", default="default")

    def get_painn_config(self, size: str = "default") -> Dict[str, Any]:
        """
        Get PaiNN model configuration.

        Args:
            size: Model size variant ("default", "large", "small")

        Returns:
            Dictionary of PaiNN hyperparameters passed to painn_neighborlist_pp
        """
        if size == "default":
            return self.get("model", "painn", default={})
        else:
            key = f"painn_{size}"
            return self.get("model", key, default=self.get("model", "painn", default={}))

    # ----- Training Configuration (New) -----

    def pretrain_prior_enabled(self) -> bool:
        """Check if prior pre-training is enabled."""
        return self.get("training", "pretrain_prior", default=False)

    def set_pretrain_prior_enabled(self, enabled: bool) -> None:
        """Set prior pre-training flag at runtime."""
        self._config.setdefault("training", {})
        self._config["training"]["pretrain_prior"] = bool(enabled)

    def get_pretrain_prior_max_steps(self) -> int:
        """Get maximum LBFGS steps for prior pre-training."""
        return self.get("training", "pretrain_prior_max_steps", default=200)

    def get_pretrain_prior_tol_grad(self) -> float:
        """Get gradient tolerance for prior pre-training convergence."""
        return self.get("training", "pretrain_prior_tol_grad", default=1e-6)

    def get_pretrain_prior_min_steps(self) -> int:
        """Get minimum LBFGS steps before convergence check."""
        return self.get("training", "pretrain_prior_min_steps", default=10)

    def get_stage1_optimizer(self) -> str:
        """Get stage 1 optimizer name."""
        return self.get("training", "stage1_optimizer", default="adabelief")

    def get_stage2_optimizer(self) -> str:
        """Get stage 2 optimizer name."""
        return self.get("training", "stage2_optimizer", default="yogi")

    # ----- Ensemble Training Configuration -----

    def is_ensemble_enabled(self) -> bool:
        """Check if ensemble training is enabled."""
        return self.get("ensemble", "enabled", default=False)

    def get_ensemble_config(self) -> Dict[str, Any]:
        """
        Get ensemble training configuration.

        Returns:
            Dictionary with ensemble settings:
                - enabled: Whether ensemble training is enabled
                - n_models: Number of models to train
                - base_seed: Base seed for generating model seeds
                - save_all_models: Whether to save all models or just the best
        """
        return {
            "enabled": self.get("ensemble", "enabled", default=False),
            "n_models": self.get("ensemble", "n_models", default=5),
            "base_seed": self.get("ensemble", "base_seed", default=42),
            "save_all_models": self.get("ensemble", "save_all_models", default=False),
        }

    def get_ensemble_n_models(self) -> int:
        """Get number of models in ensemble."""
        return self.get("ensemble", "n_models", default=5)

    def get_ensemble_base_seed(self) -> int:
        """Get base seed for ensemble (models use base_seed, base_seed+1, ...)."""
        return self.get("ensemble", "base_seed", default=42)

    def get_ensemble_save_all(self) -> bool:
        """Check if all ensemble models should be saved (vs just the best)."""
        return self.get("ensemble", "save_all_models", default=False)

    # ----- Utility Methods -----

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()

    def save(self, output_path: Union[str, Path]):
        """
        Save configuration to a new YAML file.

        Args:
            output_path: Path to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"ConfigManager('{self.config_path}')"

    def __str__(self) -> str:
        return f"ConfigManager with {len(self._config)} sections"
