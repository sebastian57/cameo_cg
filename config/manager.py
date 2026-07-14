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
    """Load and access training configuration."""

    ML_MODEL_ALIASES = {
        "allegro": "allegro",
        "allegro_cueq": "allegro_cueq",
        "allegro_cueq_opt": "allegro_cueq",
        "allegro_cueq_b1": "allegro_cueq",
        "allegro_cueq_fast": "allegro_cueq_fast",
        "allegro_cueq_fast_1103": "allegro_cueq_fast",
        "mace": "mace",
        "painn": "painn",
    }

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        required_sections = ['data', 'model', 'training', 'optimizer']
        missing = [s for s in required_sections if s not in self._config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

    def set(self, *keys_and_value) -> None:
        """Set a nested config value.  Last argument is the value.

        Example::
            config.set("model", "use_priors", False)
        """
        *keys, value = keys_and_value
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @staticmethod
    def _env_bool(var: str) -> Optional[bool]:
        """Read an env var as bool; return None if unset."""
        val = os.environ.get(var)
        if val is None:
            return None
        return val.strip().lower() in ("1", "true", "yes", "on")

    def debug_neighbor_logging(self) -> bool:
        env = self._env_bool("CHEMTRAIN_DEBUG_NEIGHBOR")
        if env is not None:
            return env
        return bool(self.get("debug", "neighbor_logging", default=False))

    def debug_neighbor_rank0_only(self) -> bool:
        env = self._env_bool("CHEMTRAIN_DEBUG_NEIGHBOR_RANK0_ONLY")
        if env is not None:
            return env
        return bool(self.get("debug", "neighbor_rank0_only", default=True))

    def debug_shape_trace(self) -> bool:
        env = self._env_bool("CHEMTRAIN_DEBUG_SHAPE_TRACE")
        if env is not None:
            return env
        return bool(self.get("debug", "shape_trace", default=False))

    def debug_model_logging(self) -> bool:
        return bool(self.get("debug", "model_logging", default=False))

    def get_seed(self) -> int:
        return self.get("seed", default=42)

    def get_model_context(self) -> str:
        return self.get("model_context", default="allegro_cg_protein")

    def get_model_id(self) -> str:
        return self.get("model_id", default="default")

    def get_data_path(self) -> str:
        return self.get("data", "path", default=None)

    def get_max_frames(self) -> Optional[int]:
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

    def get_tile_target_edges(self) -> Optional[int]:
        value = self.get("data", "tile_target_edges", default=None)
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"data.tile_target_edges must be > 0, got {parsed}.")
        return parsed

    def get_tile_bucket_edges(self) -> Optional[list[int]]:
        values = self.get("data", "tile_bucket_edges", default=None)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                "data.tile_bucket_edges must be a list of positive integers."
            )
        parsed = [int(v) for v in values]
        if any(v <= 0 for v in parsed):
            raise ValueError(
                f"data.tile_bucket_edges must contain positive integers, got {values}."
            )
        return parsed

    def get_tile_edge_estimate_scale(self) -> float:
        value = float(self.get("data", "tile_edge_estimate_scale", default=15.0))
        if value <= 0.0:
            raise ValueError(
                f"data.tile_edge_estimate_scale must be > 0, got {value}."
            )
        return value

    def get_tile_edge_estimate_mode(self) -> str:
        raw = str(self.get("data", "tile_edge_estimate_mode", default="valid_scaled"))
        normalized = raw.strip().lower()
        if normalized not in ("valid_scaled", "distance_cutoff"):
            raise ValueError(
                f"Unsupported data.tile_edge_estimate_mode='{raw}'. "
                "Expected one of: valid_scaled, distance_cutoff."
            )
        return normalized

    def get_tile_edge_estimate_cutoff(self) -> Optional[float]:
        value = self.get("data", "tile_edge_estimate_cutoff", default=None)
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0.0:
            raise ValueError(
                f"data.tile_edge_estimate_cutoff must be > 0, got {parsed}."
            )
        return parsed

    def tile_sort_by_estimated_edges_enabled(self) -> bool:
        return bool(self.get("data", "tile_sort_by_estimated_edges", default=False))

    def tile_isolate_large_structures_enabled(self) -> bool:
        return bool(self.get("data", "tile_isolate_large_structures", default=False))

    def get_tile_large_structure_threshold(self) -> Optional[int]:
        value = self.get("data", "tile_large_structure_threshold", default=None)
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(
                f"data.tile_large_structure_threshold must be > 0, got {parsed}."
            )
        return parsed

    def get_tile_large_structure_edge_threshold(self) -> Optional[float]:
        value = self.get("data", "tile_large_structure_edge_threshold", default=None)
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0.0:
            raise ValueError(
                f"data.tile_large_structure_edge_threshold must be > 0, got {parsed}."
            )
        return parsed

    def tile_spatial_separation_enabled(self) -> bool:
        return bool(self.get("data", "tile_spatial_separation", default=False))

    def get_tile_structure_gap(self) -> float:
        value = float(self.get("data", "tile_structure_gap", default=25.0))
        if value <= 0.0:
            raise ValueError(f"data.tile_structure_gap must be > 0, got {value}.")
        return value

    def tile_shuffle_structures_enabled(self) -> bool:
        return bool(self.get("data", "tile_shuffle_structures", default=False))

    def tile_sort_by_size_enabled(self) -> bool:
        return bool(self.get("data", "tile_sort_by_size", default=True))

    def tile_rebuild_each_epoch_enabled(self) -> bool:
        return bool(self.get("data", "tile_rebuild_each_epoch", default=False))

    def tile_drop_incomplete_enabled(self) -> bool:
        return bool(self.get("data", "tile_drop_incomplete", default=False))

    def tile_train_only_enabled(self) -> bool:
        return bool(self.get("data", "tile_train_only", default=True))

    # ----- Preprocessing Section -----

    def get_buffer_multiplier(self) -> float:
        return self.get("preprocessing", "buffer_multiplier", default=2.0)

    def get_park_multiplier(self) -> float:
        return self.get("preprocessing", "park_multiplier", default=0.95)

    # ----- Model Section -----

    def get_cutoff(self) -> float:
        return self.get("model", "cutoff", default=10.0)

    def get_dr_threshold(self) -> float:
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

    def neighbor_disable_cell_list_enabled(self) -> bool:
        return bool(self.get("model", "neighbor_disable_cell_list", default=False))

    def use_pbc_enabled(self) -> bool:
        return bool(self.get("model", "pbc", default=False))

    def get_allegro_config(self, size: str = "default") -> Dict[str, Any]:
        """Get Allegro model hyperparameters from model.allegro,
        with cuEq-specific overrides layered on top when applicable."""
        use_cueq_cfg = self.get_ml_model_type() in ("allegro_cueq", "allegro_cueq_fast")
        cfg: Dict[str, Any] = dict(self.get("model", "allegro", default={}))

        if use_cueq_cfg:
            for key in ("allegro_cuEq", "allegro_cueq"):
                overlay = self.get("model", key, default=None)
                if isinstance(overlay, dict):
                    cfg.update(overlay)

        cfg.setdefault("mlp_activation", "mish")
        cfg.setdefault("mlp_hidden_activation", cfg.get("mlp_activation", "mish"))
        cfg.setdefault("mlp_output_activation", "linear")
        return cfg

    def get_prior_params(self) -> Dict[str, Any]:
        model_priors = self.get("model", "priors", default=None)
        if model_priors is not None:
            return model_priors
        return self.get("priors", default={})

    _DEFAULT_PRIOR_WEIGHTS: Dict[str, float] = {
        "bond": 0.5,
        "angle": 0.1,
        "repulsive": 0.25,
        "dihedral": 0.15,
        "excluded_volume": 1.0,
        "wca": 0.0,
        "lj": 0.0,
        "fene": 0.0,
        "leash": 0.0,
        "local_in": 0.0,
        "local_bond_in": 0.0,
        "crowding_wall": 0.0,
        "dh": 0.0,
        "stickiness": 0.0,
        "salt_bridge": 0.0,
        "five_particle_flat_bottom": 0.0,
        "ala2_feature_recovery": 0.0,
        "ala2_rama_recovery": 0.0,
    }

    def get_prior_weights(self) -> Dict[str, float]:
        user = self.get("model", "priors", "weights", default=None)
        if user is None:
            user = self.get("priors", "weights", default={})
        merged = dict(self._DEFAULT_PRIOR_WEIGHTS)
        merged.update(user)
        return merged

    def get_min_repulsive_sep(self) -> int:
        value = self.get("model", "priors", "min_repulsive_sep", default=None)
        if value is None:
            value = self.get("priors", "min_repulsive_sep", default=6)
        return int(value)

    def use_spline_priors_enabled(self) -> bool:
        """True if spline priors are on (explicit flag or spline_file present)."""
        explicit = self.get("model", "priors", "use_spline_priors", default=None)
        if explicit is not None:
            return bool(explicit)
        return self.get("model", "priors", "spline_file", default=None) is not None

    def get_spline_file_path(self) -> Optional[str]:
        return self.get("model", "priors", "spline_file", default=None)

    def get_residue_specific_angles(self) -> bool:
        return self.get("model", "priors", "residue_specific_angles", default=False)

    # ----- Optimizer Section -----

    def get_optimizer_config(self, name: str) -> Dict[str, Any]:
        return self.get("optimizer", name, default={})

    def get_grad_clip(self) -> float:
        return self.get("optimizer", "grad_clip", default=1.0)

    # ----- Training Section -----

    def get_epochs(self, optimizer: str) -> int:
        return self.get("training", f"epochs_{optimizer}", default=100)

    def get_val_fraction(self) -> float:
        return self.get("training", "val_fraction", default=0.1)

    def get_batch_per_device(self) -> int:
        return self.get("training", "batch_per_device", default=4)

    def get_batch_cache(self) -> int:
        return self.get("training", "batch_cache", default=10)

    def mixed_precision_enabled(self) -> bool:
        """Explicit flag or inferred from compute_dtype != float32."""
        explicit = self.get("training", "enable_mixed_precision", default=None)
        if explicit is not None:
            return bool(explicit)
        return self.get_compute_dtype() != "float32"

    def get_compute_dtype(self) -> str:
        raw = str(self.get("training", "compute_dtype", default="float32")).lower()
        if raw not in ("float32", "bfloat16"):
            raise ValueError(
                f"Unsupported training.compute_dtype='{raw}'. "
                "Expected one of: float32, bfloat16."
            )
        return raw

    def get_param_dtype(self) -> str:
        raw = str(self.get("training", "param_dtype", default="float32")).lower()
        if raw not in ("float32",):
            raise ValueError(
                f"Unsupported training.param_dtype='{raw}'. "
                "Currently only float32 is supported."
            )
        return raw

    def get_reduce_dtype(self) -> str:
        raw = str(self.get("training", "reduce_dtype", default="float32")).lower()
        if raw not in ("float32", "bfloat16"):
            raise ValueError(
                f"Unsupported training.reduce_dtype='{raw}'. "
                "Expected one of: float32, bfloat16."
            )
        return raw

    def buffer_donation_enabled(self) -> bool:
        return bool(self.get("training", "enable_buffer_donation", default=False))

    def get_donate_mode(self) -> str:
        raw = str(self.get("training", "donate_mode", default="state_only")).lower()
        if raw not in ("state_only", "state_and_batch"):
            raise ValueError(
                f"Unsupported training.donate_mode='{raw}'. "
                "Expected one of: state_only, state_and_batch."
            )
        return raw

    def get_remat_level(self) -> int:
        raw = int(self.get("training", "remat_level", default=0))
        if raw not in (0, 1, 2):
            raise ValueError(
                f"Unsupported training.remat_level='{raw}'. "
                "Expected one of: 0, 1, 2."
            )
        return raw

    def get_remat_policy(self) -> str:
        raw = str(self.get("training", "remat_policy", default="none")).lower()
        if raw not in ("none", "allegro_blocks_coarse", "allegro_blocks_deep"):
            raise ValueError(
                f"Unsupported training.remat_policy='{raw}'. "
                "Expected one of: none, allegro_blocks_coarse, allegro_blocks_deep."
            )
        return raw

    def get_gammas(self) -> Dict[str, float]:
        return self.get("training", "gammas", default={"F": 1.0, "U": 0.0})

    def get_force_loss_normalization(self) -> str:
        raw = str(
            self.get("training", "force_loss_normalization", default="legacy_mean")
        ).strip().lower()
        if raw not in ("legacy_mean", "valid_components", "per_structure_components"):
            raise ValueError(
                f"Unsupported training.force_loss_normalization='{raw}'. "
                "Expected one of: legacy_mean, valid_components, per_structure_components."
            )
        return raw

    def relative_entropy_enabled(self) -> bool:
        return bool(self.get("training", "relative_entropy", "enabled", default=False))

    def get_relative_entropy_config(self) -> Dict[str, Any]:
        cfg = self.get("training", "relative_entropy", default={}) or {}
        if not isinstance(cfg, dict):
            raise ValueError("training.relative_entropy must be a mapping.")
        return cfg

    def get_relative_entropy_reference_data_path(self) -> Optional[str]:
        path = self.get("training", "relative_entropy", "reference_data_path", default=None)
        if path is None or str(path).strip() == "":
            return self.get_data_path()
        return str(path)

    def get_dsm_refresh_interval_steps(self) -> int:
        value = int(self.get("training", "dsm", "refresh_interval_steps", default=0))
        if value < 0:
            raise ValueError(
                "training.dsm.refresh_interval_steps must be >= 0, "
                f"got {value}."
            )
        return value

    def prior_residual_enabled(self) -> bool:
        """Residual mode: F_target = F_ref - F_prior (precomputed)."""
        return bool(self.get("training", "prior_residual", "enabled", default=False))

    def prior_residual_cache_enabled(self) -> bool:
        return bool(
            self.get("training", "prior_residual", "cache_enabled", default=True)
        )

    def get_prior_residual_cache_path(self) -> Optional[str]:
        path = self.get("training", "prior_residual", "cache_path", default=None)
        if path is None:
            return None
        path = str(path).strip()
        if path == "":
            return None
        return path

    def prior_residual_force_recompute(self) -> bool:
        return bool(
            self.get("training", "prior_residual", "force_recompute", default=False)
        )

    def get_prior_residual_compute_batch_size(self) -> int:
        value = int(
            self.get("training", "prior_residual", "compute_batch_size", default=128)
        )
        if value <= 0:
            raise ValueError(
                "training.prior_residual.compute_batch_size must be > 0, "
                f"got {value}."
            )
        return value

    # ----- Paths Section -----

    def get_output_dir(self) -> str:
        return self.get("paths", "output_dir", default="./outputs/default")

    def get_checkpoint_dir(self) -> str:
        """Read from paths.checkpoint_dir, fall back to paths.output_dir/checkpoints,
        then legacy training.checkpoint_path."""
        explicit = self.get("paths", "checkpoint_dir", default=None)
        if explicit is not None:
            return str(explicit)
        if self.get("paths", default=None) is not None:
            return str(Path(self.get_output_dir()) / "checkpoints")
        return self.get("training", "checkpoint_path", default="./checkpoints")

    def get_export_dir(self) -> str:
        """Read from paths.export_dir, fall back to paths.output_dir/exports,
        then legacy training.export_path."""
        explicit = self.get("paths", "export_dir", default=None)
        if explicit is not None:
            return str(explicit)
        if self.get("paths", default=None) is not None:
            return str(Path(self.get_output_dir()) / "exports")
        return self.get("training", "export_path", default="./exported_models")

    def get_profile_dir(self) -> str:
        explicit = self.get("paths", "profile_dir", default=None)
        if explicit is not None:
            return str(explicit)
        if self.get("paths", default=None) is not None:
            return str(Path(self.get_output_dir()) / "profiles")
        return self.get("training", "profiling", "trace_dir", default="./profiles")

    def get_slurm_dir(self) -> str:
        explicit = self.get("paths", "slurm_dir", default=None)
        if explicit is not None:
            return str(explicit)
        return str(Path(self.get_output_dir()) / "slurm")

    # Backward-compat aliases
    def get_checkpoint_path(self) -> str:
        return self.get_checkpoint_dir()

    def get_export_path(self) -> str:
        return self.get_export_dir()

    def get_checkpoint_freq(self) -> int:
        return self.get("training", "checkpoint_freq", default=0)

    def export_combined_ml_priors_enabled(self) -> bool:
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
            "trace_dir": self.get_profile_dir(),
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
        return self.get("model", "use_priors", default=True)

    def train_priors_enabled(self) -> bool:
        return self.get("model", "train_priors", default=False)

    def prior_only_enabled(self) -> bool:
        return self.get("model", "prior_only", default=False)

    def get_ml_model_type(self) -> str:
        raw = str(self.get("model", "ml_model", default="allegro"))
        normalized = raw.strip().lower().replace("-", "_")
        canonical = self.ML_MODEL_ALIASES.get(normalized)
        if canonical is None:
            allowed = ", ".join(sorted(self.ML_MODEL_ALIASES.keys()))
            raise ValueError(
                f"Unsupported model.ml_model='{raw}'. "
                f"Expected one of: {allowed}"
            )
        return canonical

    def get_mace_config(self) -> Dict[str, Any]:
        return self.get("model", "mace", default={})

    def get_painn_config(self) -> Dict[str, Any]:
        return self.get("model", "painn", default={})

    # ----- Training Configuration (New) -----

    def pretrain_prior_enabled(self) -> bool:
        return self.get("training", "pretrain_prior", default=False)

    def set_pretrain_prior_enabled(self, enabled: bool) -> None:
        self._config.setdefault("training", {})
        self._config["training"]["pretrain_prior"] = bool(enabled)

    def get_pretrain_prior_max_steps(self) -> int:
        return self.get("training", "pretrain_prior_max_steps", default=200)

    def get_pretrain_prior_tol_grad(self) -> float:
        return self.get("training", "pretrain_prior_tol_grad", default=1e-6)

    def get_pretrain_prior_min_steps(self) -> int:
        return self.get("training", "pretrain_prior_min_steps", default=10)

    def get_training_stages(self) -> list:
        """Return ordered list of training stages.

        New format (preferred):
            training:
              stages:
                - optimizer: adabelief
                  epochs: 80
                - optimizer: lamb
                  epochs: 0

        Legacy format (auto-converted):
            training:
              stage1_optimizer: adabelief
              stage2_optimizer: yogi
              epochs_adabelief: 80
              epochs_yogi: 0
        """
        from training.optimizers import get_available_optimizers

        stages = self.get("training", "stages", default=None)
        if stages is not None:
            result = [
                {"optimizer": s["optimizer"], "epochs": int(s.get("epochs", 0))}
                for s in stages
            ]
        else:
            s1 = self.get("training", "stage1_optimizer", default="adabelief")
            s2 = self.get("training", "stage2_optimizer", default=None)
            result = [{"optimizer": s1, "epochs": self.get_epochs(s1)}]
            if s2 and s2 != s1:
                result.append({"optimizer": s2, "epochs": self.get_epochs(s2)})

        available = get_available_optimizers()
        for stage in result:
            name = stage["optimizer"]
            if name not in available:
                raise ValueError(
                    f"Unknown optimizer '{name}' in training stages. "
                    f"Available: {available}"
                )
        return result

    def get_swa_config(self) -> Dict[str, Any]:
        """Return normalized stochastic weight averaging config."""
        raw = self.get("training", "swa", default={}) or {}
        stages = self.get_training_stages()
        nonzero_stages = [s for s in stages if int(s.get("epochs", 0)) > 0]
        default_stage = nonzero_stages[-1]["optimizer"] if nonzero_stages else None

        cfg = {
            "enabled": bool(raw.get("enabled", False)),
            "stage": raw.get("stage", default_stage),
            "start_epoch": raw.get("start_epoch", None),
            "start_fraction": float(raw.get("start_fraction", 0.75)),
            "sample_freq_epochs": int(raw.get("sample_freq_epochs", 1)),
            "save_checkpoint": bool(raw.get("save_checkpoint", True)),
            "use_best_params": bool(raw.get("use_best_params", False)),
        }

        if cfg["stage"] is None:
            cfg["stage"] = default_stage
        if cfg["stage"] is not None and cfg["stage"] not in {s["optimizer"] for s in stages}:
            raise ValueError(
                f"training.swa.stage={cfg['stage']!r} is not present in training.stages."
            )
        if not 0.0 <= cfg["start_fraction"] <= 1.0:
            raise ValueError(
                "training.swa.start_fraction must be between 0.0 and 1.0, "
                f"got {cfg['start_fraction']}."
            )
        if cfg["start_epoch"] is not None:
            cfg["start_epoch"] = int(cfg["start_epoch"])
            if cfg["start_epoch"] < 0:
                raise ValueError(
                    "training.swa.start_epoch must be >= 0 when provided, "
                    f"got {cfg['start_epoch']}."
                )
        if cfg["sample_freq_epochs"] < 1:
            raise ValueError(
                "training.swa.sample_freq_epochs must be >= 1, "
                f"got {cfg['sample_freq_epochs']}."
            )
        return cfg

    def get_msam_config(self) -> Dict[str, Any]:
        """Return normalized micro-batch SAM config."""
        raw = self.get("training", "msam", default={}) or {}
        stages = self.get_training_stages()
        nonzero_stages = [s for s in stages if int(s.get("epochs", 0)) > 0]
        default_stage = nonzero_stages[-1]["optimizer"] if nonzero_stages else None

        cfg = {
            "enabled": bool(raw.get("enabled", False)),
            "stage": raw.get("stage", default_stage),
            "start_epoch": raw.get("start_epoch", None),
            "start_fraction": float(raw.get("start_fraction", 0.80)),
            "rho": float(raw.get("rho", 0.01)),
            "epsilon": float(raw.get("epsilon", 1.0e-12)),
        }

        if cfg["stage"] is None:
            cfg["stage"] = default_stage
        if cfg["stage"] is not None and cfg["stage"] not in {s["optimizer"] for s in stages}:
            raise ValueError(
                f"training.msam.stage={cfg['stage']!r} is not present in training.stages."
            )
        if not 0.0 <= cfg["start_fraction"] <= 1.0:
            raise ValueError(
                "training.msam.start_fraction must be between 0.0 and 1.0, "
                f"got {cfg['start_fraction']}."
            )
        if cfg["start_epoch"] is not None:
            cfg["start_epoch"] = int(cfg["start_epoch"])
            if cfg["start_epoch"] < 0:
                raise ValueError(
                    "training.msam.start_epoch must be >= 0 when provided, "
                    f"got {cfg['start_epoch']}."
                )
        if cfg["rho"] <= 0.0:
            raise ValueError(
                f"training.msam.rho must be > 0, got {cfg['rho']}."
            )
        if cfg["epsilon"] <= 0.0:
            raise ValueError(
                f"training.msam.epsilon must be > 0, got {cfg['epsilon']}."
            )
        return cfg

    def get_stage1_optimizer(self) -> str:
        stages = self.get_training_stages()
        return stages[0]["optimizer"] if stages else "adabelief"

    def get_stage2_optimizer(self) -> str:
        stages = self.get_training_stages()
        return stages[1]["optimizer"] if len(stages) > 1 else stages[0]["optimizer"]

    # ----- Ensemble Training Configuration -----

    def is_ensemble_enabled(self) -> bool:
        return self.get("ensemble", "enabled", default=False)

    def get_ensemble_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.get("ensemble", "enabled", default=False),
            "n_models": self.get("ensemble", "n_models", default=5),
            "base_seed": self.get("ensemble", "base_seed", default=42),
            "save_all_models": self.get("ensemble", "save_all_models", default=False),
        }

    # ----- Utility Methods -----

    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()

    def save(self, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"ConfigManager('{self.config_path}')"

    def __str__(self) -> str:
        return f"ConfigManager with {len(self._config)} sections"
