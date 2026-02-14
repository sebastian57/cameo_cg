"""
Configuration Manager for Chemtrain Pipeline

Handles loading, validation, and access to YAML configuration files.
"""

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

    def get_allegro_config(self, size: str = "default") -> Dict[str, Any]:
        """
        Get Allegro model configuration.

        Args:
            size: Model size variant ("default", "large", "med")

        Returns:
            Dictionary of Allegro hyperparameters
        """
        if size == "default":
            return self.get("model", "allegro", default={})
        else:
            key = f"allegro_{size}"
            return self.get("model", key, default=self.get("model", "allegro", default={}))

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

    def get_gammas(self) -> Dict[str, float]:
        """
        Get force matching weights (gammas).

        Returns:
            Dictionary with 'F' (force) and 'U' (energy) weights
        """
        return self.get("training", "gammas", default={"F": 1.0, "U": 0.0})

    def get_checkpoint_path(self) -> str:
        """Get checkpoint directory path."""
        return self.get("training", "checkpoint_path", default="./checkpoints_allegro")

    def get_checkpoint_freq(self) -> int:
        """Get checkpoint frequency in epochs (0 = only at end)."""
        return self.get("training", "checkpoint_freq", default=0)

    def get_export_path(self) -> str:
        """Get model export directory path."""
        return self.get("training", "export_path", default="./exported_models")

    # ----- Model Configuration (New) -----

    def use_priors(self) -> bool:
        """Check if prior energy terms should be used."""
        return self.get("model", "use_priors", default=True)

    def train_priors_enabled(self) -> bool:
        """Check if prior parameters should be trained during force matching."""
        return self.get("model", "train_priors", default=False)

    def get_ml_model_type(self) -> str:
        """
        Get which ML model backbone to use.

        Returns:
            Model type: "allegro", "mace", or "painn"
        """
        return self.get("model", "ml_model", default="allegro")

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
