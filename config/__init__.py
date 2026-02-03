"""Configuration management for the chemtrain pipeline."""

from .manager import ConfigManager
from .types import (
    PathLike,
    as_path,
    PretrainResult,
    StageResult,
    TrainingResults,
    EnergyComponents,
    ForceComponents,
    SingleFrameMetrics,
    BatchMetrics,
    DatasetDict,
    TopologyDict,
    PriorParams,
    ModelParams,
    AllegroConfig,
    OptimizerConfig,
)

__all__ = [
    "ConfigManager",
    # Type aliases
    "PathLike",
    "as_path",
    # Training results
    "PretrainResult",
    "StageResult",
    "TrainingResults",
    # Evaluation results
    "EnergyComponents",
    "ForceComponents",
    "SingleFrameMetrics",
    "BatchMetrics",
    # Data structures
    "DatasetDict",
    "TopologyDict",
    # Model parameters
    "PriorParams",
    "ModelParams",
    # Config structures
    "AllegroConfig",
    "OptimizerConfig",
]
