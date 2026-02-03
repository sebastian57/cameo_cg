"""
Chemtrain Clean Code Base - Refactored CG Protein Force Field Pipeline

A clean, object-oriented implementation of the coarse-grained protein
machine learning force field pipeline combining prior energy terms with
Allegro equivariant neural networks.

Modules:
    config: Configuration management
    data: Dataset loading and preprocessing
    models: Energy models (Prior, Allegro, Combined)
    training: Training orchestration and optimizers
    evaluation: Evaluation and visualization
    export: Model export to MLIR
    scripts: CLI entry points
"""

__version__ = "1.0.0"
__author__ = "Schmidt36 & Claude"

# Import key classes for convenience
from .config.manager import ConfigManager
from .data.loader import DatasetLoader
from .data.preprocessor import CoordinatePreprocessor
from .models.topology import TopologyBuilder
from .models.prior_energy import PriorEnergy
from .models.allegro_model import AllegroModel
from .models.combined_model import CombinedModel
from .training.trainer import Trainer
from .training.optimizers import create_optimizer
from .evaluation.evaluator import Evaluator
from .evaluation.visualizer import LossPlotter, ForceAnalyzer
from .export.exporter import AllegroExporter

__all__ = [
    # Config
    "ConfigManager",
    # Data
    "DatasetLoader",
    "CoordinatePreprocessor",
    # Models
    "TopologyBuilder",
    "PriorEnergy",
    "AllegroModel",
    "CombinedModel",
    # Training
    "Trainer",
    "create_optimizer",
    # Evaluation
    "Evaluator",
    "LossPlotter",
    "ForceAnalyzer",
    # Export
    "AllegroExporter",
]
