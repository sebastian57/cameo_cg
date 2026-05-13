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

from importlib import import_module

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
    "AllegroModelCuEq",
    "CombinedModel",
    # Training
    "Trainer",
    "create_optimizer",
    # Evaluation
    "Evaluator",
    "LossPlotter",
    "ForceAnalyzer",
    # Export
    "ModelExporter",
    "AllegroExporter",
    # MD
    "MDRunner",
]

_LAZY_SYMBOLS = {
    "ConfigManager": ("cameo_cg.config.manager", "ConfigManager"),
    "DatasetLoader": ("cameo_cg.data.loader", "DatasetLoader"),
    "CoordinatePreprocessor": ("cameo_cg.data.preprocessor", "CoordinatePreprocessor"),
    "TopologyBuilder": ("cameo_cg.models.topology", "TopologyBuilder"),
    "PriorEnergy": ("cameo_cg.models.prior_energy", "PriorEnergy"),
    "AllegroModel": ("cameo_cg.models.allegro_model", "AllegroModel"),
    "AllegroModelCuEq": ("cameo_cg.models.allegro_cueq_model", "AllegroModelCuEq"),
    "CombinedModel": ("cameo_cg.models.combined_model", "CombinedModel"),
    "Trainer": ("cameo_cg.training.trainer", "Trainer"),
    "create_optimizer": ("cameo_cg.training.optimizers", "create_optimizer"),
    "Evaluator": ("cameo_cg.analysis_tests.evaluator", "Evaluator"),
    "LossPlotter": ("cameo_cg.analysis_tests.visualizer", "LossPlotter"),
    "ForceAnalyzer": ("cameo_cg.analysis_tests.visualizer", "ForceAnalyzer"),
    "ModelExporter": ("cameo_cg.export.exporter", "ModelExporter"),
    "AllegroExporter": ("cameo_cg.export.exporter", "ModelExporter"),
    "MDRunner": ("cameo_cg.md.runner", "MDRunner"),
}


def __getattr__(name):
    if name in _LAZY_SYMBOLS:
        module_name, symbol_name = _LAZY_SYMBOLS[name]
        module = import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'cameo_cg' has no attribute '{name}'")
