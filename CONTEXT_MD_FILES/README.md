# Chemtrain Clean Code Base

A clean, object-oriented refactoring of the coarse-grained protein machine learning force field pipeline. Combines physics-based prior energy terms with Allegro equivariant neural networks for accurate protein force field prediction.

## 🎯 Features

- **Fully object-oriented architecture**: Clean class-based design replacing monolithic scripts
- **Configurable via YAML**: All hyperparameters and options controlled through configuration files
- **Modular energy models**: Prior, Allegro, and combined models with clean interfaces
- **Multi-stage training**: Configurable optimizers and training stages
- **Prior pre-training**: Optional physics-based initialization
- **Auto-scaling**: Single-node or multi-node execution auto-detected from SLURM environment
- **Comprehensive evaluation**: Force analysis, error metrics, and visualization
- **MLIR export**: Direct export for LAMMPS integration via chemtrain-deploy

## 🆕 Phase 6 Improvements (2026-01-26)

### Code Quality Enhancements
- ✅ **Professional logging framework** - Module-specific loggers (`[Training]`, `[Model]`, `[Data]`, `[Eval]`, `[Export]`)
- ✅ **Modern type hints** - `jax.Array`, `PathLike`, `TypedDict` for IDE support and type safety
- ✅ **Clean configuration** - Removed unused `system.box`, added preprocessing params
- ✅ **Type-safe APIs** - Structured return types with TypedDict for better contracts

### New Features
- ✅ **Resume training** - Continue from checkpoint after interruption (`--resume checkpoint.pkl`)
- ✅ **Configurable preprocessing** - Buffer and park multipliers now in config
- ✅ **Robust data access** - Removed fragile internal API dependencies

### Comprehensive Documentation (9 files)
- 📖 **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing procedures
- 📖 **[SCIENTIFIC_REVIEW.md](SCIENTIFIC_REVIEW.md)** - 10,000+ word living document on theory and implementation
- 📖 **[SCIENTIFIC_ANALYSIS.md](SCIENTIFIC_ANALYSIS.md)** - Critical analysis identifying 4 implementation issues
- 📖 **[REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md)** - Complete change log
- 📖 **[BOX_HANDLING.md](BOX_HANDLING.md)** - Box computation documentation
- 📖 **[PARAMETER_OWNERSHIP.md](PARAMETER_OWNERSHIP.md)** - Parameter lifecycle guide
- 📖 **[ERROR_MESSAGES.md](ERROR_MESSAGES.md)** - Future improvement patterns
- 📖 **[CODE_QUALITY_REFACTORING.md](CODE_QUALITY_REFACTORING.md)** - Original refactoring plan
- 📖 **[LITERATURE.md](LITERATURE.md)** - Key scientific papers

### Benefits
- 🎯 **100+ print statements** replaced with professional logging
- 🎯 **Better IDE support** with modern type hints and TypedDict
- 🎯 **Easier debugging** with structured, module-specific logging
- 🎯 **More maintainable** with clear API contracts and documentation
- 🎯 **Backwards compatible** - all original functionality preserved

## 📁 Directory Structure

```
clean_code_base/
├── config/
│   ├── __init__.py
│   └── manager.py              # YAML config loading and validation
├── data/
│   ├── __init__.py
│   ├── loader.py               # NPZ dataset loading
│   └── preprocessor.py         # Coordinate preprocessing and masking
├── models/
│   ├── __init__.py
│   ├── topology.py             # Bond/angle/dihedral generation
│   ├── prior_energy.py         # Physics-based energy terms
│   ├── allegro_model.py        # Allegro ML model wrapper
│   └── combined_model.py       # Prior + Allegro composition
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training orchestration
│   └── optimizers.py           # Optimizer factory (AdaBelief, Yogi, etc.)
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py            # Frame evaluation and metrics
│   └── visualizer.py           # Loss and force analysis plots
├── export/
│   ├── __init__.py
│   └── exporter.py             # MLIR model export
├── utils/                      # NEW in v1.1
│   ├── __init__.py
│   └── logging.py              # Professional logging infrastructure
└── scripts/
    ├── __init__.py
    ├── train.py                # Main training CLI (supports --resume)
    ├── evaluate.py             # Evaluation CLI
    └── run_training.sh         # Unified SLURM script
```

## 🚀 Quick Start

### Training

**Single-node (4 GPUs):**
```bash
cd clean_code_base
sbatch scripts/run_training.sh path/to/config.yaml
```

**Multi-node (2+ nodes):**
```bash
sbatch --nodes=2 scripts/run_training.sh path/to/config.yaml
```

### Resume Training (NEW in v1.1)

**Resume from interrupted training:**
```bash
python scripts/train.py config.yaml --resume exports/model_checkpoint.pkl
```

Checkpoints are automatically saved after training completes. Use this to continue from an interrupted run.

### Evaluation

**Evaluate single frame:**
```bash
python scripts/evaluate.py config.yaml model_params.pkl --frame 0
```

**Evaluate full dataset:**
```bash
python scripts/evaluate.py config.yaml model_params.pkl --full
```

## ⚙️ Configuration

All training, model, and system parameters are controlled via YAML configuration files. See [config_template.yaml](config_template.yaml) for a complete example.

### Key Configuration Options

**Enable/disable prior energy terms:**
```yaml
model:
  use_priors: true  # Set to false for pure Allegro mode
```

**Prior pre-training:**
```yaml
training:
  pretrain_prior: true
  pretrain_prior_epochs: 50
  pretrain_prior_optimizer: "adam"
```

**Multi-stage training:**
```yaml
training:
  stage1_optimizer: "adabelief"
  stage2_optimizer: "yogi"
  epochs_adabelief: 100
  epochs_yogi: 50
```

**Model size:**
```yaml
model:
  allegro_size: "default"  # Options: "default", "large", "med"
```

**Prior energy weights:**
```yaml
model:
  priors:
    weights:
      bond: 0.5
      angle: 0.1
      repulsive: 0.25
      dihedral: 0.15
```

**Preprocessing parameters (NEW in v1.1):**
```yaml
preprocessing:
  buffer_multiplier: 2.0  # Box size = extent + cutoff * buffer_multiplier
  park_multiplier: 0.95   # Padded atoms placed at box_extent * park_multiplier
```

## 🔧 Programmatic Usage

```python
from clean_code_base import (
    ConfigManager,
    DatasetLoader,
    CoordinatePreprocessor,
    CombinedModel,
    Trainer,
    Evaluator,
    AllegroExporter
)

# 1. Load configuration
config = ConfigManager("config.yaml")

# 2. Load and preprocess data
loader = DatasetLoader(config.get_data_path())
loader.load()

preprocessor = CoordinatePreprocessor(cutoff=config.get_cutoff())
box, R_processed = preprocessor.process_dataset(loader.get_coordinates(), loader.get_masks())

# 3. Initialize model
model = CombinedModel(
    config=config,
    R0=R_processed[0],
    box=box,
    species=loader.get_species()[0],
    N_max=loader.N_max
)

params = model.initialize(jax.random.PRNGKey(42))

# 4. Train
trainer = Trainer(model, config, loaders)
results = trainer.train_full_pipeline()

# 5. Evaluate
evaluator = Evaluator(model, params, config)
metrics = evaluator.evaluate_frame(R, F_ref, mask, species)

# 6. Export to MLIR
exporter = AllegroExporter.from_combined_model(model, params, box, species)
exporter.export_to_file("model.mlir")
```

## 📊 Outputs

### Training Outputs

- **MLIR model**: `exported_models/<model_name>.mlir` - For LAMMPS deployment
- **Parameters**: `exported_models/<model_name>_params.pkl` - Trained parameters
- **Config**: `exported_models/<model_name>_config.yaml` - Configuration snapshot
- **Loss curves**: `exported_models/loss_curve_<job_id>.png` - Training visualization
- **Loss data**: `exported_models/loss_data_<job_id>.txt` - Numerical loss values
- **Training log**: `train_allegro_<job_id>.log` - Complete training output

### Evaluation Outputs

- **Force components**: Scatter plots for x/y/z force components
- **Force magnitude**: Magnitude comparison and error vs distance plots
- **Force distribution**: Error distribution histograms
- **Statistics**: Aggregated metrics for batch evaluation

## 🔬 Available Energy Models

### 1. Pure Allegro
```yaml
model:
  use_priors: false
```
- ML-only force field
- Fastest training
- Requires more data

### 2. Prior + Allegro (Default)
```yaml
model:
  use_priors: true
```
- Physics-informed ML
- Better data efficiency
- Includes bond, angle, dihedral, and repulsive terms

### 3. With Prior Pre-training
```yaml
model:
  use_priors: true
training:
  pretrain_prior: true
  pretrain_prior_epochs: 50
```
- Initialize with physics-based fit
- Then train combined model
- Best for small datasets

## 🎓 Training Stages

The pipeline supports flexible multi-stage training:

1. **Optional: Prior pre-training**
   - Fit prior parameters to reference forces
   - Uses configured optimizer (default: Adam)

2. **Stage 1: Coarse training**
   - Default: AdaBelief optimizer
   - Higher learning rate, longer warmup
   - Configurable epochs and optimizer

3. **Stage 2: Fine-tuning**
   - Default: Yogi optimizer
   - Lower learning rate, shorter run
   - Configurable epochs and optimizer

All stages use force matching with configurable weights for energy and force components.

## 🧪 Testing

**📋 Complete testing guide available:** See [TESTING_GUIDE.md](TESTING_GUIDE.md) for:
- 7 comprehensive test phases (smoke, unit, integration, training, evaluation, export, multi-GPU)
- Step-by-step validation procedures
- Regression testing against pre-refactoring code
- Troubleshooting guide
- Test results template

**Quick smoke test:**
```bash
cd clean_code_base
python -c "from config import ConfigManager; from data import DatasetLoader; from models import CombinedModel; from training import Trainer; print('✓ All imports successful')"
```

**Full testing:** Follow [TESTING_GUIDE.md](TESTING_GUIDE.md) to validate all changes

See also [TODO_AND_NOTES.md](TODO_AND_NOTES.md) for legacy notes and known limitations.

## 📝 Notes

- **Box sizing**: Automatically computed from dataset extent + cutoff buffer
- **Species indexing**: Internal 0-based, converted to 1-based for LAMMPS export
- **Multi-GPU**: Automatic batch parallelization across all visible GPUs
- **Checkpointing**: Automatic during training, stored in configured directory
- **JAX compilation**: First epoch slower due to JIT compilation

## 🔄 Migration from Original Code

The clean code base is fully backwards compatible with existing YAML configurations. No changes required to existing configs, though new features (like `use_priors`, `pretrain_prior`) are available if desired.

To migrate:
1. Use existing YAML config file
2. Run with new training script: `sbatch scripts/run_training.sh config.yaml`
3. Results will match original implementation

## 📚 Additional Documentation

### For Testing and Validation
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing procedures (7 test phases)
- **[REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md)** - Complete change log and testing status

### For Scientific Understanding
- **[SCIENTIFIC_REVIEW.md](SCIENTIFIC_REVIEW.md)** - 10-section living document covering:
  - Theoretical foundation of coarse-grained modeling
  - Literature review (12+ papers)
  - Prior fitting methodology (Boltzmann inversion)
  - Force matching validation
  - Transferability challenges
  - Future directions
- **[SCIENTIFIC_ANALYSIS.md](SCIENTIFIC_ANALYSIS.md)** - Critical analysis:
  - 4 implementation issues identified
  - Priority-ranked recommendations
  - Analysis of energy term weighting

### For Developers
- **[BOX_HANDLING.md](BOX_HANDLING.md)** - Box computation algorithm
- **[PARAMETER_OWNERSHIP.md](PARAMETER_OWNERSHIP.md)** - Parameter lifecycle documentation
- **[ERROR_MESSAGES.md](ERROR_MESSAGES.md)** - Error message patterns
- **[CODE_QUALITY_REFACTORING.md](CODE_QUALITY_REFACTORING.md)** - Original refactoring plan
- [TODO_AND_NOTES.md](TODO_AND_NOTES.md) - Implementation notes (legacy)
- [config_template.yaml](config_template.yaml) - Complete configuration reference
- Docstrings in all modules provide detailed API documentation

### Scientific References
- **[LITERATURE.md](LITERATURE.md)** - Key papers on coarse-grained modeling and ML force fields

## 🤝 Contributing

This is a refactored research codebase. For questions or issues:
1. Check [TODO_AND_NOTES.md](TODO_AND_NOTES.md) for known issues
2. Review module docstrings for detailed API info
3. Contact: schmidt36

## 📄 License

Same license as parent chemtrain project.

---

**Version:** 1.1.0
**Last Updated:** 2026-01-26
**Status:** ✅ Phase 6 refactoring complete - Ready for comprehensive testing
