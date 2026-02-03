# TODO and Notes for Clean Code Base

**Last Updated:** 2026-01-23
**Status:** Phases 1-5 Complete - Ready for testing

---

## ✅ Fixed Issues (Phases 1-3)

### Issue 1: Prior Energy Weights
**Status:** FIXED ✓
- Added default weights in `prior_energy.py` (0.5, 0.1, 0.25, 0.15)
- Made configurable via `model.priors.weights` in YAML
- Backwards compatible with existing configs

### Issue 2: Config Convenience Methods
**Status:** FIXED ✓
- Added to `config/manager.py`:
  - `use_priors()` - Check if priors enabled
  - `get_allegro_size()` - Get model size variant
  - `pretrain_prior_enabled()` - Check if prior pre-training enabled
  - `get_pretrain_prior_epochs()` - Get pre-training epochs
  - `get_pretrain_prior_optimizer()` - Get pre-training optimizer
  - `get_stage1_optimizer()` - Get stage 1 optimizer name
  - `get_stage2_optimizer()` - Get stage 2 optimizer name

### Issue 3: Unused theta0/k_theta Parameters
**Status:** DOCUMENTED (No Action) ✓
- Parameters loaded but unused (matching original code behavior)
- Kept for potential future switch to harmonic angle potential
- Original code also has this commented out

### Issue 4: YAML Template
**Status:** FIXED ✓
- Created `config_template.yaml` with all new options documented
- Includes examples for different use cases
- Fully backwards compatible

### Issue 5: Allegro Size Selection
**Status:** FIXED ✓
- Updated `allegro_model.py` to use `config.get_allegro_size()`
- Added logging of selected size
- Works with existing configs (defaults to "default")

### Issue 6: NumpyDataLoader Patch
**Status:** KEPT AS-IS ✓
- Patch preserved from original code
- Applied in `trainer.py` on first instantiation
- Works correctly for chemtrain compatibility

---

## 📝 Important Notes

### Box Preprocessing Strategy

**Current Behavior:**
- `CoordinatePreprocessor` computes box extent from data
- `AllegroModel` receives box as parameter (can be from config or computed)
- Training script should handle the flow

**Recommended Workflow for Training Scripts:**
```python
# 1. Load config
config = ConfigManager("config.yaml")

# 2. Load data
loader = DatasetLoader(config.get_data_path())

# 3. Preprocess coordinates and compute box
preprocessor = CoordinatePreprocessor(cutoff=config.get_cutoff())
R_processed, box_computed, shift = preprocessor.process_dataset(loader.R, loader.mask)

# 4. Use computed box for model (not config box)
model = CombinedModel(config, R_processed[0], box_computed, loader.species[0], loader.N_max)

# 5. Update loader with processed coordinates
loader.R = R_processed

# 6. Train
trainer = Trainer(model, config, loader)
```

**Why not use config box?**
- Config box might not match actual data extent
- Computed box ensures proper buffering for cutoff
- Safer to derive from data

**Action:** Document this in training script (Phase 5)

---

### Energy Component Format

**Fixed to match original behavior:**
- `PriorEnergy.compute_energy()` returns WEIGHTED individual components
- Components are: `E_bond`, `E_angle`, `E_repulsive`, `E_dihedral`, `E_total`
- All except `E_total` are individual weighted terms (can be summed to get total)
- Matches original code: `return 0.5*E_bond, 0.1*E_angle, 0.25*E_rep, 0.15*E_dih`

---

## 🔧 Minor Issues (Low Priority)

### 1. Logging Consistency
**Current:** Mix of `print()` statements with varying formats
**TODO:** Replace with Python `logging` module for better control
**Priority:** Low (works fine, just not ideal)

### 2. Type Hints
**Current:** Most code has type hints, some helper functions missing them
**TODO:** Add complete type hints to all functions
**Priority:** Low (IDE support already good)

### 3. Unit Tests
**Current:** No unit tests yet
**TODO:** Add tests for:
- Topology generation
- Prior energy computation
- Config loading
- Data preprocessing
**Priority:** Medium (test with real data first)

### 4. Documentation
**Current:** Good docstrings in classes, some functions need more detail
**TODO:** Add more examples in docstrings
**Priority:** Low

---

## 🚧 Known Limitations

### 1. Species Indexing
- Currently assumes species are 0-indexed
- Original LAMMPS exporter does `species - 1`
- Need to verify with real data when implementing exporter (Phase 4)

### 2. Multi-Node Training
- Code structure supports it (via JAX distributed in trainer.py)
- Not tested yet on multi-node setup
- Should work based on original code

### 3. Prior Pre-Training
- Implemented but not tested with real data
- Uses same ForceMatching trainer as main training
- May need tuning of learning rates

---

## ✨ New Features Available (Not in Original)

### 1. Configurable Prior Weights
```yaml
model:
  priors:
    weights:
      bond: 0.5
      angle: 0.1
      repulsive: 0.25
      dihedral: 0.15
```

### 2. Toggle Priors On/Off
```yaml
model:
  use_priors: false  # Pure Allegro mode
```

### 3. Prior Pre-Training
```yaml
training:
  pretrain_prior: true
  pretrain_prior_epochs: 50
  pretrain_prior_optimizer: "adam"
```

### 4. Configurable Training Stages
```yaml
training:
  stage1_optimizer: "adabelief"
  stage2_optimizer: "yogi"
  epochs_adabelief: 100
  epochs_yogi: 50
```

### 5. Allegro Size Selection
```yaml
model:
  allegro_size: "large"  # or "default" or "med"
```

---

## ✅ Completed Phases

### Phase 4: Evaluation & Export (COMPLETE)
- [x] `evaluation/evaluator.py` - Frame evaluation with comprehensive metrics
  - Single-frame and batch evaluation
  - Energy, force RMSE, MAE, max error
  - Component breakdown (if available)
- [x] `evaluation/visualizer.py` - Complete visualization suite
  - `LossPlotter`: Parse training logs, create annotated loss curves
  - `ForceAnalyzer`: Force component scatter plots, magnitude comparison, error distributions
  - All plots saved as high-resolution PNGs
- [x] `export/exporter.py` - Clean Allegro-only MLIR export
  - Removed MACE and PaiNN exporters
  - `AllegroExporter.from_combined_model()` factory method
  - Handles species indexing (0-based → 1-based for LAMMPS)
  - Compatible with chemtrain-deploy

### Phase 5: User Scripts (COMPLETE)
- [x] `scripts/train.py` - Unified training CLI (400+ lines)
  - Auto-detects single vs multi-node from SLURM environment
  - JAX distributed initialization for multi-node
  - Handles data loading, box computation, model init, training, export
  - Applies NumpyDataLoader patch
  - Generates loss plots automatically
  - Saves parameters, config, and MLIR model
- [x] `scripts/evaluate.py` - Comprehensive evaluation CLI (300+ lines)
  - Single-frame evaluation with detailed metrics
  - Full dataset batch evaluation
  - Generates all force analysis plots
  - Aggregated statistics for batch evaluation
- [x] `scripts/run_training.sh` - Unified SLURM script
  - Auto-detects single vs multi-node from `--nodes` flag
  - Sets up JAX distributed coordinator for multi-node
  - Handles CUDA/XLA environment setup
  - Pipes output to timestamped log files
- [x] Module structure finalized
  - Created `scripts/__init__.py`
  - Updated `clean_code_base/__init__.py` with all exports
  - All modules properly documented

---

## 🎯 Testing Checklist (After Phase 5)

### Basic Functionality
- [ ] ConfigManager loads existing YAML configs
- [ ] DatasetLoader loads real NPZ files
- [ ] CoordinatePreprocessor computes correct box sizes
- [ ] TopologyBuilder generates correct indices
- [ ] PriorEnergy computes reasonable energies
- [ ] AllegroModel initializes without errors
- [ ] CombinedModel switches between modes correctly

### Training
- [ ] Single-stage training (AdaBelief only)
- [ ] Two-stage training (AdaBelief → Yogi)
- [ ] Prior pre-training (if enabled)
- [ ] Checkpointing and resuming
- [ ] Multi-GPU training (single node)
- [ ] Multi-node training (if available)

### Modes
- [ ] Pure Allegro (use_priors=false)
- [ ] Prior + Allegro (use_priors=true)
- [ ] With prior pre-training
- [ ] Different Allegro sizes (default, large, med)

### Output
- [ ] Model export to MLIR
- [ ] Parameter saving/loading
- [ ] Loss curves
- [ ] Force analysis plots

---

## 📖 Usage Examples

### Training

**Single-node (4 GPUs):**
```bash
sbatch scripts/run_training.sh config.yaml
```

**Multi-node (2 nodes, 8 GPUs total):**
```bash
sbatch --nodes=2 scripts/run_training.sh config.yaml
```

**Direct execution (for testing):**
```bash
python scripts/train.py config.yaml
```

### Evaluation

**Single frame:**
```bash
python scripts/evaluate.py config.yaml exported_models/model_params.pkl --frame 0
```

**Full dataset:**
```bash
python scripts/evaluate.py config.yaml exported_models/model_params.pkl --full
```

**Custom output directory:**
```bash
python scripts/evaluate.py config.yaml model_params.pkl --frame 5 --output-dir ./my_eval
```

### Programmatic Usage

```python
from clean_code_base import (
    ConfigManager, DatasetLoader, CombinedModel, Trainer, AllegroExporter
)

# Load configuration
config = ConfigManager("config.yaml")

# Load and preprocess data
loader = DatasetLoader(config.get_data_path())
loader.load()
dataset = loader.get_dataset()

# Initialize model
model = CombinedModel(config, R0, box, species, N_max)
params = model.initialize(rng_key)

# Train
trainer = Trainer(model, config, loaders)
results = trainer.train_full_pipeline()

# Export
exporter = AllegroExporter.from_combined_model(model, params, box, species)
exporter.export_to_file("model.mlir")
```

---

**Status:** Phases 1-5 complete - ready for real-world testing
