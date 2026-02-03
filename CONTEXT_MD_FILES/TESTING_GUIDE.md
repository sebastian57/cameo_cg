# Testing Guide for Refactored Codebase

**Purpose:** Validate all Phase 1-4 refactoring changes with comprehensive testing
**Status:** Ready for user execution on HPC cluster
**Date:** 2026-01-26

---

## Overview

This guide provides step-by-step instructions for testing the refactored codebase. All changes have been implemented and are backwards compatible. Testing will verify functionality, performance, and scientific correctness.

---

## Testing Environment

**Requirements:**
- HPC cluster with GPU nodes (JUWELS)
- SLURM job scheduler
- Multi-GPU support (tested with 4 GPUs previously)
- Python environment with JAX, chemtrain, Allegro

**Baseline for Comparison:**
- Previous training runs from pre-refactoring code
- Known working configurations from `config_allegro_exp2.yaml`

---

## Test Plan

### Phase 1: Smoke Tests (5 minutes) ✅

**Goal:** Verify all modules import and instantiate correctly

**Commands:**
```bash
cd clean_code_base

# Test imports
python -c "from config import ConfigManager, TopologyBuilder"
python -c "from data import DatasetLoader, CoordinatePreprocessor"
python -c "from models import PriorEnergy, AllegroModel, CombinedModel"
python -c "from training import Trainer"
python -c "from evaluation import Evaluator, LossPlotter, ForceAnalyzer"
python -c "from export import AllegroExporter"
python -c "from utils.logging import training_logger"

# Test config loading
python -c "
from config import ConfigManager
config = ConfigManager('../config_allegro_exp2.yaml')
print('Config loaded:', config.get('model', 'type'))
"
```

**Expected Output:**
- No ImportError
- No syntax errors
- Config prints: `Config loaded: allegro`

**Status:** ✅ Verified during development

---

### Phase 2: Unit Tests (15 minutes)

**Goal:** Test individual components with small data

#### 2.1 Config Manager
```bash
python -c "
from config import ConfigManager

config = ConfigManager('../config_allegro_exp2.yaml')

# Test new methods
assert config.get_buffer_multiplier() == 2.0
assert config.get_park_multiplier() == 0.95
assert config.get_pretrain_prior_min_steps() == 10

# Verify removed method is gone
try:
    config.get_box()
    print('ERROR: get_box() should not exist')
except AttributeError:
    print('✓ get_box() correctly removed')

print('✓ ConfigManager tests passed')
"
```

#### 2.2 Data Loading with PathLike
```bash
python -c "
from pathlib import Path
from data import DatasetLoader

# Test with string path
loader1 = DatasetLoader('../data/protein1.npz', N_max=100)
print(f'✓ Loaded with str: {loader1.R.shape}')

# Test with Path object
loader2 = DatasetLoader(Path('../data/protein1.npz'), N_max=100)
print(f'✓ Loaded with Path: {loader2.R.shape}')

print('✓ PathLike tests passed')
"
```

#### 2.3 Logging Framework
```bash
python -c "
from utils.logging import training_logger, model_logger, data_logger

training_logger.info('Test training message')
model_logger.info('Test model message')
data_logger.info('Test data message')

print('✓ Logging framework working')
"
```

**Expected Output:**
- All assertions pass
- Logging messages appear with [Module] prefix
- No errors

---

### Phase 3: Integration Tests (30 minutes)

**Goal:** Test full pipeline components working together

#### 3.1 Single Frame Evaluation

Create test script `test_single_frame.py`:
```python
#!/usr/bin/env python
"""Test single frame evaluation with refactored code."""

from config import ConfigManager
from data import DatasetLoader, CoordinatePreprocessor
from models import TopologyBuilder, PriorEnergy, AllegroModel, CombinedModel
from jax_md import space
import jax.numpy as jnp
from utils.logging import training_logger

# Load config
config = ConfigManager('../config_allegro_exp2.yaml')
training_logger.info("Testing single frame evaluation...")

# Load data
npz_path = config.get('data', 'train_data')
N_max = config.get('model', 'N_max')
loader = DatasetLoader(npz_path, N_max=N_max)
training_logger.info(f"Loaded data: {loader.R.shape}")

# Preprocess
cutoff = config.get('model', 'allegro', 'r_max')
buffer_mult = config.get_buffer_multiplier()
park_mult = config.get_park_multiplier()
preprocessor = CoordinatePreprocessor(cutoff=cutoff, buffer_multiplier=buffer_mult, park_multiplier=park_mult)

data = preprocessor.preprocess_dataset(loader.R, loader.mask, loader.F)
training_logger.info(f"Box: {data['box']}")

# Initialize model
R0 = data["R"][0]
mask0 = data["mask"][0]
species0 = loader.species[0] if hasattr(loader, 'species') else jnp.zeros(N_max, dtype=jnp.int32)

displacement, shift = space.free()
topology = TopologyBuilder(N_max)
model = CombinedModel(config, R0, data["box"], species0, N_max, displacement)

# Initialize params (test TypedDict returns)
params = model.initialize()
training_logger.info(f"Initialized params: {list(params.keys())}")

# Compute energy
nbrs = model.allegro_model.nneigh_fn.allocate(R0)
energy_dict = model.compute_energy(params, R0, mask0, species0, nbrs)
training_logger.info(f"✓ Energy computed: {energy_dict['E_total']:.2f}")

# Compute force components (test ForceComponents TypedDict)
force_dict = model.compute_force_components(params, R0, mask0, species0)
training_logger.info(f"✓ Force components: {list(force_dict.keys())}")

print("\n✓ Integration test passed: Single frame evaluation works")
```

Run:
```bash
cd clean_code_base
python test_single_frame.py
```

**Expected Output:**
- All log messages with [Training] prefix
- Energy value printed
- Force components listed
- "✓ Integration test passed" message

---

### Phase 4: Training Tests (2-4 hours)

**Goal:** Full training pipeline with refactored code

#### 4.1 Standard Training Run

```bash
cd clean_code_base

# Submit job
sbatch scripts/run_training.sh ../config_allegro_exp2.yaml
```

**Monitor:**
```bash
# Check job status
squeue -u $USER

# Watch log file
tail -f slurm-*.out

# Look for logging format
grep "\[Training\]" slurm-*.out | head -20
grep "\[Model\]" slurm-*.out | head -20
```

**Expected Output:**
- Professional logging with [Module] prefixes (no raw print statements)
- Two-stage training (AdaBelief → Yogi)
- Loss plots generated
- Model exported to MLIR
- Checkpoint saved automatically
- Final message: "Training complete"

**Success Criteria:**
- Training converges without errors
- Loss values comparable to pre-refactoring runs
- All outputs generated (params.pkl, model.mlir, loss plots)

#### 4.2 Prior Pre-training Test

Modify config temporarily:
```yaml
# In config_allegro_exp2.yaml
model:
  priors:
    pretrain: true
    pretrain_prior_max_steps: 100
    pretrain_prior_min_steps: 10
```

Run:
```bash
sbatch scripts/run_training.sh ../config_allegro_exp2.yaml
```

**Expected Output:**
- Pre-training phase runs before main training
- LBFGS optimization completes
- Fitted prior params logged
- PretrainResult TypedDict returned (check code structure)

#### 4.3 Resume Training Test

**Step 1:** Start training and interrupt
```bash
# Submit job
job_id=$(sbatch scripts/run_training.sh ../config_allegro_exp2.yaml | awk '{print $4}')

# Wait 30 minutes, then cancel
sleep 1800
scancel $job_id
```

**Step 2:** Resume from checkpoint
```bash
# Find checkpoint
ls exports/model_checkpoint.pkl

# Resume training
python scripts/train.py ../config_allegro_exp2.yaml --resume exports/model_checkpoint.pkl
```

**Expected Output:**
- Log message: "Resuming from checkpoint: ..."
- Training continues from previous state
- No reinitialization of parameters

---

### Phase 5: Evaluation Tests (30 minutes)

**Goal:** Test evaluation and visualization tools

#### 5.1 Single Frame Evaluation
```bash
cd clean_code_base

python scripts/evaluate.py ../config_allegro_exp2.yaml \
    exports/trained_params.pkl \
    --frame 0
```

**Expected Output:**
- Energy components printed
- Force RMSE/MAE computed
- Log messages with [Eval] prefix
- SingleFrameMetrics TypedDict structure (internal)

#### 5.2 Batch Evaluation
```bash
python scripts/evaluate.py ../config_allegro_exp2.yaml \
    exports/trained_params.pkl \
    --batch
```

**Expected Output:**
- Statistics over all frames
- Mean/std of metrics
- BatchMetrics TypedDict structure (internal)

#### 5.3 Loss Plotting
```bash
python -c "
from evaluation import LossPlotter
from pathlib import Path

plotter = LossPlotter(Path('exports/training_log.txt'))
plotter.plot(Path('exports/loss_curve_test.png'))
print('✓ Loss plot saved')
"
```

**Expected Output:**
- Plot file created
- PathLike arguments accepted (both str and Path)

---

### Phase 6: Export Tests (10 minutes)

**Goal:** Verify MLIR export works

```bash
cd clean_code_base

python -c "
from export import AllegroExporter
from config import ConfigManager
from pathlib import Path
import pickle

config = ConfigManager('../config_allegro_exp2.yaml')

# Load trained params
with open('exports/trained_params.pkl', 'rb') as f:
    params = pickle.load(f)

# Export (test PathLike)
exporter = AllegroExporter(config, params['allegro'])
exporter.export_to_file(Path('exports/test_export.mlir'))
print('✓ MLIR export successful')
"
```

**Expected Output:**
- File `exports/test_export.mlir` created
- No errors
- PathLike argument accepted

---

### Phase 7: Multi-GPU Tests (1 hour)

**Goal:** Verify distributed training works

```bash
# Test with 4 GPUs
sbatch --nodes=1 --ntasks-per-node=4 scripts/run_training.sh ../config_allegro_exp2.yaml
```

**Expected Output:**
- JAX distributed initialized
- Training runs on all 4 GPUs
- No deadlocks or communication errors
- Performance similar to pre-refactoring

---

## Regression Testing

**Goal:** Ensure refactored code produces identical results

### Comparison Checklist

Compare refactored vs original code:

1. **Energy Values**
   - Same initial energy on frame 0
   - Same final loss after training
   - Energy components match

2. **Force Predictions**
   - Same RMSE/MAE on test frame
   - Force distributions identical

3. **Parameter Values**
   - Final trained params numerically close (within 1e-5)
   - Prior params identical if pre-trained

4. **Performance**
   - Training time within 10% of original
   - Memory usage unchanged

5. **Outputs**
   - MLIR file structure identical
   - Checkpoint format compatible

### Running Comparison

```bash
# Original code
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain
python train_fm_multiple_proteins.py config_allegro_exp2.yaml --job-id original

# Refactored code
cd clean_code_base
python scripts/train.py ../config_allegro_exp2.yaml --job-id refactored

# Compare
python compare_results.py exports/original/ exports/refactored/
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'utils'
```
**Solution:** Ensure you're running from `clean_code_base/` directory or adjust PYTHONPATH

**2. Config Parameter Missing**
```
KeyError: 'preprocessing.buffer_multiplier'
```
**Solution:** Add to config:
```yaml
preprocessing:
  buffer_multiplier: 2.0
  park_multiplier: 0.95
```

**3. Type Errors**
```
TypeError: expected PathLike, got ...
```
**Solution:** Check that Path or str is passed, not other types

**4. Logging Not Appearing**
**Solution:** Check logging level in utils/logging.py, set to INFO or DEBUG

**5. Resume Fails**
```
FileNotFoundError: exports/model_checkpoint.pkl
```
**Solution:** Ensure checkpoint was saved during previous run

---

## Test Results Template

After running tests, document results:

```markdown
## Test Results - [Date]

### Environment
- Node: [hostname]
- GPUs: [number and type]
- JAX version: [version]
- Chemtrain commit: [hash]

### Smoke Tests
- [✓/✗] All imports successful
- [✓/✗] Config loading works

### Unit Tests
- [✓/✗] ConfigManager
- [✓/✗] PathLike support
- [✓/✗] Logging framework

### Integration Tests
- [✓/✗] Single frame evaluation
- [✓/✗] Energy computation
- [✓/✗] Force computation

### Training Tests
- [✓/✗] Standard training (wall time: ___ hours)
- [✓/✗] Prior pre-training
- [✓/✗] Resume training

### Evaluation Tests
- [✓/✗] Single frame eval
- [✓/✗] Batch eval
- [✓/✗] Loss plotting

### Export Tests
- [✓/✗] MLIR export

### Multi-GPU Tests
- [✓/✗] 4-GPU training

### Regression Tests
- [✓/✗] Energy values match
- [✓/✗] Force predictions match
- [✓/✗] Parameters match
- [✓/✗] Performance comparable

### Issues Found
[List any problems encountered]

### Notes
[Additional observations]
```

---

## Next Steps After Testing

1. **If all tests pass:**
   - Update REFACTORING_PROGRESS.md with test results
   - Mark all phases as verified
   - Document any performance differences
   - Archive original code

2. **If issues found:**
   - Document specific failures
   - Create issue list with priority
   - Fix critical issues first
   - Rerun affected tests

3. **Scientific validation:**
   - Run MD simulation with exported model
   - Check stability (energy conservation, no explosions)
   - Validate on held-out proteins
   - Compare with reference simulations

4. **Consider scientific recommendations:**
   - Review SCIENTIFIC_ANALYSIS.md
   - Decide on implementing fixes (repulsive prior, excluded volume)
   - Plan ablation studies

---

## Useful Commands

```bash
# Check SLURM job status
squeue -u $USER
scontrol show job $JOB_ID

# Monitor log file in real-time
tail -f slurm-*.out

# Check GPU usage
watch -n 1 nvidia-smi

# Find checkpoints
find exports -name "*.pkl" -type f

# Compare parameter files
python -c "
import pickle
with open('exports/original_params.pkl', 'rb') as f:
    p1 = pickle.load(f)
with open('exports/refactored_params.pkl', 'rb') as f:
    p2 = pickle.load(f)
print('Keys match:', set(p1.keys()) == set(p2.keys()))
"

# Check log file for errors
grep -i error slurm-*.out
grep -i warning slurm-*.out
grep -i exception slurm-*.out
```

---

## Success Criteria Summary

**Minimum Requirements (Must Pass):**
- ✓ All modules import without errors
- ✓ Training completes without crashes
- ✓ Loss converges to similar value as original
- ✓ Model exports successfully
- ✓ Multi-GPU training works

**Desired Outcomes:**
- ✓ Professional logging throughout
- ✓ Resume training works correctly
- ✓ PathLike arguments accepted everywhere
- ✓ TypedDict return types improve IDE experience
- ✓ Performance within 10% of original

**Scientific Validation (Follow-up):**
- MD simulation runs stably for >100k steps
- Energy conservation in NVE ensemble
- Model transfers to new proteins
- Structural properties match reference

---

**Ready to Test:** All code complete, documented, and prepared for validation.
