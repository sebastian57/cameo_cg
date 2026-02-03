# Summary of Changes - 2026-01-26

## Issues Addressed

### 0a. ✅ Bug Fix: Module Loading Failure (libpython3.12.so.1.0 not found)
**Problem:** Training failed with `error while loading shared libraries: libpython3.12.so.1.0: cannot open shared object file`
**Root Cause:** The `load_modules.sh` script used `module purge` which doesn't fully remove Stage modules. If the compute node had `Stages/2026` pre-loaded (instead of `Stages/2025`), all subsequent 2025-stage modules (GCC, Python, CUDA, etc.) would fail to load.
**Solution:** Updated `run_training.sh` to:
1. Use `module --force purge` to completely clear all modules
2. Explicitly load `Stages/2025` first
3. Load all required modules directly in the script (not via external load_modules.sh)
4. Add `module list` verification step for debugging
**Files:** `scripts/run_training.sh`
**Status:** FIXED - Environment will now be consistent regardless of pre-loaded modules

### 0b. ✅ Bug Fix: Multi-Node Training Only Using 2 GPUs Instead of 8
**Problem:** With 2 nodes × 4 GPUs, training was only using 2 GPUs total (1 per node), not 8
**Symptom:** Log showed `Local=1 Global=2`, `Using 1 GPUs`, and ~12 min/epoch (should be ~3 min with 4x speedup)
**Root Cause:** Wrong SLURM configuration for JAX distributed:
- Old: `--ntasks-per-node=1`, `--gpus-per-task=4` → 1 process per node with 4 GPUs allocated but only 1 used
- JAX distributed assigns 1 GPU per process, so 2 processes = 2 GPUs
**Solution:** Changed to 1 process per GPU:
- New: `--ntasks-per-node=4`, `--gpus-per-task=1` → 4 processes per node, each with 1 GPU
- 1 node × 4 processes × 1 GPU = 4 GPUs
- 2 nodes × 4 processes × 1 GPU = 8 GPUs
**Expected Speedup:** ~4x faster per epoch when using 2 nodes (8 GPUs vs 2 GPUs previously)
**Files:** `scripts/run_training.sh`, `scripts/train.py`
**Status:** FIXED - All GPUs will now be utilized

### 1. ✅ Bug Fix: Training Data Access
**Problem:** Training crashed with `AttributeError: 'NumpyDataLoader' object has no attribute 'R'`
**Solution:** Added `train_data` parameter to `Trainer.__init__()`, passed explicitly from `scripts/train.py`
**Files:** `training/trainer.py`, `scripts/train.py`
**Status:** FIXED - Training should now complete without errors

### 2. ✅ Bug Fix: MLIR Export Failed
**Problem:** MLIR export crashed after training completed
**Error:** `AttributeError: 'CombinedModel' object has no attribute 'compute_total_energy'`
**Cause:** Exporter calls `compute_total_energy()`, but CombinedModel only had `compute_energy()`
**Solution:** Added `compute_total_energy()` as alias method in `CombinedModel`
**Files:** `models/combined_model.py`
**Status:** FIXED - MLIR export will now succeed
**See:** [BUGFIX_export.md](BUGFIX_export.md) for details

### 3. ✅ Loss Plotting Added
**Problem:** User wanted automatic loss plotting after training
**Solution:** Already exists! LossPlotter automatically runs after training
**Features:**
- Annotated plot with hyperparameters
- Vertical line at optimizer transition
- Loss data saved to text file
- Matches old `extract_and_plot_loss.py` format
**Files:** Lines 357-370 of `scripts/train.py`, `evaluation/visualizer.py`
**Status:** WORKING - will run after training completes

### 4. ✅ Scientific Fixes with Toggle Comments
**Solution:** Added clearly-marked scientific improvements (easy to enable/disable)

All fixes marked with:
```python
# ========================================================================
# SCIENTIFIC FIX: [Description]
# ========================================================================
```

---

## Scientific Fixes Added

### Fix 1: Repulsive Prior Strength (Config Change)
**File:** `config_template.yaml` lines 85-100
**Change:**
```yaml
# RECOMMENDED (to enable, uncomment):
# epsilon: 5.0   # Increase from 1.0
# sigma: 4.0     # Increase from 3.0

# CURRENT (original, for comparison):
epsilon: 1.0
sigma: 3.0
```
**How to Enable:** Change values in your config file
**Impact:** Much stronger repulsion (0.25 → 5 kcal/mol effective)

### Fix 2: Energy Term Weights (User's Preference)
**File:** `config_template.yaml` lines 63-85
**Change:**
```yaml
# RECOMMENDED (already set):
weights:
  bond: 0.5        # Fitted from histograms
  angle: 0.1       # Fitted from histograms
  dihedral: 0.15   # Fitted from histograms
  repulsive: 1.0   # NOT fitted → full strength (user's preference)

# ORIGINAL (for comparison):
# repulsive: 0.25  # Was scaled down
```
**Status:** ✅ Already set per user's preference
**Rationale:** Weight fitted terms, keep unfitted repulsion at full strength

### Fix 3: Excluded Volume (Optional - Commented Out)
**Files:**
- `models/topology.py` - Method added (lines 221-260)
- `models/prior_energy.py` - Implementation commented out (lines 287-360)

**Status:** Code ready, instructions provided, NOT enabled by default
**How to Enable:** Follow 4-step instructions in [SCIENTIFIC_FIXES.md](SCIENTIFIC_FIXES.md)
**Impact:** Prevents backbone self-intersection for residues 2-5 apart in sequence

---

## Files Modified

### Modified (7):
1. `training/trainer.py` - Added `train_data` parameter, validation
2. `scripts/train.py` - Pass `train_data` explicitly, JAX distributed init at top
3. `scripts/run_training.sh` - Explicit module loading with `--force purge`, unified for 1/N nodes
4. `config_template.yaml` - Added scientific fix comments
5. `models/topology.py` - Added `get_excluded_volume_pairs()` method
6. `models/prior_energy.py` - Added commented-out excluded volume method
7. `models/combined_model.py` - Added `compute_total_energy()` alias for exporter compatibility

### Created (4):
1. `SCIENTIFIC_FIXES.md` - Complete guide to all scientific improvements
2. `BUGFIX_train_data.md` - Training data access bug fix documentation
3. `BUGFIX_export.md` - MLIR export bug fix documentation
4. `CHANGES_SUMMARY.md` - This file

---

## What Works Now

✅ **Training will complete** (bug fixed)
✅ **MLIR export automatic** (line 343 of train.py)
✅ **Loss plotting automatic** (lines 357-370 of train.py)
✅ **Loss data saved** (line 367 of train.py)
✅ **Annotated plots** (hyperparameters in text box)
✅ **Stage separator** (vertical line at optimizer change)
✅ **Professional logging** (module-specific `[Training]`, `[Model]`, etc.)

---

## What to Test

### Immediate (Next Training Run):
1. **Verify bug fix works:**
   ```bash
   cd clean_code_base
   sbatch scripts/run_training.sh ../config_allegro_exp2.yaml
   ```

2. **Check outputs appear:**
   - `exported_models/<model_name>.mlir`
   - `exported_models/loss_curve_<job_id>.png`
   - `exported_models/loss_data_<job_id>.txt`
   - No `AttributeError` in log

### Scientific Testing (Ablation Study):

**Model 1: Pure Allegro**
```yaml
model:
  use_priors: false
```

**Model 2: Prior + Allegro (original)**
```yaml
model:
  use_priors: true
  priors:
    epsilon: 1.0
    sigma: 3.0
    weights:
      repulsive: 1.0  # Already updated
```

**Model 3: Prior + Allegro (Fix 1 - stronger repulsion)**
```yaml
model:
  use_priors: true
  priors:
    epsilon: 5.0  # CHANGED
    sigma: 4.0    # CHANGED
    weights:
      repulsive: 1.0
```

**Model 4: Prior + Allegro (Fixes 1+3 - repulsion + excluded volume)**
- Use Model 3 config
- Uncomment excluded volume code per SCIENTIFIC_FIXES.md

---

## User's Confirmed Preferences

1. ✅ **Angle prior doesn't dominate** - User verified, no fix needed
2. ✅ **Weight fitted terms, keep repulsion full** - Implemented in config_template.yaml
3. ✅ **Plan ablation study** - 4 models to compare
4. ✅ **Scientific fixes with easy toggle** - All marked with `# ========` comments

---

## Expected Training Output

After training completes successfully:

```
exported_models/
├── <model_name>.mlir              # For LAMMPS (MOST IMPORTANT)
├── <model_name>_params.pkl        # Trained parameters
├── <model_name>_checkpoint.pkl    # Resume checkpoint
├── loss_curve_<job_id>.png        # Annotated loss plot
└── loss_data_<job_id>.txt         # Epoch, train_loss, val_loss
```

**Loss Plot Features:**
- Train and validation curves
- Vertical line at optimizer transition (AdaBelief → Yogi)
- Annotated text box with:
  - Frames count
  - Cutoff distance
  - Batch per device
  - Val fraction
  - Learning rates for both optimizers

---

## Testing Checklist

Before next training run:
- [x] Bug fix implemented (train_data parameter)
- [x] Scientific fixes added with toggle comments
- [x] MLIR export code verified (exists at line 343)
- [x] Loss plotting code verified (exists at lines 357-370)
- [x] Documentation created (SCIENTIFIC_FIXES.md)

After training run:
- [ ] Training completes without AttributeError
- [ ] MLIR file appears in exported_models/
- [ ] Loss plot appears with annotations
- [ ] Loss data text file appears
- [ ] Can load checkpoint with --resume

Scientific validation:
- [ ] Run MD simulation with exported .mlir file
- [ ] Check stability (no crashes, energy conservation)
- [ ] Run ablation study (4 models)
- [ ] Compare force RMSE/MAE across models
- [ ] Test transferability to new proteins

---

## Quick Start

**Test the bug fix:**
```bash
cd clean_code_base
sbatch scripts/run_training.sh ../config_allegro_exp2.yaml

# Wait for completion, then check:
ls exported_models/
# Should see: .mlir, .pkl, loss_curve.png, loss_data.txt
```

**Enable stronger repulsion (recommended):**
```bash
# Edit your config file:
vim ../config_allegro_exp2.yaml

# Change:
epsilon: 5.0  # from 1.0
sigma: 4.0    # from 3.0
```

**For detailed instructions:** See [SCIENTIFIC_FIXES.md](SCIENTIFIC_FIXES.md)

---

**Status: ✅ ALL CHANGES COMPLETE - Ready for Testing**
