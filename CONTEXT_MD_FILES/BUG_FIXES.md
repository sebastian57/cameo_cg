# Bug Fixes - Training Script Debugging Session

**Date:** 2026-01-23
**Status:** All identified issues fixed, ready for testing

---

## Issues Found and Fixed

### 1. ❌ **Path Resolution Issue** → ✅ FIXED
**Error:** `FileNotFoundError: NPZ file not found: data_prep/datasets/...`
**Cause:** Relative paths in config not being resolved relative to clean_code_base directory
**Fix:** Added path resolution logic in both train.py and evaluate.py
```python
if not Path(data_path).is_absolute():
    script_dir = Path(__file__).parent
    clean_code_base_dir = script_dir.parent
    data_path = clean_code_base_dir / data_path
```
**Files:** `scripts/train.py:159-164`, `scripts/evaluate.py:247-252`

---

### 2. ❌ **DatasetLoader API Mismatch** → ✅ FIXED
**Error:** `AttributeError: 'DatasetLoader' object has no attribute 'load'`
**Cause:** DatasetLoader loads data in `__init__`, doesn't have separate `.load()` or `.shuffle()` methods
**Fixes:**
- Removed `.load()` call (data loads automatically in constructor)
- Removed `.shuffle()` call (handled by constructor params)
- Pass `max_frames` and `seed` to constructor instead
- Use direct properties (`.R`, `.mask`, `.species`) instead of method calls
- Use `.get_all()` instead of `.get_dataset()`

**Before:**
```python
loader = DatasetLoader(path)
loader.load()
loader.shuffle(seed=seed, max_frames=max_frames)
species0 = loader.get_species()[0]
all_R = loader.get_coordinates()
dataset = loader.get_dataset()
```

**After:**
```python
loader = DatasetLoader(path, max_frames=max_frames, seed=seed)
species0 = loader.species[0]
all_R = loader.R
dataset = loader.get_all()
```

**Files:** `scripts/train.py:166-177`, `scripts/evaluate.py:254-263`

---

### 3. ❌ **Trainer Initialization Mismatch** → ✅ FIXED
**Error:** `TypeError: __init__() got unexpected keyword argument 'loaders'`
**Cause:** Trainer expects separate `train_loader` and `val_loader` params, not `loaders` object
**Also:** Trainer initializes params internally, doesn't need `init_params` parameter

**Before:**
```python
trainer = Trainer(
    model=model,
    config=config,
    loaders=loaders,
    init_params=params
)
```

**After:**
```python
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)
```

**Files:** `scripts/train.py:258-262`

---

### 4. ❌ **Model Method Name Mismatches** → ✅ FIXED
**Errors:**
- `AttributeError: 'CombinedModel' object has no attribute 'initialize'`
- `AttributeError: 'CombinedModel' object has no attribute 'compute_energy_components'`

**Cause:** Wrong method names being called

**Fixes:**
1. Removed explicit `model.initialize()` call - Trainer handles initialization
2. Changed `compute_energy_components()` → `compute_components()`

**Before:**
```python
params = model.initialize(jax.random.PRNGKey(seed))
components = model.compute_energy_components(params, R, mask, species)
```

**After:**
```python
# Trainer initializes params internally
components = model.compute_components(params, R, mask, species)
```

**Files:** `scripts/train.py:210`, `scripts/train.py:279`

---

## Summary of Changes

### Files Modified:
1. **scripts/train.py** - 6 bug fixes
   - Path resolution (lines 159-164)
   - DatasetLoader API (lines 166-177)
   - Validation set size check (lines 217-250)
   - Trainer initialization (lines 258-262)
   - Model initialization removed (line 210)
   - Method name fix (line 279)

2. **scripts/evaluate.py** - 2 bug fixes
   - Path resolution (lines 247-252)
   - DatasetLoader API (lines 254-263)

3. **scripts/run_training.sh** - Path resolution (previously fixed)
   - Uses `SLURM_SUBMIT_DIR` to find scripts
   - Verifies train.py exists before running

---

## API Corrections Summary

### DatasetLoader (data/loader.py)
**Constructor:**
- `__init__(npz_path, max_frames=None, seed=42)` - Loads data immediately
- No separate `.load()` or `.shuffle()` methods

**Properties:**
- `.R` - Coordinates (not `.get_coordinates()`)
- `.F` - Forces (not `.get_forces()`)
- `.mask` - Validity mask (not `.get_masks()`)
- `.species` - Species IDs (not `.get_species()`)
- `.N_max` - Max atoms

**Methods:**
- `.get_all()` → dict (not `.get_dataset()`)
- `.get_frame(idx)` → single frame dict
- `.split_train_val(fraction)` → train_loader, val_loader

### CombinedModel (models/combined_model.py)
**Methods:**
- `.initialize_params(rng_key)` - Initialize model parameters (not `.initialize()`)
- `.compute_components(params, R, mask, species)` - Get energy components (not `.compute_energy_components()`)
- `.compute_force_components(params, R, mask, species)` - Get force breakdown
- `.compute_total_energy(params, R, mask, species, neighbor)` - Total energy

### Trainer (training/trainer.py)
**Constructor:**
- `__init__(model, config, train_loader, val_loader=None)`
- NOT: `__init__(model, config, loaders, init_params)`
- Initializes params internally

**Methods:**
- `.train_full_pipeline()` - Run complete training
- `.train_stage(optimizer_name, epochs)` - Single training stage
- `.pretrain_prior(epochs, optimizer_name)` - LBFGS prior fitting (TODO: needs fix)
- `.get_best_params()` - Get best parameters from training

---

### 5. ❌ **Validation Set Too Small for Batch Size** → ✅ FIXED
**Error:** `OverflowError: cannot convert float infinity to integer`
**Cause:** With only 10 frames and val_fraction=0.1, validation set has 1 sample, but batch_per_device=16 * 4 GPUs = 64 batch size causes division by zero in NumpyDataLoader
**Fix:** Detect when validation set is too small (`N_val < batch_per_device * n_devices`) and use training data for validation instead

**Before:**
```python
N_val = len(dataset["R"]) - N_train
# Always create separate val_loader even if too small
val_loader = NumpyDataLoader(R=dataset["R"][N_train:])
```

**After:**
```python
min_val_samples = batch_per_device * n_devices
if N_val < min_val_samples:
    print(f"[Split] Warning: Validation set too small, using training data")
    N_train = len(dataset["R"])
    N_val = 0
    val_loader = train_loader  # Reuse training loader
```

**Files:** `scripts/train.py:217-250`

---

## Known Issues

### ✅ Prior Pre-training Implementation - FIXED!
**Status:** ✅ IMPLEMENTED - Ready for testing
**Previous Issue:** Used ForceMatching with configurable optimizer instead of dedicated LBFGS
**Fix:** Completely rewrote `pretrain_prior()` method with proper LBFGS implementation

**Implementation Details:**
- ✅ Uses `optax.lbfgs()` optimizer specifically
- ✅ Runs `jax.lax.while_loop` with convergence criteria
- ✅ Only optimizes prior parameters (not Allegro)
- ✅ Matches original `pre_training_priors.py` algorithm
- ✅ Convergence based on gradient norm threshold
- ✅ Returns fitted parameters and loss history

**Files Modified:**
- `training/trainer.py` - Rewrote pretrain_prior method (206-398)
- `models/prior_energy.py` - Added compute_total_energy_from_params (366-389)
- `config/manager.py` - Added LBFGS-specific config methods
- `config_template.yaml` - Updated with LBFGS parameters

**See:** [PRIOR_PRETRAINING_FIX.md](PRIOR_PRETRAINING_FIX.md) for complete documentation

---

## Testing Status

✅ **Syntax Check:** Both scripts compile without errors
✅ **Path Resolution:** Datasets now found correctly
✅ **API Compatibility:** All method calls corrected
✅ **Data Loading:** Successfully loads datasets with 10 frames
✅ **Model Initialization:** Allegro model initializes correctly (18 species detected)
✅ **Batch Size Handling:** Validation set size checking works
⏳ **Training Loop:** Needs testing

---

## Next Steps

1. **Submit test job** from clean_code_base directory:
   ```bash
   cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base
   sbatch scripts/run_training.sh ../config_allegro_exp2.yaml
   ```

2. **Monitor log file:** `train_allegro_<JOB_ID>.log`

3. **If successful:** Test with prior pre-training enabled (after fixing LBFGS implementation)

4. **Expected outputs:**
   - `exported_models/<model_name>.mlir`
   - `exported_models/<model_name>_params.pkl`
   - `exported_models/<model_name>_config.yaml`
   - `exported_models/loss_curve_<JOB_ID>_<model_id>.png`
   - `train_allegro_<JOB_ID>.log`

---

**Ready for testing!**
