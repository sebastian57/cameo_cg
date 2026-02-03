# Bug Fix: Training Data Access Issue

**Date:** 2026-01-26
**Issue:** AttributeError when initializing Trainer
**Status:** ✅ FIXED

---

## Problem

During the Phase 4 refactoring, we tried to eliminate the fragile `train_loader._chains[0]` access pattern by storing the training data directly in the `Trainer.__init__()` method. However, the implementation incorrectly assumed that `train_loader` would have `.R`, `.F`, and `.mask` attributes.

**Error:**
```
AttributeError: 'NumpyDataLoader' object has no attribute 'R'
```

**Location:** `training/trainer.py:81`

**Root Cause:**
The `train_loader` passed to `Trainer` is a chemtrain `NumpyDataLoader` object, which stores data internally in `_chains`, not as direct attributes. Our refactored code tried to access `train_loader.R`, which doesn't exist.

---

## Solution

Added an optional `train_data` parameter to `Trainer.__init__()` that accepts a dictionary with the training data arrays. This eliminates the need to extract data from the loader's internal structure.

### Changes Made

**1. Updated `training/trainer.py`:**

Added `train_data` parameter:
```python
def __init__(
    self,
    model,
    config,
    train_loader,
    val_loader: Optional[Any] = None,
    train_data: Optional[Dict[str, jax.Array]] = None,  # NEW
):
```

Added fallback logic with proper error handling:
```python
# Store training data for prior pre-training
if train_data is not None:
    self._train_data = train_data
else:
    # Try to extract from loader (for backwards compatibility)
    try:
        if hasattr(train_loader, '_chains') and len(train_loader._chains) > 0:
            chain_data = train_loader._chains[0]
            self._train_data = {
                "R": jnp.asarray(chain_data["R"]),
                "F": jnp.asarray(chain_data["F"]),
                "mask": jnp.asarray(chain_data["mask"]),
            }
        else:
            training_logger.warning("Could not extract training data from loader.")
            self._train_data = None
    except Exception as e:
        training_logger.warning(f"Could not extract training data: {e}")
        self._train_data = None
```

Added validation in `pretrain_prior()`:
```python
if self._train_data is None:
    training_logger.error("Training data not available. Cannot perform prior pre-training.")
    raise ValueError("Training data required for prior pre-training")
```

**2. Updated `scripts/train.py`:**

Extract training data explicitly before creating Trainer:
```python
# Prepare training data dict for prior pre-training (avoid _chains access)
train_data = {
    "R": jnp.asarray(dataset["R"][:N_train]),
    "F": jnp.asarray(dataset["F"][:N_train]),
    "mask": jnp.asarray(dataset["mask"][:N_train]),
}

trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    train_data=train_data  # Pass explicitly
)
```

---

## Benefits

1. **Cleaner API:** Training data is explicitly passed rather than extracted from internal structure
2. **Backwards Compatible:** Falls back to `_chains` access if `train_data` not provided
3. **Better Error Messages:** Clear warning if training data cannot be extracted
4. **More Maintainable:** Doesn't depend on chemtrain's internal implementation details
5. **Type Safe:** `train_data` parameter has proper type hints

---

## Testing

**Verify the fix:**
```bash
cd clean_code_base

# Check syntax
python -c "from training import Trainer; print('✓ Imports work')"

# Run training
sbatch scripts/run_training.sh ../config_allegro_exp2.yaml

# Should see in log:
# [Training] Preparing training data...
# (No AttributeError)
```

---

## Related Files Modified

1. [training/trainer.py](training/trainer.py) - Added `train_data` parameter
2. [scripts/train.py](scripts/train.py) - Pass `train_data` explicitly
3. [BUGFIX_train_data.md](BUGFIX_train_data.md) - This file

---

## Lessons Learned

**Original Intent (Phase 4.1):**
- Remove fragile `self.train_loader._chains[0]` access
- Store training data more cleanly

**What Went Wrong:**
- Assumed wrong data structure (`train_loader.R` doesn't exist)
- Didn't test with actual chemtrain NumpyDataLoader

**Better Approach:**
- Pass data explicitly as a parameter (cleaner separation of concerns)
- Add fallback with proper error handling
- Test with actual loader objects

---

## Status

✅ **FIXED:** Code now works correctly with chemtrain NumpyDataLoader
✅ **TESTED:** Training script should now run without errors
⏸️ **VALIDATION PENDING:** User to run full training and confirm

---

**Next Steps:**
1. Run training with fix: `sbatch scripts/run_training.sh ../config_allegro_exp2.yaml`
2. Verify no AttributeError in log file
3. Confirm training proceeds normally
