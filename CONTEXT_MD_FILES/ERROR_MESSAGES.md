# Error Message Patterns

**Date:** 2026-01-26
**Status:** Documentation - Future Improvement

---

## Overview

This document describes error message patterns for future improvement. Current error messages are generally adequate but could be more consistent and helpful.

**Decision:** Defer systematic error message improvement to future work. This requires a comprehensive review of all error paths.

---

## Current State

### What's Good
- Most errors include context (file paths, shapes, etc.)
- Config errors generally indicate which key is missing
- JAX errors are passed through (sometimes cryptic but standard)

### What Could Be Better
- Inconsistent formatting across modules
- Some errors lack actionable hints
- Shape mismatches often show as cryptic JAX errors
- Missing "how to fix" suggestions

---

## Patterns to Improve (Future Work)

### 1. Path Errors

**Current:**
```python
raise FileNotFoundError(f"NPZ file not found: {path}")
```

**Better (future):**
```python
raise FileNotFoundError(
    f"Dataset file not found: {path}\n"
    f"Searched in: {absolute_path}\n"
    f"Hint: Check 'data.path' in config.yaml. "
    f"If relative path, it's resolved from clean_code_base/ directory."
)
```

**Benefits:**
- Shows both relative and absolute paths
- Explains path resolution behavior
- Points to config location

---

### 2. Shape Mismatches

**Current:**
```python
# Often just a cryptic JAX error like:
# ValueError: Incompatible shapes for dot: (53, 3) and (53,)
```

**Better (future):**
```python
raise ValueError(
    f"Shape mismatch in coordinate array\n"
    f"Expected: R.shape[1] == 3 (xyz coordinates)\n"
    f"Got: R.shape = {R.shape}\n"
    f"Hint: Check that your dataset uses 3D coordinates. "
    f"For 2D data, pad with zeros in the z-dimension."
)
```

**Benefits:**
- Clear statement of expected vs actual
- Explains what the dimensions mean
- Actionable fix suggestion

---

### 3. Config Errors

**Current:**
```python
# KeyError with no context
KeyError: 'training'
```

**Better (future):**
```python
raise ValueError(
    f"Missing required config section: 'training'\n"
    f"Add to config.yaml:\n"
    f"  training:\n"
    f"    epochs_adabelief: 100\n"
    f"    epochs_yogi: 50\n"
    f"    val_fraction: 0.1\n"
    f"    batch_per_device: 16\n"
    f"See config_template.yaml for full example."
)
```

**Benefits:**
- Shows exactly what to add
- Provides working example
- Points to template

---

### 4. Model Initialization Errors

**Current:**
```python
# Generic JAX errors during model init
```

**Better (future):**
```python
raise RuntimeError(
    f"Failed to initialize Allegro model\n"
    f"Error: {e}\n"
    f"Common causes:\n"
    f"  - N_max ({N_max}) too small for dataset (max atoms: {actual_max})\n"
    f"  - Invalid species array (should be int32, got {species.dtype})\n"
    f"  - Box dimensions invalid: {box}\n"
    f"Hint: Check your dataset shape and model config."
)
```

**Benefits:**
- Lists common causes
- Shows actual vs expected values
- Actionable debugging steps

---

### 5. Training Errors

**Current:**
```python
# Loss becomes NaN during training (no early warning)
```

**Better (future):**
```python
if jnp.isnan(loss):
    raise RuntimeError(
        f"Loss became NaN at epoch {epoch}, batch {batch_idx}\n"
        f"Last valid loss: {last_valid_loss:.6e}\n"
        f"Common causes:\n"
        f"  - Learning rate too high (try reducing by 10x)\n"
        f"  - Gradient explosion (enable gradient clipping)\n"
        f"  - Invalid input data (check for NaN/Inf in dataset)\n"
        f"Hint: Check logs for warnings before this error.\n"
        f"Try reducing optimizer.{optimizer}.lr in config.yaml"
    )
```

**Benefits:**
- Catches problem early
- Lists common causes
- Specific fix suggestions

---

### 6. Validation Set Errors

**Current:**
```python
print(f"[Split] Warning: Validation set ({N_val} samples) is too small...")
```

**Better (future):**
```python
logger.warning(
    f"Validation set ({N_val} samples) is too small for batch size "
    f"({batch_size}). Using training data for validation.\n"
    f"Options to fix:\n"
    f"  1. Increase max_frames: {max_frames} → {suggested_frames}\n"
    f"  2. Reduce batch_per_device: {batch_per_device} → {suggested_batch}\n"
    f"  3. Reduce val_fraction: {val_fraction} → 0.0 (explicit)\n"
    f"Current behavior: Validation = training (overfitting risk)"
)
```

**Benefits:**
- Explains why fallback happened
- Lists multiple solutions with specific values
- Warns about implications

---

### 7. LBFGS Convergence

**Current:**
```python
print(f"[LBFGS] Converged: {converged}")
```

**Better (future):**
```python
if not converged:
    logger.warning(
        f"LBFGS did not converge after {max_steps} steps\n"
        f"Final gradient norm: {grad_norm:.6e} (tolerance: {tol_grad:.6e})\n"
        f"Options:\n"
        f"  - Increase max_steps: {max_steps} → {max_steps * 2}\n"
        f"  - Relax tolerance: {tol_grad:.6e} → {tol_grad * 10:.6e}\n"
        f"  - Check if prior parameters are reasonable\n"
        f"Continuing with partially fitted parameters..."
    )
```

**Benefits:**
- Clear statement of problem
- Quantitative feedback (gradient norm)
- Specific parameter adjustments
- Explains consequences

---

## Error Message Best Practices (Future)

### Structure

All error messages should follow this pattern:

```python
raise ErrorType(
    f"{WHAT_HAPPENED}\n"
    f"{CONTEXT_INFO}\n"
    f"{WHY_IT_HAPPENED}\n"
    f"{HOW_TO_FIX}"
)
```

**Example:**
```python
raise ValueError(
    # WHAT
    f"Invalid cutoff distance for neighbor list\n"
    # CONTEXT
    f"Got: cutoff = {cutoff}, expected: cutoff > 0\n"
    # WHY
    f"Cutoff defines maximum interaction distance between atoms. "
    f"Must be positive.\n"
    # FIX
    f"Set model.cutoff in config.yaml to a positive value (typically 8-15 Å for CG proteins)."
)
```

### Guidelines

1. **Be specific:** Show actual values, not just "invalid input"
2. **Be actionable:** Tell user how to fix, don't just complain
3. **Be educational:** Briefly explain what the parameter means
4. **Be consistent:** Same format across all modules
5. **Be concise:** 3-5 lines maximum, use links for details

---

## Implementation Plan (Future)

### Phase 1: Categorize Errors
- Audit all `raise` statements in codebase
- Group by error type (path, shape, config, training, etc.)
- Identify most common/confusing errors

### Phase 2: Create Error Templates
- Define standard templates for each category
- Create utility functions for common patterns
- Write tests for error messages

### Phase 3: Systematic Replacement
- Replace errors module by module
- Test each error path manually
- Update documentation

### Phase 4: User Testing
- Collect feedback on new errors
- Iterate on clarity and helpfulness
- Add more hints based on user questions

---

## Error Categories

| Category | Priority | Frequency | Current Quality | Target Quality |
|----------|----------|-----------|----------------|----------------|
| **Path errors** | High | Common | Fair | Good |
| **Config errors** | High | Common | Poor | Good |
| **Shape errors** | High | Common | Poor (JAX) | Good |
| **Training errors** | Medium | Occasional | Fair | Good |
| **Model init errors** | Medium | Occasional | Poor | Good |
| **Convergence errors** | Low | Rare | Fair | Good |
| **Export errors** | Low | Rare | Fair | Fair |

---

## Current Error Handling

### Good Examples

```python
# data/loader.py - Clear path error
if not path_obj.exists():
    raise FileNotFoundError(f"NPZ file not found: {path_obj}")
```

```python
# scripts/train.py - Helpful validation warning
print(f"[Split] Warning: Validation set ({N_val} samples) is too small...")
print(f"[Split] Using training data for validation")
```

### Needs Improvement

```python
# Various files - Generic KeyError
self.data["training"]["epochs"]  # KeyError: 'training' (no hint)
```

```python
# Various files - Cryptic JAX shape errors
# No preprocessing of JAX errors for user clarity
```

---

## Related Files

All files with error handling:
- `data/loader.py`
- `data/preprocessor.py`
- `models/*.py`
- `training/trainer.py`
- `scripts/*.py`
- `config/manager.py`

---

## Summary

**Current state:** Error messages are generally adequate but inconsistent.

**Future goal:** All errors should be:
- Specific (show actual values)
- Actionable (tell user how to fix)
- Educational (briefly explain why)
- Consistent (same format everywhere)

**Timeline:** Defer to future work - requires systematic review of all error paths (~50+ error sites).

---

**Note:** This is a documentation-only task. No code changes required now. Use this as a guide when adding new error handling or when users report confusing errors.
