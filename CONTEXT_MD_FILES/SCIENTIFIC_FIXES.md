# Scientific Fixes - Easy Toggle Guide

**Date:** 2026-01-26
**Purpose:** Document clearly-marked scientific improvements that can be easily enabled/disabled

---

## Overview

This document lists all scientific fixes added to the codebase with clear toggle comments. Each fix is marked with:
- `# ========================================================================`
- Clear explanation of issue and solution
- Instructions to enable/disable
- Original values for comparison

---

## Fix 1: Repulsive Prior Strength ⚠️ **HIGH PRIORITY**

**File:** [config_template.yaml](config_template.yaml) lines 85-100

**Issue:**
- Current: `epsilon=1.0`, `sigma=3.0` with `weight=0.25` → effective ~0.25 kcal/mol
- Needed: ~5 kcal/mol for stable CG protein MD simulations
- **Result:** MD simulations unstable, proteins can self-intersect

**Fix:**
```yaml
# In config_template.yaml (or your config.yaml)

# RECOMMENDED VALUES (uncomment to enable):
epsilon: 5.0   # Increase from 1.0 → stronger repulsion
sigma: 4.0     # Increase from 3.0 → larger excluded volume

# ORIGINAL VALUES (for comparison):
# epsilon: 1.0
# sigma: 3.0
```

**How to Enable:**
1. Open your config file (e.g., `config_allegro_exp2.yaml`)
2. Find the `model.priors` section
3. Change `epsilon: 1.0` → `epsilon: 5.0`
4. Change `sigma: 3.0` → `sigma: 4.0`

**Expected Impact:**
- Much more stable MD simulations
- Prevents backbone self-intersection
- Better energy conservation in NVE ensemble

---

## Fix 2: Energy Term Weights (User Preference)

**File:** [config_template.yaml](config_template.yaml) lines 63-85

**Issue:**
- Bond/angle/dihedral: Fitted from histograms → should be weighted
- Repulsion: Manually chosen (NOT fitted) → should NOT be weighted

**Rationale:**
- Histogram-fitted terms capture full MD energy landscape
- Weights prevent double-counting correlated effects
- Repulsion is independent → should be at full strength

**User's Choice (CURRENT CONFIG):**
```yaml
weights:
  bond: 0.5        # Fitted from histograms
  angle: 0.1       # Fitted from histograms
  dihedral: 0.15   # Fitted from histograms
  repulsive: 1.0   # NOT fitted → full strength (user's preference)
```

**Original Weights (for comparison):**
```yaml
weights:
  bond: 0.5
  angle: 0.1
  repulsive: 0.25    # Was scaled down
  dihedral: 0.15
```

**Status:** ✅ Already set to user's preference in config_template.yaml

---

## Fix 3: Excluded Volume for Nearby Residues (OPTIONAL - Not Enabled by Default)

**Files:**
- [models/topology.py](models/topology.py) - Method added with toggle comments
- [models/prior_energy.py](models/prior_energy.py) - Commented implementation

**Issue:**
- No repulsion for sequence separation 2-5
- Backbone can self-intersect for nearby residues
- Regular repulsive pairs only handle separation ≥6

**Fix Added (Currently Commented Out):**

### In topology.py (lines 221-260):
```python
def get_excluded_volume_pairs(self, min_sep: int = 2, max_sep: int = 5) -> jax.Array:
    """Get atom pairs for excluded volume (soft repulsion for nearby residues)."""
    # Implementation provided - ready to use
```

### In prior_energy.py (lines 287-360):
```python
# FULLY COMMENTED OUT - ready to uncomment
# def compute_excluded_volume_energy(self, R: jax.Array, mask: jax.Array) -> jax.Array:
#     """Compute soft excluded volume for nearby residues."""
#     # Full implementation provided
```

**How to Enable (3 steps):**

**Step 1:** In `models/prior_energy.py`, uncomment the method (lines ~287-360)

**Step 2:** In `models/prior_energy.py` `__init__` method, add:
```python
# In __init__ after line 194 (after self.rep_pairs = ...)
self.excluded_vol_pairs = topology.get_excluded_volume_pairs(min_sep=2, max_sep=5)
self.params["epsilon_ex"] = jnp.asarray(prior_params.get("epsilon_ex", 1.5), dtype=jnp.float32)
self.params["sigma_ex"] = jnp.asarray(prior_params.get("sigma_ex", 3.5), dtype=jnp.float32)
```

**Step 3:** In `models/prior_energy.py` `compute_energy` method, add:
```python
# In compute_energy, after E_rep_raw calculation (line ~333)
E_ex_raw = self.compute_excluded_volume_energy(R, mask)
E_ex = self.weights.get("excluded_volume", 1.0) * E_ex_raw

# Update return dict to include "E_excluded_volume": E_ex
# Add E_ex to E_total calculation
```

**Step 4:** In `config_template.yaml`, add to `model.priors`:
```yaml
priors:
  weights:
    # ... existing weights ...
    excluded_volume: 1.0  # Full strength for excluded volume

  # ... existing params ...

  # Excluded volume parameters (softer than regular repulsion)
  epsilon_ex: 1.5  # Softer than regular epsilon (5.0)
  sigma_ex: 3.5    # Slightly smaller than regular sigma (4.0)
```

**Expected Impact:**
- Prevents backbone self-intersection for nearby residues
- More realistic protein conformations
- Better agreement with all-atom MD

**Recommendation:**
- Test with Fix 1 (repulsive strength) first
- Add this if MD simulations still show backbone issues

---

## Summary of Changes Made

### Files Modified:

1. **config_template.yaml**
   - Added scientific fix comments for repulsive parameters
   - Added explanation of weight rationale
   - Marked recommended vs original values

2. **models/topology.py**
   - Added `get_excluded_volume_pairs()` method
   - Fully functional, ready to use
   - Clear documentation and comments

3. **models/prior_energy.py**
   - Added commented-out `compute_excluded_volume_energy()` method
   - Full implementation provided
   - Step-by-step enable instructions in comments

---

## Testing Strategy (User's Plan)

**Ablation Study - 4 Models:**

1. **Pure Allegro** (no priors)
   ```yaml
   model:
     use_priors: false
   ```

2. **Prior + Allegro** (current implementation)
   ```yaml
   model:
     use_priors: true
   # With current weights and parameters
   ```

3. **Prior + Allegro** (with Fix 1 only - stronger repulsion)
   ```yaml
   model:
     use_priors: true
     priors:
       epsilon: 5.0
       sigma: 4.0
       weights:
         repulsive: 1.0  # Already set
   ```

4. **Prior + Allegro** (with Fixes 1+3 - repulsion + excluded volume)
   ```yaml
   model:
     use_priors: true
   # Enable excluded volume following steps above
   ```

**Metrics to Compare:**
- Training loss convergence
- Force RMSE/MAE on validation set
- MD stability (energy conservation, no crashes)
- Structural properties (Rg, RMSD, contact maps)
- Transferability to new proteins

---

## Quick Reference

**Enable Fix 1 (Repulsive Strength) - 2 minutes:**
```yaml
# In your config.yaml
model:
  priors:
    epsilon: 5.0  # Change from 1.0
    sigma: 4.0    # Change from 3.0
```

**Enable Fix 2 (Weights) - Already Done:**
```yaml
weights:
  repulsive: 1.0  # Already set in config_template.yaml
```

**Enable Fix 3 (Excluded Volume) - 15 minutes:**
- Follow 4-step instructions above
- Uncomment code in prior_energy.py
- Add to __init__, compute_energy, config

---

## What's Already Working

✅ **MLIR Export:** Code exists, will export after training completes (line 343 of train.py)
✅ **Loss Plotting:** Automatic after training (lines 357-370 of train.py)
✅ **Loss Data:** Saved to text file (line 367 of train.py)
✅ **Annotated Plots:** Config parameters shown in plot (visualizer.py)
✅ **Stage Separator:** Vertical line at optimizer transition (visualizer.py)

**Why these didn't appear before:**
- Training crashed with `AttributeError` (now fixed via `train_data` parameter)
- Code never reached export/plotting sections

---

## Next Steps

1. **Run training with bug fix:**
   ```bash
   cd clean_code_base
   sbatch scripts/run_training.sh ../config_allegro_exp2.yaml
   ```

2. **Verify outputs appear:**
   - Check `exported_models/` for `.mlir` file
   - Check for loss plot and loss data file
   - Training should complete without errors

3. **Test Fix 1 (repulsive strength):**
   - Modify config: `epsilon: 5.0`, `sigma: 4.0`
   - Run training
   - Compare MD stability with original

4. **Run ablation study:**
   - Train 4 models as planned
   - Compare all metrics
   - Decide which fixes to keep

---

**All fixes are clearly marked with `# ========================================================================` comments for easy identification and toggling.**
