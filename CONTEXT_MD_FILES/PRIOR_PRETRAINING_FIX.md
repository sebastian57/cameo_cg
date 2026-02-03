# Prior Pre-Training Implementation - LBFGS Based

**Date:** 2026-01-23
**Status:** ✅ IMPLEMENTED - Ready for testing

---

## Overview

Implemented proper LBFGS-based prior pre-training matching the original `pre_training_priors.py` implementation.

### Key Differences from Original (Incorrect) Implementation:

| Aspect | ❌ Old (Wrong) | ✅ New (Correct) |
|--------|---------------|------------------|
| **Optimizer** | Configurable (Adam, etc.) | **Always LBFGS** |
| **Training Loop** | ForceMatching trainer | `jax.lax.while_loop` |
| **Convergence** | Fixed epochs | Gradient norm threshold |
| **Parameters** | All model params | **Only prior params** |
| **Based on** | Generic training | `pre_training_priors.py` |

---

## Implementation Details

### Files Modified:

1. **training/trainer.py**
   - Completely rewrote `pretrain_prior()` method (lines 206-398)
   - Added `optax` import
   - Implements LBFGS with convergence criteria

2. **models/prior_energy.py**
   - Added `compute_total_energy_from_params()` method (lines 366-389)
   - Allows computing energy with different parameters (needed for LBFGS)

3. **config/manager.py**
   - Changed `get_pretrain_prior_optimizer()` default to `"lbfgs"`
   - Added `get_pretrain_prior_max_steps()` → default 200
   - Added `get_pretrain_prior_tol_grad()` → default 1e-6

4. **config_template.yaml**
   - Updated prior pre-training section
   - Removed `pretrain_prior_epochs` and `pretrain_prior_optimizer`
   - Added `pretrain_prior_max_steps` and `pretrain_prior_tol_grad`
   - Documented LBFGS-only behavior

---

## How It Works

### 1. Force Matching Loss

```python
def force_matching_loss(params):
    """Compute L2 loss between predicted and reference forces."""
    # Predict forces from prior energy only
    F_pred = jax.vmap(lambda R_f, m_f: prior_forces(params, R_f, m_f))(R, mask)

    # Masked squared error
    diff = (F_pred - F_ref) * mask[..., None]

    # Normalize by number of real atoms
    return jnp.sum(diff * diff) / jnp.sum(mask[..., None])
```

### 2. LBFGS Optimization

```python
# Initialize LBFGS optimizer
opt = optax.lbfgs(learning_rate=1.0)
value_and_grad = optax.value_and_grad_from_state(force_matching_loss)

# Run optimization loop
st0 = init_state(params0)
stF = jax.lax.while_loop(cond_fn, body_fn, st0)
```

### 3. Convergence Criteria

```python
def cond_fn(st: FitState):
    not_done = st.step < max_steps

    # Check gradient norm
    grad_norm = optax.tree.norm(st.opt_state["grad"])
    not_converged_grad = (st.step < min_steps) or (grad_norm >= tol_grad)

    return not_done and not_converged_grad
```

**Stops when:**
- Gradient norm < `tol_grad` (default 1e-6), OR
- Reached `max_steps` (default 200)

---

## Configuration

### YAML Example:

```yaml
model:
  use_priors: true
  priors:
    # Initial prior parameters (will be optimized)
    r0: 3.8375435
    kr: 154.50629
    # ... other prior params

training:
  # Enable prior pre-training
  pretrain_prior: true
  pretrain_prior_max_steps: 200  # Max LBFGS iterations
  pretrain_prior_tol_grad: 1.0e-6  # Convergence threshold
```

### Programmatic Usage:

```python
trainer = Trainer(model, config, train_loader, val_loader)

# Run LBFGS prior pre-training
results = trainer.pretrain_prior(
    max_steps=200,
    tol_grad=1e-6
)

# Check convergence
print(f"Converged: {results['converged']}")
print(f"Final loss: {results['train_loss']:.6e}")
print(f"Steps: {results['steps']}")
print(f"Fitted params: {results['fitted_params']}")
```

---

## Workflow

### Full Training Pipeline with Prior Pre-training:

```python
# 1. Initialize model and trainer
model = CombinedModel(config, R0, box, species, N_max)
trainer = Trainer(model, config, train_loader, val_loader)

# 2. Run full pipeline (handles prior pre-training automatically if enabled)
results = trainer.train_full_pipeline()

# Results structure:
# {
#   "prior_pretrain": {
#     "train_loss": ...,
#     "steps": ...,
#     "converged": True/False,
#     "fitted_params": {...}
#   },
#   "stage1": {"train_loss": ..., "val_loss": ...},
#   "stage2": {"train_loss": ..., "val_loss": ...}
# }
```

### Order of Operations:

1. **If `training.pretrain_prior: true` and `model.use_priors: true`:**
   - Run LBFGS on prior parameters only
   - Fit to reference forces in training data
   - Update `model.prior.params` with fitted values

2. **Stage 1 Training (e.g., AdaBelief):**
   - Train full model (priors + Allegro)
   - Use fitted prior params as initialization
   - Optimize both prior AND Allegro parameters

3. **Stage 2 Training (e.g., Yogi):**
   - Fine-tune with different optimizer
   - Continue from Stage 1 parameters

---

## Output Example

```
============================================================
Prior Pre-Training (LBFGS, max_steps=200)
============================================================
[LBFGS] Starting optimization...
[LBFGS] Completed: 47 steps
[LBFGS] Final loss: 1.234567e-03
[LBFGS] Grad norm: 5.432e-07 (tol=1.000e-06)
[LBFGS] Converged: True

[LBFGS] Fitted parameters:
  r0: 3.842156
  kr: 156.234521
  epsilon: 1.123456
  sigma: 2.987654
  a: [...]
  b: [...]
  k_dih: [...]
  gamma_dih: [...]
```

---

## What Gets Optimized

### Prior Parameters (Optimized by LBFGS):
- `r0` - Equilibrium bond distance
- `kr` - Bond force constant
- `epsilon` - LJ epsilon
- `sigma` - LJ sigma
- `a`, `b` - Repulsive potential coefficients
- `k_dih`, `gamma_dih` - Dihedral parameters

### NOT Optimized (Fixed):
- `theta0`, `k_theta` - Angle parameters (unused, kept for future)

### Allegro Parameters:
- **Not touched** during prior pre-training
- Only optimized in Stage 1 and Stage 2

---

## Advantages of LBFGS vs Gradient Descent

1. **Faster Convergence** - Second-order method, fewer iterations
2. **Automatic Convergence** - Stops when gradient norm is small
3. **No Learning Rate Tuning** - LBFGS handles step sizes automatically
4. **Physics-Based Initialization** - Priors start from reasonable values

---

## Testing Checklist

### ✅ To Test:

1. **Syntax:** ✅ Compiles without errors
2. **Small dataset:** Test with 100 frames, LBFGS pre-training enabled
3. **Convergence:** Check that it stops before max_steps (gradient threshold)
4. **Parameter update:** Verify fitted params are different from initial
5. **Full pipeline:** Ensure Stage 1/2 training works after pre-training
6. **Disable pre-training:** Verify `pretrain_prior: false` skips LBFGS

### Configuration for Testing:

```yaml
model:
  use_priors: true
  # ... prior initial values

training:
  pretrain_prior: true
  pretrain_prior_max_steps: 50  # Small for quick test
  pretrain_prior_tol_grad: 1.0e-5  # Looser tolerance

  epochs_adabelief: 5
  epochs_yogi: 0
```

---

## Comparison with Original Implementation

### Original `pre_training_priors.py`:

```python
params_fit, loss_hist = fit_priors_lbfgs(
    prior_params,
    displacement,
    dataset,
    bonds, angles, rep_pairs,
    apply_fn_priors
)
```

### New `Trainer.pretrain_prior()`:

```python
results = trainer.pretrain_prior(max_steps=200, tol_grad=1e-6)
# Automatically:
# - Extracts training data from loader
# - Gets topology from model
# - Computes forces via jax.grad
# - Runs LBFGS with while_loop
# - Updates model.prior.params
```

**Same algorithm, integrated into OOP design!**

---

## Related Files

- Original implementation: `../pre_training_priors.py`
- Original usage: `../train_fm_multiple_proteins.py` (lines 252-263, commented out)
- New implementation: `training/trainer.py` (method `pretrain_prior`)
- Helper method: `models/prior_energy.py` (method `compute_total_energy_from_params`)

---

**Status:** Fully implemented and ready for testing with real data! ✅
