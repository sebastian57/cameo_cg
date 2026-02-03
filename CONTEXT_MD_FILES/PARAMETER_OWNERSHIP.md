# Parameter Ownership in Chemtrain

**Date:** 2026-01-26
**Status:** Documentation

---

## Overview

This document clarifies **who owns parameters** in the chemtrain codebase and how they flow through the training pipeline.

Understanding parameter ownership is critical for:
- Avoiding duplicate parameter storage
- Ensuring parameters are updated correctly
- Preventing stale parameter bugs
- Understanding the training state

---

## Parameter Lifecycle

### Phase 1: Initialization

```python
# scripts/train.py
model = CombinedModel(config=config, R0=R0, box=box, species=species0, N_max=N_max)
```

**What happens:**
1. `CombinedModel.__init__()` creates initial parameters
2. Prior params initialized from config
3. Allegro params initialized randomly (via nequip)
4. Both stored in `model.params` dict

**Owner:** `CombinedModel` instance

**Structure:**
```python
model.params = {
    "prior": {...},      # Prior energy parameters
    "allegro": {...}     # Allegro ML parameters
}
```

### Phase 2: Training Setup

```python
trainer = Trainer(model, config, train_loader, val_loader)
```

**What happens:**
1. Trainer stores reference to model
2. Trainer creates its own copy for tracking best params
3. Initial parameters: `self.params = model.params`

**Owner:** `Trainer` instance

**Important:** Model parameters are **passed by reference**, not copied. Changes to `trainer.params` affect `model.params` unless explicitly copied.

### Phase 3: Prior Pre-training (Optional)

```python
results = trainer.pretrain_prior()
```

**What happens:**
1. LBFGS optimizes **only** prior parameters
2. Fitted parameters returned in `results["fitted_params"]`
3. Model parameters updated: `model.prior.params = fitted_params`
4. Trainer parameters updated: `trainer.params["prior"] = fitted_params`

**Owner:** Model and Trainer (synchronized)

**Critical:** Prior pre-training modifies parameters **in place**. The allegro parameters remain unchanged.

### Phase 4: Full Training

```python
results = trainer.train_full_pipeline()
```

**What happens:**
1. ForceMatching trainer optimizes **all** parameters (prior + allegro)
2. Best parameters tracked in `trainer.best_params`
3. Model parameters updated after each epoch
4. Best checkpoint saved to disk

**Owner:** `Trainer.best_params` is the source of truth

**Important:** After training, use `trainer.get_best_params()` to retrieve final parameters, **not** `model.params` (which may contain last epoch, not best).

### Phase 5: Export

```python
best_params = trainer.get_best_params()
exporter = AllegroExporter.from_combined_model(model, params=best_params, ...)
```

**What happens:**
1. Best parameters passed explicitly to exporter
2. Exporter does **not** modify parameters
3. Parameters embedded in MLIR file

**Owner:** Exporter receives parameters but does not own them (stateless)

---

## Who Owns What?

| Component | Owns Parameters? | Modifies Parameters? | Source of Truth? |
|-----------|------------------|---------------------|------------------|
| `CombinedModel` | ✅ Yes (`self.params`) | ❌ No (stateless compute) | ❌ Only during init |
| `PriorEnergy` | ✅ Yes (`self.params`) | ❌ No (stateless compute) | ❌ Only during init |
| `AllegroModel` | ✅ Yes (`self.params`) | ❌ No (stateless compute) | ❌ Only during init |
| `Trainer` | ✅ Yes (`self.best_params`) | ✅ Yes (via optimizers) | ✅ **YES** - after training |
| `Evaluator` | ❌ No (accepts params) | ❌ No (stateless) | ❌ No |
| `Exporter` | ❌ No (accepts params) | ❌ No (stateless) | ❌ No |

---

## Design Philosophy

### Models are Stateless

All model classes (`CombinedModel`, `PriorEnergy`, `AllegroModel`) are **stateless** for energy and force computation:

```python
# CORRECT - parameters passed explicitly
energy = model.compute_energy(params, R, mask, species, neighbor)

# WRONG - using internal params directly (not recommended after training starts)
energy = model.compute_energy(model.params, R, mask, species, neighbor)
```

**Why?**
- JAX functional programming paradigm
- Enables gradient computation via `jax.grad`
- Prevents stale parameter bugs
- Thread-safe and reentrant

### Trainer Owns Training State

The `Trainer` is the **single source of truth** for parameters during and after training:

```python
# CORRECT
best_params = trainer.get_best_params()

# WRONG - may not be best params
params = model.params
```

**Why?**
- Trainer tracks best validation loss
- Trainer saves checkpoints
- Trainer manages optimizer state

---

## Parameter Update Flow

```
Initialization:
  CombinedModel.params (random + config)
    ↓
  Trainer.params (copy reference)

Prior Pre-training (if enabled):
  LBFGS optimization
    ↓
  fitted_params (new prior params)
    ↓
  model.prior.params = fitted_params (update in place)
  trainer.params["prior"] = fitted_params (update in place)

Stage 1 Training (e.g., AdaBelief):
  ForceMatching optimizer
    ↓
  Updated params each epoch
    ↓
  trainer.best_params (best by validation loss)

Stage 2 Training (e.g., Yogi):
  ForceMatching optimizer (continues from Stage 1)
    ↓
  Updated params each epoch
    ↓
  trainer.best_params (updated if improved)

Export:
  trainer.get_best_params()
    ↓
  AllegroExporter (receives params, does not own)
    ↓
  MLIR file (frozen params)
```

---

## Common Pitfalls

### ❌ Using model.params after training

```python
# WRONG
results = trainer.train_full_pipeline()
params = model.params  # May not be best params!
```

**Fix:**
```python
# CORRECT
results = trainer.train_full_pipeline()
params = trainer.get_best_params()  # Best params by validation loss
```

### ❌ Modifying params without synchronization

```python
# WRONG - only updates model, not trainer
model.prior.params["r0"] = 5.0
```

**Fix:**
```python
# CORRECT - update both
new_params = {...}
model.prior.params = new_params
trainer.params["prior"] = new_params
```

### ❌ Forgetting to pass params to stateless functions

```python
# WRONG - which params are used?
energy = model.compute_energy(R, mask, species, neighbor)
```

**Fix:**
```python
# CORRECT - explicit params
best_params = trainer.get_best_params()
energy = model.compute_energy(best_params, R, mask, species, neighbor)
```

---

## Best Practices

### 1. Always pass parameters explicitly

```python
# Energy computation
energy = model.compute_energy(params, R, mask, species, neighbor)

# Force computation
forces = -jax.grad(lambda R_: model.compute_energy(params, R_, ...))(R)
```

### 2. Use Trainer as source of truth

```python
# During training
current_params = trainer.params

# After training
best_params = trainer.get_best_params()
```

### 3. Don't modify model.params after training starts

Once `Trainer` is created, modifications should go through `Trainer`, not `Model`.

### 4. Be explicit about parameter provenance

```python
# GOOD - clear where params come from
fitted_prior_params = results["fitted_params"]
model.prior.params = fitted_prior_params

# BAD - unclear
model.prior.params = results["fitted_params"]
```

---

## Related Files

- `models/combined_model.py` - Model initialization
- `models/prior_energy.py` - Prior parameter storage
- `models/allegro_model.py` - Allegro parameter storage
- `training/trainer.py` - Parameter optimization and tracking
- `export/exporter.py` - Parameter export (stateless)
- `evaluation/evaluator.py` - Parameter evaluation (stateless)

---

**Summary:** Models initialize and store parameters but are **stateless** for computation. Trainer **owns** parameters during and after training. Always use `trainer.get_best_params()` for final parameters.
