# Ensemble Training Plan

## Overview

Add ensemble training capability to train multiple models with different random seeds,
compute variance across models, and select the best performing one.

**Implementation approach:** Use chemtrain's native `EnsembleOfModels` class for sequential
training of multiple models. This is Phase 1; parallel GPU training can be added later.

---

## Architecture

### Current Flow (Single Model)
```
config.yaml → train.py → CombinedModel → ForceMatching trainer → export best model
```

### New Flow (Ensemble)
```
config.yaml (with ensemble settings)
    ↓
train_ensemble.py
    ↓
┌─────────────────────────────────────────┐
│  For each seed in [base_seed, base_seed+1, ..., base_seed+n-1]:  │
│    1. Initialize CombinedModel with seed                          │
│    2. Create ForceMatching trainer                                │
│    3. Train model                                                 │
│    4. Record validation loss                                      │
└─────────────────────────────────────────┘
    ↓
Compute variance of validation losses
    ↓
Select best model (lowest validation loss)
    ↓
Export best (or all, based on config)
```

---

## Config Changes

New section in `config_template.yaml`:

```yaml
ensemble:
  enabled: false              # Set to true to enable ensemble training
  n_models: 5                 # Number of models to train
  base_seed: 42               # Base seed (models use seeds: 42, 43, 44, 45, 46)
  save_all_models: false      # If true, save all models; if false, only save best
```

The `seed` field at the top level will be ignored when ensemble mode is enabled;
instead, `ensemble.base_seed` will be used.

---

## Implementation Details

### 1. Config Manager Updates

Add methods to `config/manager.py`:

```python
def get_ensemble_config(self) -> dict:
    """Get ensemble configuration."""
    return {
        "enabled": self.get("ensemble", "enabled", default=False),
        "n_models": self.get("ensemble", "n_models", default=5),
        "base_seed": self.get("ensemble", "base_seed", default=42),
        "save_all_models": self.get("ensemble", "save_all_models", default=False),
    }

def is_ensemble_enabled(self) -> bool:
    """Check if ensemble training is enabled."""
    return self.get("ensemble", "enabled", default=False)
```

### 2. New Training Script: `train_ensemble.py`

Based on `train.py` but modified for ensemble:

```python
def train_ensemble(config_file: str, job_id: str = None):
    """
    Train an ensemble of models with different seeds.

    1. Load config
    2. For each seed in range:
       - Initialize model with seed
       - Create ForceMatching trainer
       - Train model
       - Save checkpoint
    3. Compute variance of validation losses
    4. Select best model
    5. Export best (or all) models
    """

    config = ConfigManager(config_file)
    ensemble_config = config.get_ensemble_config()

    n_models = ensemble_config["n_models"]
    base_seed = ensemble_config["base_seed"]
    save_all = ensemble_config["save_all_models"]

    # Storage for results
    all_params = []
    all_val_losses = []
    all_trainers = []

    for i in range(n_models):
        seed = base_seed + i
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL {i+1}/{n_models} (seed={seed})")
        print(f"{'='*60}\n")

        # Initialize model with this seed
        model = create_model(config, seed=seed)

        # Create trainer
        trainer = create_trainer(model, config, loaders)

        # Train
        trainer.train(epochs=total_epochs)

        # Store results
        all_params.append(trainer.best_params)
        all_val_losses.append(trainer.val_losses[-1])
        all_trainers.append(trainer)

    # Compute statistics
    val_losses = np.array(all_val_losses)
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    variance_loss = np.var(val_losses)
    best_idx = np.argmin(val_losses)

    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Validation losses: {val_losses}")
    print(f"Mean: {mean_loss:.4f}")
    print(f"Std:  {std_loss:.4f}")
    print(f"Var:  {variance_loss:.4f}")
    print(f"Best model: {best_idx} (seed={base_seed + best_idx}, loss={val_losses[best_idx]:.4f})")

    # Export
    if save_all:
        for i, (params, trainer) in enumerate(zip(all_params, all_trainers)):
            export_model(params, config, suffix=f"_ensemble_{i}")
    else:
        export_model(all_params[best_idx], config, suffix="_best")

    # Save ensemble metadata
    save_ensemble_metadata(...)
```

### 3. Seed Handling

Seeds affect:
1. **Model initialization** - Weight initialization in Allegro
2. **Data shuffling** - Order of training batches

For reproducibility, each ensemble member will use:
- Model seed: `base_seed + i`
- Data shuffle seed: Same for all (to ensure fair comparison)

### 4. Output Structure

```
exported_models/
├── ensemble_metadata.json          # Statistics, seeds, losses
├── allegro_cg_protein_best.mlir    # Best model (always saved)
├── allegro_cg_protein_best_params.pkl
│
# If save_all_models=true:
├── allegro_cg_protein_ensemble_0.mlir
├── allegro_cg_protein_ensemble_0_params.pkl
├── allegro_cg_protein_ensemble_1.mlir
├── allegro_cg_protein_ensemble_1_params.pkl
└── ...
```

### 5. Ensemble Metadata JSON

```json
{
    "n_models": 5,
    "base_seed": 42,
    "seeds": [42, 43, 44, 45, 46],
    "validation_losses": [3.45, 3.21, 3.67, 3.12, 3.89],
    "mean_loss": 3.468,
    "std_loss": 0.287,
    "variance_loss": 0.082,
    "best_model_index": 3,
    "best_model_seed": 45,
    "best_model_loss": 3.12,
    "training_config": { ... }
}
```

---

## Usage

### Enable Ensemble Training

In `config_template.yaml`:
```yaml
ensemble:
  enabled: true
  n_models: 5
  base_seed: 42
  save_all_models: false  # Only save best
```

### Run Training

```bash
# Same SLURM script, just use train_ensemble.py instead
sbatch scripts/run_training.sh config.yaml  # If run_training.sh is updated

# Or directly:
python scripts/train_ensemble.py config.yaml
```

### Backward Compatibility

- If `ensemble.enabled: false` (default), `train_ensemble.py` behaves like `train.py`
- Original `train.py` is unchanged and can still be used

---

## Future Enhancements (Phase 2)

### Parallel GPU Training

Split GPUs across ensemble members:
- 8 GPUs, 4 models → 2 GPUs per model
- Requires modified distributed initialization
- Significant speedup for large ensembles

### Uncertainty Quantification at Inference

Use all ensemble models for prediction:
```python
predictions = [model_i(x) for model_i in ensemble]
mean_pred = np.mean(predictions, axis=0)
uncertainty = np.std(predictions, axis=0)
```

### Early Stopping Across Ensemble

Stop training models that are clearly underperforming:
- Monitor validation loss across all models
- Stop a model if it's > 2σ worse than the best

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `config_template.yaml` | Modify | Add `ensemble` section |
| `config/manager.py` | Modify | Add `get_ensemble_config()` method |
| `scripts/train_ensemble.py` | Create | New ensemble training script |
| `ENSEMBLE_TRAINING_PLAN.md` | Create | This document |

---

## Testing Plan

1. **Unit test**: Verify config parsing for ensemble settings
2. **Integration test**: Run with `n_models=2` to verify flow
3. **Full test**: Run with `n_models=5` overnight

---

## Estimated Time

- Config changes: ~10 min
- train_ensemble.py: ~30 min
- Testing: ~1 hour (mostly waiting for training)

Total implementation: ~45 min of active work
