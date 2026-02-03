# Code Quality Refactoring Plan

**Date:** 2026-01-26
**Status:** 🟡 PLANNING - Awaiting approval
**Previous Work:** Phases 1-5 complete, LBFGS prior pre-training fixed

---

## Issues to Address

### Priority 1: Critical (Breaking Changes)

1. **Remove `system.box` from config** - Unused and confusing
2. **Fix fragile `_chains[0]` data access** - Internal API dependency in LBFGS pretrain
3. **Add proper logging framework** - Replace inconsistent print statements
4. **Fix `pretrain_prior()` signature** - Remove misleading unused parameters
5. **Standardize path handling** - Accept `Union[str, Path]`, use `Path` internally
6. **Add TypedDict for return types** - Better type safety for complex dicts
7. **Add JAX typing** - Use `jax.Array` instead of deprecated `jnp.ndarray`

### Priority 2: Important (Feature Additions)

8. **Add resume training feature** - Continue from checkpoint when training incomplete
9. **Add preprocessor params to config** - Make `buffer_multiplier`, `park_multiplier`, `min_steps` configurable

### Priority 3: Documentation Only

10. **Document parameter ownership** - Clarify who owns params (model vs trainer)
11. **Document box handling** - Explain computed vs config box
12. **Document error message patterns** - Note for future improvement

---

## Current State Analysis

### Logging Patterns
- **Current:** All `print()` statements with various prefixes: `[Split]`, `[LBFGS]`, `[Model]`, etc.
- **Files affected:** 12 files use print statements
- **Pattern:** Prefix-based categorization already exists

### Path Handling
- **Current:** Mix of `str` and `Path` objects
- **Inconsistencies:**
  - `config.get_data_path()` returns `str`
  - Scripts convert to `Path` for operations
  - Some functions accept only `str`, others only `Path`
- **Files affected:** 12 files handle paths

### Type Hints
- **Current:** Extensive use (120+ occurrences)
- **Issues:**
  - Uses `jnp.ndarray` (deprecated in favor of `jax.Array`)
  - Return types use `Dict[str, Any]` instead of TypedDict
  - No consistent type aliases
- **Files affected:** All model, training, and data files

### Config Structure
- **Current:** Nested dict with `.get(*keys, default)` pattern
- **Extension approach:** Add new keys without breaking old configs
- **Files affected:** 1 file (config/manager.py)

---

## Implementation Plan

### Issue 1: Remove `system.box` from Config

**Current behavior:**
- `config.get_box()` reads `system.box` from YAML
- Never actually used (box is computed from data)
- Confusing to users

**Solution:**
1. Remove `get_box()` method from `ConfigManager`
2. Remove `system:` section from `config_template.yaml`
3. Add migration note to docs

**Files to modify:**
- `config/manager.py` - Remove `get_box()` method
- `config_template.yaml` - Remove `system:` section
- `BOX_HANDLING.md` - New doc explaining box computation

**Testing:** Verify existing configs still load (box section ignored if present)

**Backwards compatibility:** Config files with `system.box` will still load (section just ignored)

---

### Issue 2: Fix Fragile `_chains[0]` Data Access

**Current behavior:**
```python
# trainer.py line 247 - FRAGILE!
train_data = {
    "R": jnp.asarray(self.train_loader._chains[0]["data"]["R"]),
    "F": jnp.asarray(self.train_loader._chains[0]["data"]["F"]),
    "mask": jnp.asarray(self.train_loader._chains[0]["data"]["mask"]),
}
```

**Problem:** Accesses internal `NumpyDataLoader` structure (`_chains[0]`)

**Solution:** Store full dataset in Trainer during initialization

**Implementation:**
```python
class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        # ... existing code ...

        # Store training dataset for LBFGS pretrain
        self._train_dataset = {
            "R": jnp.asarray(train_loader._chains[0]["data"]["R"]),
            "F": jnp.asarray(train_loader._chains[0]["data"]["F"]),
            "mask": jnp.asarray(train_loader._chains[0]["data"]["mask"]),
        }

    def pretrain_prior(self, max_steps=200, tol_grad=1e-6):
        # Use stored dataset instead of accessing _chains
        train_data = self._train_dataset
        # ... rest of method
```

**Files to modify:**
- `training/trainer.py` - Add `_train_dataset` to `__init__`, use in `pretrain_prior()`

**Testing:** Run LBFGS pretrain with small dataset

**Risk:** Low - encapsulates internal access to one location

---

### Issue 3: Fix `pretrain_prior()` Signature

**Current behavior:**
```python
def pretrain_prior(
    self,
    epochs: int = None,           # UNUSED!
    optimizer_name: str = None,   # UNUSED!
    max_steps: int = 200,
    min_steps: int = 10,
    tol_grad: float = 1e-6
) -> Dict[str, Any]:
```

**Problem:** `epochs` and `optimizer_name` are never used (LBFGS hardcoded)

**Solution:** Remove unused parameters

**Implementation:**
```python
def pretrain_prior(
    self,
    max_steps: int = None,
    tol_grad: float = None
) -> Dict[str, Any]:
    """Pre-train prior using LBFGS force matching.

    Args:
        max_steps: Maximum LBFGS iterations (default from config)
        tol_grad: Gradient tolerance for convergence (default from config)

    Returns:
        Dictionary with keys: train_loss, steps, converged, fitted_params
    """
    # Read defaults from config
    if max_steps is None:
        max_steps = self.config.get_pretrain_prior_max_steps()
    if tol_grad is None:
        tol_grad = self.config.get_pretrain_prior_tol_grad()

    # ... rest of method
```

**Files to modify:**
- `training/trainer.py` - Update signature and docstring

**Testing:** Call with and without parameters

**Backwards compatibility:** Breaking change - but method is new (just added), so acceptable

---

### Issue 4: Add Proper Logging Framework

**Current behavior:**
- All logging done via `print()` with manual prefixes
- No control over verbosity
- No log levels

**Solution:** Use Python `logging` module with consistent patterns

**Implementation approach:**

1. **Create logging utility:**
```python
# utils/logging.py
import logging
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Format: [Module] Message
    formatter = logging.Formatter('[%(name)s] %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

# Module-level loggers
data_logger = setup_logger("Data")
model_logger = setup_logger("Model")
training_logger = setup_logger("Training")
export_logger = setup_logger("Export")
```

2. **Replace print statements systematically:**

**Pattern mapping:**
```python
# Before:
print("[Data] Loading dataset...")
print(f"[Model] Initialized: {model}")
print(f"[LBFGS] Completed: {steps} steps")

# After:
data_logger.info("Loading dataset...")
model_logger.info(f"Initialized: {model}")
training_logger.info(f"[LBFGS] Completed: {steps} steps")
```

3. **Add log levels:**
```python
# INFO - Normal progress messages
logger.info("Starting training...")

# WARNING - Non-fatal issues
logger.warning("Validation set too small, using training data")

# ERROR - Errors that don't stop execution
logger.error("Failed to save checkpoint")

# DEBUG - Verbose debugging info (disabled by default)
logger.debug(f"Batch shape: {batch.shape}")
```

**Files to modify:**
- `utils/__init__.py` - New file
- `utils/logging.py` - New file with logger setup
- `data/loader.py` - Replace 6 print statements
- `data/preprocessor.py` - Replace 4 print statements
- `models/combined_model.py` - Replace 3 print statements
- `training/trainer.py` - Replace 25+ print statements
- `export/exporter.py` - Replace 2 print statements
- `scripts/train.py` - Replace 20+ print statements
- `scripts/evaluate.py` - Replace 10+ print statements

**Testing:** Run training and verify log output matches old behavior

**Risk:** Medium - affects all modules, but purely cosmetic change

---

### Issue 5: Standardize Path Handling

**Current behavior:**
- Some functions accept `str`, others `Path`
- Config returns `str`, scripts convert to `Path`
- Inconsistent internal usage

**Solution:**
1. Define `PathLike = Union[str, Path]` type alias
2. All public APIs accept `PathLike`
3. Convert to `Path` at API boundary
4. Use `Path` for all internal operations

**Implementation:**

```python
# config/types.py (new file)
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

def as_path(p: PathLike) -> Path:
    """Convert str or Path to Path object."""
    return Path(p) if not isinstance(p, Path) else p
```

**Update all path-accepting functions:**

```python
# Before:
def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        ...

# After:
def load_npz(path: PathLike) -> Dict[str, np.ndarray]:
    path = as_path(path)
    with np.load(path) as data:
        ...
```

**Files to modify:**
- `config/types.py` - New file with PathLike definition
- `config/manager.py` - Update return types (still return str for backwards compat)
- `data/loader.py` - Accept PathLike
- `export/exporter.py` - Accept PathLike
- `evaluation/visualizer.py` - Accept PathLike
- All scripts - Use PathLike in type hints

**Testing:** Pass both str and Path objects to all functions

**Backwards compatibility:** Full - accepting Union[str, Path] is strictly more permissive

---

### Issue 6: Add TypedDict for Return Types

**Current behavior:**
```python
def pretrain_prior(...) -> Dict[str, Any]:
    return {
        "train_loss": ...,
        "steps": ...,
        "converged": ...,
        "fitted_params": ...
    }
```

**Problem:** No type safety, keys/types not documented

**Solution:** Define TypedDict classes for all complex return types

**Implementation:**

```python
# config/types.py
from typing import TypedDict, Dict
import jax.numpy as jnp

class PretrainResult(TypedDict):
    """Result from prior pre-training."""
    train_loss: float
    steps: int
    converged: bool
    fitted_params: Dict[str, jnp.ndarray]

class StageResult(TypedDict):
    """Result from a training stage."""
    train_loss: float
    val_loss: float
    best_epoch: int

class TrainingResults(TypedDict):
    """Results from full training pipeline."""
    prior_pretrain: PretrainResult | None
    stage1: StageResult
    stage2: StageResult | None

class EnergyComponents(TypedDict):
    """Energy component breakdown."""
    E_total: float
    E_prior: float
    E_allegro: float
    # Prior components
    E_bond: float
    E_angle: float
    E_rep: float
    E_dih: float

class EvaluationMetrics(TypedDict):
    """Single-frame evaluation metrics."""
    energy: float
    force_rmse: float
    force_mae: float
    components: EnergyComponents
```

**Update method signatures:**

```python
# Before:
def pretrain_prior(...) -> Dict[str, Any]:

# After:
def pretrain_prior(...) -> PretrainResult:

# Before:
def train_full_pipeline(...) -> Dict[str, Any]:

# After:
def train_full_pipeline(...) -> TrainingResults:
```

**Files to modify:**
- `config/types.py` - Add TypedDict definitions
- `training/trainer.py` - Update return types
- `models/combined_model.py` - Update `compute_components()` return type
- `evaluation/evaluator.py` - Update return types

**Testing:** Run mypy type checking

**Backwards compatibility:** Full - TypedDict is compatible with Dict at runtime

---

### Issue 7: Add JAX Typing

**Current behavior:**
- Type hints use `jnp.ndarray` (deprecated)
- JAX recommends `jax.Array` (or `jaxtyping` for shapes)

**Solution:**
1. Replace all `jnp.ndarray` with `jax.Array`
2. Optional: Use `jaxtyping` for shape annotations

**Implementation:**

```python
# Before:
def compute_energy(self, R: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:

# After (simple):
import jax
def compute_energy(self, R: jax.Array, mask: jax.Array) -> jax.Array:

# After (with jaxtyping - optional):
from jaxtyping import Array, Float
def compute_energy(
    self,
    R: Float[Array, "n_atoms 3"],
    mask: Float[Array, "n_atoms"]
) -> Float[Array, ""]:
```

**Decision:** Start with simple `jax.Array`, add `jaxtyping` later if desired

**Files to modify:**
- All files with type hints (12 files)
- Simple find-replace: `jnp.ndarray` → `jax.Array`

**Testing:** Import all modules, verify no type errors

**Backwards compatibility:** Full - pure type hint change, no runtime effect

---

### Issue 8: Add Resume Training Feature

**Current behavior:**
- Training always starts from scratch
- If `epochs_adabelief` too low, must re-train from beginning
- Checkpoints saved but not used for resuming

**Solution:** Add checkpoint loading and resume capability

**Use case:**
```bash
# First run (discovers 50 epochs not enough)
python train.py config.yaml

# Resume with more epochs (update config, then resume)
python train.py config.yaml --resume checkpoints/latest.pkl
```

**Implementation:**

1. **Add checkpoint saving in Trainer:**

```python
class Trainer:
    def save_checkpoint(self, path: PathLike, stage: str, epoch: int):
        """Save full training state."""
        checkpoint = {
            "model_params": self.model.params,
            "stage": stage,
            "epoch": epoch,
            "config": self.config.to_dict(),
            "best_params": self.best_params,
            "best_loss": self.best_loss,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: PathLike) -> Dict[str, Any]:
        """Load training state from checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Restore state
        self.model.params = checkpoint["model_params"]
        self.best_params = checkpoint.get("best_params")
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        return checkpoint
```

2. **Add resume logic to training pipeline:**

```python
def train_full_pipeline(self, resume_from: PathLike | None = None) -> TrainingResults:
    """Run full training pipeline with optional resume."""

    start_stage = "prior_pretrain"
    start_epoch = 0
    results = {}

    # Load checkpoint if resuming
    if resume_from is not None:
        checkpoint = self.load_checkpoint(resume_from)
        start_stage = checkpoint["stage"]
        start_epoch = checkpoint["epoch"]
        logger.info(f"Resuming from {start_stage} epoch {start_epoch}")

    # Run pipeline from checkpoint
    if start_stage == "prior_pretrain":
        if self.config.get_pretrain_prior():
            results["prior_pretrain"] = self.pretrain_prior()
        start_stage = "stage1"

    if start_stage == "stage1":
        results["stage1"] = self._train_stage(
            "adabelief",
            start_epoch=start_epoch if start_stage == "stage1" else 0
        )
        start_stage = "stage2"
        start_epoch = 0

    if start_stage == "stage2":
        if self.config.get_epochs_yogi() > 0:
            results["stage2"] = self._train_stage(
                "yogi",
                start_epoch=start_epoch if start_stage == "stage2" else 0
            )

    return results
```

3. **Update scripts to accept --resume flag:**

```python
# scripts/train.py
def main(config_file: str, job_id: str = None, resume_from: str = None):
    # ... existing setup ...

    trainer = Trainer(model, config, train_loader, val_loader)

    # Run with resume
    results = trainer.train_full_pipeline(resume_from=resume_from)

    # ... rest of script ...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint")
    args = parser.parse_args()

    main(args.config_file, args.job_id, args.resume)
```

**Files to modify:**
- `training/trainer.py` - Add save/load checkpoint, update train_full_pipeline
- `scripts/train.py` - Add --resume argument

**Testing:**
1. Train for 5 epochs
2. Update config to 10 epochs
3. Resume from checkpoint
4. Verify starts at epoch 6

**Risk:** Medium - touches training loop, but well-isolated feature

---

### Issue 9: Add Preprocessor Params to Config

**Current behavior:**
```python
# data/preprocessor.py - HARDCODED
buffer_multiplier: float = 1.2
park_multiplier: float = 10.0
min_steps: int = 10  # in pretrain_prior
```

**Solution:** Add to config with backward-compatible defaults

**Implementation:**

1. **Update ConfigManager:**

```python
# config/manager.py
def get_preprocessing_buffer_multiplier(self) -> float:
    """Get box buffer multiplier for preprocessing."""
    return self.get("preprocessing", "buffer_multiplier", default=1.2)

def get_preprocessing_park_multiplier(self) -> float:
    """Get parking multiplier for preprocessing."""
    return self.get("preprocessing", "park_multiplier", default=10.0)

def get_pretrain_prior_min_steps(self) -> int:
    """Get minimum LBFGS steps before convergence check."""
    return self.get("training", "pretrain_prior_min_steps", default=10)
```

2. **Update config template:**

```yaml
# config_template.yaml
preprocessing:
  buffer_multiplier: 1.2  # Box extent buffer (R_max * multiplier)
  park_multiplier: 10.0   # Parking distance for masked atoms

training:
  pretrain_prior_min_steps: 10     # Minimum LBFGS steps
  pretrain_prior_max_steps: 200    # Maximum LBFGS iterations
  pretrain_prior_tol_grad: 1.0e-6  # Convergence threshold
```

3. **Update preprocessor to read from config:**

```python
# data/preprocessor.py
class CoordinatePreprocessor:
    def __init__(self, cutoff: float, config: ConfigManager):
        self.cutoff = cutoff
        self.buffer_multiplier = config.get_preprocessing_buffer_multiplier()
        self.park_multiplier = config.get_preprocessing_park_multiplier()

    # Use self.buffer_multiplier instead of hardcoded 1.2
    # Use self.park_multiplier instead of hardcoded 10.0
```

4. **Update Trainer to read min_steps from config:**

```python
# training/trainer.py
def pretrain_prior(self, max_steps=None, tol_grad=None):
    if max_steps is None:
        max_steps = self.config.get_pretrain_prior_max_steps()
    if tol_grad is None:
        tol_grad = self.config.get_pretrain_prior_tol_grad()

    min_steps = self.config.get_pretrain_prior_min_steps()  # NEW

    # ... use min_steps in cond_fn
```

**Files to modify:**
- `config/manager.py` - Add 3 new getter methods
- `config_template.yaml` - Add preprocessing section
- `data/preprocessor.py` - Accept config, use config values
- `training/trainer.py` - Read min_steps from config
- `scripts/train.py` - Pass config to CoordinatePreprocessor

**Testing:** Verify defaults match hardcoded values, try custom values

**Backwards compatibility:** Full - new config keys have defaults matching old hardcoded values

---

### Issue 10-12: Documentation

**Create new documentation files:**

1. **BOX_HANDLING.md** - Explain box computation vs config

```markdown
# Box Handling in Chemtrain

## Overview
The simulation box is **computed from data**, not read from config.

## Why?
- Ensures box fits all structures in dataset
- Prevents atoms outside box boundaries
- Automatically adapts to different systems

## How It Works
1. **Compute extent**: Find max coordinate range across all frames
2. **Add buffer**: `box = extent * buffer_multiplier` (default 1.2)
3. **Center coordinates**: Shift all atoms to box center
4. **Park masked atoms**: Move padded atoms to `box * park_multiplier`

## Config (Old - Removed)
Previously `system.box` existed in config but was never used.
This has been removed to avoid confusion.

## Customization
To adjust box behavior, modify:
- `preprocessing.buffer_multiplier` - How much padding around atoms
- `preprocessing.park_multiplier` - Where to place masked atoms
```

2. **PARAMETER_OWNERSHIP.md** - Clarify parameter lifecycle

```markdown
# Parameter Ownership in Chemtrain

## Parameter Lifecycle

### Initialization
- `CombinedModel.__init__()` creates initial parameters
- Both prior and Allegro params initialized
- Prior params from config, Allegro params random

### During Training
- `Trainer` owns current parameters
- Stored in `self.model.params`
- Updated each training step
- Best params stored in `self.best_params`

### After Training
- `Trainer.get_best_params()` returns best params
- These are passed to exporter
- Also saved to pickle file

## Who Owns What?

| Component | Owns Parameters? | Modifies Parameters? |
|-----------|------------------|---------------------|
| `CombinedModel` | ✅ Yes (`self.params`) | ❌ No (stateless compute) |
| `PriorEnergy` | ✅ Yes (`self.params`) | ❌ No (stateless compute) |
| `AllegroModel` | ✅ Yes (`self.params`) | ❌ No (stateless compute) |
| `Trainer` | ✅ Yes (`self.best_params`) | ✅ Yes (via optimizers) |
| `Evaluator` | ❌ No (accepts params) | ❌ No (stateless) |
| `Exporter` | ❌ No (accepts params) | ❌ No (stateless) |

## Best Practices
- Models are **stateless** - always pass params explicitly
- Trainer **owns** the training state
- Use `get_best_params()` to retrieve final params
```

3. **ERROR_MESSAGES.md** - Document patterns for future improvement

```markdown
# Error Message Patterns

## Current State
Error messages are generally good but could be more consistent.

## Patterns to Improve (Future Work)

### Path Errors
```python
# Current:
raise FileNotFoundError(f"NPZ file not found: {path}")

# Better (future):
raise FileNotFoundError(
    f"Dataset file not found: {path}\n"
    f"Searched in: {search_paths}\n"
    f"Hint: Check data.path in config.yaml"
)
```

### Shape Mismatches
```python
# Current:
# (often just cryptic JAX error)

# Better (future):
raise ValueError(
    f"Shape mismatch: expected R.shape[1] == 3 (xyz), got {R.shape[1]}\n"
    f"Full shape: {R.shape}"
)
```

### Config Errors
```python
# Current:
# KeyError with no context

# Better (future):
raise ValueError(
    f"Missing required config key: {key_path}\n"
    f"Add to config.yaml:\n"
    f"  {suggested_yaml}"
)
```

## Decision
Defer to future work - requires systematic review of all error paths.
```

**Files to create:**
- `BOX_HANDLING.md`
- `PARAMETER_OWNERSHIP.md`
- `ERROR_MESSAGES.md`

---

## Implementation Order

### Phase 1: Simple Fixes (Low Risk)
**Time estimate:** Quick fixes
**Files:** 4 files

1. Remove `system.box` from config (Issue 1)
2. Fix `pretrain_prior()` signature (Issue 3)
3. Add JAX typing - simple replace (Issue 7)
4. Create documentation files (Issues 10-12)

**Testing:** Import all modules, run type checker

---

### Phase 2: Logging Framework (Medium Risk)
**Time estimate:** Systematic replacement
**Files:** 10 files

1. Create `utils/logging.py` (Issue 4)
2. Replace print statements file by file
3. Test each file after replacement

**Testing:** Run full training, compare output with old logs

---

### Phase 3: Type Safety (Low-Medium Risk)
**Time estimate:** Definition and updates
**Files:** 6 files

1. Create `config/types.py` with PathLike (Issue 5)
2. Add TypedDict definitions (Issue 6)
3. Update function signatures to accept PathLike
4. Update return types to use TypedDict

**Testing:** Run mypy, test with both str and Path inputs

---

### Phase 4: New Functionality (Medium Risk)
**Time estimate:** Feature implementation
**Files:** 5 files

1. Fix `_chains[0]` access by storing dataset (Issue 2)
2. Add resume training feature (Issue 8)
3. Add preprocessor params to config (Issue 9)

**Testing:**
- Run LBFGS pretrain
- Test resume from checkpoint
- Test custom preprocessing params

---

## Testing Strategy

### After Each Phase:
1. **Import check:** All modules import without errors
2. **Type check:** Run `mypy clean_code_base/`
3. **Smoke test:** Initialize model, run 1 training step
4. **Integration test:** Run full pipeline on small dataset

### Final Validation:
1. Run training with 1000 frames
2. Enable LBFGS pretrain
3. Test resume from checkpoint
4. Verify all outputs (MLIR, plots, params)
5. Compare results with pre-refactoring run

### Backwards Compatibility:
- Old config files must still work
- All API changes must be additive (accept more, not less)
- Default behavior must match old behavior

---

## Risk Assessment

### Low Risk:
- JAX typing (pure type hint change)
- Documentation files (no code changes)
- Fix pretrain signature (new method, safe to change)
- Remove system.box (unused code removal)
- TypedDict (runtime compatible with Dict)

### Medium Risk:
- Logging framework (affects all output)
- Path handling (affects all file I/O)
- Fix _chains[0] access (changes data flow)
- Resume feature (touches training loop)
- Preprocessor config (changes initialization)

### Mitigation:
- Test after each file modification
- Keep old code in comments during transition
- Run side-by-side comparison with old version
- User tests on real datasets after implementation

---

## Success Criteria

- [✓] All 12 issues addressed
- [ ] No print() statements remain (except scripts for user output)
- [ ] All paths use PathLike type
- [ ] All return types use TypedDict
- [ ] All arrays use jax.Array type
- [ ] Resume training works
- [ ] Config has preprocessing section
- [ ] Documentation files complete
- [ ] All tests pass
- [ ] Results match pre-refactoring run
- [ ] CODE_QUALITY_REFACTORING.md kept up to date

---

## Files Requiring Changes

### New Files (5):
1. `utils/__init__.py`
2. `utils/logging.py`
3. `config/types.py`
4. `BOX_HANDLING.md`
5. `PARAMETER_OWNERSHIP.md`
6. `ERROR_MESSAGES.md`

### Modified Files (15):
1. `config/manager.py` - Remove get_box(), add preprocessing getters
2. `config_template.yaml` - Remove system, add preprocessing
3. `data/loader.py` - Logging, PathLike
4. `data/preprocessor.py` - Logging, read from config
5. `models/topology.py` - Logging, jax.Array
6. `models/prior_energy.py` - Logging, jax.Array, TypedDict
7. `models/allegro_model.py` - Logging, jax.Array
8. `models/combined_model.py` - Logging, jax.Array, TypedDict
9. `training/trainer.py` - Logging, fix signature, store dataset, resume, TypedDict
10. `export/exporter.py` - Logging, PathLike
11. `evaluation/evaluator.py` - Logging, TypedDict
12. `evaluation/visualizer.py` - Logging, PathLike
13. `scripts/train.py` - Logging, add --resume flag
14. `scripts/evaluate.py` - Logging
15. All `__init__.py` files - Export new types

---

**Next Step:** Review and approve this plan, then begin Phase 1 implementation.
