# MACE Model Integration Plan

> **Status**: Implemented
> **Last Updated**: 2026-02-09

---

## 1. Overview

Integrate the MACE (Multi Atomic Cluster Expansion) model from chemutils as an alternative ML backbone to Allegro within the cameo_cg framework. MACE is already implemented in JAX/Haiku within chemutils and follows the **exact same `(init_fn, apply_fn)` pattern** as Allegro, making this a relatively clean integration.

---

## 2. Why This Is Straightforward

Both `allegro_neighborlist_pp` and `mace_neighborlist_pp` in chemutils share:
- Same `@hk.without_apply_rng` + `@hk.transform` pattern
- Same function signature: `(displacement, r_cutoff, n_species, positions_test, neighbor_test, ...) -> (init_fn, apply_fn)`
- Same `init_fn(rng, R, nbrs, species)` and `apply_fn(params, R, nbrs, species, mask)` interfaces
- Same `AtomicEnergyLayer` for per-species energy offsets
- Same jax_md neighbor list handling (Dense/Sparse)
- Same energy output: scalar total energy (when `mode="energy"`)

**Key difference**: Allegro computes **per-edge** energies then aggregates via `segment_sum`, while MACE computes **per-node** energies directly. This difference is internal to the chemutils implementation and transparent to our wrapper.

---

## 3. Files to Create / Modify

### 3.1 New File: `models/mace_model.py`

Create a `MACEModel` class mirroring `AllegroModel`, but calling `mace_neighborlist_pp` instead of `allegro_neighborlist_pp`.

**Source to import from**: `chemutils.models.mace.model.mace_neighborlist_pp`

**Structure** (mirrors `allegro_model.py`):

```python
"""
MACE Equivariant Neural Network Model Wrapper

Wraps the MACE model initialization and inference for force field training.
Handles neighbor lists, species types, and coordinate masking.
"""

import jax
import jax.numpy as jnp
from jax_md import space, partition
from chemutils.models.mace.model import mace_neighborlist_pp
from typing import Optional, Tuple, Any

from utils.logging import model_logger


class MACEModel:
    """
    Wrapper for MACE equivariant graph neural network.

    Same interface as AllegroModel — can be used as a drop-in replacement
    in CombinedModel.
    """

    def __init__(self, config, R0, box, species, N_max):
        self.config = config
        self.N_max = N_max

        self.cutoff = config.get_cutoff()
        self.dr_threshold = config.get_dr_threshold()

        # Get MACE hyperparameters from config
        mace_size = config.get_mace_size()
        self.mace_config = config.get_mace_config(size=mace_size)

        model_logger.info(f"Using MACE size: {mace_size}")

        # Setup JAX-MD displacement and neighbor list (same as Allegro)
        self.displacement, self.shift = space.free()
        safe_box = jnp.asarray(box, dtype=jnp.float32)

        self.nneigh_fn = partition.neighbor_list(
            self.displacement,
            box=safe_box,
            r_cutoff=self.cutoff,
            dr_threshold=self.dr_threshold,
            fractional_coordinates=False
        )

        self.nbrs_init = self.nneigh_fn.allocate(R0, extra_capacity=64)

        self.n_species = int(jnp.max(species)) + 1
        species_safe = jnp.asarray(species, dtype=jnp.int32)

        model_logger.info(f"Detected {self.n_species} unique species")

        # Initialize MACE model via mace_neighborlist_pp
        self.init_mace, self.apply_mace = mace_neighborlist_pp(
            displacement=self.displacement,
            r_cutoff=self.cutoff,
            n_species=self.n_species,
            positions_test=R0,
            neighbor_test=self.nbrs_init,
            max_edge_multiplier=1.25,
            mode="energy",
            **self.mace_config
        )

        self._R0 = R0
        self._species0 = species_safe

    def initialize_params(self, rng_key):
        return self.init_mace(rng_key, self._R0, self.nbrs_init, self._species0)

    def get_neighborlist(self, R, nbrs=None):
        if nbrs is None:
            nbrs = self.nneigh_fn.allocate(R)
        nbrs = self.nneigh_fn.update(R, nbrs)
        return nbrs

    def compute_energy(self, params, R, mask, species, neighbor=None):
        mask_3d = mask[:, None]
        R_masked = jnp.where(mask_3d > 0, R, jax.lax.stop_gradient(R))

        if neighbor is None:
            nbrs = self.nneigh_fn.allocate(R_masked)
            nbrs = self.nneigh_fn.update(R_masked, nbrs)
        else:
            nbrs = neighbor

        species_masked = jnp.where(mask > 0, species, 0).astype(jnp.int32)
        E_mace = self.apply_mace(params, R_masked, nbrs, species_masked)
        return E_mace

    def compute_energy_and_forces(self, params, R, mask, species, neighbor=None):
        def energy_fn(R_):
            return self.compute_energy(params, R_, mask, species, neighbor)
        E = energy_fn(R)
        F = -jax.grad(energy_fn)(R)
        return E, F

    @property
    def initial_neighbors(self):
        return self.nbrs_init

    def __repr__(self):
        return f"MACEModel(cutoff={self.cutoff}, n_species={self.n_species}, N_max={self.N_max})"
```

### 3.2 Modify: `models/combined_model.py`

Generalize `CombinedModel` to accept either `AllegroModel` or `MACEModel` as the ML backbone.

**Changes**:

1. Add a `get_ml_model_type()` config accessor (returns `"allegro"` or `"mace"`).
2. In `__init__`, branch on model type:

```python
from .allegro_model import AllegroModel
from .mace_model import MACEModel

class CombinedModel:
    def __init__(self, config, R0, box, species, N_max):
        ...
        # Determine which ML backbone to use
        self.ml_model_type = config.get_ml_model_type()  # "allegro" or "mace"

        if self.ml_model_type == "mace":
            self.ml_model = MACEModel(config, R0, box, species, N_max)
        else:
            self.ml_model = AllegroModel(config, R0, box, species, N_max)

        # Keep backward-compatible alias
        self.allegro = self.ml_model
        ...
```

3. All downstream code that references `self.allegro` continues to work because both `AllegroModel` and `MACEModel` expose the same interface:
   - `initialize_params(rng_key)`
   - `compute_energy(params, R, mask, species, neighbor)`
   - `compute_energy_and_forces(params, R, mask, species, neighbor)`
   - `initial_neighbors` property
   - `displacement` property (inherited from jax_md setup)
   - `nneigh_fn` property
   - `cutoff` attribute

4. The `initialize_params` method needs to store ML params under a generic key:

```python
def initialize_params(self, rng_key):
    params = {
        'ml': self.ml_model.initialize_params(rng_key),
    }
    if self.use_priors:
        params['prior'] = self.prior.params
    return params
```

**IMPORTANT: Backward compatibility decision**

Option A (recommended): Keep `'allegro'` key in params dict for backward compat, regardless of model type:
```python
params = {'allegro': self.ml_model.initialize_params(rng_key)}
```
This means existing checkpoints and the exporter continue to work without changes.

Option B: Use generic `'ml'` key, but then must update:
- `trainer.py` (pretrain_prior references `self.params['prior']`)
- `exporter.py` (references `params['allegro']`)
- All existing checkpoints become incompatible

**Recommendation**: Use Option A for now. The param key `'allegro'` becomes a misnomer when using MACE, but it avoids a large refactoring effort. A later refactor can rename the key to `'ml'` across the entire codebase.

### 3.3 Modify: `models/__init__.py`

Add `MACEModel` to the public exports.

```python
from .mace_model import MACEModel
```

### 3.4 Modify: `config/manager.py`

Add methods for MACE configuration:

```python
# In ConfigManager:

def get_ml_model_type(self) -> str:
    """Get which ML model to use: 'allegro' or 'mace'."""
    return self.get("model", "ml_model", default="allegro")

def get_mace_size(self) -> str:
    """Get MACE model size variant."""
    return self.get("model", "mace_size", default="default")

def get_mace_config(self, size: str = "default") -> Dict[str, Any]:
    """Get MACE model configuration."""
    if size == "default":
        return self.get("model", "mace", default={})
    else:
        key = f"mace_{size}"
        return self.get("model", key, default=self.get("model", "mace", default={}))
```

### 3.5 New Config Section: MACE defaults in YAML

Add to `config_template.yaml` (and create a `config_mace.yaml` example):

```yaml
model:
  ml_model: "mace"  # Options: "allegro", "mace"

  # MACE model configurations
  mace_size: "default"

  mace:  # Default MACE config
    num_interactions: 2         # Number of message-passing layers
    hidden_irreps: "128x0e + 128x1o"  # Hidden representation irreps
    readout_irreps: "16x0e"    # MLP readout hidden irreps
    max_ell: 2                 # Max spherical harmonic degree
    n_radial_basis: 8          # Radial basis functions
    envelope_p: 6              # Envelope polynomial order
    correlation: 3             # Correlation order
    embed_dim: 32              # Species embedding dimension
    avg_num_neighbors: 21      # Average neighbors (estimate)

  mace_large:
    num_interactions: 3
    hidden_irreps: "256x0e + 256x1o"
    readout_irreps: "32x0e"
    max_ell: 3
    n_radial_basis: 12
    envelope_p: 6
    correlation: 3
    embed_dim: 64
    avg_num_neighbors: 21
```

**MACE default kwargs reference** (from `mace_default_kwargs` in `model.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 32 | Species embedding dimension |
| `input_irreps` | "1o" | Input vector irreps |
| `output_irreps` | "1x0e" | Output irreps (scalar energy) |
| `hidden_irreps` | "128x0e + 128x1o" | Hidden layer irreps |
| `readout_irreps` | "16x0e" | Readout MLP hidden irreps |
| `num_interactions` | 2 | Number of MACE layers |
| `mlp_n_hidden` | 64 | MLP hidden width |
| `mlp_n_layers` | 2 | MLP depth |
| `max_ell` | 2 | Max spherical harmonic degree |
| `avg_num_neighbors` | 1 | Average neighbor count |
| `n_radial_basis` | 8 | Number of radial basis functions |
| `envelope_p` | 6 | Envelope polynomial order |
| `correlation` | 3 | Correlation order |

### 3.6 Modify: `export/exporter.py`

The exporter currently hardcodes `allegro_model.apply_allegro`. Generalize:

```python
# In from_combined_model():
ml_model = model.ml_model  # Works for both AllegroModel and MACEModel

# Replace:
#   apply_model=allegro_model.apply_allegro,
# With:
#   apply_model=ml_model.apply_allegro if hasattr(ml_model, 'apply_allegro') else ml_model.apply_mace,
```

Or better: add a generic `apply_fn` property to both `AllegroModel` and `MACEModel`:

```python
# In AllegroModel:
@property
def apply_fn(self):
    return self.apply_allegro

# In MACEModel:
@property
def apply_fn(self):
    return self.apply_mace
```

Then the exporter uses `ml_model.apply_fn` uniformly.

### 3.7 No Changes Needed

These files need **no modification**:
- `training/trainer.py` — uses `model.energy_fn_template()` which is model-agnostic
- `training/optimizers.py` — optimizer factory is model-independent
- `scripts/train.py` — creates `CombinedModel` which handles the dispatch internally
- `scripts/train_ensemble.py` — same as above
- `evaluation/evaluator.py` — uses `model.compute_energy()` / `model.compute_components()`
- `data/loader.py` — model-independent
- `data/preprocessor.py` — model-independent
- `models/prior_energy.py` — independent of ML backbone
- `models/topology.py` — independent of ML backbone

---

## 4. Implementation Steps

### Step 1: Create `models/mace_model.py`
- Copy `allegro_model.py` as template
- Replace `allegro_neighborlist_pp` import with `mace_neighborlist_pp`
- Replace all `allegro` naming with `mace`
- Adjust kwargs mapping (MACE uses different hyperparameter names)

### Step 2: Add MACE config methods to `config/manager.py`
- `get_ml_model_type()` — new method
- `get_mace_size()` — new method
- `get_mace_config()` — new method

### Step 3: Modify `models/combined_model.py`
- Import `MACEModel`
- Branch on `config.get_ml_model_type()` to create the right ML model
- Keep `self.allegro` as an alias for backward compat
- Keep `'allegro'` as the param dict key for backward compat

### Step 4: Update `models/__init__.py`
- Export `MACEModel`

### Step 5: Update `export/exporter.py`
- Use generic `ml_model.apply_fn` instead of hardcoded `apply_allegro`
- Add `apply_fn` property to both model wrappers

### Step 6: Create example config
- `config_mace.yaml` — example MACE configuration

### Step 7: Test
- Run training with `ml_model: "mace"` on a small dataset
- Verify force evaluation produces reasonable results
- Verify export works

---

## 5. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MACE kwargs don't match chemutils defaults | Low | Map our config names to `mace_default_kwargs` keys |
| Different energy scale between models | Medium | Same `AtomicEnergyLayer` handles this, plus optimizer adapts |
| Neighbor list format mismatch | Low | Both use same jax_md Dense neighbor lists |
| Existing checkpoints break | Low | Keep `'allegro'` param key for backward compat |
| Import errors (missing MACE deps) | Low | MACE is in same chemutils package as Allegro |

---

## 6. MACE vs Allegro: Key Architectural Differences

| Aspect | Allegro | MACE |
|--------|---------|------|
| Energy computation | Per-edge → segment_sum → per-atom | Per-node directly |
| Message passing | Edge-based (pair) | Node-based with higher-order correlations |
| Equivariance | E(3) via edge features | E(3) via spherical harmonics + tensor products |
| Body order | 2-body (pair) | N-body via correlation parameter |
| Key hyperparameter | `num_layers` | `num_interactions` + `correlation` |
| Skip connections | Within layers | Species-dependent linear skip |
| Readout | Direct from edges | Linear (intermediate) + MLP (final layer) |

Both are well-suited for CG protein force fields.

---

## 7. Verification Checklist

- [ ] `MACEModel` can be instantiated with test config
- [ ] `MACEModel.initialize_params()` produces valid params
- [ ] `MACEModel.compute_energy()` returns scalar energy
- [ ] `CombinedModel` with `ml_model: "mace"` trains without errors
- [ ] `CombinedModel` with `ml_model: "allegro"` still works (regression test)
- [ ] Force evaluation produces physically reasonable forces
- [ ] MLIR export works with MACE model
- [ ] Loss converges during training (at least on single protein)

---

## 8. Implementation Notes (Post-Implementation)

### Files Created
- **`models/mace_model.py`** — `MACEModel` class mirroring `AllegroModel` interface
- **`config_mace.yaml`** — Example MACE configuration with all hyperparameters

### Files Modified
- **`models/combined_model.py`** — Dispatches between `AllegroModel` and `MACEModel` via `config.get_ml_model_type()`
- **`models/__init__.py`** — Added `MACEModel` to public exports
- **`config/manager.py`** — Added `get_ml_model_type()`, `get_mace_size()`, `get_mace_config()` methods
- **`export/exporter.py`** — Generalized to use `ml_model.model_apply_fn` instead of `allegro_model.apply_allegro`
- **`models/allegro_model.py`** — Added `model_apply_fn` property for exporter compatibility
- **`models/mace_model.py`** — Has `model_apply_fn` property returning raw Haiku apply function

### Key Design Decisions

1. **Backward compatibility via `'allegro'` key**: The params dict still uses `params['allegro']` as the key for ML model parameters, even when using MACE. This prevents breaking existing checkpoints, the exporter, and the trainer. `self.allegro` is an alias for `self.ml_model`.

2. **Generic `model_apply_fn` property**: Both `AllegroModel` and `MACEModel` expose a `model_apply_fn` property returning their raw Haiku apply function. The exporter uses this instead of hardcoded `apply_allegro`.

3. **Config-driven dispatch**: `model.ml_model` in YAML selects the backbone (`"allegro"` or `"mace"`). Default is `"allegro"` for full backward compat.

4. **MACE kwargs flow directly**: MACE hyperparameters in YAML (`mace:` section) are unpacked directly as `**kwargs` to `mace_neighborlist_pp()`, matching how Allegro config works.

5. **No changes needed to trainer/optimizer/data**: The `energy_fn_template()`, `ForceMatching` trainer, optimizer factory, and data loading are model-agnostic — they work unchanged with MACE.

### Files NOT Modified (model-agnostic)
- `training/trainer.py` — uses `model.energy_fn_template()` which is model-agnostic
- `training/optimizers.py` — optimizer factory is model-independent
- `scripts/train.py` — creates `CombinedModel` which handles dispatch internally
- `evaluation/evaluator.py` — uses `model.compute_energy()` / `model.compute_components()`
- `data/loader.py`, `data/preprocessor.py` — model-independent
- `models/prior_energy.py`, `models/topology.py` — independent of ML backbone
