# Implementation Plan: Spline-Based Priors + Train Priors via Force Matching

## Status of Existing Work

The `train_priors` feature is **partially implemented** (changes already in the codebase):
- `config/manager.py`: `train_priors_enabled()` reads `model.train_priors`
- `config_template.yaml`: `train_priors: true` already present
- `prior_energy.py`: `params` argument threaded through all compute methods
- `combined_model.py`: `self.train_priors` flag, branching in `compute_energy()` and `compute_components()`
- `trainer.py`: sync-back of `self.model.prior.params = self.params["prior"]` after training

**Remaining train_priors work:** Update `compute_force_components()` in `combined_model.py` to pass `params['prior']` through when `train_priors` is enabled (currently only `compute_energy` and `compute_components` branch on this flag).

---

## Compatibility Rules

| Combination | Behavior |
|---|---|
| Parametric priors, `train_priors: false` | Default. Frozen priors from config/YAML. |
| Parametric priors, `train_priors: true` | Prior params differentiated during force matching. |
| Parametric priors, `pretrain_prior: true` | LBFGS warm-start of prior params. |
| **Spline priors**, `train_priors: true` | **Incompatible.** Guard forces `train_priors=false`, logs warning. |
| **Spline priors**, `pretrain_prior: true` | **Incompatible.** Guard forces `pretrain_prior=false`, logs warning. |
| Spline priors, both false | Normal spline prior operation. |

Guards are placed in `scripts/train.py` after model initialization.

---

## Step 1: Create `models/spline_eval.py` (new file)

Pure-JAX cubic spline evaluator. No scipy at runtime.

### Functions:

```python
def evaluate_cubic_spline(x, knots, coeffs) -> Array:
    """Evaluate cubic spline. x clamped to [knots[0], knots[-1]].
    Uses jnp.searchsorted for interval lookup (JIT-compatible)."""

def evaluate_cubic_spline_periodic(x, knots, coeffs, period=2*pi) -> Array:
    """Wraps x into [knots[0], knots[0]+period) then evaluates."""

def evaluate_cubic_spline_by_type(x, species, all_knots, all_coeffs,
                                   type_mask, global_knots, global_coeffs) -> Array:
    """Per-species spline with fallback to global where type_mask==0."""
```

Implementation details:
- `x_clamped = jnp.clip(x, knots[0], knots[-1] - eps)`
- `i = jnp.searchsorted(knots, x_clamped, side='right') - 1`; clip to `[0, N-2]`
- `dx = x_clamped - knots[i]`
- `U = c0 + c1*dx + c2*dx^2 + c3*dx^3`
- For type-indexed: `U = jnp.where(type_mask[species], U_typed, U_global)`

All functions are JIT-compilable and autodiff-compatible.

---

## Step 2: Modify `data_prep/prior_fitting_script.py`

Add a `--spline` flag. When set, run the KDE -> Boltzmann inversion -> CubicSpline pipeline **in addition to** the existing parametric fitting. Existing parametric fitting is preserved and always runs (the user might want both outputs).

### New CLI arguments:
- `--spline` (flag): Enable spline fitting, output `.npz`
- `--spline_out` (str, default `"fitted_priors_spline.npz"`): Output NPZ path
- `--residue_specific_angles` (flag): Per-AA angle typing
- `--angle_min_samples` (int, default 500): Min samples per AA type before fallback
- `--kde_bandwidth_factor` (float, default 1.0): Multiplier on Silverman bandwidth
- `--spline_grid_points` (int, default 500): Grid points for spline fitting

### New functions:

**`fit_bond_spline(all_bonds, T, kB, n_grid, bw_factor)`**
1. `scipy.stats.gaussian_kde(all_bonds, bw_method='silverman' * bw_factor)`
2. Evaluate on regular grid spanning `[percentile_1, percentile_99]` with padding
3. Boltzmann inversion: `U = -kT * ln(max(P, floor))`, shift so `min(U)=0`
4. `scipy.interpolate.CubicSpline(grid, U, bc_type='natural')`
5. Return `knots, coeffs` (extract `.x` and polynomial coefficients from CubicSpline)

**`fit_angle_spline(all_angles, T, kB, n_grid, bw_factor)`**
1. KDE with `weights=1/sin(theta)` for Jacobian correction
2. Evaluate on grid over `[theta_min, theta_max]` (avoid `sin(theta)~0` boundaries)
3. Same BI + CubicSpline as bond (natural BC)

**`fit_angle_spline_by_type(all_angles, all_central_species, ...)`**
1. Group angles by central-atom species
2. For each type with >= `min_samples`: fit its own spline on the same grid
3. Types below threshold: flag as fallback to global
4. Stack into `(n_types, N_grid)` knots and `(n_types, N_grid-1, 4)` coeffs arrays
5. Return `type_knots, type_coeffs, type_mask`

**`fit_dihedral_spline(all_dihedrals, T, kB, n_grid, bw_factor)`**
1. Periodic KDE: duplicate data at `phi-2pi` and `phi+2pi`, KDE on extended data, evaluate on `[-pi, pi]`
2. BI + CubicSpline with `bc_type='periodic'`

**`extract_spline_coeffs(cs)`**: Helper to extract knots and `(N-1, 4)` coefficient array from a scipy CubicSpline object. Scipy stores coefficients as `(4, N-1)` in descending power order; we transpose and reverse to get `(N-1, 4)` in ascending order `[c0, c1, c2, c3]`.

### Output NPZ format:
```
bond_knots, bond_coeffs,
angle_knots, angle_coeffs,
angle_n_types, angle_type_knots, angle_type_coeffs, angle_type_mask,  # if residue_specific
dih_knots, dih_coeffs,
temperature, kB, grid_points, kde_bandwidth_factor, residue_specific_angles
```

### Diagnostic plots (when `--spline` and not `--skip_plots`):
For each term: PMF overlay (histogram bins + KDE PMF + spline), implied distribution comparison, force (derivative) sanity check. For residue-specific angles: one subplot per type.

### Data requirements for species:
When `--residue_specific_angles` is used, the script needs species data per angle. The NPZ dataset contains `species` arrays. The script already loads positions — also load `species` and `mask`. For each angle triplet `(i, i+1, i+2)`, the central species is `species[frame, i+1]`. Filter out angles where any atom is masked (`mask==0`).

---

## Step 3: Modify `models/prior_energy.py`

### 3a. Add `species` argument

Thread `species` through the interface. Only `compute_angle_energy` actually uses it (for residue-specific type lookup). Other methods ignore it.

Signature changes:
- `compute_angle_energy(self, R, mask, species=None, params=None)` — **adds `species`**
- `compute_energy(self, R, mask, species=None, params=None)` — **adds `species`**, passes it to `compute_angle_energy`
- `compute_total_energy(self, R, mask, species=None, params=None)` — **adds `species`**, passes to `compute_energy`
- `compute_total_energy_from_params(self, params, R, mask, species=None)` — **adds `species`**, passes to `compute_total_energy`
- `compute_bond_energy`, `compute_repulsive_energy`, `compute_dihedral_energy`: NO change (don't need species)

Default `species=None` preserves backward compatibility for the parametric path.

### 3b. Constructor branching

```python
def __init__(self, config, topology, displacement):
    # ... existing topology and weight setup (unchanged) ...

    spline_path = config.get("model", "priors", "spline_file", default=None)

    if spline_path is not None:
        self.uses_splines = True
        spline_data = np.load(spline_path)

        self.bond_knots = jnp.asarray(spline_data["bond_knots"])
        self.bond_coeffs = jnp.asarray(spline_data["bond_coeffs"])
        self.angle_knots = jnp.asarray(spline_data["angle_knots"])
        self.angle_coeffs = jnp.asarray(spline_data["angle_coeffs"])
        self.dih_knots = jnp.asarray(spline_data["dih_knots"])
        self.dih_coeffs = jnp.asarray(spline_data["dih_coeffs"])

        self.residue_specific_angles = bool(spline_data.get("residue_specific_angles", False))
        if self.residue_specific_angles:
            self.angle_type_knots = jnp.asarray(spline_data["angle_type_knots"])
            self.angle_type_coeffs = jnp.asarray(spline_data["angle_type_coeffs"])
            self.angle_type_mask = jnp.asarray(spline_data["angle_type_mask"])

        # Only repulsive stays parametric
        prior_params = config.get_prior_params()
        self.params = {
            "epsilon": jnp.asarray(prior_params.get("epsilon", 1.0), dtype=jnp.float32),
            "sigma": jnp.asarray(prior_params.get("sigma", 3.0), dtype=jnp.float32),
        }
    else:
        self.uses_splines = False
        self.residue_specific_angles = False
        # ... existing parametric params loading (unchanged) ...
```

### 3c. Energy method branching

Each method gets an `if self.uses_splines:` branch. Example for bond:
```python
def compute_bond_energy(self, R, mask, params=None):
    p = params if params is not None else self.params
    # ... existing distance computation (unchanged) ...
    r = _safe_norm(dR)

    if self.uses_splines:
        U_bond = evaluate_cubic_spline(r, self.bond_knots, self.bond_coeffs)
    else:
        U_bond = (r - p["r0"]) ** 2
        U_bond = 0.5 * p["kr"] * ...  # existing

    E_bond = jnp.sum(jnp.where(bond_valid, U_bond, 0.0))
    return E_bond
```

Similarly for angle (uses `evaluate_cubic_spline_by_type` when `residue_specific_angles`), dihedral (uses `evaluate_cubic_spline_periodic`), repulsive (unchanged — always parametric).

---

## Step 4: Modify `models/combined_model.py`

### 4a. Pass `species` to prior methods

In `compute_energy()`, `compute_components()`, and `compute_force_components()`, pass the `species` argument through to prior energy calls:

```python
# In compute_energy:
if self.train_priors and "prior" in params:
    E_prior = self.prior.compute_total_energy(R_masked, mask, species=species, params=params["prior"])
else:
    E_prior = self.prior.compute_total_energy(R_masked, mask, species=species)
```

Same pattern for `compute_components()` and `compute_force_components()`.

### 4b. Fix `compute_force_components()` for train_priors

Currently this method doesn't branch on `self.train_priors`. Update it to pass `params['prior']` when training priors, consistent with `compute_energy()` and `compute_components()`.

---

## Step 5: Config changes

### `config/manager.py` — add methods:
```python
def get_spline_file_path(self) -> Optional[str]:
    """Get path to spline prior NPZ file (None = use parametric)."""
    return self.get("model", "priors", "spline_file", default=None)

def uses_spline_priors(self) -> bool:
    """Check if spline-based priors are configured."""
    return self.get_spline_file_path() is not None

def get_residue_specific_angles(self) -> bool:
    """Check if residue-specific angle priors are enabled."""
    return self.get("model", "priors", "residue_specific_angles", default=False)
```

### `config_template.yaml` — add under `model.priors`:
```yaml
priors:
    # Spline-based priors (optional, overrides parametric bond/angle/dihedral)
    # spline_file: "data_prep/datasets/fitted_priors_spline.npz"
    # residue_specific_angles: false

    weights: ...  # unchanged
    # Parametric params below still used if spline_file is absent
```

### `config/types.py` — add TypedDict:
```python
class SplineArrays(TypedDict, total=False):
    bond_knots: jax.Array
    bond_coeffs: jax.Array
    angle_knots: jax.Array
    angle_coeffs: jax.Array
    dih_knots: jax.Array
    dih_coeffs: jax.Array
    # Per-type angle arrays (if residue_specific_angles)
    angle_type_knots: Optional[jax.Array]
    angle_type_coeffs: Optional[jax.Array]
    angle_type_mask: Optional[jax.Array]
```

---

## Step 6: Compatibility guards in `scripts/train.py`

Add **after model initialization** (after `model = CombinedModel(...)`, around line 346):

```python
# === Compatibility guards ===
if model.use_priors and hasattr(model.prior, 'uses_splines') and model.prior.uses_splines:
    if config.pretrain_prior_enabled():
        training_logger.warning(
            "Spline priors detected with pretrain_prior=true. "
            "LBFGS pretraining is incompatible with spline priors — disabling."
        )
        config._config["training"]["pretrain_prior"] = False

    if config.train_priors_enabled():
        training_logger.warning(
            "Spline priors detected with train_priors=true. "
            "Training priors is incompatible with spline priors — disabling."
        )
        config._config["model"]["train_priors"] = False
        model.train_priors = False
```

This mutates the config in-memory only (the saved YAML copy is written earlier). The guard runs after data import and model creation, as requested.

---

## Step 7: Export verification

In `export/exporter.py`, the `from_combined_model` factory currently extracts `prior_params = params.get("prior", model.prior.params)`. With spline priors:
- `params["prior"]` only contains `{epsilon, sigma}`
- Spline arrays are instance attributes on `model.prior` (bond_knots, bond_coeffs, etc.)
- When the `default_apply_fn` closure calls `model.compute_total_energy(params_, R_, mask_, species_, neighbor)`, JAX traces through the prior computation and captures spline arrays as constants

**Verify:** The existing export path should work without modification because the energy function closure already captures all instance attributes. However, need to verify that:
1. Spline arrays are baked into the MLIR as constants
2. The exported model produces identical energies/forces to the Python model

No code changes expected, but verification is required.

---

## Execution Order

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `models/spline_eval.py` | CREATE | JAX cubic spline evaluator (3 functions) |
| 2 | `models/prior_energy.py` | MODIFY | Add `species` arg + spline branching + constructor |
| 3 | `models/combined_model.py` | MODIFY | Pass `species` to prior + fix `compute_force_components` |
| 4 | `config/manager.py` | MODIFY | Add spline config accessors |
| 5 | `config/types.py` | MODIFY | Add SplineArrays TypedDict |
| 6 | `config_template.yaml` | MODIFY | Add spline config section |
| 7 | `scripts/train.py` | MODIFY | Add compatibility guards |
| 8 | `data_prep/prior_fitting_script.py` | MODIFY | Add spline fitting pipeline |
| 9 | `export/exporter.py` | VERIFY | Confirm spline arrays export correctly |

Steps 1-7 deliver the runtime capability. Step 8 provides the data pipeline. Step 9 is verification.

---

## What Stays Unchanged

- All existing parametric prior code (harmonic bond, Fourier angle, periodic dihedral)
- All NaN-safety infrastructure (`_safe_norm`, `_safe_atan2`, `stop_gradient`)
- TopologyBuilder
- Repulsive energy (always parametric)
- LBFGS pretraining code (guarded, not deleted)
- chemtrain ForceMatching trainer internals
- Data loading, preprocessing, padding, parking
- Ensemble training
- Multi-node distributed training
