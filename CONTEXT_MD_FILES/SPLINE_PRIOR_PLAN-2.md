# Plan: Spline-Based PMF Priors with Residue-Specific Angle Terms

> Status: Planning
> Date: 2026-02-11

---

## 1. Motivation

### 1.1 Problem with Current Parametric Priors

The current prior fitting pipeline performs Boltzmann inversion (BI) on bond/angle/dihedral histograms, then fits **parametric** functional forms to the resulting PMF:

- **Bond:** harmonic `½kr(r - r₀)²` — 2 parameters
- **Angle:** 10-term Fourier series `Σ[aₙcos(nθ) + bₙsin(nθ)]` — 20 parameters
- **Dihedral:** 2-term periodic cosine `Σ[kₙ(1 + cos(nφ - γₙ))]` — 4 parameters

These parametric forms are truncated approximations of the true PMF. The consequences:

**Dihedral (most severe):** The Cα dihedral PMF is multi-modal and asymmetric (reflecting Ramachandran basin structure projected onto the CG coordinate). Two Fourier terms cannot represent this. The prior injects a systematically wrong bias that Allegro must learn to correct — wasting model capacity and potentially creating artifacts in regions where training data is sparse.

**Angle:** 10 Fourier terms are better but the global-support basis means fitting one region perturbs all others. The weighted least-squares fit is sensitive to ridge regularization, pseudocount, and bin-count thresholds. It can ring or overshoot in sparsely populated regions.

**Bond:** The harmonic approximation is actually adequate (Cα–Cα bond length distributions are near-Gaussian). The change here is for consistency.

### 1.2 Proposed Replacement

Replace all parametric fitting with a non-parametric pipeline:

```
Raw samples → KDE smoothing → Boltzmann inversion → Cubic spline interpolation
```

This preserves the full PMF shape with no truncation bias. The cubic spline gives C² continuity (smooth energy, continuous forces, existing Hessian).

### 1.3 Residue-Specific Angle Terms

Currently all residues share one global angle prior U(θ). In reality, different amino acids have very different backbone flexibility — glycine is highly flexible, proline is rigid, bulky hydrophobic residues differ from small polar ones, etc. A single U(θ) averages over this variation, which either under-constrains flexible residues or over-constrains rigid ones.

The spline infrastructure makes residue-specific priors natural: instead of one spline, store one spline per residue type (or type-group), and select the appropriate one at evaluation time based on the central atom's species ID.

---

## 2. Design Decisions

### 2.1 Spline Representation

Each prior term is stored as a set of cubic spline coefficients:
- `knots`: array of shape `(N,)` — the grid points
- `coeffs`: array of shape `(N-1, 4)` — cubic polynomial coefficients per interval

Evaluation: given coordinate value `x`, find interval via `searchsorted`, compute `dx = x - knots[i]`, evaluate `c₀ + c₁·dx + c₂·dx² + c₃·dx³`.

**Boundary conditions:**
- Bond, angle: `natural` (second derivative = 0 at boundaries)
- Dihedral: `periodic` (value and derivatives match at ±π)

### 2.2 Species-Specific Typing for Angles

For the angle triplet at positions (i, i+1, i+2), the type is determined by the **central atom** species `species[i+1]`. Rationale:

- The backbone angle at a residue is primarily governed by that residue's identity (side-chain size, proline ring constraint, glycine flexibility).
- Central-atom typing gives up to 20 types (one per standard amino acid) — manageable both in data requirements and storage.
- Full triplet typing (species[i] × species[i+1] × species[i+2]) would yield up to 8000 types, most with insufficient samples.

**Fallback mechanism:** If a particular AA type has fewer than a configurable minimum number of angle samples (e.g., < 500), fall back to the global (all-type) spline for that type. This prevents noisy fits for rare residues.

### 2.3 What Happens to LBFGS Pretraining

Currently, LBFGS optimizes the parametric prior parameters (r0, kr, a, b, k_dih, γ_dih) to minimize force-matching error. With spline priors, the PMF is deterministic from the data — there are no free parameters to optimize.

**Decision: Guard LBFGS pretraining with a runtime check.** The existing config already supports `pretrain_prior: false`, which is the expected setting when using spline priors. The code change is a **runtime guard** in the training pipeline: if the `PriorEnergy` instance uses spline priors (detected by checking for the presence of a `spline_file` in config, or a flag on the `PriorEnergy` instance such as `self.uses_splines`), the LBFGS pretraining step is skipped with a log message, regardless of the `pretrain_prior` config value. This prevents accidental misconfiguration (spline priors + `pretrain_prior: true`) from attempting to optimize non-existent parametric parameters.

The parametric prior path (harmonic bond, Fourier angle, periodic dihedral) remains fully functional behind this check — no code is deleted. Users can still use the old parametric priors by omitting `spline_file` from the config and setting `pretrain_prior: true`.

No config file changes are needed for this — the guard is purely in the trainer code.

### 2.4 What About Bonds and Dihedrals — Residue-Specific?

**Bonds:** Cα–Cα bond lengths are nearly identical across all residue types (the peptide bond geometry is universal). Residue-specific bond priors would add complexity for negligible benefit. Keep as one global spline.

**Dihedrals:** Residue-specific dihedral priors are scientifically well-motivated (Ramachandran preferences differ dramatically by residue). However, a dihedral involves 4 residues, making the typing question harder. Central-pair typing (species[i+1], species[i+2]) gives up to 400 combinations — many with low counts. This is a natural **future extension** once the angle-specific infrastructure is validated, but is out of scope for this plan.

---

## 3. Implementation Plan

### 3.1 Overview of Changes

```
Files modified:
  data_prep/prior_fitting_script.py   — rewrite fitting logic
  models/prior_energy.py              — replace energy functions with spline eval
  models/combined_model.py            — pass species to prior
  config/manager.py                   — add spline config accessors
  config/types.py                     — add spline-related TypedDicts
  config_template.yaml                — new prior config section
  training/trainer.py                 — add LBFGS spline guard
  export/exporter.py                  — pass spline arrays in export

Files created:
  models/spline_eval.py               — JAX-compatible cubic spline evaluator
```

---

### Phase 1: Spline Fitting Script

**File:** `data_prep/prior_fitting_script.py`

**Replace** the existing `fit_bond_harmonic`, `fit_fourier_angles_stable`, and `fit_dihedral_fourier_2term` functions. Keep the data collection logic (extracting bond lengths, angles, dihedrals from trajectory data) unchanged.

#### 1a. Bond Prior

```
Input:  all_bonds — array of bond lengths from all frames/proteins
Output: bond_knots (N,), bond_coeffs (N-1, 4)
```

Steps:
1. Compute KDE of bond length samples using `scipy.stats.gaussian_kde` with default bandwidth (Silverman's rule). Expose bandwidth as a CLI argument for manual override.
2. Evaluate KDE on a regular grid of ~500 points spanning `[percentile_1, percentile_99]` of the data (with small padding).
3. Boltzmann inversion: `U(r) = -kT · ln(P_KDE(r))`. Apply a density floor at `1e-8 × max(P)` before the log to avoid divergence in unpopulated regions. Shift so `min(U) = 0`.
4. Fit `scipy.interpolate.CubicSpline(grid, U, bc_type='natural')`.
5. Extract knots and polynomial coefficients.

#### 1b. Angle Prior — Global

```
Input:  all_angles — array of angle values (radians) from all frames/proteins
Output: angle_knots (N,), angle_coeffs (N-1, 4)
```

Steps:
1. Apply Jacobian correction: the intrinsic probability is `P_intrinsic(θ) = P_raw(θ) / sin(θ)`. Rather than correcting the histogram, apply a weight `1/sin(θ)` to each sample during KDE (via `scipy.stats.gaussian_kde` with `weights` parameter). This is more principled than the current approach of correcting histogram bins after the fact.
2. Evaluate KDE on a regular grid of ~500 points over `[θ_min, θ_max]` (e.g., `[0.5, π - 0.01]` — stay away from boundaries where `sin(θ) → 0`).
3. Boltzmann inversion with density floor, shift to min=0.
4. Fit `CubicSpline(grid, U, bc_type='natural')`.
5. Extract knots and coefficients.

#### 1c. Angle Prior — Residue-Specific

```
Input:  all_angles, all_angle_central_species — per-angle species ID of the central atom
Output: per-type angle splines stored in a dict keyed by species ID
```

Steps:
1. Group angle samples by the species of the central atom (index i+1 in each triplet). This requires the fitting script to know the species array — it already has access to the NPZ which contains species.
2. For each species type `s`:
   a. Extract subset: `angles_s = all_angles[all_angle_central_species == s]`
   b. If `len(angles_s) < min_samples` (configurable, default 500), mark this type as "fallback to global".
   c. Otherwise, fit a spline using the same KDE → BI → CubicSpline procedure as 1b.
3. Store: for each type, either its own spline coefficients or a flag indicating global fallback.

**CLI additions to `prior_fitting_script.py`:**
- `--residue_specific_angles` (flag, default False): enable per-type angle fitting.
- `--angle_min_samples` (int, default 500): minimum samples per type before falling back to global.
- `--kde_bandwidth_factor` (float, default 1.0): multiplier on Silverman's rule bandwidth.
- `--spline_grid_points` (int, default 500): number of grid points for spline fitting.

#### 1d. Dihedral Prior

```
Input:  all_dihedrals — array of dihedral angles (radians, range [-π, π])
Output: dih_knots (N,), dih_coeffs (N-1, 4)
```

Steps:
1. Periodic KDE: duplicate data at boundaries by appending `φ - 2π` and `φ + 2π`, run standard KDE on the extended data, then evaluate only on `[-π, π]`. This avoids edge artifacts. Alternative: use `scipy.stats.vonmises_kde` if available. The duplication approach is simpler and robust.
2. Evaluate KDE on a regular grid of ~500 points over `[-π, π]`.
3. Boltzmann inversion with density floor, shift to min=0.
4. Fit `CubicSpline(grid, U, bc_type='periodic')`. The periodic boundary condition ensures `U(-π) = U(π)` and `U'(-π) = U'(π)`.
5. Extract knots and coefficients.

#### 1e. Output Format

Save to a single `.npz` file:

```python
np.savez(
    output_path,
    # Bond (global)
    bond_knots=bond_knots,              # (N_bond,)
    bond_coeffs=bond_coeffs,            # (N_bond-1, 4)

    # Angle — global fallback
    angle_knots=angle_knots,            # (N_angle,)
    angle_coeffs=angle_coeffs,          # (N_angle-1, 4)

    # Angle — residue-specific (only if --residue_specific_angles)
    angle_n_types=n_types,              # scalar: number of species types
    angle_type_knots=type_knots,        # (n_types, N_angle) — padded to same grid size
    angle_type_coeffs=type_coeffs,      # (n_types, N_angle-1, 4)
    angle_type_mask=type_mask,          # (n_types,) — 1 if type has own spline, 0 if fallback

    # Dihedral (global)
    dih_knots=dih_knots,                # (N_dih,)
    dih_coeffs=dih_coeffs,             # (N_dih-1, 4)

    # Metadata
    temperature=T,
    kB=kB,
    grid_points=n_grid,
    kde_bandwidth_factor=bw_factor,
    residue_specific_angles=True/False,
)
```

**Key constraint for JAX compatibility:** All per-type angle splines must use the **same knot grid** (same number of points, same domain). This allows stacking into a single `(n_types, N, ...)` array and indexing by species ID without dynamic shapes. Achieve this by evaluating all per-type KDEs on the same grid before fitting splines.

#### 1f. Diagnostic Plots

For each term, generate:
1. **PMF overlay:** raw histogram bins (thin gray), KDE-implied PMF (blue), spline PMF (red dashed). These should be nearly identical — any visible deviation is a red flag.
2. **Implied distribution:** empirical histogram (gray), spline-implied `P(x) ∝ exp(-βU_spline(x))` (red). For angles, include the `sin(θ)` Jacobian in the implied distribution.
3. **Residue-specific angles (if enabled):** one subplot per species type, showing type-specific KDE density overlaid with global density for comparison. Types that fell back to global should be marked.
4. **Force comparison:** numerical derivative of spline PMF vs analytical derivative (cubic spline derivative is exact — this is a sanity check on the coefficient extraction).

---

### Phase 2: JAX-Compatible Spline Evaluator

**New file:** `models/spline_eval.py`

This is a pure-JAX module — no scipy dependency at runtime.

#### Core Function

```python
def evaluate_cubic_spline(
    x: jax.Array,           # (...,) — coordinates to evaluate
    knots: jax.Array,       # (N,)   — sorted knot positions
    coeffs: jax.Array,      # (N-1, 4) — [c0, c1, c2, c3] per interval
) -> jax.Array:             # (...,) — energy values
    """
    Evaluate a cubic spline at given coordinates.

    For each x, finds interval i such that knots[i] <= x < knots[i+1],
    then computes: U = c0 + c1*dx + c2*dx² + c3*dx³  where dx = x - knots[i].

    Clamps x to [knots[0], knots[-1]] for boundary safety.
    JIT-compilable and autodiff-compatible.
    """
```

Implementation:
```
x_clamped = jnp.clip(x, knots[0], knots[-1] - epsilon)
i = jnp.searchsorted(knots, x_clamped, side='right') - 1
i = jnp.clip(i, 0, len(knots) - 2)
dx = x_clamped - knots[i]
c = coeffs[i]  # (... , 4)
U = c[..., 0] + c[..., 1] * dx + c[..., 2] * dx**2 + c[..., 3] * dx**3
```

`jnp.searchsorted` is JIT-compatible in JAX >= 0.4.1. Autodiff through polynomial arithmetic is standard.

#### Periodic Variant

```python
def evaluate_cubic_spline_periodic(
    x: jax.Array,
    knots: jax.Array,
    coeffs: jax.Array,
    period: float = 2.0 * jnp.pi,
) -> jax.Array:
    """Evaluate periodic spline. Wraps x into [knots[0], knots[0] + period) before lookup."""
```

Implementation: `x_wrapped = knots[0] + (x - knots[0]) % period`, then call `evaluate_cubic_spline`.

#### Species-Indexed Variant

```python
def evaluate_cubic_spline_by_type(
    x: jax.Array,              # (M,)        — coordinate values
    species: jax.Array,        # (M,)        — species ID per entry (int)
    all_knots: jax.Array,      # (n_types, N) — knots per type (same grid)
    all_coeffs: jax.Array,     # (n_types, N-1, 4) — coeffs per type
    type_mask: jax.Array,      # (n_types,)  — 1 = use type-specific, 0 = use global
    global_knots: jax.Array,   # (N,)
    global_coeffs: jax.Array,  # (N-1, 4)
) -> jax.Array:                # (M,)
    """
    Evaluate type-specific spline, falling back to global where type_mask == 0.
    """
```

Implementation:
```
U_typed = evaluate_cubic_spline(x, all_knots[species], all_coeffs[species])
U_global = evaluate_cubic_spline(x, global_knots, global_coeffs)
use_typed = type_mask[species]  # (M,) — 1 or 0
return jnp.where(use_typed, U_typed, U_global)
```

The `all_knots[species]` indexing selects the row for each entry's type. Because all types share the same grid size, this is a simple gather — no dynamic shapes.

#### Verification

Write a standalone test (can be run with pytest or as a script):
1. Fit a known function (e.g., `U = cos(2θ) + 0.5 cos(3θ)`) with CubicSpline in scipy.
2. Load the coefficients into `evaluate_cubic_spline` in JAX.
3. Verify: max absolute error < 1e-6 across a dense evaluation grid.
4. Verify: JAX gradient matches scipy's spline `.derivative()` evaluation to < 1e-5.
5. Verify: `jax.jit(evaluate_cubic_spline)` produces identical output to non-JIT.
6. Verify: `jax.vmap` over a batch of coordinates works correctly.

---

### Phase 3: Integrate into PriorEnergy

**File:** `models/prior_energy.py`

#### 3a. Interface Change — Species Threading

`PriorEnergy.compute_energy` currently has signature `(R, mask)`. It needs `(R, mask, species)`.

This is a **breaking interface change**. The `species` array is always available at every call site — threading it through is mechanical but must be thorough. Here is the exhaustive list of call sites that must be updated:

**`models/combined_model.py`** — primary consumer of PriorEnergy:
- `compute_energy(params, R, mask, species)` — already receives `species`, passes it to ML model. Add pass-through to `self.prior.compute_energy(R, mask, species)`.
- `compute_components(params, R, mask, species)` — same pattern. Currently calls `self.prior.compute_energy(R, mask)`. Add `species`.
- `compute_force_components(params, R, mask, species)` — same.
- `energy_fn_template` closure — this is the closure returned for chemtrain's `ForceMatching`. It captures `species` from the outer scope (it is a per-batch quantity set during data iteration). Verify that the closure passes `species` into the prior energy call. This closure is the most subtle call site because it's used inside `jax.grad` and must carry `species` through the differentiation.

**`training/trainer.py`**:
- `pretrain_prior()` — the LBFGS loss function currently takes `(params, R, mask)`. Since the LBFGS path is guarded (see Phase 4c) and will skip when splines are active, this call site only matters for the parametric path. For cleanliness, add `species` anyway so both paths have a consistent interface. The species array is available in the training data batch.

**`models/prior_energy.py`** — internal methods:
- `compute_energy(R, mask, species)` — new signature.
- `compute_total_energy(R, mask, species)` — delegates to `compute_energy`.
- `compute_total_energy_from_params(params, R, mask, species)` — LBFGS helper; add `species` for consistency.
- `compute_angle_energy(R, mask, species)` — the only sub-method that actually uses `species`. Bond, dihedral, and repulsive sub-methods keep `(R, mask)` signatures since they don't need species.

**Pattern:** Only `compute_angle_energy` consumes `species` internally. The top-level methods (`compute_energy`, `compute_total_energy`) accept it and route it to the angle method. This keeps the change minimal inside `prior_energy.py` while providing a clean external interface.

#### 3b. Constructor Changes

The constructor branches based on whether `spline_file` is present in config:

```python
def __init__(self, config, topology, displacement):
    # ... existing topology and weight setup ...

    spline_path = config.get("model", "priors", "spline_file", default=None)

    if spline_path is not None:
        # --- Spline prior path ---
        self.uses_splines = True
        spline_data = np.load(spline_path)

        # Bond spline (global)
        self.bond_knots = jnp.asarray(spline_data["bond_knots"])
        self.bond_coeffs = jnp.asarray(spline_data["bond_coeffs"])

        # Angle spline (global fallback)
        self.angle_knots = jnp.asarray(spline_data["angle_knots"])
        self.angle_coeffs = jnp.asarray(spline_data["angle_coeffs"])

        # Angle splines (per-AA, if available)
        self.residue_specific_angles = bool(spline_data.get("residue_specific_angles", False))
        if self.residue_specific_angles:
            self.angle_type_knots = jnp.asarray(spline_data["angle_type_knots"])
            self.angle_type_coeffs = jnp.asarray(spline_data["angle_type_coeffs"])
            self.angle_type_mask = jnp.asarray(spline_data["angle_type_mask"])

        # Dihedral spline (global)
        self.dih_knots = jnp.asarray(spline_data["dih_knots"])
        self.dih_coeffs = jnp.asarray(spline_data["dih_coeffs"])

        # Only repulsive params from YAML (still parametric)
        prior_params = config.get_prior_params()
        self.params = {
            "epsilon": jnp.asarray(prior_params.get("epsilon", 1.0)),
            "sigma": jnp.asarray(prior_params.get("sigma", 3.0)),
        }

    else:
        # --- Parametric prior path (legacy, unchanged) ---
        self.uses_splines = False
        self.residue_specific_angles = False
        prior_params = config.get_prior_params()
        self.params = {
            "r0": jnp.asarray(prior_params.get("r0", 3.8), dtype=jnp.float32),
            "kr": jnp.asarray(prior_params.get("kr", 150.0), dtype=jnp.float32),
            "a": jnp.asarray(prior_params.get("a", [0.0]), dtype=jnp.float32),
            "b": jnp.asarray(prior_params.get("b", [0.0]), dtype=jnp.float32),
            "epsilon": jnp.asarray(prior_params.get("epsilon", 1.0), dtype=jnp.float32),
            "sigma": jnp.asarray(prior_params.get("sigma", 3.0), dtype=jnp.float32),
            "k_dih": jnp.asarray(prior_params.get("k_dih", [0.5]), dtype=jnp.float32),
            "gamma_dih": jnp.asarray(prior_params.get("gamma_dih", [0.0]), dtype=jnp.float32),
        }
```

The `self.uses_splines` flag is used by:
- Each `compute_*_energy` method to select evaluation path.
- The trainer's LBFGS guard (Phase 4c) to skip pretraining.

#### 3c. Energy Method Replacements

Each method branches on `self.uses_splines`. The existing parametric code stays as the `else` branch.

**Bond:**
```python
def compute_bond_energy(self, R, mask):
    # ... existing distance computation (unchanged) ...
    r = _safe_norm(dR)

    if self.uses_splines:
        U_bond = evaluate_cubic_spline(r, self.bond_knots, self.bond_coeffs)
    else:
        U_bond = 0.5 * self.params["kr"] * (r - self.params["r0"]) ** 2

    E_bond = jnp.sum(jnp.where(bond_valid, U_bond, 0.0))
    return E_bond
```

**Angle:**
```python
def compute_angle_energy(self, R, mask, species=None):
    # ... existing angle computation (unchanged through to theta) ...
    theta = jnp.where(angle_valid, theta, jax.lax.stop_gradient(theta))

    if self.uses_splines:
        if self.residue_specific_angles and species is not None:
            central_species = species[self.angles[:, 1]]
            U_angle = evaluate_cubic_spline_by_type(
                theta, central_species,
                self.angle_type_knots, self.angle_type_coeffs, self.angle_type_mask,
                self.angle_knots, self.angle_coeffs,
            )
        else:
            U_angle = evaluate_cubic_spline(theta, self.angle_knots, self.angle_coeffs)
    else:
        U_angle = _angular_fourier_energy(theta, self.params["a"], self.params["b"])

    E_angle = jnp.sum(jnp.where(angle_valid, U_angle, 0.0))
    return E_angle
```

Note: `species` defaults to `None` so that the parametric path (which never passes species) does not break. When splines are active and `residue_specific_angles` is true, `species` must be provided — the caller (`compute_energy`) is responsible for this.

**Dihedral:**
```python
def compute_dihedral_energy(self, R, mask):
    # ... existing dihedral computation (unchanged through to phi) ...
    phi = jnp.where(dih_valid, phi, jax.lax.stop_gradient(phi))

    if self.uses_splines:
        U_dih = evaluate_cubic_spline_periodic(phi, self.dih_knots, self.dih_coeffs)
    else:
        U_dih = _dihedral_periodic_energy(phi, self.params["k_dih"], self.params["gamma_dih"])

    E_dih = jnp.sum(jnp.where(dih_valid, U_dih, 0.0))
    return E_dih
```

**Repulsive:** Unchanged — always parametric `ε(σ/r)⁴`.

**`compute_energy` signature:**
```python
def compute_energy(self, R, mask, species=None) -> Dict[str, jax.Array]:
    E_bond = self.compute_bond_energy(R, mask)
    E_angle = self.compute_angle_energy(R, mask, species)
    E_dih = self.compute_dihedral_energy(R, mask)
    E_rep = self.compute_repulsive_energy(R, mask)
    # ... weighting and return unchanged ...
```

The `species=None` default preserves backward compatibility for the parametric path. All callers on the spline path (CombinedModel) pass species explicitly.

#### 3d. Retain Parametric Functions as Fallback

Keep the existing functions intact:
- `_angular_fourier_energy()`
- `_dihedral_periodic_energy()`
- The harmonic bond computation inside `compute_bond_energy`

These remain the active code path when `spline_file` is absent from config (parametric fallback). The `PriorEnergy.__init__` constructor branches based on whether `spline_file` is present: if yes, load spline arrays and set `self.uses_splines = True`; if no, load parametric params from YAML as before and set `self.uses_splines = False`. Each `compute_*_energy` method checks `self.uses_splines` to choose the evaluation path.

#### 3e. Preserve `compute_total_energy_from_params` for Parametric Path

This method exists for LBFGS pretraining — it temporarily swaps parametric prior params and recomputes energy. With spline priors it is unused (the LBFGS guard in Phase 4c prevents it from being called). **Keep it as-is** for the parametric fallback path. Update its signature to `(params, R, mask, species)` for interface consistency.

---

### Phase 4: Config and Training Changes

#### 4a. Config Template

```yaml
model:
  priors:
    # Spline-based priors (new)
    spline_file: "data_prep/datasets/fitted_priors_spline.npz"
    residue_specific_angles: true   # per-AA typing (20 types); must match spline_file contents

    weights:
      bond: 0.5
      angle: 0.25
      dihedral: 0.25
      repulsive: 1.0

    # Repulsive parameters (still parametric)
    epsilon: 1.0
    sigma: 3.0

    # Legacy parametric params — REMOVED when using spline priors
    # r0, kr, a, b, k_dih, gamma_dih no longer present
    # These are still supported if spline_file is omitted (parametric fallback path)
```

#### 4b. ConfigManager Updates

Add:
```python
def get_spline_file_path(self) -> str:
    return self.get("model", "priors", "spline_file")

def get_residue_specific_angles(self) -> bool:
    return self.get("model", "priors", "residue_specific_angles", default=False)
```

#### 4c. Trainer — LBFGS Pretraining Guard

**File:** `training/trainer.py`

Add a runtime guard at the top of the `pretrain_prior()` method (or wherever LBFGS pretraining is invoked in the training pipeline). The logic:

```
if prior_energy.uses_splines:
    logger.info("Spline priors detected — skipping LBFGS prior pretraining (no parametric params to optimize).")
    return
```

The `uses_splines` flag is set on `PriorEnergy.__init__` based on whether a `spline_file` was loaded. This guard fires even if the config has `pretrain_prior: true`, preventing a misconfiguration from attempting to optimize non-existent parameters.

**No code is deleted.** The existing LBFGS pretraining logic remains intact for the parametric prior path. The guard is a single early-return check. The `compute_total_energy_from_params` method on `PriorEnergy` also stays — it is only relevant for the parametric path and will simply never be called when splines are active.

---

### Phase 5: Export

**File:** `export/exporter.py`

The exporter currently passes `prior_params` (dict with r0, kr, a, b, etc.) into the traced function. With spline priors, the spline arrays (knots, coeffs) are attributes of the `PriorEnergy` instance. Since JAX traces through the actual computation graph (which now calls `evaluate_cubic_spline` on `self.bond_knots`, etc.), the spline arrays become **constants baked into the MLIR** — no special handling is needed.

**What to verify:**
1. The spline arrays are captured as constants during `jax.make_jaxpr` / StableHLO lowering.
2. The exported MLIR produces identical energies and forces as the Python model on a test frame.
3. LAMMPS simulation runs stably with the spline-based exported model.

The `from_combined_model` factory may need to stop extracting individual prior params from the model and instead just rely on the energy function closure capturing everything.

---

## 4. Residue-Specific Angle Priors — Scientific Detail

### 4.1 Physical Justification

The backbone angle at Cα bead i (formed by beads i-1, i, i+1) is predominantly governed by the **identity of residue i**:

- **Glycine (G):** No side chain → maximal backbone flexibility → broad, shallow U(θ).
- **Proline (P):** Pyrrolidine ring constrains φ → very rigid, narrow U(θ) with a single sharp minimum.
- **β-branched residues (V, I, T):** Bulky Cβ branching restricts backbone → narrower U(θ) than average.
- **Small polar (S, A, C):** Moderate flexibility → U(θ) close to global average.
- **Large aromatic (W, F, Y):** Side-chain bulk creates distinct angular preferences.

A global U(θ) averages over all of these, producing a PMF that is too broad for proline and too narrow for glycine. The ML model must then learn residue-specific corrections — but these corrections are systematic and predictable, exactly the kind of thing a physics-based prior should capture.

### 4.2 Typing Strategy

**Decision: Per-amino-acid typing (20 types)**

Each of the 20 standard amino acids gets its own U(θ). This is the most scientifically informative and is feasible given the mdCATH dataset scale.

Rough estimate: with ~2500 frames and proteins of ~50-200 residues, each frame contributes ~50-200 angle samples. With 20 types and roughly equal frequency, each type gets ~(2500 × 100 / 20) ≈ 12,500 samples (order of magnitude). This is sufficient for KDE + spline fitting.

For rare amino acids (e.g., tryptophan in small proteins), the fallback-to-global mechanism handles insufficient data gracefully. The fitting script reports per-type sample counts and warns for any type below the threshold.

Note: the codebase also has `group_amino_acids_4way()` which groups residues by charge character. This could serve as a future alternative if a dataset has very few samples, but is not used in this implementation.

### 4.3 Data Requirements

For reliable KDE → BI → spline fitting, each type needs enough samples to resolve the PMF shape. Rules of thumb:

- **Minimum ~500 samples** per type for a reasonable 1D KDE. Below this, the KDE bandwidth dominates and the PMF becomes over-smoothed.
- **Ideal ~5000+ samples** for capturing multi-modal features.

The fitting script should report per-type sample counts and warn for any type below the threshold.

### 4.4 Interaction with Multi-Protein Datasets

When training on multiple proteins (combined padded NPZ), the angle samples for fitting come from **all proteins** pooled together. The species array in the combined dataset provides the type labels. This is correct — the prior should capture the average behavior across proteins for each residue type.

For **padded atoms** (species often set to 0 or a dummy value), their angle samples must be excluded during fitting. The fitting script should filter by `mask > 0` before collecting samples.

### 4.5 Expected Impact

- **Glycine angles:** Prior will be broader → less bias, ML model needs less correction.
- **Proline angles:** Prior will be much narrower and correctly positioned → strong physical constraint, ML model learns finer corrections on a better baseline.
- **Overall:** Reduced burden on Allegro to learn systematic per-residue angular preferences. This should improve data efficiency, especially for proteins with residue compositions different from the training set (better transferability).

### 4.6 Future Extension: Residue-Specific Dihedrals

Once the per-type infrastructure is validated for angles, extending to dihedrals is straightforward:

- Type by central pair (species[i+1], species[i+2]) → up to 400 types.
- Use the same `evaluate_cubic_spline_by_type` machinery, but with a 2D type index mapped to a flat index: `type_idx = species[i+1] * n_types + species[i+2]`.
- Fallback to global for rare pairs.
- Periodic spline evaluation throughout.

This is the natural next step but should be deferred until the angle-specific system is working and validated.

---

## 5. Testing and Validation

### 5.1 Unit Tests (Phase 2)

- Spline evaluator matches scipy on known functions.
- Gradients match analytical cubic derivatives.
- Periodic spline wraps correctly at boundaries.
- Type-indexed evaluation selects correct splines.
- Fallback to global works when `type_mask == 0`.

### 5.2 Integration Tests (Phase 3)

- `PriorEnergy` with spline data produces finite energies and forces on a real dataset frame.
- Padded atom masking still prevents NaN gradients (the `_safe_norm` / `stop_gradient` infrastructure is unchanged, but verify end-to-end).
- Energy components (bond, angle, dihedral, repulsive) are all reasonable magnitudes.
- Force-matching loss with spline priors is comparable to or better than parametric priors on the same dataset.

### 5.3 Scientific Validation (Phase 5)

- **PMF fidelity:** overlay the spline U(x) against the raw BI histogram — they should be near-identical for well-sampled regions.
- **Implied distributions:** `P_spline(x) ∝ exp(-βU_spline(x))` should match the empirical histogram closely.
- **Residue-specific angles:** for glycine and proline specifically, compare the type-specific U(θ) to the global U(θ) and verify that the expected physical differences are captured.
- **Force RMSE:** compare training convergence (force RMSE vs epoch) with spline priors vs old parametric priors. Spline priors should converge at least as fast, likely faster (less systematic bias for ML to correct).
- **MD stability:** run a short LAMMPS simulation with the spline-prior-trained model and verify energy conservation and structural stability.

---

## 6. Execution Order

| Step | Phase | Description | Estimated effort |
|------|-------|-------------|-----------------|
| 1 | 2 | Create `models/spline_eval.py` with unit tests | Small — isolated, testable |
| 2 | 1 | Modify `prior_fitting_script.py` — global splines only (bond, angle, dihedral) | Medium |
| 3 | 3 | Modify `PriorEnergy` — replace parametric with spline (global only, no species yet) | Medium |
| 4 | 3 | Thread `species` through `PriorEnergy` interface → `CombinedModel` | Small — mechanical |
| 5 | 1+3 | Add residue-specific angle fitting and evaluation | Medium |
| 6 | 4 | Update config, add LBFGS spline guard | Small |
| 7 | 5 | Verify MLIR export and LAMMPS integration | Small-Medium |
| 8 | 5 | Scientific validation (PMF plots, training comparison, MD test) | Medium |

Steps 1–3 deliver the core value (non-parametric PMF priors). Steps 4–5 add the residue-specific capability. Steps 6–8 are integration and validation.

---

## 7. What Stays Unchanged

- **Parametric prior path:** Fully preserved behind `self.uses_splines == False`. Harmonic bond, Fourier angle, periodic dihedral, and LBFGS pretraining all remain functional when `spline_file` is absent from config.
- Repulsive energy term: still parametric `ε(σ/r)⁴`. Not data-derived.
- All NaN-safety infrastructure: `_safe_norm`, `_safe_atan2`, `stop_gradient` on invalid entries.
- TopologyBuilder: generates the same index arrays.
- CombinedModel orchestration logic (modes, ML backbone selection).
- chemtrain ForceMatching trainer, optimizers, learning rate schedules.
- Data loading, preprocessing, padding, parking.
- LAMMPS deployment workflow (only the baked-in constants change).
- Ensemble training.
