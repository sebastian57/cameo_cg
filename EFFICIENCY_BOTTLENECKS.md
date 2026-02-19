# Efficiency Bottleneck Analysis — cameo_cg

> Audit date: 2026-02-18
> Scope: `cameo_cg/` code only (excludes chemtrain and chemutils internals)

---

## Critical

### 1. JAX-to-NumPy round-trip in `_create_chemtrain_loaders`

**File:** `training/trainer.py:254-271`

Data is loaded as JAX arrays in `DatasetLoader` (`jnp.asarray`, loader.py:169-172), then converted back to NumPy via `np.array()` to create chemtrain's `NumpyDataLoader`. This triggers a full device-to-host transfer of the entire dataset. Called at the start of every training stage.

**Fix:** Store data as NumPy arrays from the start in `DatasetLoader`, or pass through without conversion if chemtrain accepts JAX arrays.

---

### 2. Host-side parameter norm computation every epoch

**File:** `training/trainer.py:397-403`

After each training stage, `total_param_norm` is computed by flattening every parameter leaf to host via `np.asarray(v)` and calling `np.linalg.norm()` in a Python generator loop. Each leaf transfer blocks on the device.

**Fix:** Compute entirely on-device: `jnp.sqrt(sum(jnp.vdot(v, v) for v in leaves))`, transfer only the final scalar.

---

## High

### 3. Neighbor list allocated AND updated every forward pass

**File:** `models/allegro_model.py:213-217`

When `neighbor is None` (the common case during training), `allocate()` builds the full neighbor structure O(N^2), then `update()` is called immediately after. Allocation is expensive and redundant if the structure is reused across frames with the same atom count.

**Fix:** Allocate once during model init or at the start of each training stage; pass the pre-allocated neighbor list to `update()` on each call.

---

### 4. Multiple separate `jax.vmap(displacement)` calls in prior energy

**File:** `models/prior_energy.py:418, 176-177, 207-209, 519, 574`

Each energy term (bond, angle, dihedral, repulsive) independently vmaps the `displacement` function over its pair/triplet/quadruplet arrays. This means 5+ separate vmap compilations for what is essentially the same operation on different index subsets. JAX cannot fuse operations across separate vmap boundaries.

**Fix:** Compute all needed displacements in a single batched call (concatenate all index pairs, vmap once, slice results).

---

### 5. Incomplete `_block_until_ready` fallback

**File:** `training/trainer.py:239-243`

The fallback path (when `jax.block_until_ready(tree)` raises) iterates over leaves but `return`s after blocking the **first** leaf only, leaving all other arrays potentially unsynced. This can cause silent race conditions or incorrect timing measurements.

**Fix:** Remove the early `return` — block all leaves, or just rely on the `jax.block_until_ready(tree)` call in the try block (which handles the full tree).

---

## Medium

### 6. `stop_gradient` masking creates unnecessary array copy

**File:** `models/combined_model.py:164-168`

`jnp.where(mask, R, jax.lax.stop_gradient(R))` evaluates both branches, so `stop_gradient(R)` allocates a full copy of R even though it's only used for padded positions. This happens every energy evaluation (forward + backward).

**Fix:** Use a custom VJP or apply `stop_gradient` only to the gradient of padded positions, avoiding the extra allocation in the forward pass.

---

### 7. Redundant `jnp.asarray` calls in train.py

**File:** `scripts/train.py:429-432`

Data already converted to JAX arrays by `DatasetLoader` is wrapped again in `jnp.asarray()` when building `train_data`/`val_data` dicts. While `jnp.asarray` on an existing JAX array is a no-op, the slicing (`[:N_train]`) creates new arrays.

**Fix:** Use the loader's pre-split data directly instead of re-slicing.

---

### 8. Padded coordinate spreading recomputes every call

**File:** `models/allegro_model.py:142-153`

`_spread_padded_coordinates` builds `offsets` via `jnp.arange` + `jnp.stack` + `jnp.where` on every `compute_energy()` call. The offset array is deterministic for a given N and could be precomputed once.

**Fix:** Precompute and cache the offset array at init time; only apply the `jnp.where` mask at call time.

---

### 9. Logging triggers device-to-host transfers

**File:** `scripts/train.py:341-342`

`np.asarray(box)` and `np.asarray(R_shift)` pull JAX arrays to host for string formatting in log messages. Harmless at init, but the pattern is also used elsewhere (e.g., trainer.py:722-727 logs entire parameter arrays).

**Fix:** Use `float()` for scalars or `jax.device_get()` explicitly; for arrays, log shapes/norms instead of full contents.

---

## Low

### 10. `tree_map(jnp.asarray, ...)` on checkpoint load

**File:** `training/trainer.py:991, 1002`

Checkpoint params are wrapped in `jnp.asarray` via `tree_map` even if they're already the correct type. Minor overhead, but only happens on resume.

**Fix:** Check dtype before converting, or accept the one-time cost.

---

### 11. Preprocessor reshape may copy

**File:** `data/preprocessor.py:77-78`

`R_for_min.reshape(-1, 3)` can create a copy if the array isn't contiguous. One-time cost at data-prep time.

**Fix:** Use `.reshape` on contiguous arrays or accept the negligible cost since it runs once.

---

## Summary

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Critical | trainer.py:254-271 | JAX→NumPy round-trip for chemtrain loaders |
| 2 | Critical | trainer.py:397-403 | Host-side param norm via np.linalg.norm per leaf |
| 3 | High | allegro_model.py:213-217 | Neighbor list allocate+update every forward pass |
| 4 | High | prior_energy.py (multiple) | 5+ separate vmap(displacement) compilations |
| 5 | High | trainer.py:239-243 | Incomplete _block_until_ready (returns after 1 leaf) |
| 6 | Medium | combined_model.py:164-168 | stop_gradient creates full R copy in forward pass |
| 7 | Medium | train.py:429-432 | Redundant jnp.asarray on already-converted data |
| 8 | Medium | allegro_model.py:142-153 | Padded offsets recomputed every energy call |
| 9 | Medium | train.py:341, trainer.py:722 | Logging pulls arrays to host |
| 10 | Low | trainer.py:991 | Unnecessary tree_map on checkpoint load |
| 11 | Low | preprocessor.py:77 | Reshape may copy non-contiguous array |
