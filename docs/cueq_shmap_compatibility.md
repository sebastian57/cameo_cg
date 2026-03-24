# cuEquivariance + chemtrain shmap Compatibility Notes

Date: 2026-03-09

## Goal

Run the full cuEquivariance Allegro model with `disable_shmap=False` so we keep the shmap distribution path and preserve expected performance gains versus the old e3nn model.

## What Failed So Far

### 1) `disable_shmap=True` (pmap fallback) on multi-node

- Single-node could run.
- Multi-node failed with:
  - `INVALID_ARGUMENT: CopyArrays only supports destination device list of the same size as the array device lists.`
- This is a pmap/multi-host data movement mismatch on the fallback path and does not validate shmap compatibility.

### 2) `disable_shmap=False` (shmap path) on cuEquivariance model

- Fails with:
  - `ValueError: Context mesh ... axis_types=(Manual) should match ... axis_types=(Auto) passed to broadcast_in_dim`
- Trace points into `jax.nn.softplus` called from `models/allegro_cueq_v2.py`, especially:
  - `Allegro.__init__` normalization (`self.epsilon`)
  - residual mixing (`self.alpha`)

## Root Cause (Current Understanding)

`chemtrain` shmap path builds a Manual mesh context (`shard_map` over `Mesh(..., axis_name='batch')`).

Inside cuEquivariance Allegro, some scalar ops currently call `jax.nn.softplus`. In the current JAX stack, these can lower through a `broadcast_in_dim` path that carries Auto-sharding metadata, which conflicts with the active Manual mesh context.

So the failure is a sharding-semantics mismatch at operation lowering boundaries, not a fundamental incompatibility of cuEquivariance with shmap.

## Patch Strategy Applied Here

1. Keep `disable_shmap=False` so the shmap path remains active.
2. Replace `jax.nn.softplus` at the affected sites with a mesh-safe softplus helper that:
   - avoids the problematic `jax.nn.softplus` lowering path,
   - builds shape-matched constants from `zeros_like`/`ones_like`,
   - keeps numerically stable softplus behavior.
3. Keep all heavy model kernels unchanged (tensor products, graph operations, segment reductions).

### Follow-up after first patch

After patching the cuEq Allegro `softplus` sites, the next 2-node shmap run moved the failure to:

- `chemutils/models/layers/scale_shift.py` (`nn.softplus(self.scale)` inside `ScaleShiftLayer`)

That call is on the active traceback path through `AtomicEnergyLayer`, so the same mesh-safe softplus strategy was applied there as the next minimal fix.

## Why This Should Work

- The traceback identifies `softplus`/broadcast lowering as the direct source.
- Replacing only these scalar transform sites removes the known trigger while leaving the model architecture and the shmap orchestration unchanged.
- This is a minimal compatibility patch focused on sharding semantics, not algorithmic behavior.

## Performance Impact Expectation

- Expected impact is negligible to very small.
- Patched operations are scalar or small broadcast-like transforms, not dominant GPU kernels.
- The shmap execution model remains unchanged, so distribution/performance characteristics should stay close to baseline shmap behavior.

## Validation Plan

1. Run 1-node shmap sanity.
2. Run 2-node shmap test.
3. Confirm:
   - no Manual-vs-Auto mesh error,
   - no fallback to `disable_shmap=True`,
   - normal epoch logging and non-zero loss progression.
