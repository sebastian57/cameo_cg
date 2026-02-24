# Efficiency Optimizations & Profiling Guide

**Session date:** 2026-02-20

---

## Overview

This session covered two phases:
1. **Implementing bottleneck fixes** identified in `EFFICIENCY_BOTTLENECKS.md`
2. **Adding profiling infrastructure** to quantify the remaining host-sync bottleneck before fixing it

---

## Changes Made

### `models/combined_model.py` — `compute_force_components`

**Problem:** The method called `jax.grad(E_component)(R)` seven times separately. Each call
triggered a full forward pass through the model (Allegro + priors), making diagnostic force
decomposition 7× slower than necessary.

**Fix:** Replaced with a single `jax.vjp` call:
```python
_, vjp_fn = jax.vjp(all_energies, R)   # ONE forward pass, residuals cached
F_bond = -vjp_fn((0,0,1,0,0,0,0))[0]  # backward-only, no re-forward
```
`jax.vjp` separates the forward pass from the backward pass. The stored residuals are reused for
each of the 7 component backward passes. Cost goes from `7 × (fwd + bwd)` to `1 × fwd + 7 × bwd`.

**Scope:** Evaluation/analysis only — not in the training hot path.

---

### `config/manager.py` — new profiling keys

Three new keys added to `get_profiling_config()`:

| Key | Default | Meaning |
|-----|---------|---------|
| `batch_profiler_enabled` | `false` | Enable per-batch dispatch/barrier timing |
| `batch_profiler_warmup` | `5` | Skip first N batches (JIT compilation noise) |
| `batch_profiler_samples` | `50` | How many batches to profile per stage |

---

### `training/trainer.py` — `_attach_batch_profiler` + `_report_batch_profiler`

Two new methods that monkey-patch `trainer._update_fn` before `trainer.train()` is called.
No changes to chemtrain code. The patch is inserted in `train_stage()` and reports after training.

---

### `config_profile.yaml` — batch profiler enabled

```yaml
profiling:
  batch_profiler_enabled: true
  batch_profiler_warmup: 5
  batch_profiler_samples: 50
```

---

## How the Batch Profiler Works

### What it measures

For each profiled batch, three timestamps are captured around the `_update_fn` call:

```
t0 ──────────► t1 ──────────────────────────► t2
     dispatch        effects_barrier wait
     (async)         (GPU compute)
```

| Metric | Formula | Meaning |
|--------|---------|---------|
| `dispatch_ms` | `t1 − t0` | Time for `_update_fn()` to return. Should be ~0 ms if JAX's async dispatch is working — the GPU work is queued but not yet executed. |
| `gpu_barrier_ms` | `t2 − t1` | Time spent in `jax.effects_barrier()`. This is the true GPU compute time per batch. |
| `inter-batch gap_ms` | `t0[i+1] − t0[i]` | Wall time between the start of consecutive batches. |

### The key diagnostic — gap / barrier ratio

```
gap / barrier  ≈  1.0   →  CPU blocks every step   (current code, sync bottleneck)
gap / barrier  ≈  0.0   →  GPU fully pipelined      (target after fix)
```

**Why this matters:**

With the current chemtrain code, `_update` calls `onp.asarray(train_loss)` immediately after
`_update_fn` returns. `onp.asarray` forces a device-to-host transfer, which blocks the Python
thread until the GPU finishes the current batch. The next batch cannot be dispatched until the
current one is fully complete.

Timeline (current — blocking):
```
GPU: [batch 1 compute ████████████████][batch 2 compute ████████████████]
CPU:  dispatch → wait_for_GPU (onp.asarray) → dispatch → wait_for_GPU → ...
      ^─────────────────────────────────────^  gap ≈ barrier
```

Timeline (after fix — async):
```
GPU: [batch 1 ████████████████][batch 2 ████████████████]
CPU:  dispatch → load_data → dispatch → load_data → ...  (flush at epoch end)
      ^─────────^  gap ≈ data_load_time << barrier
```

### Reading the profiler output

The log will contain a block like:

```
[BatchProfiler] Per-batch timing (50 samples, 5 warmup skipped):
  dispatch_fn  : mean=0.15 ± 0.04 ms  p50=0.13  p95=0.24
  gpu_barrier  : mean=312.4 ± 18.3 ms  p50=310.1  p95=348.2
  inter-batch gap: mean=313.1 ± 18.5 ms  p50=310.9  p95=349.0
  gap / barrier ratio: 1.002  (1.0 = CPU blocks each step; 0.0 = GPU fully pipelined)
  [!!] CPU is BLOCKING on every batch step. ...
```

**Interpretation guide:**

| What you see | What it means |
|---|---|
| `dispatch_ms ≈ 0` | JAX async dispatch is working — GPU work is queued without waiting |
| `dispatch_ms >> 0` | Dispatch is synchronous for some reason (unexpected, check for shape changes or Python-traced conditionals) |
| `gpu_barrier_ms` | Actual GPU compute time per batch — this is your baseline throughput |
| `gap_ms ≈ barrier_ms`, ratio ≈ 1.0 | CPU is idle waiting for GPU each step due to `onp.asarray` syncs — **fix this** |
| `gap_ms << barrier_ms`, ratio ≈ 0.0 | CPU keeps GPU busy, data loading is faster than compute — good |
| `gap_ms > barrier_ms`, ratio > 1.0 | Data loading or Python overhead is the bottleneck (GPU is waiting for CPU) |

### Planned fix (pending profiling confirmation)

Remove `onp.asarray()` from the per-batch `_update` loop in
`chemtrain/trainers/base.py:1023–1026`. Instead, accumulate raw JAX arrays and
flush with `jax.device_get()` at the end of each epoch. The batch profiler will
confirm whether the ratio drops toward 0 after the fix.

---

## Pending Architectural Change — Separate Prior from Training Graph

When `train_priors=False` (default), the prior energy `E_prior(R)` does not depend on
`params_ML`. It therefore contributes **zero** to `grad(loss, params_ML)`. Despite this,
it currently lives inside the JIT gradient graph, meaning XLA traces and stores residuals
for the full prior subgraph (bonds, angles, dihedrals, repulsion) on every training step.

**Planned fix:** Precompute `F_prior(R_i)` for every training configuration once at setup
time and store corrected force targets:

```
F_corrected[i] = F_ref[i] - F_prior(R_i)
```

Train the ML model against `F_corrected`. The `compute_energy` JIT graph becomes `E_ML` only,
removing the `stop_gradient + jnp.where` pattern and all prior subgraph residual storage.
This will be implemented after profiling confirms the sync fix is the primary bottleneck.
