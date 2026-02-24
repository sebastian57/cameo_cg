# Model Execution Optimizations: Mixed Precision + Activation Checkpointing

Date: 2026-02-23  
Scope: `cameo_cg` training stack (JAX + chemtrain + chemutils Allegro wrapper)

## Goal

Reduce per-step model execution time and memory pressure so we can push `batch_per_device` higher while preserving force-matching quality and training stability.

Primary tracks:
1. Mixed precision (prefer `bfloat16` compute with safe `float32` reductions/state).
2. Activation checkpointing (`jax.remat`) in heavy Allegro compute regions.
3. Buffer donation (`donate_argnums`) for state-heavy update paths.

## Current Baseline Assumptions

- Training path is `scripts/train.py -> training/trainer.py -> chemtrain ForceMatching -> model energy_fn_template`.
- Allegro wrapper lives in `models/allegro_model.py`, with `init_allegro` / `apply_allegro` from `chemutils.models.allegro.model.allegro_neighborlist_pp`.
- Current code is largely hard-wired to `float32` in model/prior wrappers.
- Prior terms are disabled for the profiling baseline, so ML path dominates.

## Track A: Mixed Precision Implementation Plan

### A1. Add Config Knobs

Files:
- `config/manager.py`
- `config_template.yaml` (and profile config(s) as needed)

Add:
- `training.compute_dtype`: `"float32" | "bfloat16"` (default `"float32"`).
- `training.param_dtype`: `"float32"` (keep FP32 master params initially).
- `training.reduce_dtype`: `"float32"` (safe all-reduce + optimizer math).
- Optional guard: `training.enable_mixed_precision: bool`.

Acceptance:
- Config parser returns validated dtypes and sane defaults.

### A2. Thread Dtype Policy Through Model Construction

Files:
- `models/allegro_model.py`
- `models/combined_model.py`
- `models/mace_model.py`, `models/painn_model.py` (consistency pass)

Steps:
1. Introduce model-level dtype attributes (`compute_dtype`, `param_dtype`).
2. Replace hard-coded `jnp.float32` casts where safe with dtype policy:
   - Geometry tensors and model inputs -> `compute_dtype`.
   - Keep indices/species integer types unchanged.
3. Keep numerically sensitive constants/reductions in FP32 unless explicitly validated.

Acceptance:
- Model builds and runs in FP32 unchanged.
- `bfloat16` mode executes without dtype-related crashes.

### A3. Safe Gradient/Optimizer Path

Files:
- `chemtrain-deploy/external/chemtrain/chemtrain/learn/max_likelihood.py` (local vendor copy used in runs)

Steps:
1. Keep optimizer state and master params in FP32.
2. If gradients are produced in BF16, upcast to FP32 before optimizer update.
3. Keep collective reduction dtype explicit and stable (`lax.pmean` inputs cast as needed).
4. Add logging of dtype policy per run start.

Acceptance:
- No optimizer instability from BF16 state corruption.
- Loss/grad norm curves remain close to FP32 reference for short A/B runs.

### A4. Validation Protocol for Mixed Precision

Run matrix (short):
- FP32 baseline vs BF16 compute (same seed, same data slice, same batch config).
- 1-node quick check, then 2-node check.

Metrics:
- Step time, structures/s, time/structure.
- Max memory usage (per GPU).
- Final train loss trajectory (first N epochs).
- Gradient norm behavior (spikes/NaNs).

Go/No-Go:
- No NaNs/divergence.
- Throughput improvement and/or memory headroom gain.

## Track B: Activation Checkpointing / Rematerialization Plan

## What It Means

Activation checkpointing (`jax.remat`) trades compute for memory:
- Forward pass stores fewer intermediate activations.
- Backward pass recomputes marked subgraphs on demand.
- Result: lower peak memory, potentially enabling larger `batch_per_device`.

### B1. Identify Candidate Remat Boundaries

Primary targets:
- Deep equivariant message/update blocks in Allegro model apply path.
- Large tensor-product / interaction blocks with high activation footprint.

Files:
- `chemutils` Allegro implementation (inside `chemtrain-deploy` environment).
- Wrapper touchpoint in `models/allegro_model.py` if boundary insertion is externalized.

Steps:
1. Inspect Allegro forward graph structure.
2. Mark 1-2 coarse `jax.remat` boundaries first (not every small op).
3. Keep neighbor-list update and indexing logic outside remat unless needed.

Acceptance:
- Remat-enabled graph compiles and runs with identical outputs (within tolerance).

### B2. Add Configurable Remat Levels

Add config knob:
- `training.remat_level`: `0` (off), `1` (coarse), `2` (deeper blocks).
- `training.remat_policy`: `"none" | "allegro_blocks_coarse" | "allegro_blocks_deep"` (optional explicit policy mapping).

Behavior:
- `0`: current execution.
- `1`: remat around main interaction stack.
- `2`: additional remat inside repeated block internals.

Acceptance:
- Easy runtime switching without code edits.

### B3. Validate Memory vs Compute Tradeoff

Run matrix:
- Fixed model + fixed dataset + fixed seed.
- Compare remat levels at same `batch_per_device`.
- Then increase `batch_per_device` until memory limit.

Metrics:
- Peak memory.
- Step time and throughput.
- Effective best achievable `batch_per_device`.
- Final time/structure at the best stable setting.

Decision rule:
- Keep remat setting only if end-to-end time/structure improves at target scale.

## Track C: Buffer Donation (`donate_argnums`) Plan

## What It Means

JAX buffer donation allows compiled update functions to reuse input buffers for outputs:
- Reduces device memory pressure and device-side copies.
- Can improve step time if allocator/copy overhead is nontrivial.
- No math change; this is primarily a memory/runtime optimization.

### C1. Add Config Knobs

Files:
- `config/manager.py`
- `config_template.yaml`

Add:
- `training.enable_buffer_donation: bool` (default `true` for profiling runs, `false` for conservative fallback).
- `training.donate_mode: "state_only" | "state_and_batch"` (start with `state_only`).

### C2. Apply Donation to JIT/PJIT Update Entry Point

Files:
- `chemtrain-deploy/external/chemtrain/chemtrain/trainers/base.py` (and any wrappers where `_update_fn` is compiled)

Steps:
1. Identify compiled update function (`jit`/`pjit`) call that takes `(state, batch, ...)`.
2. Set `donate_argnums` for state-like arguments first (optimizer state, params container).
3. Avoid donating inputs that are reused outside the compiled step unless lifecycle is explicit.
4. Log active donation mode at training start.

Acceptance:
- Functional parity with donation off.
- No buffer-aliasing runtime errors.
- Lower peak memory and/or reduced per-step overhead.

### C3. Validate Donation + Remat Interaction

Run matrix:
- FP32 baseline
- FP32 + donation
- BF16 + donation
- BF16 + donation + remat (best remat level)

Metrics:
- Peak memory per GPU.
- Update-step timing (`local_grad`, `collective`, `optimizer` buckets).
- Time/structure at fixed global batch and at max feasible `batch_per_device`.

Decision rule:
- Keep donation enabled if it is neutral or positive for stability and improves memory/time.

## Integration Sequence (Recommended)

1. Land config + dtype plumbing with behavior identical in FP32.
2. Enable BF16 compute with FP32 optimizer/reduction safety.
3. Enable `donate_argnums` (`state_only`), benchmark memory/time.
4. Benchmark and lock best mixed-precision + donation baseline.
5. Add remat level 1, benchmark memory/time.
6. Add remat level 2 only if level 1 is promising.
7. Re-sweep `batch_per_device` and `microbatch_count` under best setting.

## Risks and Mitigations

- Risk: Numerical drift/instability in BF16.
  - Mitigation: FP32 master params/state, FP32 reductions, strict A/B checks.

- Risk: Remat increases compute too much.
  - Mitigation: Use coarse boundaries first; retain toggle.

- Risk: Aggressive donation causes accidental buffer reuse bugs.
  - Mitigation: Start with state-only donation and expand only after parity checks.

- Risk: Hidden dtype assumptions in external libraries.
  - Mitigation: Add explicit startup logging and assert dtypes in critical nodes.

- Risk: Compile overhead explosion from too many variants.
  - Mitigation: Keep static shapes, fixed microbatch policy, minimal knob combinations per run.

## Deliverables

1. Config knobs + parser support for mixed precision and remat.
2. Model/training dtype policy implementation with safe optimizer path.
3. `donate_argnums` integration on update step with runtime toggles.
4. Remat-enabled Allegro execution path with level control.
5. Benchmark report:
   - FP32 baseline
   - FP32 + donation
   - BF16 + donation best
   - BF16 + donation + remat best
   - Throughput, memory, and stability comparison.
