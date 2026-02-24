# Unified Optimization Plan (Training + Export Compatible)

Date: 2026-02-23  
Scope: `cameo_cg` force-matching training stack on multi-node JAX runs

## Why This Order

Latest profiling shows optimization priority should be:
1. Local model compute path (dominant).
2. Memory headroom to increase `batch_per_device`.
3. Host-side non-kernel overhead.
4. Lower-priority structural work (only if relevant for active workflow).

From `slurm-13303518.out` steady steps:
- `local_grad_total_ms` is dominant.
- `collective_total_ms` is comparatively small.
- `optimizer_total_ms` is negligible.
- Large inter-batch gap remains outside `_update_fn` and should be treated as a separate host-pipeline track.

---

## Phase 0: Guardrails and Baseline Lock (Mandatory)

## Objective
Prevent regressions and keep training/export behavior stable while optimizing.

## Actions
1. Freeze a baseline config for A/B:
   - 2 nodes, fixed seed, fixed data slice, static shapes.
2. Add one compatibility check command bundle:
   - short train run
   - checkpoint save
   - MLIR export
   - one inference sanity check from exported artifacts.
3. Keep all new optimization features behind config/env flags, default-off unless explicitly enabled.

## Acceptance
1. FP32 baseline reproduces current loss/grad behavior.
2. Export succeeds with unchanged artifact format and inference result tolerance.

---

## Phase 1: Highest ROI, Lowest Risk

## 1A. Mixed Precision (BF16 Compute, FP32 Master/Reduce)

## Objective
Reduce execution cost and memory bandwidth while preserving optimizer stability.

## Actions
1. Add dtype policy knobs:
   - `training.compute_dtype`: `float32|bfloat16`
   - `training.param_dtype`: `float32`
   - `training.reduce_dtype`: `float32`
2. Thread dtype policy through model wrappers (`allegro_model`, `combined_model`), keeping index tensors integer.
3. Keep optimizer state and parameter master copy in FP32.
4. Ensure gradient path upcasts to FP32 before optimizer update if needed.
5. Log active dtype policy at startup.

## Acceptance
1. No NaN/Inf or instability in short and medium checks.
2. Export path remains functional and numerically close to FP32 baseline.
3. Improved time/structure and/or memory headroom.

## 1B. Buffer Donation (`donate_argnums`, state-only first)

## Objective
Lower memory pressure and copy overhead in update step.

## Actions
1. Add:
   - `training.enable_buffer_donation`
   - `training.donate_mode: state_only|state_and_batch`
2. Apply donation only on compiled training update entrypoint first (`state_only`).
3. Keep strict lifecycle discipline for donated inputs.

## Acceptance
1. No aliasing/runtime errors.
2. Equal training trajectory within tolerance.
3. Neutral-to-positive time/structure and/or memory use.

---

## Phase 2: Memory-for-Throughput Tradeoff

## 2A. Activation Checkpointing (`jax.remat`)

## Objective
Reduce activation memory to enable larger `batch_per_device` and better throughput.

## Actions
1. Add knobs:
   - `training.remat_level: 0|1|2`
   - `training.remat_policy` (optional mapping)
2. Start with coarse remat boundaries in heavy Allegro blocks.
3. Keep neighbor-list/index orchestration outside remat initially.
4. Run memory/throughput sweep:
   - remat off vs level1 vs level2
   - then increase `batch_per_device` to limit.

## Acceptance
1. Identical semantics within tolerance.
2. Best chosen setting improves end-to-end time/structure at target scale.

---

## Phase 3: Host/Loop Overhead (Separate Track)

## Objective
Address non-kernel overhead seen as inter-batch gap outside `_update_fn`.

## Actions
1. Data/loop overhead profiling pass with minimal model instrumentation.
2. Prioritize:
   - data pipeline prefetch/dequeue overlap
   - Python-loop overhead in training step orchestration
   - optional validation frequency reduction (`val_every`)
3. Keep async validation as optional follow-up if validation remains a measurable wall-time fraction.

## Acceptance
1. Inter-batch gap drops materially without changing training semantics.
2. Throughput improves at same model config.

---

## Phase 4: Structural Work (Conditional)

## 4A. DiffTRe Statepoint Parallelization (Point 2 from prior plan)

Only execute this phase if DiffTRe/RE workflows are active bottlenecks for the target workload.

## 4B. Point 6 (state device cache) and Point 7 (async validation)

Treat as secondary:
1. Point 6 currently appears low-impact in current FM runs (steady `put_state_ms` already tiny).
2. Point 7 can still help wall time if validation share is high enough.

---

## Explicit De-Prioritization for Now

1. Custom Allegro kernels / deep source rewrites before completing Phases 1-3.
2. Fine-grained micro-optimizations with unclear measured impact.

These remain valid long-term options, but not first implementation targets given current profiling evidence.

---

## Compatibility Contract (Must Hold for Every Phase)

1. Default behavior unchanged unless config flag enabled.
2. Short-train parity check passes.
3. Export path (`.mlir` + params) succeeds after training with feature enabled.
4. One exported inference comparison stays within numeric tolerance against baseline.

---

## Implementation Order (Final)

1. Phase 0 guardrails and fixed A/B harness.
2. Phase 1A mixed precision.
3. Phase 1B buffer donation (`state_only`).
4. Phase 2 remat level 1, then level 2 only if justified.
5. Re-sweep `batch_per_device` under best Phase 1/2 settings.
6. Phase 3 host-overhead work (inter-batch gap reduction).
7. Phase 4 conditional structural items (DiffTRe/async validation) as needed.
