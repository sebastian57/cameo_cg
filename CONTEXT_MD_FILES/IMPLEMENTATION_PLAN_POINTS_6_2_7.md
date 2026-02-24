# Implementation Plan: Points 6 -> 2 -> 7

**Date:** 2026-02-20  
**Scope:** Chemtrain performance/scaling work for large multi-node runs  
**Priority order:**  
1. Point 6: Cache sharded `params` / `opt_state` (remove repeated `device_put`)  
2. Point 2: Parallelize DiffTRe statepoint updates (replace sequential loop)  
3. Point 7: Async validation overlap with next-epoch training/data loading
4. Additional concern: Replace `stop_gradient` masking copy in `combined_model.py` with a custom VJP boundary

---

## Goals

1. Minimize per-step host overhead and host-device synchronization.
2. Reduce communication pressure and improve scaling toward many nodes.
3. Keep training semantics stable (or make semantic changes explicit and configurable).

---

## Baseline (Current State)

1. MLE path currently performs repeated `device_put(params, replicate)` and `device_put(opt_state, replicate)` in the update function.
2. DiffTRe main trainer path still processes statepoints in Python loops (per-statepoint calls).
3. Validation/evaluation runs synchronously in post-epoch flow.
4. New gradient-accumulation support (`CHEMTRAIN_GRAD_ACCUM_STEPS`) is available for the shmap MLE path and reduces collective frequency.

---

## Phase 1 (Point 6): Cache Sharded `params` / `opt_state`

### Objective

Remove repeated per-step state sharding/transfer for parameters and optimizer state.

### Design

1. Introduce a small device-cache layer in trainer state:
   1. `params_dev`, `opt_state_dev`, `cache_valid` (or equivalent).
2. Only refresh cache when host-side params/opt_state change outside normal update flow:
   1. checkpoint load
   2. explicit setter
   3. potential external mutation paths
3. Keep batch `device_put` in place (input data changes every step), but avoid repeated state `device_put`.
4. Ensure `update_fn` receives already-sharded state and returns same layout.

### Steps

1. Add cached-device-state fields and invalidation hooks in trainer.
2. Refactor shmap update entry to skip state `device_put` when cache valid.
3. Add a feature flag for fast rollback:
   1. `CHEMTRAIN_CACHE_SHARDED_STATE=1` (default off initially).
4. Validate correctness by comparing:
   1. losses
   2. grad norms
   3. parameter deltas over fixed short runs.

### Acceptance Criteria

1. `put_state_ms` in `[UpdateFnInternal]` drops near zero in steady state.
2. No change in training outputs beyond expected numeric noise.
3. No memory leak over long run.

### Risks

1. Stale cached state after checkpoint/load or manual state mutation.
2. Hidden code paths replacing host params directly.

### Mitigation

1. Centralized invalidation helper.
2. Explicit cache reset in all parameter-setting and checkpoint-loading paths.

---

## Phase 2 (Point 2): Parallelize DiffTRe Across Statepoints

### Objective

Replace sequential per-statepoint loops with batched/device-parallel execution.

### Design Direction

1. Use batched DiffTRe path (`DifftreParallel` pattern) as reference and converge to one performant path.
2. Batch trajectory states, targets, and statepoint metadata so gradient/loss are computed in vectorized form.
3. Keep recompute logic batched where possible (or grouped by mask).
4. Minimize per-statepoint Python `print`/host conversions in hot path.

### Steps

1. Audit current DiffTRe and RelativeEntropy `_update` loops:
   1. identify statepoint-wise operations that can be vectorized
   2. identify unavoidable scalar/control-flow steps.
2. Implement batched gradient/loss computation:
   1. batch statepoint inputs
   2. run one compiled function over batch
   3. aggregate outputs on device.
3. Move step-size decision to batched logic where feasible:
   1. compute candidate alphas per statepoint in one pass
   2. reduce to chosen global alpha per update.
4. Keep fallback mode:
   1. `CHEMTRAIN_DIFFTRE_SEQUENTIAL=1` for rollback/testing.

### Acceptance Criteria

1. Significant drop in host-side loop time per update.
2. Fewer dispatches and fewer tiny collectives.
3. Same convergence trend on fixed benchmark.

### Risks

1. Behavior differences if per-statepoint control flow diverges.
2. Memory pressure for large batched trajectory payloads.

### Mitigation

1. Support microbatching over statepoints.
2. Add batched-path equivalence tests against sequential for small deterministic cases.

---

## Phase 3 (Point 7): Async Validation Overlap

### Objective

Avoid blocking training progression on validation when possible.

### Design

1. Snapshot params at epoch boundary (`params_eval = stop_gradient/copy` semantics).
2. Launch validation with that snapshot in a background worker (thread/process, rank0 only).
3. Start next epoch training immediately.
4. Consume validation result later with ordering guarantees before:
   1. early-stopping decision
   2. checkpoint selection marked "best".

### Steps

1. Add async validation controller:
   1. start job
   2. poll status
   3. collect result
   4. handle exceptions.
2. Gate behavior with flag:
   1. `CHEMTRAIN_ASYNC_VALIDATION=1` (default off initially).
3. Define semantics clearly:
   1. training at epoch N can overlap validation of epoch N-1 snapshot.
4. Ensure deterministic fallback to synchronous mode.

### Acceptance Criteria

1. Validation wall-time overlap visible in logs/traces.
2. Correct early stopping behavior preserved.
3. No race conditions on shared trainer state.

### Risks

1. Metric lag complicates early-stop/checkpoint logic.
2. Additional host memory use for snapshots.

### Mitigation

1. Strict snapshot/result ownership.
2. Queue depth of 1 (single outstanding validation job).

---

## Cross-Cutting Profiling and Verification

For each phase, collect:

1. Per-update timing (`UpdateFnInternal`, `UpdateBreakdown`).
2. Collective profile (NCCL volume and count).
3. Throughput metrics (updates/s, epoch wall time).
4. Convergence sanity (loss curves against baseline).

Suggested benchmark matrix:

1. 1 node x 4 GPUs
2. 2 nodes x 8 GPUs
3. larger-node test after regression checks pass

---

## Execution Order and Milestones

1. Milestone A (Point 6):
   1. implement cache + invalidation + flag
   2. verify no-regression
   3. profile delta.
2. Milestone B (Point 2):
   1. batched DiffTRe update path
   2. equivalence tests
   3. profile scaling delta.
3. Milestone C (Point 7):
   1. async validation controller
   2. correctness guards
   3. overlap benchmark.

---

## Expected Impact (Qualitative)

1. Point 6 should reduce avoidable host-device state-transfer overhead immediately.
2. Point 2 is the largest structural gain for large node counts (reduces Python orchestration and improves device utilization).
3. Point 7 improves pipeline utilization and wall-time efficiency, especially when validation is non-trivial.

---

## Additional Optimization Track: Validation Every N Epochs

### Objective

Avoid running the full validation forward pass every epoch. The validation pass (`_evaluate_convergence`) costs ~31 s/epoch (8% of epoch wall time on 2 nodes), but the full frequency is only needed near the end of training for early stopping decisions.

### Design

1. Add a `val_freq` parameter to the trainer (e.g. `val_every: 5` in config).
2. Wrap the `_evaluate_convergence` post-epoch task to skip when `epoch % val_freq != 0`.
3. When skipped: reuse the last known val loss for logging and early-stop checks (or simply suppress the print and not update the early-stop window).
4. Gate with a config key so default behaviour is unchanged (`val_every: 1`).

### Steps

1. Add `val_every` to trainer init (read from config).
2. In `base.py` or the ForceMatching subclass, gate the `_evaluate_convergence` body on `self._epoch % self.val_every == 0`.
3. Update `run_training.sh` / config to expose the parameter.
4. Verify: early stopping still fires correctly when validation runs every N epochs.

### Acceptance Criteria

1. With `val_every=5`, validation runs only on epochs 0, 5, 10, ...; intermediate epochs show no validation cost in task timing.
2. No change in training or validation loss values on epochs where validation does run.
3. Early stopping correctness preserved.

### Expected Impact

~31 s × (1 - 1/N) saved per epoch. At N=5 on the current 2-node setup (~364 s/epoch): saves ~25 s/epoch (~7% speedup). More impactful on 1 node where validation is a larger fraction of the (faster) epoch.

---

## Additional Optimization Track: Mixed Precision (bfloat16)

### Objective

Halve the bytes transferred during NCCL all-reduce by computing and reducing gradients in bfloat16 instead of float32. The gradient tensor dominates cross-node IB communication cost.

### Background

On 2 nodes, `block_loss_ms` is ~37 s/step, of which ~98% is NCCL all-reduce time (confirmed by profiling). The all-reduce transfers the full gradient tensor (float32, 4 bytes/param) across InfiniBand. bfloat16 reduces this to 2 bytes/param → up to 2× reduction in NCCL transfer time, targeting ~18 s/step.

### Design

1. Cast model parameters and activations to bfloat16 for forward/backward passes.
2. Keep optimizer state in float32 (standard mixed-precision practice — avoids optimizer instability).
3. Accumulate gradients in float32 before the all-reduce, or reduce in bfloat16 and upcast before optimizer step.
4. Use JAX's `jax.lax.convert_element_type` or `flax.linen` dtype policies.

### Risks

1. Allegro uses spherical harmonic features and tensor products — precision loss could degrade model accuracy.
2. bfloat16 has limited dynamic range; gradient norms of ~20 (seen in training) are safe, but occasional spikes (seen in AdaBelief run up to 24000) could overflow.
3. chemtrain/chemutils may not yet support end-to-end bfloat16 training.

### Steps

1. Audit chemutils Allegro implementation for dtype assumptions.
2. Add a dtype config flag (`use_bfloat16: false` default).
3. Run short convergence test and compare loss curves against float32 baseline.
4. Profile `block_loss_ms` before and after.

### Acceptance Criteria

1. `block_loss_ms` drops by ≥30% (not necessarily full 2× due to non-NCCL compute still in float32).
2. Loss curves agree with float32 within reasonable tolerance over 10 epochs.
3. No NaN/overflow events.

---

## Additional Optimization Track: `stop_gradient` Masking Copy

### Objective

Reduce overhead from repeated `R_detached = stop_gradient(R)` + `jnp.where(...)` masking in `combined_model.py` while preserving gradient blocking for padded atoms.

### Current Observation

1. Current pattern appears in multiple paths (e.g. prior-only and prior-enabled energy paths).
2. It is functionally correct, but may create extra array movement/transform overhead in compiled graphs.

### Proposed Direction

1. Introduce a custom VJP wrapper around masked coordinate preparation:
   1. Forward pass keeps numerically safe masked coordinates.
   2. Backward pass explicitly zeros gradients on invalid atom entries.
2. Replace repeated local `stop_gradient`/`where` snippets with one reusable helper.

### Steps

1. Add helper in model layer (e.g. `masked_coords_with_blocked_grad`).
2. Implement custom VJP:
   1. forward returns masked coordinates
   2. backward multiplies cotangent by `mask[:, None]`.
3. Swap call sites in `combined_model.py`.
4. Verify equality:
   1. energies match baseline
   2. gradients are zero on padded atoms
   3. training loss trajectory unchanged within tolerance.

### Acceptance Criteria

1. No semantic change in force/energy outputs.
2. Invalid/padded atom gradients remain strictly blocked.
3. Profiling shows no regression and ideally lower overhead in prior-enabled paths.
