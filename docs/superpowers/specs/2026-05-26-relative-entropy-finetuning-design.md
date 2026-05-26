# Relative-Entropy Fine-Tuning V1 Design

## Context

`cameo_cg` currently trains coarse-grained potentials through a Chemtrain
force-matching path in `scripts/train.py` and `training/trainer.py`. That path
is built around static reference batches, with optional DSM, safety
regularization, prior-residual targets, tiled batches, and Chemtrain
checkpointing.

Relative-entropy (RE) fine-tuning needs a different training signal:

```text
grad S_rel = beta * (E_ref[grad_theta U_theta] - E_model[grad_theta U_theta])
```

The reference expectation comes from mapped atomistic CG frames. The model
expectation comes from short rollouts under the current CG model. Because this
requires parameter-dependent sampling, RE fine-tuning will be implemented as a
separate testing-phase entry point rather than as a new `Trainer.train_stage`
mode.

## Scope

V1 implements RE fine-tuning for a warm-started model that is already stable
enough for short CG rollouts. It focuses on ML-residual updates only and keeps
prior parameters frozen. The implementation must not alter existing force
matching behavior.

Out of scope for V1:

- mean-force matching,
- reweighting / effective sample size reuse,
- enhanced sampling,
- tiled-batch RE,
- multi-host distributed RE,
- adaptive trust-region rollback beyond basic update rejection,
- smoothing/Hessian regularization.

## Files

Add:

- `scripts/train_relative_entropy.py`: RE entry point.
- `training/relative_entropy.py`: RE config parsing, sampler, loss/gradient
  update, diagnostics, and checkpoint helpers.

Update:

- `configs/base_config.yaml`: document the optional
  `training.relative_entropy` block.
- `config/manager.py`: small accessors for RE config values, following the
  existing config style.

No changes are planned for the current Chemtrain force-matching path except
shared helper reuse if a small extraction is required.

## Configuration

The RE script reads the same YAML config type as `scripts/train.py`.

Required for RE:

- `training.relative_entropy.enabled: true`
- `training.init_from_checkpoint.enabled: true`
- `training.init_from_checkpoint.path: <checkpoint.pkl>`

Recommended config block:

```yaml
training:
  relative_entropy:
    enabled: false
    reference_data_path: null
    optimizer: adam
    iterations: 100
    reference_batch_size: 16
    n_replicas: 8
    steps_per_iteration: 200
    burn_in_steps: 50
    sample_stride: 10
    dt: 0.02045
    kT: 0.636
    gamma: 0.000977
    mass: 12.011
    start_frame_mode: reference_random
    checkpoint_freq: 10
    max_force: 1.0e4
    min_pair_distance: 1.5
    reject_on_instability: true
```

`reference_data_path` defaults to `data.path`. If set, it loads the reference
ensemble from a separate mapped-AA dataset while retaining model architecture
and default preprocessing from the main config.

## Data Flow

1. Load `ConfigManager`.
2. Resolve reference dataset from `training.relative_entropy.reference_data_path`
   or `data.path`.
3. Apply the same coordinate preprocessing conventions as `scripts/train.py`:
   PBC uses dataset box and wrapping; non-PBC uses `CoordinatePreprocessor`
   centering and parking.
4. Build `CombinedModel` with the same model config, species cardinality,
   `id_to_aa`, box, and initial mask conventions as force matching.
5. Load warm-start params from the configured checkpoint. Accepted checkpoint
   formats match existing training checkpoints: `params`, `best_params`, or
   Chemtrain `trainer_state.params`.
6. Initialize model replicas from sampled reference frames.
7. Per RE iteration:
   - sample a reference batch,
   - run short in-process JAX-MD Langevin rollouts under current params,
   - discard burn-in and stride model samples,
   - compute the RE ML-parameter gradient,
   - apply an Optax update to `params["ml"]` only,
   - run diagnostics and save periodic checkpoints.

## Model And Gradient

The RE objective updates only the ML parameter subtree:

```text
params = {"ml": ..., "prior": ...}
```

`params["prior"]`, if present, is copied through unchanged. This keeps the
physics prior as the stable support and avoids RE weakening bonded or
excluded-volume terms.

The gradient is computed from total model energy with respect to ML parameters:

```text
g_ml = beta * (mean grad_ml U_total(R_ref) - mean grad_ml U_total(R_model))
```

If priors are active, `U_total = U_prior + gate * U_ml`, but only the ML subtree
receives updates. This preserves robustness-gate behavior and any configured
ML/prior energy scales.

The sampler treats generated positions as samples from the current model
distribution. Gradients are not differentiated through MD integration steps.
Model samples are stop-gradient coordinates for the RE expectation term.

## In-Process Sampler

The sampler uses JAX-MD Langevin dynamics in-process. It should be leaner than
`MDRunner` but follow its physical conventions:

- AKMA units,
- `dt`, `kT`, `gamma`, `mass`,
- optional per-species mass support if practical in V1,
- zero center-of-mass velocity by default,
- masks respected for padded atoms,
- same model `energy_fn_template`.

Replica initialization uses reference frames by default. Early defaults should
favor many short rollouts over one long rollout.

## Diagnostics And Safety

Each iteration logs:

- RE reference energy mean,
- RE model energy mean,
- gradient norm,
- update norm and update/parameter norm,
- max force,
- min valid bead pair distance,
- NaN/Inf counts,
- simple temperature/energy summaries from rollouts,
- number of retained model samples.

If `reject_on_instability` is enabled, updates are rejected when diagnostics
show NaN/Inf, max force above `max_force`, or min valid pair distance below
`min_pair_distance`. Rejection restores the previous ML params and optimizer
state for that iteration.

## Checkpointing

RE checkpoints use the existing top-level style:

```python
{
    "params": params,
    "best_params": best_params,
    "metadata": {...}
}
```

Metadata records RE iteration, config path, checkpoint source, diagnostics
history, and whether prior params were frozen. This keeps downstream export and
MD scripts compatible with the existing checkpoint extraction logic.

## Error Handling

The RE script should fail early with clear messages when:

- `training.relative_entropy.enabled` is false,
- no warm-start checkpoint is configured,
- reference data shape/species are incompatible with model initialization,
- `data.batch_mode: tiled` is requested for RE V1,
- PBC is requested but the reference dataset has no box,
- rollout settings would retain zero model samples,
- optimizer name is unknown.

## Testing

Initial tests should cover:

- RE config defaults and validation.
- Checkpoint parameter extraction for existing checkpoint formats.
- ML-only update behavior: prior subtree unchanged after an update.
- RE gradient sign on a small analytic or mocked-energy example.
- Sampler shape and diagnostic checks on a tiny masked system.

Full production validation remains physical rather than only numeric:

- compare reference and model distributions,
- check prior-only and force-matched baselines,
- inspect bond/angle/torsion/contact/FES diagnostics,
- run longer post-RE trajectories for stability.

## Open Decisions

No blocking decisions remain for V1. Later versions can add reweighting,
stronger rollback/trust-region logic, MFM handoff integration, and merging RE
as a first-class `training.stages` type if the isolated script proves useful.
