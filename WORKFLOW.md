# cameo_cg workflow

This guide explains the normal lifecycle from mapped data to a validated,
enhanced model. Exact invocations are in `COMMANDS.md`; environment creation is
in `env_setup/SETUP_ENV.md`; simulation safety and LAMMPS details are in
`md_setup/README.md`.

## 1. Organize a project

Run shared launchers from the repository root and keep each experiment under
`local_work/`:

```text
local_work/
├── input_data/             # local or linked datasets
├── experiment_name/
│   ├── config.yaml
│   └── notes/
├── outputs/                # launcher-created training/analysis runs
├── md/                     # simulation configs and trajectories
└── sampling/               # generated campaigns and collected frames
```

Commit reusable code and reference configs. Do not commit datasets,
checkpoints, generated campaigns, or personal absolute paths.

## 2. Prepare data

### From GROMACS/HDF5 output

The full pipeline converts source HDF5 systems to NPZ, applies the selected CG
mapping and AggForce force map, pads/combines or buckets systems, and optionally
fits priors. Submit `data_prep/run_pipeline_gpu.sh` with environment variables;
see `COMMANDS.md` for all switches.

Choose the mapping deliberately:

- `1bead`: CA-only representation
- `backbone_cb`: backbone plus side-chain/CB representation

For diverse molecule sizes, prefer bucketed output (`N_BUCKETS` or explicit
`BUCKET_BOUNDARIES`) so padding does not dominate training. `NO_COMBINE=1` is
useful for inspecting per-system results. `SKIP_PRIOR_FITTING=1` separates
mapping/assembly from a later prior fit.

### Dataset contract and checks

A training NPZ normally supplies frame-major coordinates `R`, forces `F`,
`species`, and `mask` (or enough information for the loader to construct the
last two). Keep units and mapping consistent between data, priors, checkpoints,
export, and MD. Do not slice all-atom forces as a substitute for the fitted
AggForce map.

Before training:

```bash
python data_prep/analyze_dataset.py --npz DATASET.npz
```

Check frame/bead shapes, finite coordinates and forces, species identities,
padding masks, force scale, geometry tails, mapping metadata, and train/holdout
provenance. Split related trajectory frames with a guard interval when building
cross-fit teachers to avoid near-duplicate leakage.

### Assemble enhanced datasets

Keep source datasets immutable. Use `build_mixed_training_set.py` when enhanced
sources have priorities/quotas and `build_region_balanced_set.py` when explicit
FES regions need equal coverage. Record source paths, selection rules, seed,
force/geometry caps, and final counts with the generated dataset.

Transition or distorted frames are useful only when their coordinates and
unbiased mapped forces remain physically meaningful. Collection drops broken
PBC frames and can apply a force cap; review the summary rather than treating
all generated frames as training data.

## 3. Configure a model and training run

Copy `configs/base_config.yaml` into the experiment directory. Treat that file
as the current schema reference, then change only the sections relevant to the
run:

- `paths`: output/export/checkpoint locations
- `data`: dataset, standard/tiled batching, bucketing/static neighbors
- `model`: backbone, cutoff, neighbor format, and architecture
- prior/topology sections: representation-consistent prior energies
- `optimizer`: schedules and optimizer parameters
- `training`: stages, batch size, losses, precision, accumulation, mSAM/REM
- `export`: artifact behavior

Use relative paths that resolve from the config directory or repository root.
A copied `config_input.yaml` records intent; `config_runtime.yaml` records the
resolved paths actually used by the job.

## 4. Choose batching and loss semantics

### Standard versus tiled batches

`data.batch_mode: standard` batches padded structures normally. Use it for
small, similarly sized systems and as the correctness baseline.

`data.batch_mode: tiled` packs multiple structures into a bead-budget tile.
`tile_target_beads` controls the approximate tile capacity; bucket settings,
sorting, shuffling, spatial layout, and structure gaps control packing. Tiling
changes how much padding and how many structures each optimizer item contains,
so compare effective structures, valid beads, and force components—not only the
nominal tile count.

Static tiled neighbors can remove repeated graph construction, but are valid
only when coordinates used by the loss match the stored graph. The config
rejects incompatible DSM, noised-residual, REM, and PBC cases. Establish tiled
and static-neighbor equivalence with the supplied diagnostic scripts before a
large run.

### Batch size and accumulation

`training.batch_per_device` is the local device batch. The ordinary global
batch is local batch times global device count. `training.grad_accum_steps`
combines multiple optimizer microsteps (`stack_scan`) before an update, trading
memory/compile behavior against update frequency. When comparing experiments,
record devices, batch per device, tiles/structures per item, accumulation
steps, and optimizer updates per epoch.

Do not tune all of these simultaneously. First obtain a correct single-node
baseline, then change tiling, accumulation, or node count one at a time and run
profiling/equivalence checks.

### Force losses

`training.gammas.F` normally drives force matching; energy loss is commonly off
for CG force matching. `force_loss_normalization` controls padding and molecule
size weighting:

- `legacy_mean`: historical tensor mean; padding can dilute the loss
- `valid_components`: average active force components
- `per_structure_components`: normalize within each structure before averaging

Select the definition that matches the scientific weighting objective, and do
not compare raw loss magnitudes across definitions as if they were identical.
Optional force masks/weights, DSM, HVP, noised-residual, and safety losses must
be backed by correctly generated dataset fields and are opt-in.

## 5. Train and inspect force matching

Submit standard FM or mSAM with the common launcher:

```bash
sbatch scripts/run_training.sh local_work/experiment/config.yaml
```

A healthy run should show the expected Python/JAX path, devices and process
count, dataset split/counts, neighbor capacity, effective batch accounting,
finite losses/gradients, checkpoints, and a final export. Use `--resume auto`
only when the runtime config is compatible with the saved state.

mSAM is a late-stage sharpness-aware update, not a separate model. Enable
`training.msam`, choose its optimizer stage and start epoch/fraction, then set
`rho`. Its micro-batch logic requires the local batch to divide into the
configured microbatch count. Compare it to a matched FM baseline; mSAM does not
replace data coverage or model expressivity.

## 6. Evaluate before MD

Run force/suite analysis on held-out data and inspect more than aggregate MSE:
per-component correlations, molecule/region breakdowns, force tails, prior and
ML residuals, neighbor behavior, and physically important FES regions. Use the
run registry to find the runtime config and artifacts instead of guessing from
folder names.

Failures in rollout despite good average force error often indicate uncovered
regions, poor tail behavior, a prior/mapping mismatch, or an MD integration
issue. Diagnose which layer fails before changing the architecture.

## 7. Run short safe JAX-MD

Copy `configs/example_md.yaml`, point it to the training config, parameters,
and starting dataset, and submit `scripts/submit_md.sh`. Start with diverse
reference frames, 1 fs for current Ala2 bb6 tests, short equilibration and
production, frequent output, and enabled stability aborts.

Only scale to persistent replicas or longer trajectories after inspecting
forces, temperature, minimum distances, geometry, NaNs, and neighbor overflow.
Analyze trajectories with `md/analyze_traj.py`. See `md_setup/README.md` for the
full safety gate and LAMMPS alternative.

## 8. Acquire difficult configurations

The enhanced-sampling architecture generates GROMACS campaigns from YAML and
adds decomposable CG-space biases while retaining all-atom dynamics/force
labels. The main bias families have different roles:

- standard TICA MetaD deposits history-dependent bias and expands explored CV
  space
- regional/static TICA bias favors edges or transition corridors represented
  in a reference grid
- the negative teacher contribution pushes away from structures the current CG
  model already represents comfortably
- local-inversion umbrellas steer a signed-volume chirality coordinate through
  a window ladder to probe otherwise inaccessible inversion barriers
- corridor/harvest campaigns shoot short trajectories from selected edge or
  transition starts

Teacher and TICA biases benefit from a normal MetaD component when the goal is
continued exploration rather than attraction to a fixed map. A local inversion
bias is a diagnostic/acquisition device, not a permanent modification of an
O(3)-equivariant Allegro energy.

Typical loop:

1. select diverse starting frames (farthest-point selection is available)
2. copy and edit a campaign YAML
3. generate cases with `sampling/cases.py`
4. run the generated smoke/pilot jobs
5. inspect per-replica logs/CVs and stability before full submission
6. collect bias-free mapped forces with `sampling/collect.py`
7. validate and balance the harvested frames before training

Collection requires the named CG mapping and recovered AggForce weight matrix
unless using coordinates-only mode. Keep raw trajectories until the aggregate
NPZ and summary have been verified.

## 9. Relative-entropy fine-tuning

REM starts from a competent force-matched checkpoint; it is not the first
training stage. Enable `training.relative_entropy`, configure the checkpoint,
reference and initial-state data, replica count, persistent trajectory/state
behavior, MD safety, sampling cadence, and optimizer, then submit
`scripts/run_relative_entropy.sh`.

Use enough diverse replicas and time for the model distribution to expose its
failure regions. If rare mirror/inversion transitions appear only in long REM,
that is evidence for targeted transition acquisition, not automatically a
reason to add a non-transferable chirality label or break O(3) equivariance.
The transferable strategy is to teach the same scalar energy from unbiased
examples of allowed and disallowed local environments, while targeted biases
help reach the barriers during data generation.

Reject or stop unstable iterations rather than optimizing through broken
structures. Compare the fine-tuned model to its FM checkpoint on held-out force
metrics, FES populations/barriers, chirality diagnostics, and fresh rollouts.

## 10. Export and preserve provenance

Export ML-only or combined ML+prior energy according to the deployment target.
Keep together:

- parameters/checkpoint
- input and resolved config
- mapping/species/topology definition
- fitted prior artifacts
- dataset selection manifest
- JAX/package snapshot and git revision
- evaluation and MD safety diagnostics

Use `export/reexport_mlir.py` for post-hoc MLIR reconstruction. Validate an
export on known frames before using it in LAMMPS.

## 11. Scale only after the small proof passes

Before requesting a larger server campaign, require:

- preprocessing and mapping checks pass
- standard/tiled or static-neighbor equivalence is established if used
- a short training job produces finite learning and a loadable checkpoint
- held-out force and region diagnostics are credible
- a short MD smoke run is stable
- enhanced-sampling pilot replicas move the intended CV without widespread
  LINCS/PBC/force-cap failures
- collection retains unbiased forces and reproducible provenance

Then increase one dimension at a time: frames, model width, devices/nodes,
replicas, or simulation length. This keeps performance changes attributable
and makes failures cheap to localize.
