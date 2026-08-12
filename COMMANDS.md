# Command reference

Run these commands from `$CAMEO_CG_PROJECT_ROOT`. Replace placeholders and
inspect `--help` before using uncommon options. Environment setup lives in
`env_setup/SETUP_ENV.md`; workflow explanations live in `WORKFLOW.md`.

Choose a configuration by its canonical role:

- `configs/example_training.yaml` is the first-FM starter; shorten its
  production-scale optimizer schedule when preparing a smoke run.
- `configs/base_config.yaml` is the complete stable annotated training lookup,
  not a ready-to-run experiment.
- `configs/example_md.yaml` is the stable annotated MD lookup and safe smoke
  template.

## Shell setup and checks

```bash
source ~/.bashrc
cd "$CAMEO_CG_PROJECT_ROOT"
source env_setup/load_modules_2026.sh
source "$CAMEO_STANDARD_VENV/bin/activate"
python -c "import jax; print(jax.__version__, jax.__file__, jax.devices())"
bash scripts/configure_user_env.sh
```

Override environment selection for one submission:

```bash
CAMEO_ACTIVE_VENV=/path/to/venv sbatch scripts/run_training.sh CONFIG.yaml
```

## Training: force matching and mSAM

Create a first-FM config from the starter:

```bash
mkdir -p local_work/my_run
cp configs/example_training.yaml local_work/my_run/config.yaml
sbatch scripts/run_training.sh local_work/my_run/config.yaml
```

The YAML selects standard force matching or mSAM and defines batch size,
tiling, gradient accumulation, optimizer stages, losses, priors, and model.
Use the same launcher for either method. Consult `configs/base_config.yaml`
before enabling additional stable settings or methods.

```bash
# Two nodes
sbatch --nodes=2 scripts/run_training.sh local_work/my_run/config.yaml

# Resume the newest compatible checkpoint
sbatch scripts/run_training.sh local_work/my_run/config.yaml --resume auto

# Resume an explicit checkpoint
sbatch scripts/run_training.sh local_work/my_run/config.yaml \
  --resume local_work/outputs/RUN/checkpoints/CHECKPOINT.pkl

# Explicit bucketed multi-protein directory
sbatch scripts/run_training.sh local_work/my_run/config.yaml \
  --multi-protein-dir data_prep/datasets/buckets

# Profile the config
sbatch scripts/run_profiling.sh local_work/my_run/config.yaml
```

Submit every YAML in a directory as a bounded Slurm array:

```bash
bash scripts/submit_suite.sh \
  --input_dir local_work/my_suite/configs \
  --name my_suite --max_concurrent 4 --nodes 1 --time 10:00:00
```

## Relative entropy / REM fine-tuning

Configure `training.relative_entropy` and checkpoint initialization in the
YAML, then submit:

```bash
sbatch scripts/run_relative_entropy.sh local_work/my_rem/config.yaml
```

Use persistent trajectories, multiple diverse initial frames/replicas, and a
short diagnostic stage before a long REM run. The training config controls
simulation length, state reuse, clipping/stability behavior, and optimizer.

## Direct-force teacher and ensemble preparation

Build guarded cross-fit folds:

```bash
python scripts/build_crossfit_manifest.py DATASET.npz MANIFEST.json \
  --n-folds 5 --guard-frames 10 --group-key trajectory_id
```

Materialize an ensemble teacher locally:

```bash
python scripts/materialize_direct_force_teacher.py \
  DATASET.npz MANIFEST.json ENSEMBLE_SPEC.yaml TEACHER.npz --batch-size 32
```

Or submit materialization:

```bash
sbatch scripts/submit_teacher_materialization.sh \
  DATASET.npz MANIFEST.json ENSEMBLE_SPEC.yaml TEACHER.npz 32
```

## Data preprocessing

Run the complete HDF5-to-CG pipeline on Jupiter:

```bash
H5_DIR=/path/to/h5_inputs \
OUT_DIR=local_work/data/pipeline_run \
NFRAMES=2500 TEMP_GROUPS='298 320' PRIOR_FIT_T=320 \
MAPPING=backbone_cb N_BUCKETS=4 \
sbatch data_prep/run_pipeline_gpu.sh
```

Useful environment switches are:

```bash
# Exact size boundaries instead of N_BUCKETS
BUCKET_BOUNDARIES='64 128 256'
# Keep per-system NPZs and do not combine
NO_COMBINE=1
# Skip the prior-fitting stage
SKIP_PRIOR_FITTING=1
# Optional fitting/mapping switches
ENABLE_SPLINE=1 RESIDUE_SPECIFIC_ANGLES=1 NORMALIZE_FORCES=1 \
USE_4WAY_GROUPING=1 VERBOSE=1
```

`N_BUCKETS`, `BUCKET_BOUNDARIES`, and `NO_COMBINE=1` are mutually constrained;
the wrapper validates incompatible combinations. Run the pipeline directly in
an allocated environment when debugging:

```bash
python data_prep/run_pipeline.py --h5_dir INPUT --out_dir OUTPUT \
  --nframes 2500 --temp 298 320 --T 320 --mapping backbone_cb --n_buckets 4
```

Individual operations:

```bash
python data_prep/cg_1bead.py --infile AA.npz --outfile CA_CG.npz --verbose
python data_prep/cg_backbone_cb.py --help
python data_prep/pad_and_combine_datasets.py --help
python data_prep/prior_fitting_script.py --help
python data_prep/analyze_dataset.py --npz DATASET.npz
```

## Enhanced/region-balanced dataset assembly

Mix reference and prioritized enhanced sources:

```bash
python -m data_prep.build_mixed_training_set \
  --reference REFERENCE.npz --n-reference 100000 \
  --enhanced inversion.npz:25000:priority --enhanced tica.npz:25000 \
  --mapping ala2_backbone_cb_6 --chi-window 0.15 --enhanced-basin-frac 0.25 \
  --seed 17 --out local_work/data/mixed.npz
```

Build a fixed-size region-balanced set (repeat `--source` and `--region`):

```bash
python -m data_prep.build_region_balanced_set \
  --source reference=REFERENCE.npz --source enhanced=ENHANCED.npz \
  --region alpha=-120:-20:-100:40 \
  --region mirror=20:120:-40:100 \
  --n-per-region 10000 --mapping ala2_backbone_cb_6 --max-bond 6.0 \
  --seed 17 --out local_work/data/region_balanced.npz
```

Precompute special targets/features only when enabled by the training config:

```bash
python data_prep/precompute_hvp_targets.py --help
python data_prep/precompute_edge_distance_gate.py --help
python data_prep/noise_decoy_frames.py --help
```

## Model evaluation and run analysis

```bash
# Full run/suite analysis on Slurm
sbatch scripts/run_analysis.sh --input-dir local_work/outputs/RUN \
  --name RUN_analysis --detailed-force-eval --complete-eval \
  --detailed-batch-size 8 --complete-eval-batch-size 4

# Direct force check
python analysis_tests/evaluate_forces.py PARAMS.pkl CONFIG.yaml --frames 50

# Direct suite analysis in an allocated environment
python analysis_tests/analyze_suite.py local_work/outputs/RUN \
  --detailed-force-eval
```

Equivalence and regression diagnostics:

```bash
python analysis_tests/check_tiled_equivalence.py --help
python analysis_tests/check_static_neighbor_equivalence.py --help
python analysis_tests/check_prior_residual_equivalence.py --help
python -m pytest -q tests
```

## MLIR export

```bash
python export/reexport_mlir.py PARAMS.pkl CONFIG.yaml \
  --mode combined --prior-source config --output-dir local_work/export \
  --output-name model_with_priors --export-mode symbolic

sbatch export/run_reexport.sh PARAMS.pkl CONFIG.yaml \
  --mode combined --prior-source config --output-name model_with_priors
```

The wrapper defaults to ML-only export unless extra arguments request combined
energy/priors. Always retain the config used to reconstruct species, topology,
and priors.

## JAX-MD

Start from the short safe example:

```bash
mkdir -p local_work/my_md
cp configs/example_md.yaml local_work/my_md/md.yaml
sbatch scripts/submit_md.sh local_work/my_md/md.yaml
sbatch scripts/submit_md_parallel.sh local_work/my_md/md.yaml
sbatch scripts/submit_md_array.sh local_work/my_md/md.yaml --max_concurrent 3
```

Direct execution in an allocated shell:

```bash
export CONFIG_FILE="$CAMEO_CG_PROJECT_ROOT/local_work/my_md/md.yaml"
source scripts/slurm_env.sh
"$PYTHON_BIN" scripts/run_md.py "$CONFIG_FILE" local
```

Analyze NPZ output or a LAMMPS dump:

```bash
python md/analyze_traj.py --npz TRAJECTORY.npz --outdir ANALYSIS_DIR \
  --method tica --lagtime 10
python md/analyze_traj.py --dump TRAJECTORY.dump --outdir ANALYSIS_DIR
```

See `md_setup/README.md` before extending duration or using LAMMPS.

## Enhanced-sampling campaigns

Generate a campaign from an existing YAML:

```bash
python -m sampling.cases \
  --config sampling/campaigns/ala2_bb6_teacher_tica_pilot.yaml \
  --output-dir local_work/sampling/teacher_tica_pilot
```

Other maintained examples cover standard TICA MetaD, teacher-only,
TICA-only, teacher+TICA, local inversion smoke/window ladders, and corridor
shooting under `sampling/campaigns/`. Inspect and copy a YAML; do not edit the
shared campaign in place.

Collect a complete campaign:

```bash
python -m sampling.collect \
  --campaign local_work/sampling/teacher_tica_pilot \
  --mapping ala2_backbone_cb_6 \
  --weights local_work/input_data/ala2_bb6_aggforce_weight_matrix.npz \
  --discard-ps 2.0 --force-cap 10000 \
  --out local_work/data/teacher_tica_collected.npz
```

Collect one case by replacing `--campaign` with `--case`. Add `--coords-only`
when forces are intentionally absent; use `--delete-trr` only after verifying
the collected output.

Build bias inputs/diagnostics:

```bash
python -m sampling.build_transition_map --grid-dir GRID_DIR \
  --reference REFERENCE.npz --enhanced tica=ENHANCED.npz \
  --outdir local_work/sampling/transition_map
python -m sampling.build_reference_bias --grid-dir GRID_DIR \
  --reference REFERENCE.npz --temperature 298 --mode log_ratio \
  --out local_work/sampling/reference_bias.npz
python -m sampling.pick_start_frames --help
python -m sampling.plot_bias_landscape --help
```

The campaign generator normally starts the socket server and writes the
GROMACS case files. `sampling/server.py` is primarily a generated-campaign
runtime component, not the first user entry point.

## Run registry and monitoring

```bash
python3 runs/registry.py sync
python3 runs/registry.py status
python3 runs/registry.py show RUN_ID
python3 runs/registry.py render

squeue -u "$USER"
scontrol show job JOB_ID
scancel JOB_ID
tail -f local_work/outputs/RUN/slurm-JOB_ID.out
```
