# Molecular dynamics setup

JAX-MD is the normal validation and rollout path. LAMMPS/MLIR is an optional
connector path for deployment checks. Both use the environment variables
documented in `../env_setup/SETUP_ENV.md`; no launcher should depend on a
particular parent-directory layout.

All stable user-facing MD settings are annotated in
[`configs/example_md.yaml`](../configs/example_md.yaml). Private runtime fields
are deliberately absent; launchers derive them when needed.

## JAX-MD smoke run

Copy the safe short example and edit its paths:

```bash
cd "$CAMEO_CG_PROJECT_ROOT"
mkdir -p local_work/example_md
cp configs/example_md.yaml local_work/example_md/md.yaml
```

The MD YAML references:

- `training_config_path`: training/export config that reconstructs the model
- `params_path`: trained parameter checkpoint
- `dataset_path`: NPZ supplying starting frames, species, masks, and box data
- `output_dir`: trajectory/log destination

Paths may be absolute or repository-relative. The example uses 1 fs steps,
short equilibration/production, center-of-mass removal, initial-temperature
rescaling, continuous output, and conservative stability aborts. It disables
the cell list for its tiny Ala2 smoke system. Do not copy that neighbor-list
choice to large systems without assessing cost.

Submit one trajectory:

```bash
sbatch scripts/submit_md.sh local_work/example_md/md.yaml
```

Run directly in an already allocated GPU shell:

```bash
export CONFIG_FILE="$CAMEO_CG_PROJECT_ROOT/local_work/example_md/md.yaml"
source scripts/slurm_env.sh
"$PYTHON_BIN" scripts/run_md.py "$CONFIG_FILE" local
```

Submit the config's frames as parallel processes on one allocation:

```bash
sbatch scripts/submit_md_parallel.sh local_work/example_md/md.yaml
```

Submit a Slurm array. Replica count comes from `md.n_replicas`; optionally cap concurrency:

```bash
sbatch scripts/submit_md_array.sh local_work/example_md/md.yaml --max_concurrent 3
```

The launchers export `CONFIG_FILE`, load Jupiter modules, resolve the training
config/model type, and select the correct venv through `scripts/slurm_env.sh`.
Set `CAMEO_ACTIVE_VENV` only when deliberately overriding that choice.

## Safety gate before longer MD

Check the smoke run before increasing steps or replicas:

- no stability-abort, NaN, neighbor overflow, or repeated force warning
- temperature reaches and remains near the requested ensemble distribution
- no immediate bond/angle collapse or unphysical bead overlap
- output cadence is sufficient to diagnose the first unstable step
- force and minimum-pair-distance thresholds are appropriate for the system
- 1 fs remains the default for current Ala2 wide160/bb6 tests unless separately
  validated

Persistent or replica simulations should begin from diverse frames rather than
copies of one structure. Extend duration only after the short diagnostic is
clean.

## Analyze a JAX-MD trajectory

```bash
python md/analyze_traj.py \
  --npz local_work/example_md/output/trajectory.npz \
  --outdir local_work/example_md/analysis \
  --method tica --lagtime 10
```

For a LAMMPS dump instead, replace `--npz ...` with `--dump FILE`. See
`python md/analyze_traj.py --help` for stride, pair-feature, reference-model,
and plotting options.

## LAMMPS/MLIR path

First export or re-export an MLIR model (see `../COMMANDS.md`). Convert a
starting frame to a LAMMPS data file:

```bash
python md_setup/lmp_input_gen.py \
  --dataset /path/to/start_frames.npz \
  --frame 0 \
  --output local_work/lammps/start.data
```

Then provide explicit artifacts and submit:

```bash
export CAMEO_LAMMPS_DATA_FILE="$CAMEO_CG_PROJECT_ROOT/local_work/lammps/start.data"
export CAMEO_LAMMPS_MODEL_FILE="$CAMEO_CG_PROJECT_ROOT/local_work/lammps/model.mlir"
export CAMEO_LAMMPS_OUTPUT_DIR="$CAMEO_CG_PROJECT_ROOT/local_work/lammps/run"
sbatch md_setup/submit_lammps_chemtrain.sh
```

Optional environment overrides are `CAMEO_LAMMPS_INPUT_FILE`,
`CAMEO_LAMMPS_TEMPERATURE`, `CAMEO_LAMMPS_TIMESTEP_FS`,
`CAMEO_LAMMPS_RUN_STEPS`, and `CAMEO_LAMMPS_DUMP_FILE`, and `CAMEO_LMP_BIN`. The submitter takes
LAMMPS from `CAMEO_LAMMPS_BUILD_DIR` and refuses to run without explicit data
and model files.

`env_setup/LAMMPS_build.md` and `env_setup/CONNECTOR_REBUILD.md` record the
specialized build/rebuild process. They are not substitutes for the current
Python/Jupiter setup guide.
